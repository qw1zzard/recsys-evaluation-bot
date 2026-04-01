import asyncio
import time

from aiogram import Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
)

from src.core.config import settings
from src.services.data_loader import load_ground_truth, load_test_users
from src.services.metrics import (
    calculate_precision,
    calculate_recall,
    calculate_success_rate,
    get_latency_percentiles,
)
from src.services.sheets import write_evaluation_result
from src.services.tester import run_load_test, sanity_check
from src.states.registration import EvaluationFSM

router = Router()


def get_models_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text='dssm_model_with_popular')],
            [KeyboardButton(text='als_ann_with_features_model')],
            [KeyboardButton(text='knn_tfidf_model_with_popular')],
            [KeyboardButton(text='baseline_first_10_items')],
        ],
        resize_keyboard=True,
        one_time_keyboard=True,
    )


def get_rps_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text='Easy (5 RPS)')],
            [KeyboardButton(text='Medium (15 RPS)')],
            [KeyboardButton(text='Hard (50 RPS)')],
        ],
        resize_keyboard=True,
        one_time_keyboard=True,
    )


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()
    await message.answer(
        'Привет! Я бот для оценки рекомендательных систем.\n'
        'Давай проведем тест твоего сервиса.\n\n'
        'Для начала введи свои Имя и Фамилию (или никнейм):',
        reply_markup=ReplyKeyboardRemove(),
    )
    await state.set_state(EvaluationFSM.waiting_for_name)


@router.message(Command('cancel'))
async def cmd_cancel(message: Message, state: FSMContext):
    await state.clear()
    await message.answer(
        'Действие отменено. Введите /start чтобы начать заново.',
        reply_markup=ReplyKeyboardRemove(),
    )


@router.message(EvaluationFSM.waiting_for_name)
async def process_name(message: Message, state: FSMContext):
    await state.update_data(student_name=message.text)
    await message.answer(
        'Отлично! Теперь выбери (или введи) название модели, которую будем тестировать:',
        reply_markup=get_models_keyboard(),
    )
    await state.set_state(EvaluationFSM.waiting_for_model_name)


@router.message(EvaluationFSM.waiting_for_model_name)
async def process_model_name(message: Message, state: FSMContext):
    await state.update_data(model_name=message.text)
    await message.answer(
        'Принято. Теперь введи базовый URL твоего поднятого сервиса '
        '(например: http://192.168.1.5:8000 или https://my-recsys.ru):',
        reply_markup=ReplyKeyboardRemove(),
    )
    await state.set_state(EvaluationFSM.waiting_for_url)


@router.message(EvaluationFSM.waiting_for_url)
async def process_url(message: Message, state: FSMContext):
    url = message.text.strip().rstrip('/')
    if not url.startswith('http'):
        await message.answer(
            'Пожалуйста, введи корректный URL (с http:// или https://).'
        )
        return
    await state.update_data(base_url=url)
    await message.answer(
        'Супер! Теперь введи свой секретный API ключ (Token) для авторизации:'
    )
    await state.set_state(EvaluationFSM.waiting_for_token)


@router.message(EvaluationFSM.waiting_for_token)
async def process_token(message: Message, state: FSMContext):
    await state.update_data(token=message.text.strip())
    await message.answer(
        'Отлично. Выбери уровень сложности (от этого зависит RPS - количество запросов в секунду):',
        reply_markup=get_rps_keyboard(),
    )
    await state.set_state(EvaluationFSM.waiting_for_rps_level)


@router.message(EvaluationFSM.waiting_for_rps_level)
async def process_rps_level(message: Message, state: FSMContext):
    level_text = message.text
    target_rps = 0
    if 'Easy' in level_text:
        target_rps = settings.rps_easy
    elif 'Medium' in level_text:
        target_rps = settings.rps_medium
    elif 'Hard' in level_text:
        target_rps = settings.rps_hard
    else:
        await message.answer('Пожалуйста, используйте кнопки.')
        return

    await state.update_data(target_rps=target_rps)

    user_data = await state.get_data()

    await message.answer(
        f'Данные собраны!\n\n'
        f'Имя: {user_data.get("student_name")}\n'
        f'Модель: {user_data.get("model_name")}\n'
        f'URL: {user_data.get("base_url")}\n'
        f'Сложность: {target_rps} RPS\n\n'
        'Начинаю процесс тестирования... Сначала проверим работоспособность сервиса 🚀',
        reply_markup=ReplyKeyboardRemove(),
    )

    # Переходим в состояние чтобы игнорировать другой ввод
    await state.set_state(EvaluationFSM.running_evaluation)

    # Запускаем конвейер в фоне (чтобы не блокировать хендлер слишком уж откровенно,
    # хотя aiogram 3 выполняет хендлеры конкурентно)
    asyncio.create_task(run_evaluation_pipeline(message, state, user_data, target_rps))


async def run_evaluation_pipeline(
    message: Message, state: FSMContext, user_data: dict, target_rps: int
):
    base_url = user_data['base_url']
    token = user_data['token']
    model_name = user_data['model_name']
    student_name = user_data['student_name']

    # 1. Загрузка данных для теста
    test_users = load_test_users()
    if not test_users:
        await message.answer(
            '❌ Внутренняя ошибка бота: невозможно загрузить тестовых пользователей (test.csv отсутствует).'
        )
        await state.clear()
        return

    ground_truth = load_ground_truth()

    # 2. Sanity Check
    # Используем первого юзера из теста
    test_use_id = test_users[0]
    ok, err_msg = await sanity_check(
        base_url, token, model_name, test_user_id=test_use_id
    )
    if not ok:
        await message.answer(
            f'❌ **Sanity Check не пройден!**\n\nПричина: {err_msg}\n\nПожалуйста, исправь ошибку и попробуй снова - /start'
        )
        await state.clear()
        return

    await message.answer(
        f'✅ Sanity check пройден. Начинаю нагрузочное тестирование ({len(test_users)} запросов при {target_rps} RPS)...'
    )

    # 3. Нагрузочный прогон
    start_time = time.time()
    results = await run_load_test(base_url, token, model_name, test_users, target_rps)
    elapsed = time.time() - start_time

    # Фактический RPS = кол-во запросов / время выполнения
    actual_rps = len(test_users) / elapsed if elapsed > 0 else 0

    # 4. Расчет метрик
    statuses = [r.status_code for r in results]
    success_rate = calculate_success_rate(statuses)

    latencies = [r.latency for r in results]
    latency_stats = get_latency_percentiles(latencies)

    precision_sum = 0.0
    recall_sum = 0.0
    valuable_queries = 0

    for i, res in enumerate(results):
        if res.status_code == 200 and res.recos:
            uid = test_users[i]
            actual_items = ground_truth.get(uid, set())
            if actual_items:
                precision_sum += calculate_precision(res.recos, actual_items)
                recall_sum += calculate_recall(res.recos, actual_items)
                valuable_queries += 1

    mean_precision = precision_sum / valuable_queries if valuable_queries > 0 else 0.0
    mean_recall = recall_sum / valuable_queries if valuable_queries > 0 else 0.0

    # 5. Сообщение пользователю
    report = (
        f'📊 **Результаты Тестирования**\n\n'
        f'Модель: `{model_name}`\n\n'
        f'⚡ **Производительность:**\n'
        f'• Целевой RPS: `{target_rps}`\n'
        f'• Фактический RPS: `{actual_rps:.2f}`\n'
        f'• Успешных ответов (200): `{success_rate}%`\n'
        f'• Latency (ms): p50=`{latency_stats["p50"]}`, p95=`{latency_stats["p95"]}`, p99=`{latency_stats["p99"]}`\n\n'
        f'🎯 **Качество Рекомендаций (@10):**\n'
        f'• Precision: `{mean_precision:.4f}`\n'
        f'• Recall: `{mean_recall:.4f}`\n\n'
    )

    # 6. Запись в Google Sheets
    sheet_url = write_evaluation_result(
        student_name,
        model_name,
        target_rps,
        actual_rps,
        success_rate,
        latency_stats['p95'],
        mean_precision,
        mean_recall,
    )

    if sheet_url:
        report += f'✅ Твои результаты успешно занесены в публичную [Google Таблицу]({sheet_url})'
    else:
        report += '⚠️ Не удалось сохранить результаты в таблицу (отсутствуют настройки интеграции).'

    await message.answer(report, parse_mode='Markdown', disable_web_page_preview=True)
    await state.clear()
