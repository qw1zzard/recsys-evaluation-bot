import asyncio
import re
import time

from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    ReplyKeyboardRemove,
    User,
)

from src.core.config import settings
from src.services.data_loader import load_ground_truth, load_test_users
from src.services.metrics import (
    calculate_ap,
    calculate_f1,
    calculate_hit_rate,
    calculate_mrr,
    calculate_ndcg,
    calculate_precision,
    calculate_recall,
    calculate_success_rate,
    get_latency_percentiles,
)
from src.services.sheets import write_evaluation_result
from src.services.tester import run_load_test, sanity_check
from src.states.registration import EvaluationFSM

router = Router()


def get_rps_keyboard():
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text='🟢 Easy (5 RPS)', callback_data='rps_5')],
            [InlineKeyboardButton(text='🟡 Medium (15 RPS)', callback_data='rps_15')],
            [InlineKeyboardButton(text='🔴 Hard (50 RPS)', callback_data='rps_50')],
        ]
    )


def get_start_keyboard():
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text='🚀 Начать тестирование', callback_data='start_new'
                )
            ]
        ]
    )


def get_config_menu_keyboard():
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text='🚀 Запустить тест', callback_data='retry_check'
                )
            ],
            [
                InlineKeyboardButton(
                    text='📝 Изменить модель', callback_data='edit_model'
                )
            ],
            [
                InlineKeyboardButton(text='🔗 Изменить URL', callback_data='edit_url')
            ],
            [
                InlineKeyboardButton(
                    text='🔑 Изменить токен', callback_data='edit_token'
                )
            ],
            [
                InlineKeyboardButton(
                    text='📊 Изменить RPS', callback_data='edit_rps'
                )
            ],
            [
                InlineKeyboardButton(
                    text='🧹 Сбросить всё', callback_data='start_new'
                )
            ],
            [
                InlineKeyboardButton(
                    text='❌ Отмена', callback_data='cancel_evaluation'
                )
            ],
        ]
    )


async def show_summary_menu(message: Message, state: FSMContext):
    user_data = await state.get_data()
    await message.answer(
        f'⚙️ **Конфигурация теста**\n\n'
        f'• **Модель**: `{user_data.get("model_name", "—")}`\n'
        f'• **URL**: `{user_data.get("base_url", "—")}`\n'
        f'• **Сложность**: `{user_data.get("target_rps", "—")} RPS`\n\n'
        'Что будем делать?',
        reply_markup=get_config_menu_keyboard(),
        parse_mode='Markdown',
    )


async def start_evaluation_flow(message: Message, state: FSMContext, user: User):
    user_data = await state.get_data()

    # If user already has some data, show summary menu instead of starting from scratch
    if user_data.get('model_name') and user_data.get('base_url'):
        await show_summary_menu(message, state)
        return

    if user.username:
        student_name = f'@{user.username}'
    else:
        student_name = user.full_name

    await state.update_data(student_name=student_name)

    await message.answer(
        f'👋 Привет, {student_name}!\n\n'
        'Я бот для оценки рекомендательных систем. Давай протестируем твой сервис! 🚀\n\n'
        '📝 Введи название модели (например, `main` или `boost`):',
        reply_markup=ReplyKeyboardRemove(),
        parse_mode='Markdown',
    )
    await state.set_state(EvaluationFSM.waiting_for_model_name)


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    await start_evaluation_flow(message, state, message.from_user)


@router.message(Command('cancel'))
async def cmd_cancel(message: Message, state: FSMContext):
    await state.clear()
    await message.answer(
        '❌ Действие отменено.',
        reply_markup=get_start_keyboard(),
    )


@router.message(EvaluationFSM.waiting_for_model_name)
async def process_model_name(message: Message, state: FSMContext):
    model_name = message.text.strip()

    if not re.match(r'^[a-zA-Z0-9_-]+$', model_name):
        await message.answer(
            '❌ Некорректное название модели.\n\n'
            'Название должно состоять только из латинских букв, цифр, подчеркиваний `_` или дефисов `-`.\n'
            'Попробуй еще раз:'
        )
        return

    await state.update_data(model_name=model_name)

    data = await state.get_data()
    if data.get('target_rps'):
        await show_summary_menu(message, state)
        return

    await message.answer(
        '✅ Принято. Теперь введи базовый URL твоего сервиса\n'
        '(например: `http://recsys-service.com` или `http://localhost:8000`):',
        reply_markup=ReplyKeyboardRemove(),
        parse_mode='Markdown',
    )
    await state.set_state(EvaluationFSM.waiting_for_url)


@router.message(EvaluationFSM.waiting_for_url)
async def process_url(message: Message, state: FSMContext):
    url = message.text.strip().rstrip('/')
    if not url.startswith('http'):
        await message.answer(
            '⚠️ Пожалуйста, введи корректный URL (с `http://` или `https://`).',
            parse_mode='Markdown',
        )
        return
    await state.update_data(base_url=url)

    data = await state.get_data()
    if data.get('target_rps'):
        await show_summary_menu(message, state)
        return

    await message.answer(
        '🔑 Супер! Теперь введи свой секретный API ключ (Token) для авторизации:'
    )
    await state.set_state(EvaluationFSM.waiting_for_token)


@router.message(EvaluationFSM.waiting_for_token)
async def process_token(message: Message, state: FSMContext):
    await state.update_data(token=message.text.strip())

    data = await state.get_data()
    if data.get('target_rps'):
        await show_summary_menu(message, state)
        return

    await message.answer(
        '📊 Отлично. Выбери уровень сложности (от этого зависит RPS):',
        reply_markup=get_rps_keyboard(),
    )
    await state.set_state(EvaluationFSM.waiting_for_rps_level)


@router.message(EvaluationFSM.waiting_for_rps_level)
async def process_rps_level_text(message: Message, state: FSMContext):
    await message.answer('⚠️ Пожалуйста, используй кнопки для выбора уровня RPS.')


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
            f'❌ **Sanity Check не пройден!**\n\n'
            f'Причина: `{err_msg}`\n\n'
            'Что будем делать?',
            reply_markup=get_config_menu_keyboard(),
            parse_mode='Markdown',
        )
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

    # Инициализация сумматоров для метрик
    metrics_sums = {
        'precision@10': 0.0,
        'recall@10': 0.0,
        'ndcg@10': 0.0,
        'map@10': 0.0,
        'mrr': 0.0,
        'hitrate@10': 0.0,
        'recall@5': 0.0,
        'ndcg@5': 0.0,
    }
    valuable_queries = 0

    for i, res in enumerate(results):
        if res.status_code == 200 and res.recos:
            uid = test_users[i]
            actual_items = ground_truth.get(uid, set())
            if actual_items:
                recos = res.recos
                metrics_sums['precision@10'] += calculate_precision(recos, actual_items, k=10)
                metrics_sums['recall@10'] += calculate_recall(recos, actual_items, k=10)
                metrics_sums['ndcg@10'] += calculate_ndcg(recos, actual_items, k=10)
                metrics_sums['map@10'] += calculate_ap(recos, actual_items, k=10)
                metrics_sums['mrr'] += calculate_mrr(recos, actual_items)
                metrics_sums['hitrate@10'] += calculate_hit_rate(recos, actual_items, k=10)
                metrics_sums['recall@5'] += calculate_recall(recos, actual_items, k=5)
                metrics_sums['ndcg@5'] += calculate_ndcg(recos, actual_items, k=5)
                valuable_queries += 1

    # Усреднение метрик
    mean_metrics = {}
    for key, total in metrics_sums.items():
        mean_metrics[key] = total / valuable_queries if valuable_queries > 0 else 0.0

    # Расчет F1@10
    f1_10 = calculate_f1(mean_metrics['precision@10'], mean_metrics['recall@10'])

    # 5. Сообщение пользователю
    report = (
        f'📊 **Результаты Тестирования**\n\n'
        f'Модель: `{model_name}`\n\n'
        f'⚡ **Производительность:**\n'
        f'• Целевой RPS: `{target_rps}`\n'
        f'• Фактический RPS: `{actual_rps:.2f}`\n'
        f'• Успешных ответов (200): `{success_rate}%`\n'
        f'• Latency (ms): p50=`{latency_stats["p50"]}`, p95=`{latency_stats["p95"]}`, p99=`{latency_stats["p99"]}`\n\n'
        f'🎯 **Качество Рекомендаций:**\n'
        f'• Precision@10: `{mean_metrics["precision@10"]:.4f}`\n'
        f'• Recall@10: `{mean_metrics["recall@10"]:.4f}`\n'
        f'• F1@10: `{f1_10:.4f}`\n'
        f'• NDCG@10: `{mean_metrics["ndcg@10"]:.4f}`\n'
        f'• MAP@10: `{mean_metrics["map@10"]:.4f}`\n'
        f'• MRR: `{mean_metrics["mrr"]:.4f}`\n'
        f'• HitRate@10: `{mean_metrics["hitrate@10"]:.4f}`\n'
        f'• Recall@5: `{mean_metrics["recall@5"]:.4f}`\n'
        f'• NDCG@5: `{mean_metrics["ndcg@5"]:.4f}`\n\n'
    )

    # 6. Запись в Google Sheets
    sheet_url = write_evaluation_result(
        student_name,
        model_name,
        target_rps,
        actual_rps,
        success_rate,
        latency_stats['p95'],
        mean_metrics,
    )

    if sheet_url:
        report += f'✅ Твои результаты успешно занесены в публичную [Google Таблицу]({sheet_url})'
    else:
        report += '⚠️ Не удалось сохранить результаты в таблицу (отсутствуют настройки интеграции).'

    await message.answer(report, parse_mode='Markdown', disable_web_page_preview=True)

    # Сохраняем состояние, чтобы пользователь мог изменить параметры и запустить снова
    await show_summary_menu(message, state)


@router.callback_query(F.data == 'start_new')
async def handle_start_new(callback: CallbackQuery, state: FSMContext):
    await callback.message.edit_text('🚀 Запускаю новый тест...')
    await start_evaluation_flow(callback.message, state, callback.from_user)
    await callback.answer()


@router.callback_query(F.data.startswith('rps_'))
async def handle_rps_selection(callback: CallbackQuery, state: FSMContext):
    target_rps = int(callback.data.split('_')[1])
    await state.update_data(target_rps=target_rps)

    user_data = await state.get_data()

    await callback.message.edit_text(f'✅ Выбран уровень: `{target_rps} RPS`', parse_mode='Markdown')

    # Re-call the logic from process_rps_level but adapted for callback
    await callback.message.answer(
        f'✨ **Данные собраны!**\n\n'
        f'• **Имя**: {user_data.get("student_name")}\n'
        f'• **Модель**: `{user_data.get("model_name")}`\n'
        f'• **URL**: `{user_data.get("base_url")}`\n'
        f'• **Сложность**: `{target_rps} RPS`\n\n'
        '🚀 Начинаю процесс тестирования... Сначала проверим сервис.',
        reply_markup=ReplyKeyboardRemove(),
        parse_mode='Markdown',
    )

    await state.set_state(EvaluationFSM.running_evaluation)
    asyncio.create_task(run_evaluation_pipeline(callback.message, state, user_data, target_rps))
    await callback.answer()


@router.callback_query(F.data == 'retry_check')
async def handle_retry(callback: CallbackQuery, state: FSMContext):
    user_data = await state.get_data()
    target_rps = user_data.get('target_rps', settings.rps_easy)

    await callback.message.edit_text(
        '🔄 Повторяю проверку... 🚀',
        reply_markup=None,
    )

    asyncio.create_task(
        run_evaluation_pipeline(callback.message, state, user_data, target_rps)
    )
    await callback.answer()


@router.callback_query(F.data == 'edit_model')
async def handle_edit_model(callback: CallbackQuery, state: FSMContext):
    await state.set_state(EvaluationFSM.waiting_for_model_name)
    await callback.message.answer(
        '📝 Введите новое название модели (например, `main` или `boost`):',
        parse_mode='Markdown',
    )
    await callback.answer()


@router.callback_query(F.data == 'edit_url')
async def handle_edit_url(callback: CallbackQuery, state: FSMContext):
    await state.set_state(EvaluationFSM.waiting_for_url)
    await callback.message.answer(
        '🔗 Введите новый базовый URL (например: `http://recsys-service.com`):',
        parse_mode='Markdown',
    )
    await callback.answer()


@router.callback_query(F.data == 'edit_token')
async def handle_edit_token(callback: CallbackQuery, state: FSMContext):
    await state.set_state(EvaluationFSM.waiting_for_token)
    await callback.message.answer('🔑 Введите новый API ключ (Token):')
    await callback.answer()


@router.callback_query(F.data == 'edit_rps')
async def handle_edit_rps(callback: CallbackQuery, state: FSMContext):
    await state.set_state(EvaluationFSM.waiting_for_rps_level)
    await callback.message.answer(
        '📊 Выберите новый уровень сложности:',
        reply_markup=get_rps_keyboard(),
    )
    await callback.answer()


@router.callback_query(F.data == 'cancel_evaluation')
async def handle_cancel(callback: CallbackQuery, state: FSMContext):
    await state.clear()
    await callback.message.edit_text(
        '❌ Действие отменено.',
        reply_markup=get_start_keyboard(),
    )
    await callback.answer()
