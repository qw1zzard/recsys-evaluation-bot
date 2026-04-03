import asyncio
import re
import time
from html import escape

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
from src.services.database import db
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
            [InlineKeyboardButton(text='🔗 Изменить URL', callback_data='edit_url')],
            [
                InlineKeyboardButton(
                    text='🔑 Изменить токен', callback_data='edit_token'
                )
            ],
            [InlineKeyboardButton(text='📊 Изменить RPS', callback_data='edit_rps')],
            [InlineKeyboardButton(text='🧹 Сбросить всё', callback_data='start_new')],
            [InlineKeyboardButton(text='❌ Отмена', callback_data='cancel_evaluation')],
        ]
    )


async def show_summary_menu(message: Message, state: FSMContext):
    user_data = await state.get_data() or {}
    await message.answer(
        '⚙️ <b>Конфигурация теста</b>\n\n'
        f'• <b>Модель</b>: <code>{escape(str(user_data.get("model_name", "—")))}</code>\n'
        f'• <b>URL</b>: <code>{escape(str(user_data.get("base_url", "—")))}</code>\n'
        f'• <b>Сложность</b>: <code>{escape(str(user_data.get("target_rps", "—")))} RPS</code>\n\n'
        'Что будем делать?',
        reply_markup=get_config_menu_keyboard(),
        parse_mode='HTML',
    )


async def start_evaluation_flow(message: Message, state: FSMContext, user: User):
    user_data = await state.get_data() or {}

    if user.username:
        student_name = f'@{user.username}'
    else:
        student_name = user.full_name

    # Try to load existing config from DB
    if not user_data:
        saved_config = await db.get_user_config(user.id)
        if saved_config:
            # Reconstruct data for FSM. We need to match keys used in the code.
            # user_configs uses snake_case, but some keys in FSM might be different.
            # Based on code: model_name, base_url, target_rps, token, student_name
            fsm_data = {
                'student_name': saved_config.get('username') or student_name,
                'model_name': saved_config.get('model_name'),
                'base_url': saved_config.get('base_url'),
                'token': saved_config.get('token'),
                'target_rps': saved_config.get('target_rps'),
            }
            # Remove None values
            fsm_data = {k: v for k, v in fsm_data.items() if v is not None}
            await state.update_data(**fsm_data)
            user_data = fsm_data

    # If user already has some data, show summary menu instead of starting from scratch
    if user_data.get('model_name') and user_data.get('base_url'):
        await show_summary_menu(message, state)
        return

    await state.update_data(student_name=student_name)
    await db.upsert_user_config(user_id=user.id, username=student_name)

    await message.answer(
        f'👋 Привет, {escape(student_name)}!\n\n'
        'Я бот для оценки рекомендательных систем. Давай протестируем твой сервис! 🚀\n\n'
        '📝 Введи название модели (например, <code>main</code> или <code>boost</code>):',
        reply_markup=ReplyKeyboardRemove(),
        parse_mode='HTML',
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
    await db.upsert_user_config(user_id=message.from_user.id, model_name=model_name)

    data = await state.get_data()
    if data.get('target_rps'):
        await show_summary_menu(message, state)
        return

    await message.answer(
        '✅ Принято. Теперь введи базовый URL твоего сервиса\n'
        '(например: <code>http://recsys-service.com</code>):',
        reply_markup=ReplyKeyboardRemove(),
        parse_mode='HTML',
    )
    await state.set_state(EvaluationFSM.waiting_for_url)


@router.message(EvaluationFSM.waiting_for_url)
async def process_url(message: Message, state: FSMContext):
    url = message.text.strip().rstrip('/')
    if not url.startswith('http'):
        await message.answer(
            '⚠️ Пожалуйста, введи корректный URL (с <code>http://</code> или <code>https://</code>).',
            parse_mode='HTML',
        )
        return
    await state.update_data(base_url=url)
    await db.upsert_user_config(user_id=message.from_user.id, base_url=url)

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
    token = message.text.strip()
    await state.update_data(token=token)
    await db.upsert_user_config(user_id=message.from_user.id, token=token)

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
        error_msg = 'test.csv отсутствует'
        await message.answer(
            f'❌ Внутренняя ошибка бота: невозможно загрузить тестовых пользователей ({error_msg}).'
        )
        await db.save_test_result(
            {
                'user_id': message.chat.id,
                'model_name': model_name,
                'target_rps': target_rps,
                'status': 'error',
                'error_message': error_msg,
            }
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
            '❌ <b>Sanity Check не пройден!</b>\n\n'
            f'Причина: <code>{escape(str(err_msg))}</code>\n\n'
            'Что будем делать?',
            reply_markup=get_config_menu_keyboard(),
            parse_mode='HTML',
        )
        await db.save_test_result(
            {
                'user_id': message.chat.id,
                'model_name': model_name,
                'target_rps': target_rps,
                'status': 'failed_sanity',
                'error_message': err_msg,
            }
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
        'ndcg@5': 0.0,
    }
    valuable_queries = 0

    for i, res in enumerate(results):
        if res.status_code == 200 and res.recos:
            uid = test_users[i]
            actual_items = ground_truth.get(uid, set())
            if actual_items:
                recos = res.recos
                metrics_sums['precision@10'] += calculate_precision(
                    recos, actual_items, k=10
                )
                metrics_sums['recall@10'] += calculate_recall(recos, actual_items, k=10)
                metrics_sums['ndcg@10'] += calculate_ndcg(recos, actual_items, k=10)
                metrics_sums['map@10'] += calculate_ap(recos, actual_items, k=10)
                metrics_sums['mrr'] += calculate_mrr(recos, actual_items)
                metrics_sums['hitrate@10'] += calculate_hit_rate(
                    recos, actual_items, k=10
                )
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
        '📊 <b>Результаты Тестирования</b>\n\n'
        f'Модель: <code>{escape(str(model_name))}</code>\n\n'
        '⚡ <b>Производительность:</b>\n'
        f'• Целевой RPS: <code>{target_rps}</code>\n'
        f'• Фактический RPS: <code>{actual_rps:.2f}</code>\n'
        f'• Успешных ответов (200): <code>{success_rate}%</code>\n'
        f'• Latency (ms): p50=<code>{latency_stats["p50"]}</code>, '
        f'p95=<code>{latency_stats["p95"]}</code>, p99=<code>{latency_stats["p99"]}</code>\n\n'
        '🎯 <b>Качество Рекомендаций:</b>\n'
        f'• Precision@10: <code>{mean_metrics["precision@10"]:.4f}</code>\n'
        f'• Recall@10: <code>{mean_metrics["recall@10"]:.4f}</code>\n'
        f'• F1@10: <code>{f1_10:.4f}</code>\n'
        f'• NDCG@10: <code>{mean_metrics["ndcg@10"]:.4f}</code>\n'
        f'• MAP@10: <code>{mean_metrics["map@10"]:.4f}</code>\n'
        f'• MRR: <code>{mean_metrics["mrr"]:.4f}</code>\n'
        f'• HitRate@10: <code>{mean_metrics["hitrate@10"]:.4f}</code>\n'
        f'• NDCG@5: <code>{mean_metrics["ndcg@5"]:.4f}</code>\n\n'
    )

    # 6. Запись в БД
    await db.save_test_result(
        {
            'user_id': message.chat.id,  # В данном случае это user_id
            'model_name': model_name,
            'target_rps': target_rps,
            'actual_rps': actual_rps,
            'success_rate': success_rate,
            'latency_p50': latency_stats['p50'],
            'latency_p95': latency_stats['p95'],
            'latency_p99': latency_stats['p99'],
            'precision_at_10': mean_metrics['precision@10'],
            'recall_at_10': mean_metrics['recall@10'],
            'f1_at_10': f1_10,
            'ndcg_at_10': mean_metrics['ndcg@10'],
            'map_at_10': mean_metrics['map@10'],
            'mrr': mean_metrics['mrr'],
            'hitrate_at_10': mean_metrics['hitrate@10'],
            'duration': elapsed,
            'status': 'success',
        }
    )

    # 7. Запись в Google Sheets
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
        safe_url = escape(str(sheet_url), quote=True)
        report += f'✅ Твои результаты успешно занесены в публичную <a href="{safe_url}">Google Таблицу</a>'
    else:
        report += '⚠️ Не удалось сохранить результаты в таблицу (отсутствуют настройки интеграции).'

    await message.answer(report, parse_mode='HTML', disable_web_page_preview=True)

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
    await db.upsert_user_config(user_id=callback.from_user.id, target_rps=target_rps)

    user_data = await state.get_data()

    await callback.message.edit_text(
        f'✅ Выбран уровень: <code>{target_rps} RPS</code>', parse_mode='HTML'
    )

    # Re-call the logic from process_rps_level but adapted for callback
    await callback.message.answer(
        '✨ <b>Данные собраны!</b>\n\n'
        f'• <b>Имя</b>: {escape(str(user_data.get("student_name") or ""))}\n'
        f'• <b>Модель</b>: <code>{escape(str(user_data.get("model_name") or ""))}</code>\n'
        f'• <b>URL</b>: <code>{escape(str(user_data.get("base_url") or ""))}</code>\n'
        f'• <b>Сложность</b>: <code>{target_rps} RPS</code>\n\n'
        '🚀 Начинаю процесс тестирования... Сначала проверим сервис.',
        reply_markup=ReplyKeyboardRemove(),
        parse_mode='HTML',
    )

    await state.set_state(EvaluationFSM.running_evaluation)
    asyncio.create_task(
        run_evaluation_pipeline(callback.message, state, user_data, target_rps)
    )
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
        '📝 Введите новое название модели (например, <code>main</code> или <code>boost</code>):',
        parse_mode='HTML',
    )
    await callback.answer()


@router.callback_query(F.data == 'edit_url')
async def handle_edit_url(callback: CallbackQuery, state: FSMContext):
    await state.set_state(EvaluationFSM.waiting_for_url)
    await callback.message.answer(
        '🔗 Введите новый базовый URL (например: <code>http://recsys-service.com</code>):',
        parse_mode='HTML',
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
