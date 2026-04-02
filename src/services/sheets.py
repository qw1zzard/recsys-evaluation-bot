import json
import logging
from datetime import datetime

import gspread
from google.oauth2.service_account import Credentials

from src.core.config import settings

logger = logging.getLogger(__name__)

# Скоупы для доступа к таблицам
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive',
]


def init_gspread_client():
    creds_raw = settings.google_creds_json
    if not creds_raw:
        logger.warning('GOOGLE_CREDS_JSON не задан. Интеграция с таблицами отключена.')
        return None

    try:
        if creds_raw.startswith('{'):
            # Расцениваем как JSON строку
            creds_dict = json.loads(creds_raw)
            credentials = Credentials.from_service_account_info(
                creds_dict,
                scopes=SCOPES,
            )
        else:
            # Расцениваем как путь к файлу
            credentials = Credentials.from_service_account_file(
                creds_raw,
                scopes=SCOPES,
            )

        return gspread.authorize(credentials)
    except Exception as e:
        logger.error(f'Ошибка при авторизации gspread: {e}')
        return None


def write_evaluation_result(
    student_name: str,
    model_name: str,
    target_rps: int,
    actual_rps: float,
    success_rate: float,
    p95_latency: float,
    metrics: dict[str, float],
) -> str | None:
    """
    Записывает результаты в конец таблицы GOOGLE_SHEET_URL
    Возвращает URL таблицы при успехе, иначе None
    """
    client = init_gspread_client()
    if not client:
        return None

    if not settings.google_sheet_url:
        logger.warning('GOOGLE_SHEET_URL не задан.')
        return None

    try:
        # Открываем таблицу по URL
        sheet = client.open_by_url(settings.google_sheet_url).sheet1

        # Если таблица пустая, добавляем заголовки
        if sheet.get_all_values() == [[]]:
            sheet.append_row(
                [
                    'Timestamp',
                    'Имя/Логин',
                    'Модель',
                    'Target RPS',
                    'Actual RPS',
                    'Success Rate %',
                    'p95 Latency (ms)',
                    'Precision@10',
                    'Recall@10',
                    'NDCG@10',
                    'MAP@10',
                    'MRR',
                    'HitRate@10',
                    'Recall@5',
                    'NDCG@5',
                ]
            )

        timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        row = [
            timestamp_str,
            student_name,
            model_name,
            str(target_rps),
            f'{actual_rps:.2f}',
            f'{success_rate}%',
            f'{p95_latency}',
            f'{metrics.get("precision@10", 0.0):.4f}',
            f'{metrics.get("recall@10", 0.0):.4f}',
            f'{metrics.get("ndcg@10", 0.0):.4f}',
            f'{metrics.get("map@10", 0.0):.4f}',
            f'{metrics.get("mrr", 0.0):.4f}',
            f'{metrics.get("hitrate@10", 0.0):.4f}',
            f'{metrics.get("recall@5", 0.0):.4f}',
            f'{metrics.get("ndcg@5", 0.0):.4f}',
        ]

        sheet.append_row(row)
        logger.info(
            f'Результаты для {student_name} ({model_name}) успешно записаны в таблицу.'
        )
        return settings.google_sheet_url

    except Exception as e:
        logger.error(f'Ошибка при записи в Google табличку: {e}')
        return None
