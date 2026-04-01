import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_PATH = Path('test.csv')


def load_test_users() -> list[int]:
    """Загружает список уникальных пользователей для тестирования из test.csv"""
    if not DATA_PATH.exists():
        logger.error(f'Файл датасета {DATA_PATH} не найден!')
        return []

    try:
        df = pd.read_csv(DATA_PATH)
        if 'user_id' not in df.columns:
            logger.error("Колонка 'user_id' не найдена в test.csv")
            return []

        unique_users = df['user_id'].unique().tolist()
        logger.info(
            f'Загружено {len(unique_users)} уникальных пользователей из {DATA_PATH}'
        )
        return unique_users
    except Exception as e:
        logger.error(f'Ошибка при чтении датасета: {e}')
        return []


def load_ground_truth() -> dict[int, set[int]]:
    """Создает словарь 'user_id -> множество правильных item_id' для расчета метрик"""
    if not DATA_PATH.exists():
        return {}

    try:
        df = pd.read_csv(DATA_PATH)
        # Группируем по пользователям и собираем item_id
        gt = df.groupby('user_id')['item_id'].apply(set).to_dict()
        return gt
    except Exception as e:
        logger.error(f'Ошибка при получении ground truth: {e}')
        return {}
