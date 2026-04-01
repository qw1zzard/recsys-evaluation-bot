import numpy as np


def calculate_precision(recommended: list[int], actual: set[int], k: int = 10) -> float:
    """Вычисляет Precision@K"""
    if not recommended or not actual:
        return 0.0

    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in actual)
    return hits / k


def calculate_recall(recommended: list[int], actual: set[int], k: int = 10) -> float:
    """Вычисляет Recall@K"""
    if not recommended or not actual:
        return 0.0

    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in actual)
    return hits / len(actual)


def get_latency_percentiles(latencies: list[float]) -> dict[str, float]:
    """Возвращает квантили p50, p95, p99 (в миллисекундах)"""
    if not latencies:
        return {'p50': 0.0, 'p95': 0.0, 'p99': 0.0}

    arr = np.array(latencies) * 1000  # Перевод в мс
    return {
        'p50': round(float(np.percentile(arr, 50)), 2),
        'p95': round(float(np.percentile(arr, 95)), 2),
        'p99': round(float(np.percentile(arr, 99)), 2),
    }


def calculate_success_rate(statuses: list[int]) -> float:
    """Считает долю успешных (200) запросов"""
    if not statuses:
        return 0.0
    success = sum(1 for s in statuses if s == 200)
    return round((success / len(statuses)) * 100, 2)
