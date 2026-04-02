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


def calculate_mrr(recommended: list[int], actual: set[int]) -> float:
    """Вычисляет Reciprocal Rank"""
    if not recommended or not actual:
        return 0.0

    for i, item in enumerate(recommended):
        if item in actual:
            return 1.0 / (i + 1)
    return 0.0


def calculate_ndcg(recommended: list[int], actual: set[int], k: int = 10) -> float:
    """Вычисляет NDCG@K"""
    if not recommended or not actual:
        return 0.0

    top_k = recommended[:k]
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in actual:
            dcg += 1.0 / np.log2(i + 2)

    # Идеальный DCG (IDCG) - если бы все релевантные были в топе
    idcg = 0.0
    for i in range(min(len(actual), k)):
        idcg += 1.0 / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def calculate_ap(recommended: list[int], actual: set[int], k: int = 10) -> float:
    """Вычисляет Average Precision@K"""
    if not recommended or not actual:
        return 0.0

    top_k = recommended[:k]
    score = 0.0
    hits = 0.0
    for i, item in enumerate(top_k):
        if item in actual:
            hits += 1.0
            score += hits / (i + 1)

    return score / min(len(actual), k) if actual else 0.0


def calculate_hit_rate(recommended: list[int], actual: set[int], k: int = 10) -> float:
    """Вычисляет Hit Rate@K (1 если есть попадание, иначе 0)"""
    if not recommended or not actual:
        return 0.0

    top_k = recommended[:k]
    for item in top_k:
        if item in actual:
            return 1.0
    return 0.0


def calculate_f1(precision: float, recall: float) -> float:
    """Вычисляет F1-меру"""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


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


def calculate_status_stats(statuses: list[int]) -> dict[str, float]:
    """Считает статистику по статус-кодам"""
    if not statuses:
        return {'success_rate': 0.0, 'error_rate': 0.0}

    success = sum(1 for s in statuses if s == 200)
    total = len(statuses)
    return {
        'success_rate': round((success / total) * 100, 2),
        'error_rate': round(((total - success) / total) * 100, 2),
    }


def calculate_success_rate(statuses: list[int]) -> float:
    """Считает долю успешных (200) запросов"""
    return calculate_status_stats(statuses)['success_rate']
