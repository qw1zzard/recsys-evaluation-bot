import asyncio
import logging
import time
from dataclasses import dataclass

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    status_code: int
    latency: float
    recos: list[int]
    error: str | None = None


async def sanity_check(
    base_url: str, token: str, model_name: str, test_user_id: int = 1
) -> tuple[bool, str]:
    """Проверяет работоспособность сервиса перед запуском нагрузочного тестирования"""
    logger.info(f'Начало sanity check для {base_url}')

    headers = {'Authorization': f'Bearer {token}'}

    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5)
        ) as session:
            # 1. Health check
            health_url = f'{base_url}/health'
            try:
                async with session.get(health_url) as resp:
                    if resp.status != 200:
                        return False, f'GET /health вернул статус {resp.status}'
            except Exception as e:
                return False, f'Ошибка при запросе GET /health: {e}'

            # 2. 5 последовательных запросов к reco
            reco_url = f'{base_url}/reco/{model_name}/{test_user_id}'
            for i in range(5):
                try:
                    async with session.get(reco_url, headers=headers) as resp:
                        if resp.status == 401:
                            return False, 'Ошибка авторизации (401). Неверный токен.'
                        if resp.status == 404:
                            return False, 'Модель не найдена (404) или неверный URL.'
                        if resp.status != 200:
                            return (
                                False,
                                f'Запрос {i + 1} упал со статусом {resp.status}',
                            )

                        # Проверяем структуру ответа
                        await resp.json()
                        # Зависит от формата сервиса, предполагаем список или dict с ключом
                        # По хорошему нужна проверка схемы
                except Exception as e:
                    return False, f'Запрос {i + 1} завершился с ошибкой сети: {e}'

            return True, 'Sanity check пройден успешно!'

    except Exception as e:
        return False, f'Непредвиденная системная ошибка: {e}'


async def _fetch_reco(
    session: aiohttp.ClientSession, url: str, headers: dict
) -> TestResult:
    start_t = time.perf_counter()
    try:
        async with session.get(url, headers=headers) as resp:
            latency = time.perf_counter() - start_t
            if resp.status == 200:
                try:
                    data = await resp.json()
                    # Если ответ - список, берем его. Если словарь (типа {"recos": []}), нужно адаптировать
                    recos = data if isinstance(data, list) else data.get('recos', [])
                    return TestResult(
                        status_code=resp.status, latency=latency, recos=recos
                    )
                except Exception:
                    return TestResult(
                        status_code=resp.status,
                        latency=latency,
                        recos=[],
                        error='Invalid JSON format',
                    )
            else:
                return TestResult(status_code=resp.status, latency=latency, recos=[])
    except Exception as e:
        latency = time.perf_counter() - start_t
        return TestResult(status_code=0, latency=latency, recos=[], error=str(e))


async def run_load_test(
    base_url: str, token: str, model_name: str, user_ids: list[int], target_rps: int
) -> list[TestResult]:
    """Производит нагрузочное тестирование с заданным RPS"""
    headers = {'Authorization': f'Bearer {token}'}
    results: list[TestResult] = []

    delay_between_requests = 1.0 / target_rps if target_rps > 0 else 0

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=10)
    ) as session:
        # Для соблюдения RPS используем background tasks и небольшую задержку
        tasks = []

        for user_id in user_ids:
            url = f'{base_url}/reco/{model_name}/{user_id}'
            tasks.append(asyncio.create_task(_fetch_reco(session, url, headers)))
            # Чтобы не блокироваться, спим ровно delay
            await asyncio.sleep(delay_between_requests)

        # Ждем завершения всех отправленных запросов
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        for res in raw_results:
            if isinstance(res, TestResult):
                results.append(res)
            else:
                # На случай если задача упала (чего не должно быть, т.к. ловим всё внутри _fetch_reco)
                results.append(
                    TestResult(
                        status_code=500, latency=0, recos=[], error='Task Exception'
                    )
                )

    return results
