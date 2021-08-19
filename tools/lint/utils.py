import asyncio
import multiprocessing
from typing import List, Any


VERBOSE = False

def log(*args: Any) -> None:
    if VERBOSE:
        print(*args)


async def gather_with_concurrency(n: int, tasks: List[Any]) -> Any:
    semaphore = asyncio.Semaphore(n)

    async def sem_task(task: Any) -> Any:
        async with semaphore:
            return await task

    return await asyncio.gather(
        *(sem_task(task) for task in tasks), return_exceptions=True
    )


async def gather(tasks: List[Any]) -> Any:
    return await gather_with_concurrency(multiprocessing.cpu_count(), tasks)
