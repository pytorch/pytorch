# mypy: allow-untyped-defs
r"""Contains definitions of the methods used by the _BaseDataLoaderIter to fetch data from an iterable-style or map-style dataset.

This logic is shared in both single- and multi-processing data loading.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

# Set up logger for this module
logger = logging.getLogger(__name__)


class _BaseDatasetFetcher:
    def __init__(self, dataset, auto_collation, collate_fn, drop_last, num_threads=1):
        self.dataset = dataset
        self.auto_collation = auto_collation
        self.collate_fn = collate_fn
        self.drop_last = drop_last
        self.num_threads = num_threads

        if self.num_threads > 1:
            self._executor = ThreadPoolExecutor(self.num_threads)
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

    def _run_async(self, indices, fetch_fn):
        async def _worker(task_queue, result_queue):
            while not task_queue.empty():
                index = await task_queue.get()
                try:
                    result = await self._loop.run_in_executor(
                        self._executor, fetch_fn, index
                    )
                    await result_queue.put((index, result))
                except Exception as e:
                    logger.error(
                        "Exception during fetch for index %s: %s (%s)", 
                        index, str(e), type(e).__name__,
                        exc_info=True
                    )
                finally:
                    task_queue.task_done()

        async def _run(indices):
            task_queue = asyncio.Queue()
            result_queue = asyncio.Queue()
            for idx in indices:
                await task_queue.put(idx)

            workers = [
                asyncio.create_task(_worker(task_queue, result_queue))
                for _ in range(self.num_threads)
            ]
            await asyncio.gather(*workers)

            results = {}
            while not result_queue.empty():
                idx, value = await result_queue.get()
                results[idx] = value
            return [results[i] for i in indices]

        return self._loop.run_until_complete(_run(indices))


class _IterableDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        super().__init__(dataset, auto_collation, collate_fn, drop_last)
        self.dataset_iter = iter(dataset)
        self.ended = False

    def _fetch_one(self, _):
        return next(self.dataset_iter)

    def fetch(self, possibly_batched_index):
        if self.ended:
            raise StopIteration

        if self.auto_collation:
            data = []
            if self.num_threads > 1:
                try:
                    data = self._run_async(possibly_batched_index, self._fetch_one)
                except StopIteration:
                    self.ended = True
            else:
                for _ in possibly_batched_index:
                    try:
                        data.append(next(self.dataset_iter))
                    except StopIteration:
                        self.ended = True
                        break
            if len(data) == 0 or (
                self.drop_last and len(data) < len(possibly_batched_index)
            ):
                raise StopIteration
        else:
            data = next(self.dataset_iter)
        return self.collate_fn(data)


class _MapDatasetFetcher(_BaseDatasetFetcher):
    def _fetch_one(self, idx):
        return self.dataset[idx]

    def fetch(self, possibly_batched_index):
        if self.auto_collation:
            if hasattr(self.dataset, "__getitems__") and self.dataset.__getitems__:
                data = self.dataset.__getitems__(possibly_batched_index)
            elif self.num_threads > 1:
                data = self._run_async(possibly_batched_index, self._fetch_one)
            else:
                data = [self.dataset[idx] for idx in possibly_batched_index]
        else:
            data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)
