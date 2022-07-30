import time

from typing import Any, List

import torch.utils.data.backward_compatibility

import torch.utils.data.graph_settings
from torch.utils.data import DataLoader, IterDataPipe, communication
from torch.utils.data.datapipes.iter import IterableWrapper

__all__ = [
    "DataLoader2",
]


class _ThreadingDataLoader2:

    def __init__(self, datapipe, num_workers=0, collate_fn=None):
        self.threads = []
        self.datapipes = []
        self.collate_fn = collate_fn
        for worker_id in range(num_workers):
            (thread, req_queue, res_queue, thread_localdatapipe) = communication.eventloop.SpawnThreadForDataPipeline(datapipe)
            torch.utils.data.graph_settings.apply_sharding(thread_localdatapipe, num_workers, worker_id)
            thread.start()
            self.threads.append((thread, req_queue, res_queue))  # These queues are independent
            local_datapipe = communication.iter.QueueWrapper(
                communication.protocol.IterDataPipeQueueProtocolClient(req_queue, res_queue))
            self.datapipes.append(local_datapipe)

    def __iter__(self):
        not_available = False
        forever = True
        exclude_datapipes: List[Any] = []
        while len(exclude_datapipes) < len(self.datapipes):
            for dp in self.datapipes:
                if dp not in exclude_datapipes:
                    try:
                        value = dp.nonblocking_next()
                        yield value
                    except StopIteration:
                        exclude_datapipes.append(dp)
                    except communication.iter.NotAvailable:
                        not_available = True
            if not_available:
                time.sleep(0.001)

    def __del__(self):
        self._cleanup_all_threads()

    def _cleanup_all_threads(self):
        def clean_me(thread, req_queue, res_queue):
            req_queue.put(communication.messages.TerminateRequest())
            _ = res_queue.get()
            thread.join()

        for thread, req_queue, res_queue in self.threads:
            clean_me(thread, req_queue, res_queue)

class DataLoader2:
    def __new__(cls,
                dataset,
                batch_size=1,
                shuffle=None,
                sampler=None,
                batch_sampler=None,
                num_workers=0,
                collate_fn=None,
                pin_memory=False,
                drop_last=False,
                timeout=0,
                worker_init_fn=None,
                *,
                prefetch_factor=2,
                persistent_workers=False,
                batch_outside_worker=False,
                parallelism_mode='mp'):
        if isinstance(dataset, IterDataPipe):
            data_loader: Any = None
            if batch_sampler is not None:
                raise Exception(
                    'batch_sampler is not yet supported by DataPipes')
            if sampler is not None:
                raise Exception(
                    'sampler is not yet supported by DataPipes')
            datapipe = dataset
            datapipe = torch.utils.data.graph_settings.apply_shuffle_settings(datapipe, shuffle=shuffle)  # type: ignore[assignment]
            if batch_outside_worker and pin_memory:
                raise Exception(
                    'pin_memory is not yet compatible with batch_outside_worker')
            if not batch_outside_worker:
                if batch_size is not None:
                    datapipe = datapipe.batch(batch_size, drop_last=drop_last)
                    if collate_fn is None:
                        collate_fn = torch.utils.data._utils.collate.default_collate

                # Note: It is safe to pass shuffle=True to the old DataLoader, as shuffle does nothing
                # for Iterable, but required to set Pipes correctly.
                data_loader = DataLoader(datapipe,
                                         batch_size=None,  # Replaced by .batch DataPipe
                                         shuffle=shuffle,
                                         sampler=None,
                                         batch_sampler=None,
                                         num_workers=num_workers,
                                         collate_fn=collate_fn,
                                         pin_memory=pin_memory,
                                         drop_last=False,  # Replaced by .batch DataPipe
                                         timeout=timeout,
                                         worker_init_fn=worker_init_fn,
                                         prefetch_factor=prefetch_factor,
                                         persistent_workers=persistent_workers)
            elif parallelism_mode == 'thread':
                if collate_fn is not None and not batch_outside_worker:
                    datapipe = datapipe.map(collate_fn)
                if pin_memory:
                    raise Exception(
                        'pin_memory is not yet supported by DataPipes with Threading')
                if worker_init_fn is not None:
                    raise Exception(
                        'worker_init_fn is not yet supported by DataPipes with Threading')
                data_loader = _ThreadingDataLoader2(datapipe,
                                                    num_workers=num_workers,
                                                    collate_fn=collate_fn)
            else:
                raise Exception('Unsupported parallelism mode', parallelism_mode)
            if not batch_outside_worker:
                return data_loader
            else:
                if collate_fn is None:
                    collate_fn = torch.utils.data._utils.collate.default_collate
                datapipe = IterableWrapper(data_loader).batch(
                    batch_size, drop_last=drop_last).map(collate_fn)
                return datapipe
        else:
            if parallelism_mode == 'thread':
                raise Exception(
                    'thread parallelism mode is not supported for old DataSets')
            return DataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              sampler=sampler,
                              batch_sampler=batch_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn,
                              pin_memory=pin_memory,
                              drop_last=drop_last,
                              timeout=timeout,
                              worker_init_fn=worker_init_fn,
                              prefetch_factor=prefetch_factor,
                              persistent_workers=persistent_workers)
