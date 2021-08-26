
import functools
import time

import torch.utils.data.backward_compatibility
from torch.utils.data import DataLoader, IterDataPipe, communication
from torch.utils.data.datapipes.iter import IterableAsDataPipe


class _ThreadingDataLoader2:
    known_dataloaders = {}

    def __init__(self, datapipe, num_workers=0):
        self.threads = []
        self.datapipes = []
        for i in range(num_workers):
            (thread, req_queue, res_queue) = communication.eventloop.SpawnThreadForDataPipeline(datapipe)
            thread.start()
            self.threads.append((thread, req_queue, res_queue))
            local_datapipe = communication.iter.QueueWrapper(
                communication.protocol.IterDataPipeQueueProtocolClient(req_queue, res_queue))
            self.datapipes.append(local_datapipe)

    def __iter__(self):
        exclude_datapipes = []
        not_available = False
        forever = True
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
                shuffle=False,
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
            if batch_sampler is not None:
                raise Exception(
                    'batch_sampler is not yet supported for DataPipes')
            if sampler is not None:
                raise Exception(
                    'sampler is not yet supported for DataPipes')
            datapipe = dataset
            if shuffle:
                datapipe = datapipe.shuffle()
            if batch_outside_worker and pin_memory:
                raise Exception(
                    'pin_memory is not yet compatible with batch_outside_worker')
            if not batch_outside_worker:
                if batch_size is not None:
                    datapipe = datapipe.batch(batch_size, drop_last=drop_last)
                    if collate_fn is None:
                        collate_fn = torch.utils.data._utils.collate.default_collate
            if parallelism_mode == 'mp' or num_workers == 0:
                def sharding_worker_init_fn(worker_init_fn, worker_id):
                    if worker_init_fn is not None:
                        worker_init_fn(worker_id)
                    torch.utils.data.backward_compatibility.worker_init_fn(
                        worker_id)

                my_worker_init_fn = functools.partial(
                    sharding_worker_init_fn, worker_init_fn)

                data_loader = DataLoader(datapipe,
                                         batch_size=None,  # Replaced by .batch DataPipe
                                         shuffle=False,  # Replaced by .shuffle DataPipe
                                         sampler=None,
                                         batch_sampler=None,
                                         num_workers=num_workers,
                                         collate_fn=collate_fn,
                                         pin_memory=pin_memory,
                                         drop_last=False,  # Replaced by .batch DataPipe
                                         timeout=timeout,
                                         worker_init_fn=my_worker_init_fn,
                                         prefetch_factor=prefetch_factor,
                                         persistent_workers=persistent_workers)

                if not batch_outside_worker:
                    return data_loader
                else:
                    if collate_fn is None:
                        collate_fn = torch.utils.data._utils.collate.default_collate
                    datapipe = IterableAsDataPipe(data_loader).batch(
                        batch_size, drop_last=drop_last).map(collate_fn)
                    return datapipe
            elif parallelism_mode == 'thread':
                return _ThreadingDataLoader2(datapipe, num_workers=num_workers)
                # threads = []
                # pipes = []
                # # for i in num_workers:
                # #     (process, req_queue, res_queue) = eventloop.SpawnThreadForDataPipeline(datapipe)
                # #     # TODO(VitalyFedyunin): Add test to check that Spawn creates separate instance of datapipe
                # #     threads.append((process, req_queue, res_queue))
                # #     process.start()
                # #     thread_datapipe = communication_iter.QueueWrapper(
                # #         datapipes_protocol.IterDataPipeQueueProtocolClient(req_queue, res_queue))
                # #     pipes.append(thread_datapipe)

                # def clean_me(process, req_queue, res_queue):
                #     req_queue.put(messages.TerminateRequest())
                #     _ = res_queue.get()
                #     process.join()

            else:
                raise Exception('Unsupported parallelism mode', parallelism_mode)

        else:
            if parallelism_mode != 'thread':
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
