
import functools

import torch.utils.data.backward_compatibility
from torch.utils.data import DataLoader, IterDataPipe
from torch.utils.data.datapipes.iter import IterableAsDataPipe

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
                batch_outside_worker=False):
        if isinstance(dataset, IterDataPipe):
            datapipe = dataset
            if batch_sampler is not None:
                raise Exception(
                    'batch_sampler is not yet supported for DataPipes')
            if sampler is not None:
                raise Exception(
                    'sampler is not yet supported for DataPipes')
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

        else:
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
