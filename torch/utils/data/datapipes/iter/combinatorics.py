import random

from torch.utils.data import IterDataPipe, Sampler, SequentialSampler, functional_datapipe
from typing import TypeVar, Type, Iterator, Sized, Optional, Tuple, Dict, List

T_co = TypeVar('T_co', covariant=True)


class SamplerIterDataPipe(IterDataPipe[T_co]):
    r""" :class:`SamplerIterDataPipe`.

    Iterable DataPipe to generate sample elements.
    args:
        datapipe: IterDataPipe sampled from
        sampler: Sampler class to genereate sample elements from input DataPipe.
                    Default is :class:`SequentialSampler` for IterDataPipe
    """
    datapipe: IterDataPipe
    sampler: Sampler

    def __init__(self,
                 datapipe: IterDataPipe,
                 sampler: Type[Sampler] = SequentialSampler,
                 sampler_args: Optional[Tuple] = None,
                 sampler_kwargs: Optional[Dict] = None
                 ) -> None:
        assert isinstance(datapipe, Sized), \
            "Sampler class requires input datapipe implemented `__len__`"
        super().__init__()
        self.datapipe = datapipe
        self.sampler_args = () if sampler_args is None else sampler_args
        self.sampler_kwargs = {} if sampler_kwargs is None else sampler_kwargs
        # https://github.com/python/mypy/pull/9629 will solve
        self.sampler = sampler(data_source=self.datapipe, *self.sampler_args, **self.sampler_kwargs)  # type: ignore[misc]

    def __iter__(self) -> Iterator[T_co]:
        return iter(self.sampler)

    def __len__(self) -> int:
        # Dataset has been tested as `Sized`
        if isinstance(self.sampler, Sized) and len(self.sampler) >= 0:
            return len(self.sampler)
        raise NotImplementedError


@functional_datapipe('shuffle')
class ShuffleIterDataPipe(IterDataPipe[T_co]):
    r""" :class:`ShuffleIterDataPipe`

    Iterable DataPipe to shuffle the input DataPipe with a buffer. The buffer
    with `buffer_size` is filled with elements from the datapipe first. Then,
    each item will be yielded from the buffer by reservoir sampling via iterator.

    `buffer_size` is required to be larger than 0. For `buffer_size == 1`, the
    datapipe is not shuffled. In order to fully shuffle all elements from datapipe,
    `buffer_size` is required to be greater than or equal to the size of datapipe.

    When it is used with :class:`~torch.utils.data.DataLoader`, the methods to
    set up random seed are different based on :attr:`num_workers`.

    For single-process mode (:attr:`num_workers == 0`), the random seed is set before
    the :class:`~torch.utils.data.DataLoader` in the main process. For multi-process
    mode (:attr:`num_worker > 0`), `worker_init_fn` is used to set up a random seed
    for each worker process.

    args:
        datapipe: The IterDataPipe being shuffled
        buffer_size: The buffer size for shuffling
    """
    datapipe: IterDataPipe[T_co]
    buffer_size: int
    _buffer: List[T_co]

    def __init__(self,
                 datapipe: IterDataPipe[T_co],
                 *,
                 buffer_size: int = 10000,
                 batch_level: bool = False,
                 cross_shuffle: bool = True) -> None:
        super().__init__()
        assert buffer_size > 0, "buffer_size should be larger than 0"
        self.datapipe = datapipe
        self.buffer_size = buffer_size
        self._buffer = []
        self._sizes = []
        self.batch_level = batch_level
        self.cross_shuffle = cross_shuffle
        
        
    def buffer_replace(self, x):
        idx = random.randint(0, self.buffer_size - 1)
        val = self._buffer[idx]
        self._buffer[idx] = x
        return val

    def __iter__(self) -> Iterator[T_co]:
        # TODO: Buffer is global, should be per __iter__ !!!
        # TODO: Works bad with size = 0 batches
        if self.cross_shuffle or self.batch_level:
            batch_level = self.batch_level    
            check_first = True
            response = []
            for x in self.datapipe:
                if check_first and not isinstance(x, list):
                    batch_level = True
                check_first = False
                if batch_level:
                    if len(self._buffer) == self.buffer_size:
                        yield self.buffer_replace(x)
                    else:
                        self._buffer.append(x)
                else:
                    if len(x) > 0:
                        
                        self._sizes.append(len(x))
                        # print('sizes',self._sizes)
                        for i in x:
                            if len(self._buffer) == self.buffer_size:
                                response.append(self.buffer_replace(i))
                                if len(response) == self._sizes[0]:
                                    yield response
                                    response = []
                                    self._sizes.pop(0)
                                    # print('sizes',self._sizes)

                            else:
                                self._buffer.append(i)

            random.shuffle(self._buffer)

            if batch_level:    
                while self._buffer:
                    yield self._buffer.pop()
            else:
                for i in self._buffer:
                    response.append(i)
                    if len(response) == self._sizes[0]:
                        yield response
                        response = []
                        self._sizes.pop(0)
                        # print('sizes',self._sizes)

                    else:
                        # print('meed', self._sizes[0])
                        pass
        else:
            for x in self.datapipe:
                if isinstance(x, list):
                    l = [ i for i in x]
                    random.shuffle(l)
                    yield l
                else:
                    yield x

    
    def __len__(self) -> int:
        if isinstance(self.datapipe, Sized) and len(self.datapipe) >= 0:
            return len(self.datapipe)
        raise NotImplementedError
