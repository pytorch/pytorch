from torch.utils.data import IterDataPipe, functional_datapipe, DataChunk
from typing import Callable, TypeVar, Iterator, Optional, Tuple, Dict

from .callable import MapIterDataPipe

import pandas

T_co = TypeVar('T_co', covariant=True)


@functional_datapipe('filter')
class FilterIterDataPipe(MapIterDataPipe):
    r""" :class:`FilterIterDataPipe`.

    Iterable DataPipe to filter elements from datapipe according to filter_fn.
    args:
        datapipe: Iterable DataPipe being filterd
        filter_fn: Customized function mapping an element to a boolean.
        fn_args: Positional arguments for `filter_fn`
        fn_kwargs: Keyword arguments for `filter_fn`
    """
    def __init__(self,
                 datapipe: IterDataPipe[T_co],
                 filter_fn: Callable[..., bool],
                 fn_args: Optional[Tuple] = None,
                 fn_kwargs: Optional[Dict] = None,
                 batch_level: bool = False,
                 drop_empty_batches: bool = True,
                 nesting_level:int = 0,
                 ) -> None:
        self.drop_empty_batches = drop_empty_batches
        super().__init__(datapipe, fn=filter_fn, fn_args=fn_args, fn_kwargs=fn_kwargs, nesting_level = nesting_level)

    def _merge(self, data, mask):
        result = []
        # print('merging')
        # print(data)
        # print(mask)
        # print('merging start')
        is_df = False
        # print(data.__class__)
        # print(mask.__class__)
        chunk_type = data.__class__
        if isinstance(data, DataChunk):
            # print('using raw iterator')
            data_iterator = list(data.raw_iterator())
        else:
            data_iterator = data

        if isinstance(mask, DataChunk):
            # print('using raw iterator (mask)')
            mask_iterator = list(mask.raw_iterator())
        else:
            mask_iterator = mask
        
        # print(data_iterator)
        for i,b in zip(data_iterator, mask_iterator):
            if isinstance(b, DataChunk):
                t = self._merge(i, b)
                if len(t) > 0 or not self.drop_empty_batches:
                    result.append(t)
            if isinstance(b, pandas.core.series.Series):
                is_df = True
                # print('iterating inside DF')
                for idx,mask in enumerate(b):
                    # print(mask)
                    # print(row)
                    if mask:
                        result.append(i[idx:idx+1])
            else:
                # print('combining', type(b), type(i))
                # print(b)
                # print(i)
                if b:
                    result.append(i)

        if is_df:
            if len(result) > 0:
                return chunk_type([pandas.concat(result)])
            else:
                return chunk_type([])
        return chunk_type(result)

    def __iter__(self) -> Iterator[T_co]:
        res: bool
        for data in self.datapipe:
            if self.nesting_level == 0:
                res = self.fn(data, *self.args, **self.kwargs)
                if not isinstance(res, bool):
                    raise ValueError("Boolean output is required for "
                                    "`filter_fn` of FilterIterDataPipe")
                if res:
                    yield data
            else:
                mask = self._apply(data, self.nesting_level, self.fn, self.args, self.kwargs)
                # print(mask)
                merged = self._merge(data, mask)
                # print('merge resutl')
                # print(merged)
                # print('done')
                if len(merged) > 0 or not self.drop_empty_batches:
                    yield merged
                
            

    def __len__(self):
        raise(NotImplementedError)
