from typing import Any, Iterator

from torch.utils.data.datapipes.dataframe import dataframe_wrapper as df_wrapper
from torch.utils.data.datapipes.datapipe import DataChunk


__all__ = ["DataChunkDF"]


class DataChunkDF(DataChunk):
    """DataChunkDF iterating over individual items inside of DataFrame containers, to access DataFrames user `raw_iterator`."""

    def __iter__(self) -> Iterator[Any]:
        for df in self.items:
            yield from df_wrapper.iterate(df)

    def __len__(self) -> int:
        total_len = 0
        for df in self.items:
            total_len += df_wrapper.get_len(df)
        return total_len
