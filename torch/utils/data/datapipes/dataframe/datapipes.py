# mypy: allow-untyped-defs
import random
from typing import Any

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.dataframe import dataframe_wrapper as df_wrapper
from torch.utils.data.datapipes.datapipe import DFIterDataPipe, IterDataPipe


__all__ = [
    "ConcatDataFramesPipe",
    "DataFramesAsTuplesPipe",
    "ExampleAggregateAsDataFrames",
    "FilterDataFramesPipe",
    "PerRowDataFramesPipe",
    "ShuffleDataFramesPipe",
]


@functional_datapipe("_dataframes_as_tuples")
class DataFramesAsTuplesPipe(IterDataPipe):
    def __init__(self, source_datapipe):
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for df in self.source_datapipe:
            # for record in df.to_records(index=False):
            yield from df_wrapper.iterate(df)


@functional_datapipe("_dataframes_per_row", enable_df_api_tracing=True)
class PerRowDataFramesPipe(DFIterDataPipe):
    def __init__(self, source_datapipe):
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for df in self.source_datapipe:
            # TODO(VitalyFedyunin): Replacing with TorchArrow only API, as we are dropping pandas as followup
            for i in range(len(df)):
                yield df[i : i + 1]


@functional_datapipe("_dataframes_concat", enable_df_api_tracing=True)
class ConcatDataFramesPipe(DFIterDataPipe):
    def __init__(self, source_datapipe, batch=3):
        self.source_datapipe = source_datapipe
        self.n_batch = batch

    def __iter__(self):
        buffer = []
        for df in self.source_datapipe:
            buffer.append(df)
            if len(buffer) == self.n_batch:
                yield df_wrapper.concat(buffer)
                buffer = []
        if len(buffer):
            yield df_wrapper.concat(buffer)


@functional_datapipe("_dataframes_shuffle", enable_df_api_tracing=True)
class ShuffleDataFramesPipe(DFIterDataPipe):
    def __init__(self, source_datapipe):
        self.source_datapipe = source_datapipe

    def __iter__(self):
        size = None
        all_buffer: list[Any] = []
        for df in self.source_datapipe:
            if size is None:
                size = df_wrapper.get_len(df)
            all_buffer.extend(
                df_wrapper.get_item(df, i) for i in range(df_wrapper.get_len(df))
            )
        random.shuffle(all_buffer)
        buffer = []
        for df in all_buffer:
            buffer.append(df)
            if len(buffer) == size:
                yield df_wrapper.concat(buffer)
                buffer = []
        if len(buffer):
            yield df_wrapper.concat(buffer)


@functional_datapipe("_dataframes_filter", enable_df_api_tracing=True)
class FilterDataFramesPipe(DFIterDataPipe):
    def __init__(self, source_datapipe, filter_fn):
        self.source_datapipe = source_datapipe
        self.filter_fn = filter_fn

    def __iter__(self):
        size = None
        all_buffer = []
        filter_res = []
        # pyrefly: ignore  # bad-assignment
        for df in self.source_datapipe:
            if size is None:
                size = len(df.index)
            for i in range(len(df.index)):
                all_buffer.append(df[i : i + 1])
                filter_res.append(self.filter_fn(df.iloc[i]))

        buffer = []
        for df, res in zip(all_buffer, filter_res):
            if res:
                buffer.append(df)
                if len(buffer) == size:
                    yield df_wrapper.concat(buffer)
                    buffer = []
        if len(buffer):
            yield df_wrapper.concat(buffer)


@functional_datapipe("_to_dataframes_pipe", enable_df_api_tracing=True)
class ExampleAggregateAsDataFrames(DFIterDataPipe):
    def __init__(self, source_datapipe, dataframe_size=10, columns=None):
        self.source_datapipe = source_datapipe
        self.columns = columns
        self.dataframe_size = dataframe_size

    def _as_list(self, item):
        try:
            return list(item)
        except (
            Exception
        ):  # TODO(VitalyFedyunin): Replace with better iterable exception
            return [item]

    def __iter__(self):
        aggregate = []
        for item in self.source_datapipe:
            aggregate.append(self._as_list(item))
            if len(aggregate) == self.dataframe_size:
                yield df_wrapper.create_dataframe(aggregate, columns=self.columns)
                aggregate = []
        if len(aggregate) > 0:
            yield df_wrapper.create_dataframe(aggregate, columns=self.columns)
