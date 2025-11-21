import warnings

warnings.warn(
    "The 'torch.utils.data.datapipes.dataframe' module is deprecated and will be removed in a future version.",
    FutureWarning,
    stacklevel=2,
)

from torch.utils.data.datapipes.dataframe.dataframes import (
    CaptureDataFrame,
    DFIterDataPipe,
)
from torch.utils.data.datapipes.dataframe.datapipes import DataFramesAsTuplesPipe


__all__ = ["CaptureDataFrame", "DFIterDataPipe", "DataFramesAsTuplesPipe"]

# Please keep this list sorted
if __all__ != sorted(__all__):
    raise AssertionError("__all__ is not sorted")
