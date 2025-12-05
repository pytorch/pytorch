import warnings


warnings.warn(
    "The 'torch.utils.data.datapipes' module is deprecated and will be removed in a future version.",
    FutureWarning,
    stacklevel=2,
)

from torch.utils.data.datapipes import dataframe as dataframe, iter as iter, map as map
