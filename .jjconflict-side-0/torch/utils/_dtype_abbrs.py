import torch


# Used for testing and logging
dtype_abbrs = {dt: dt.abbr for dt in torch._C._get_all_dtypes()}
