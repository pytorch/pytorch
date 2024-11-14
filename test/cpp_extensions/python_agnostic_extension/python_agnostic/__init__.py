from pathlib import Path

import torch


so_files = list(Path(__file__).parent.glob("_C*.so"))
assert len(so_files) == 1, f"Expected one _C*.so file, found {len(so_files)}"
torch.ops.load_library(so_files[0])

from . import ops
