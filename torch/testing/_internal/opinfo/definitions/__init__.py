# mypy: ignore-errors

from typing import List

from torch.testing._internal.opinfo.core import OpInfo
from torch.testing._internal.opinfo.definitions import (
    _masked,
    fft,
    linalg,
    signal,
    special,
)


# Operator database
op_db: List[OpInfo] = [
    *fft.op_db,
    *linalg.op_db,
    *signal.op_db,
    *special.op_db,
    *_masked.op_db,
]

python_ref_db: List[OpInfo] = [
    *fft.python_ref_db,
    *linalg.python_ref_db,
    *special.python_ref_db,
]
