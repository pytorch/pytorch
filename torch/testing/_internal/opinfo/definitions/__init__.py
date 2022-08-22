from typing import List

from torch.testing._internal.opinfo.core import OpInfo
from torch.testing._internal.opinfo.definitions import fft, linalg

# Operator database
op_db: List[OpInfo] = [
    *fft.op_db,
    *linalg.op_db,
]

python_ref_db: List[OpInfo] = [
    *fft.python_ref_db,
    *linalg.python_ref_db,
]
