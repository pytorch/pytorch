from typing import List

from torch.testing._internal.opinfo.core import OpInfo
from torch.testing._internal.opinfo.definitions import fft, linalg, special

# Operator database
op_db: List[OpInfo] = [
    *fft.op_db,
    *linalg.op_db,
    *special.op_db,
]

python_ref_db: List[OpInfo] = [
    *fft.python_ref_db,
    *linalg.python_ref_db,
    *special.python_ref_db,
]
