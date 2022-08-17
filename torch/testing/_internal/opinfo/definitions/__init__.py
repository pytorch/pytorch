from typing import List

from torch.testing._internal.opinfo.core import OpInfo
from torch.testing._internal.opinfo.definitions import fft

# Operator database
op_db: List[OpInfo] = [
    *fft.op_db,
]
