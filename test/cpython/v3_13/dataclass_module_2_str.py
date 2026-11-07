from __future__ import annotations
USING_STRINGS = True

# dataclass_module_2.py and dataclass_module_2_str.py are identical
# except only the latter uses string annotations.

from dataclasses import dataclass, InitVar
from typing import ClassVar

T_CV2 = ClassVar[int]
T_CV3 = ClassVar

T_IV2 = InitVar[int]
T_IV3 = InitVar

@dataclass
class CV:
    T_CV4 = ClassVar
    cv0: ClassVar[int] = 20
    cv1: ClassVar = 30
    cv2: T_CV2
    cv3: T_CV3
    not_cv4: T_CV4  # When using string annotations, this field is not recognized as a ClassVar.

@dataclass
class IV:
    T_IV4 = InitVar
    iv0: InitVar[int]
    iv1: InitVar
    iv2: T_IV2
    iv3: T_IV3
    not_iv4: T_IV4  # When using string annotations, this field is not recognized as an InitVar.
