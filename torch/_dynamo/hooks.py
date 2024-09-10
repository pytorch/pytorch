import dataclasses
from typing import Callable, Optional

from torch._guards import GuardsSet

from .types import GuardFail


@dataclasses.dataclass
class Hooks:
    guard_export_fn: Optional[Callable[[GuardsSet], None]] = None
    guard_fail_fn: Optional[Callable[[GuardFail], None]] = None
