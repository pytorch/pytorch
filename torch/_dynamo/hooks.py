import dataclasses
from typing import Callable, List, Optional

from torch._guards import GuardsSet

from .types import GuardFail


@dataclasses.dataclass
class Hooks:
    guard_export_fn: Optional[Callable[[GuardsSet], None]] = None
    guard_fail_fn: Optional[Callable[[GuardFail], None]] = None
    guard_filter_fn: Optional[Callable[[GuardsSet], GuardsSet]] = None
    symbolic_guard_filter_fn: Optional[Callable[[List[str]], List[str]]] = None
