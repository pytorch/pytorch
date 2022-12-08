import dataclasses

from typing import Callable, Optional, Set, Tuple


@dataclasses.dataclass
class Hooks:
    guard_export_fn: Optional[Callable[[Set["Guard"]], None]] = None
    guard_fail_fn: Optional[Callable[[Tuple["GuardFail"]], None]] = None
