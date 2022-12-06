import dataclasses

from typing import Callable, Set, Tuple, Optional


@dataclasses.dataclass
class Hooks:
    guard_export_fn: Optional[Callable[[Set["Guard"]], None]]
    guard_fail_fn: Optional[Callable[[Tuple["GuardFail"]], None]]
