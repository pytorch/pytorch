from __future__ import annotations

import dataclasses
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from .package import CompilePackage


@dataclasses.dataclass
class DynamoCompileOptions:
    """Options threaded through the torch.compile → dynamo.optimize → convert_frame chain.

    Adding a new compile option only requires adding a field here and wiring it
    at the production site (torch.compile) and consumption site. No intermediate
    plumbing changes needed.
    """

    dynamic: bool | None = None
    fullgraph: bool = False
    error_on_graph_break: bool | None = None
    export: bool = False
    export_constraints: Any | None = None
    one_graph: bool = False
    recompile_limit: int | None = None
    package: CompilePackage | None = None
    compiler_config: Any | None = None
