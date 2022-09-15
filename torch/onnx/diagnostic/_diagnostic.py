from __future__ import annotations

import enum
from typing import Any, Optional, Tuple

from torch.onnx import sarif_om
from torch.onnx.diagnostic import formatter


class Level(enum.Enum):
    NONE = "none"
    NOTE = "note"
    WARNING = "warning"
    ERROR = "error"


Rule = sarif_om.ReportingDescriptor
Diagnostic = sarif_om.Result
