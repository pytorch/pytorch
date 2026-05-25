"""Backward-compatibility shim. This module was moved to
torch._inductor.heuristics.template. This shim will be removed in a future release.
"""

import warnings


warnings.warn(
    "torch._inductor.template_heuristics has been moved to "
    "torch._inductor.heuristics.template. Please update your imports. "
    "This shim will be removed in a future release.",
    FutureWarning,
    stacklevel=2,
)

from torch._inductor.heuristics.template.registry import get_template_heuristic
