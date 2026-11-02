"""The trigger under which Inductor's reinplacing pass counts a decision. It lives
here rather than in torch._dynamo.utils (which re-exports it) so that modules that
register user-kernel HOPs at import time can use it without importing torch._dynamo,
which is imported lazily."""

import enum


class ReInplaceTrigger(enum.Enum):
    AUTO_FUNC_V1 = 1
    AUTO_FUNC_V2 = 2
    TRITON_OPS = 3
    CUTEDSL_OPS = 4
