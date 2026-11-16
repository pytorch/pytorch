"""The trigger under which Inductor's reinplacing pass counts a decision. It lives
here rather than in torch._dynamo.utils (which re-exports it) so that modules that
register user-kernel HOPs at import time can use it without importing torch._dynamo,
which is imported lazily."""

import enum


class ReInplaceTrigger(enum.Enum):
    AUTO_FUNC_V1 = 1
    AUTO_FUNC_V2 = 2
    TRITON_OPS = 3
    # A user-kernel DSL without a member of its own.
    USER_KERNEL_OPS = 4


# The DSL that each DSL-specific member belongs to, for register_user_kernel_hop,
# which refuses a member to any other DSL whichever of them registers first, and a
# member without an entry to every DSL: a DSL that adds a member adds its entry.
TRIGGER_DSLS: dict[ReInplaceTrigger, str] = {
    ReInplaceTrigger.TRITON_OPS: "triton",
}
