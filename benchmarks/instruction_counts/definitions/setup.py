import enum

from core.api import GroupedSetup

_TRIVIAL_2D = GroupedSetup(
    r"x = torch.ones((4, 4))",
    r"auto x = torch::ones({4, 4});"
)

class Setup(enum.Enum):
    TRIVIAL_2D = _TRIVIAL_2D
