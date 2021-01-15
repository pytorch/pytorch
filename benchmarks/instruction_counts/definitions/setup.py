from typing import Dict, TYPE_CHECKING

from core.api import Setup, GroupedTimerArgs

if TYPE_CHECKING:
    # See core.api for an explanation.
    from torch.utils.benchmark.utils.timer import Language
else:
    from torch.utils.benchmark import Language


SETUP_MAP: Dict[Setup, Dict[Language, str]] = {
    Setup.NONE: {
        Language.PYTHON: r"pass",
        Language.CPP: r"",
    },

    Setup.TRIVIAL: {
        Language.PYTHON: r"x = torch.ones((4, 4))",
        Language.CPP: r"auto x = torch::ones({4, 4});",
    },
}

# Ensure map is complete.
assert tuple(Setup) == tuple(SETUP_MAP.keys())
for k in Setup:
    assert tuple(SETUP_MAP[k].keys()) == (Language.PYTHON, Language.CPP)
