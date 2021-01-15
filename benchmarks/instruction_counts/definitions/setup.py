from typing import Dict, TYPE_CHECKING

from core.api import Setup, GroupedTimerArgs
from core.utils import parse_stmts

if TYPE_CHECKING:
    # See core.api for an explanation.
    from torch.utils.benchmark.utils.timer import Language
else:
    from torch.utils.benchmark import Language


def _setup_from_string(setup_stmts: str) -> Dict[Language, str]:
    py_stmt, cpp_stmt = parse_stmts(setup_stmts)
    return {
        Language.PYTHON: py_stmt,
        Language.CPP: cpp_stmt,
    }


SETUP_MAP: Dict[Setup, Dict[Language, str]] = {
    Setup.NONE: {
        Language.PYTHON: r"pass",
        Language.CPP: r"",
    },

    Setup.TRIVIAL: {
        Language.PYTHON: r"x = torch.ones((4, 4))",
        Language.CPP: r"auto x = torch::ones({4, 4});",
    },

    Setup.GENERIC: _setup_from_string(
        r"""
            Python                                   | C++
            ---------------------------------------- | ----------------------------------------
            torch.manual_seed(138_10_23)             | torch::manual_seed(1381023);
            x = torch.rand((4, 4))                   | auto x = torch::rand({4, 4});
            y_float = torch.ones((4, 4))             | auto y_float = torch::ones({4, 4});
            y_int = torch.ones(                      | auto y_int = torch::ones({4, 4}, at::kInt);
                (4, 4), dtype=torch.int32)           |
        """
    ),

    Setup.INDEXING: _setup_from_string(
        r"""
            Python                                   | C++
            ---------------------------------------- | ----------------------------------------
                                                     | using namespace torch::indexing;
            torch.manual_seed(6626_10_34)            | torch::manual_seed(66261034);
                                                     |
            x = torch.randn(1, 1, 1)                 | auto x = torch::randn({1, 1, 1});
            y = torch.randn(1, 1, 1)                 | auto y = torch::randn({1, 1, 1});
            a = torch.zeros(100, 100, 1, 1, 1)       | auto a = torch::zeros({100, 100, 1, 1, 1});
            b = torch.arange(100).long().flip(0)     | auto b = torch::arange(
                                                     |     0, 100, torch::TensorOptions().dtype(torch::kLong)).flip(0);
        """
    ),

    Setup.MESOSCALE: _setup_from_string(
        r"""
            Python                                   | C++
            ---------------------------------------- | ----------------------------------------
            x = torch.ones((4, 4))                   | auto x = torch::ones({4, 4});
            y = torch.ones((4, 4))                   | auto y = torch::ones({4, 4});
            bias = torch.ones((1,))                  | auto bias = torch::ones({1});
            eye_4 = torch.eye(4)                     | auto eye_4 = torch::eye(4);
        """
    ),

    Setup.AUTOGRAD: _setup_from_string(
        r"""
            Python                                   | C++
            ---------------------------------------- | ----------------------------------------
            # Inputs                                 | // Inputs
            x = torch.ones((1,))                     | auto x = torch::ones({1});
            y = torch.ones((1,))                     | auto y = torch::ones({1});
                                                     |
            # Weights                                | // Weights
            w0 = torch.ones(                         | auto w0 = torch::ones({1});
                (1,), requires_grad=True)            | w0.set_requires_grad(true);
            w1 = torch.ones(                         | auto w1 = torch::ones({1});
                (1,), requires_grad=True)            | w1.set_requires_grad(true);
            w2 = torch.ones(                         | auto w2 = torch::ones({2});
                (2,), requires_grad=True)            | w2.set_requires_grad(true);
        """
    ),

    Setup.EXAMPLE_FOR_ADHOC: {
        Language.PYTHON: r"x = torch.ones((1,))",
        Language.CPP: r"auto x = torch::ones({1});",
    },
}


# Ensure map is complete.
assert tuple(Setup) == tuple(SETUP_MAP.keys())
for k in Setup:
    assert tuple(SETUP_MAP[k].keys()) == (Language.PYTHON, Language.CPP)
