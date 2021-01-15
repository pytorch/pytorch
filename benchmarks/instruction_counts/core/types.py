"""Type annotations for various benchmark objects."""
from typing import Any, Dict, Optional, Tuple, Union

from core.api import Mode, TimerArgs, GroupedTimerArgs
from worker.main import WorkerTimerArgs


# =============================================================================
# == Benchmark schema =========================================================
# =============================================================================
""" (There is a TL;DR at the end for ad-hoc benchmarks.)

The end state for representing a benchmark is:
  ```
  Tuple[
      Tuple[
          Tuple[str, ...],      # Primary key
          core.api.Mode,        # Secondary key
          core.api.TimerArgs,   # Value
      ],
      ...
  ]
  ```

For example:
  ```
  [
      (("pointwise", "add"), Mode.PY, TimerArgs(...)),
      (("pointwise", "add"), Mode.CPP, TimerArgs(...)),
      (("pointwise", "add", "with alpha"), Mode.PY, TimerArgs(...)),
      (("pointwise", "add", "with alpha"), Mode.CPP, TimerArgs(...)),
      (("pointwise", "mul"), Mode.PY, TimerArgs(...)),
      (("pointwise", "mul"), Mode.PY_TS, TimerArgs(...)),
      (("pointwise", "mul"), Mode.CPP, TimerArgs(...)),
      (("pointwise", "mul"), Mode.CPP_TS, TimerArgs(...)),
      (("matmul",), Mode.PY, TimerArgs(...)),
      ...
  ]
  ```

However, such a flat list is somewhat tedious to maintain (and read), because
there is significant duplication in the key structure. So instead, we would
like to define something like:
  ```
  {
      "pointwise" : {
          "add": {
              None: GroupedTimerArgs(...),
              "with alpha": GroupedTimerArgs(...),
          },
          "mul": GroupedTimerArgs(...),
      },
      "matmul": GroupedTimerArgs(...),
  }
  ```
and then parse out the flat representation. The type declarations below are
simply formalizing the structure of nested dictionaries with string or tuple
of string keys.


TL;DR
    If you only care about writing an ad-hoc benchmark for a PR, just use a
    flat dictionary and everything will work. For example:
    ```
    {
        "case 0": TimerArgs(...),
        "case 1": TimerArgs(...),
        "case 2": GroupedTimerArgs(...),
        ...
    }
    ```
"""

# Allow strings in definition for convenience, and None to signify a base
# case. (No subsequent entry needed. See the "add" example above.)
Label = Tuple[str, ...]
_Label = Union[Label, Optional[str]]

# MyPy does not currently support recursive types:
#   https://github.com/python/mypy/issues/731
#
# So while the correct type definition would be:
#   _Value = Union[
#       # Base case:
#       Union[TimerArgs, GroupedTimerArgs],
#
#       # Recursive case:
#       Dict[Label, "_Value"],
#   ]
# we instead have to use Any and rely on runtime asserts in the parser.
_Value = Union[
    Union[TimerArgs, GroupedTimerArgs],
    Dict[_Label, Any],
]

Definition = Dict[_Label, _Value]

# We initially have to parse (flatten) to an intermediate state in order to
# build TorchScript models.
FlatIntermediateDefinition = Dict[Label, Union[TimerArgs, GroupedTimerArgs]]

# Final parsed schema.
FlatDefinition = Tuple[Tuple[Label, Mode, WorkerTimerArgs], ...]
