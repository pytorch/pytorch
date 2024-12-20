"""Type annotations for various benchmark objects."""

# mypy: ignore-errors

from typing import Dict, Optional, Tuple, Union

from core.api import AutoLabels, GroupedBenchmark, TimerArgs


# =============================================================================
# == Benchmark schema =========================================================
# =============================================================================
""" (There is a TL;DR at the end for ad-hoc benchmarks.)
The end state for representing a benchmark is:
  ```
  Tuple[
      Tuple[
          Tuple[str, ...],      # Primary key
          core.api.AutoLabels,  # Secondary key
          core.api.TimerArgs,   # Value
      ],
      ...
  ]
  ```

For example:
  ```
  [
      (("pointwise", "add"), AutoLabels(..., Language.PYTHON), TimerArgs(...)),
      (("pointwise", "add"), AutoLabels(..., Language.CPP), TimerArgs(...)),
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
              None: GroupedStmts(...),
              "with alpha": GroupedStmts(...),
          },
          "mul": GroupedStmts(...),
      },
      "matmul": GroupedStmts(...),
  }
  ```
and then parse out a flat representation. The type declarations below are
simply formalizing the structure of nested dictionaries with string or tuple
of string keys.

TL;DR
    If you only care about writing an ad-hoc benchmark for a PR, just use a
    flat dictionary and everything will work. For example:
    ```
    {
        "case 0": TimerArgs(...),
        "case 1": TimerArgs(...),
        "case 2": GroupedStmts(...),
        ...
    }
    ```
"""

# Allow strings in definition for convenience, and None to signify a base
# case. (No subsequent entry needed. See the "add" example above.)
Label = Tuple[str, ...]
_Label = Union[Label, Optional[str]]

_Value = Union[
    Union[TimerArgs, GroupedBenchmark],
    Dict[_Label, "_Value"],
]

Definition = Dict[_Label, _Value]

# We initially have to parse (flatten) to an intermediate state in order to
# build TorchScript models since multiple entries will share the same model
# artifact.
FlatIntermediateDefinition = Dict[Label, Union[TimerArgs, GroupedBenchmark]]

# Final parsed schema.
FlatDefinition = Tuple[Tuple[Label, AutoLabels, TimerArgs], ...]
