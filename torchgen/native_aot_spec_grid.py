"""Spec-grid expansion shared by the AOT export tool and runtime coverage.

Lives in torchgen rather than under torch/ because both consumers need it
at times when torch may not be importable, and torchgen is the sanctioned
home for build-time-importable shared code: it is pure Python, it ships in
the wheel (pyproject.toml `packages`), and torch already depends on it at
import time (see torch/utils/_python_dispatch.py). That lets
tools/native_aot/export.py import this normally -- including in the
linter image, where there is no built torch -- instead of loading it by
file path.

Consumers: tools/native_aot/export.py (grid fan-out at export time) and
torch._native.aot_manifest (matching a live call against the grid).
"""

import itertools


def expand_specs(specs: list[dict]) -> list[dict]:
    """Cross-multiply list-valued fields of each spec block; concatenate
    blocks. Scalars are singleton axes."""
    points = []
    for spec in specs:
        keys = list(spec.keys())
        axes = [v if isinstance(v, list) else [v] for v in spec.values()]
        points.extend(dict(zip(keys, combo)) for combo in itertools.product(*axes))
    return points
