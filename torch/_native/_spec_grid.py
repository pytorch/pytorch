"""Spec-grid expansion shared by the AOT export tool and runtime coverage.

Torch-free on purpose: tools/native_aot/export.py loads this module by
file path in an environment with only the DSL wheel installed, while
torch._native.aot_manifest imports it normally.
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
