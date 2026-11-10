"""The parity fixture shared with pytorch/test-infra
(tools/torchci/tests/td/fixtures/td_scorer_parity.json there): both repos'
evaluators must give these numbers for this model."""

from __future__ import annotations

from typing import Any


NAN = float("nan")

DOC: dict[str, Any] = {
    "format": 1,
    "enabled": True,
    "trained_at": "2026-11-10T00:00:00Z",
    "data_through": "2026-11-07T00:00:00Z",
    "source": {"run_id": 1, "commit": "fixture"},
    "calibrated": True,
    "features": ["fail_n", "test_seconds", "is_pr"],
    "scorer": {
        "baseline": -3.0,
        "trees": [
            {
                "feature": [0, -1, 1, -1, -1],
                "threshold": [0.1, 0.0, 60.0, 0.0, 0.0],
                "missing_left": [True, False, False, False, False],
                "left": [1, 0, 3, 0, 0],
                "right": [2, 0, 4, 0, 0],
                "value": [0.0, -0.5, 0.0, 0.3, 0.8],
                "leaf": [False, True, False, True, True],
            },
            {
                "feature": [2, -1, -1],
                "threshold": [0.5, 0.0, 0.0],
                "missing_left": [False, False, False],
                "left": [1, 0, 0],
                "right": [2, 0, 0],
                "value": [0.0, -0.2, 0.1],
                "leaf": [False, True, True],
            },
            {
                "feature": [2, -1, -1],
                "threshold": [1e308, 0.0, 0.0],
                "missing_left": [False, False, False],
                "left": [1, 0, 0],
                "right": [2, 0, 0],
                "value": [0.0, 0.05, -0.05],
                "leaf": [False, True, True],
            },
        ],
    },
    "platt": {"coef": 0.9, "intercept": -0.4, "clip": 1e-07},
    "risk": {
        "features": ["log_files", "path_revert_max", "path_revert_mean"],
        "fill": [0.0, -2.0, -2.5],
        "mean": [1.0, -2.0, -2.5],
        "scale": [0.5, 0.6, 0.8],
        "coef": [0.7, 0.9, 0.3],
        "intercept": -1.0,
        "quantiles": [0.5, 0.7, 0.85, 0.9, 0.95],
        "share_min": 0.1,
        "share_max": 0.2,
        "paths": {
            "base": 0.1,
            "nodes": {
                "torch/": 0.12,
                "torch/_inductor/": 0.25,
                "torch/_inductor/ir.py": 0.4,
                "aten/": 0.05,
                "docs/": 0.0,
            },
        },
    },
    "gate": {},
}

ROWS = [
    [0.05, 30.0, 1.0],
    [0.2, 30.0, 0.0],
    [0.2, 120.0, 1.0],
    [NAN, NAN, 1.0],
    [0.2, NAN, NAN],
    [0.1, 60.0, 0.5],
]

EXPECTED_P = [
    0.031829952246,
    0.049032338846,
    0.095781628096,
    0.031829952246,
    0.088265460687,
    0.024482727435,
]

CHANGED = [
    "torch/_inductor/ir.py",
    "torch/_inductor/lowering.py",
    "torch/nn/modules/linear.py",
    "aten/src/ATen/native/Foo.cpp",
    "docs/source/index.rst",
    "setup.py",
    "torch/_inductor/ir.py",
]

# The risk's values: {"log_files": log1p(distinct changed files)} and the
# path features of the changed files under DOC's risk paths.
RISK_CASE: dict[str, Any] = {
    "changed": CHANGED,
    "expected_risk": 0.898818046851,
    "expected_percentile": 0.6,
    "expected_share": 0.16,
}

PATHS_CASE: dict[str, Any] = {
    "paths": DOC["risk"]["paths"],
    "changed": CHANGED,
    "expected": {
        "path_revert_max": -0.916290731874,
        "path_revert_mean": -1.875141078167,
    },
    "docs_changed": ["docs/a.md"],
    "docs_expected": {
        "path_revert_max": -13.815510557964,
        "path_revert_mean": -13.815510557964,
    },
    "long_changed": ["torch/" + "x" * 600 + "/y.py"],
    "long_expected": {
        "path_revert_max": -2.1202635362,
        "path_revert_mean": -2.1202635362,
    },
}

BUDGET_CUT_CASE: dict[str, Any] = {
    "tests": ["a", "b", "c", "d", "e", "f", "g", "h"],
    "p": {
        "a": 0.3,
        "b": 0.2,
        "c": 0.15,
        "d": 0.1,
        "e": 0.05,
        "f": 0.01,
        "g": 0.02,
        "h": 0.004,
    },
    "cost": {
        "a": 415.0,
        "b": 25.0,
        "c": 315.0,
        "d": 35.0,
        "e": 35.0,
        "f": 20.0,
        "g": 18.0,
        "h": 380.0,
    },
    "budget_share": 0.25,
    "expected_selected": ["b", "d", "e", "g", "a"],
    "expected_selected_seconds": 528.0,
    "expected_total_seconds": 1243.0,
    "expected_budget_seconds": 310.75,
}

NUMSTAT_CASE: dict[str, Any] = {
    "records": [
        {"added": 10, "deleted": 2, "path": "torch/_inductor/ir.py", "old_path": None},
        {
            "added": 5,
            "deleted": 5,
            "path": "test/inductor/test_ir.py",
            "old_path": None,
        },
        {"added": None, "deleted": None, "path": "torch/lib/blob.so", "old_path": None},
        {
            "added": 3,
            "deleted": 1,
            "path": "torch/utils/new_name.py",
            "old_path": "docs/old_name.md",
        },
        {"added": 7, "deleted": 0, "path": "docs/source/x.rst", "old_path": None},
        {"added": 4, "deleted": 4, "path": "requirements.txt", "old_path": None},
    ],
    "expected": {
        "sig_added": 13.0,
        "sig_deleted": 3.0,
        "log_sig_added": 2.639057329615,
        "log_sig_deleted": 1.38629436112,
    },
}

STATS_CASE: dict[str, Any] = {
    "changed": [
        "torch/_inductor/codegen/triton.py",
        "aten/src/ATen/native/Foo.cpp",
        "test/inductor/test_foo.py",
        "docs/source/foo.rst",
        ".github/workflows/x.yml",
        "requirements-ci.txt",
        "CMakeLists.txt",
    ],
    "expected": {
        "n_files": 7.0,
        "n_src_files": 2.0,
        "n_test_files": 1.0,
        "docs_only": 0.0,
        "area_core_cpp": 1.0,
        "area_inductor": 1.0,
        "area_dynamo": 0.0,
        "area_distributed": 0.0,
        "area_build": 1.0,
        "area_ci": 1.0,
    },
    "docs_only_changed": ["README.md", "docs/source/a.rst"],
    "docs_only_expected": 1.0,
}
