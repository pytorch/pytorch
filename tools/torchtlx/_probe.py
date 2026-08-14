#!/usr/bin/env python3
"""Report the active torch/Triton environment as JSON, for bringup.py doctor.

Run as a subprocess so `doctor` reflects what is installed right now rather
than what the parent process imported before a provider swap.
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import json


TLX_REGISTRY = "triton.language.extra.tlx.inductor.registry"

# Every distribution that can provide the top-level `triton` package.
TRITON_DISTRIBUTIONS = [
    "fbtriton",
    "triton",
    "pytorch-triton",
    "pytorch-triton-rocm",
    "triton-rocm",
]


def main() -> None:
    out: dict = {}

    import torch

    out["torch"] = torch.__version__
    out["torch_file"] = torch.__file__
    out["hip"] = torch.version.hip
    out["cuda"] = torch.version.cuda
    out["device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if out["device_count"]:
        out["device_name"] = torch.cuda.get_device_name(0)
        out["gcn_arch"] = getattr(
            torch.cuda.get_device_properties(0), "gcnArchName", None
        )
    else:
        out["device_name"] = None
        out["gcn_arch"] = None

    try:
        import triton
        import triton.backends

        out["triton"] = triton.__version__
        out["triton_file"] = triton.__file__
        out["backends"] = sorted(triton.backends.backends.keys())
    except Exception as e:
        out["triton_error"] = repr(e)

    dists = {}
    for dist in TRITON_DISTRIBUTIONS:
        try:
            dists[dist] = md.version(dist)
        except md.PackageNotFoundError:
            pass
    out["dists"] = dists

    try:
        importlib.import_module(TLX_REGISTRY)
        out["tlx_registry"] = True
    except Exception as e:
        out["tlx_registry"] = False
        out["tlx_error"] = repr(e)

    print("<<<JSON>>>" + json.dumps(out))


if __name__ == "__main__":
    main()
