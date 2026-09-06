#!/usr/bin/env python3
"""Report the active torch/Triton environment as JSON, for dev.py.

Used by `doctor`, and by `switch` -- which needs the backend to choose a
provider, and the cache directories to clear once one is installed.

A subprocess for two reasons. The report has to reflect what is installed right
now, not what the parent imported before a provider swap; and the parent is
usually run from the repo root, where `import torch` would find the source tree
rather than the installed package. dev.py runs this with cwd=/.
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import json
import os


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

    # Resolved here rather than in dev.py so the torch import happens in
    # this process, which runs from / and so cannot pick up the source tree
    # instead of the installed package. default_cache_dir() handles
    # tempfile.gettempdir() vs /var/tmp under fbcode and username sanitisation.
    cache_dirs = []
    try:
        from torch._inductor.runtime.cache_dir_utils import default_cache_dir

        cache_dirs.append(
            os.environ.get("TORCHINDUCTOR_CACHE_DIR") or default_cache_dir()
        )
    except Exception as e:
        out["cache_dir_error"] = repr(e)
    # Inductor puts the Triton cache inside the directory above, so it is a
    # separate target only when TRITON_CACHE_DIR moves it out.
    if os.environ.get("TRITON_CACHE_DIR"):
        cache_dirs.append(os.environ["TRITON_CACHE_DIR"])
    out["cache_dirs"] = cache_dirs

    print("<<<JSON>>>" + json.dumps(out))


if __name__ == "__main__":
    main()
