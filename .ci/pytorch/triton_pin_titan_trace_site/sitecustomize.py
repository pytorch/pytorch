"""Observe Triton imports and compile requests in every strict SFT subprocess."""

import functools
import importlib.metadata
import json
import os
import sys
import traceback
from pathlib import Path


def install_trace():
    import triton
    from triton.backends.amd.compiler import HIPBackend
    from triton.backends.nvidia.compiler import CUDABackend

    trace_dir = Path(os.environ["PIN_TRITON_TRACE_DIR"])
    trace_dir.mkdir(parents=True, exist_ok=True)

    def record(event, **data):
        with (trace_dir / f"{os.getpid()}.jsonl").open("a") as output:
            output.write(
                json.dumps(
                    {
                        "event": event,
                        "pid": os.getpid(),
                        **data,
                        "triton_file": triton.__file__,
                        "triton_core_file": triton._C.libtriton.__file__,
                    }
                )
                + "\n"
            )

    def flags():
        return {
            "requested_true": triton.knobs.language.fp_fusion_enabled(True),
            "requested_none": triton.knobs.language.fp_fusion_enabled(None),
        }

    record(
        "import",
        triton_file=triton.__file__,
        triton_version=importlib.metadata.version("triton"),
        executable=sys.executable,
        argv=sys.argv,
        flags=flags(),
    )

    def trace_parse_options(original_parse):
        @functools.wraps(original_parse)
        def parse_options(backend, options):
            requested = options.get("enable_fp_fusion")
            result = original_parse(backend, options)
            record(
                "parse_options",
                backend=backend.target.backend,
                requested=requested,
                resolved=result.enable_fp_fusion,
                flags=flags(),
                caller=traceback.format_stack(limit=8),
            )
            return result

        return parse_options

    for backend_class in (CUDABackend, HIPBackend):
        backend_class.parse_options = trace_parse_options(backend_class.parse_options)
    original_compile = triton.compile

    @functools.wraps(original_compile)
    def compile_kernel(src, *args, **kwargs):
        record(
            "compile_request",
            name=getattr(getattr(src, "fn", None), "__name__", None),
            requested=(kwargs.get("options") or {}).get("enable_fp_fusion"),
            flags=flags(),
            caller=traceback.format_stack(limit=8),
        )
        result = original_compile(src, *args, **kwargs)
        record(
            "compiled",
            name=result.metadata.name,
            resolved=result.metadata.enable_fp_fusion,
        )
        return result

    triton.compile = compile_kernel


if os.environ.get("PIN_TRITON_TRACE_DIR"):
    install_trace()
