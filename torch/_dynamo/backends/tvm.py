"""
This module provides TVM backend integration for TorchDynamo.

Apache TVM is a deep learning compiler framework that can optimize and execute
models on various hardware backends. This module enables:

- Compilation of PyTorch models to TVM's computation graphs
- Multiple scheduling options:
  - Default scheduler
  - Auto-scheduler for automatic optimization
  - Meta-schedule for evolutionary search-based tuning
- Hardware-specific optimizations:
  - CUDA GPU support
  - CPU support with LLVM targeting and architecture-specific tuning
  - Automatic detection of CPU capabilities (AVX2, AVX512)
- Tensor conversion utilities between PyTorch and TVM formats
- Configurable optimization levels and tuning trials

The backend can be used with torch.compile():
    model = torch.compile(model, backend="tvm")

The scheduler/trials options only apply to the legacy relay frontend
(removed in TVM 0.20). With the relax frontend, pass a TVM pipeline instead:

    pipeline = tvm.relax.get_pipeline("static_shape_tuning", target="llvm", total_trials=2000)
    model = torch.compile(model, backend="tvm", options={"pipeline": pipeline})
"""

import functools
import importlib.util
import logging
import os
import sys
import tempfile
import warnings
from collections.abc import Callable
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from torch import fx

from .common import device_from_inputs, fake_tensor_unsupported
from .registry import register_backend


log = logging.getLogger(__name__)


@register_backend
@fake_tensor_unsupported  # type: ignore[arg-type]
def tvm(
    gm: fx.GraphModule,
    example_inputs: list[torch.Tensor],
    *,
    options: MappingProxyType[str, Any] | None = None,
) -> Callable[..., Any]:
    if options is None:
        options = MappingProxyType({"scheduler": None, "trials": 20000, "opt_level": 3})
    try:
        import tvm  # type: ignore[import]  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Please install apache-tvm to use the tvm backend. "
            "See https://tvm.apache.org/docs/install/index.html for instructions."
        ) from e

    # relay was removed in TVM 0.20; newer pip wheels only ship the relax
    # frontend, so dispatch on whichever API the installed TVM provides.
    if importlib.util.find_spec("tvm.relay") is not None:
        return _tvm_relay_compile(gm, example_inputs, options)
    if importlib.util.find_spec("tvm.relax.frontend.torch") is not None:
        return _tvm_relax_compile(gm, example_inputs, options)
    raise ImportError(
        "The installed apache-tvm provides neither the legacy relay frontend nor "
        "the relax torch frontend, so the tvm backend cannot compile this graph."
    )


def _tvm_relax_compile(
    gm: fx.GraphModule,
    example_inputs: list[torch.Tensor],
    options: MappingProxyType[str, Any],
) -> Callable[..., Any]:
    from tvm.relax.frontend.torch import relax_dynamo  # type: ignore[import]

    scheduler = options.get("scheduler", None) or os.environ.get("TVM_SCHEDULER", None)
    if scheduler in ("auto_scheduler", "meta_schedule"):
        log.warning(
            "scheduler=%s has no equivalent in the relax TVM backend; "
            "falling back to the default relax pipeline.",
            scheduler,
        )
    return relax_dynamo(pipeline=options.get("pipeline", None))(gm, example_inputs)


def _tvm_relay_compile(
    gm: fx.GraphModule,
    example_inputs: list[torch.Tensor],
    options: MappingProxyType[str, Any],
) -> Callable[..., Any]:
    warnings.warn(
        "The tvm backend's relay path is deprecated and will be removed; "
        "install a recent apache-tvm to use the relax frontend instead.",
        FutureWarning,
        stacklevel=2,
    )
    import tvm  # type: ignore[import]
    from tvm import relay  # type: ignore[import]
    from tvm.contrib import graph_executor  # type: ignore[import]

    if any(
        isinstance(x, (torch.SymInt, torch.SymFloat, torch.SymBool))
        for x in example_inputs
    ):
        raise RuntimeError(
            "The TVM (relay) backend does not support dynamic shapes: got symbolic "
            "scalar graph inputs. Either compile with dynamic=False or use TVM's "
            "relax frontend instead: from tvm.relax.frontend.torch import relax_dynamo; "
            "torch.compile(model, backend=relax_dynamo())."
        )
    jit_mod = torch.jit.trace(gm, example_inputs)
    device = device_from_inputs(example_inputs)
    shape_list = [(f"inp_{idx}", i.shape) for idx, i in enumerate(example_inputs)]
    example_outputs = gm(*example_inputs)
    if len(example_outputs) == 0:
        log.warning("Explicitly fall back to eager due to zero output")
        return gm.forward
    mod, params = relay.frontend.from_pytorch(jit_mod, shape_list)
    if device.type == "cuda":
        dev = tvm.cuda(device.index)
        target = tvm.target.cuda()
    else:
        dev = tvm.cpu(0)
        target = tvm.target.Target(llvm_target())

    scheduler = options.get("scheduler", None)
    if scheduler is None:
        scheduler = os.environ.get("TVM_SCHEDULER", None)

    trials = options.get("trials", 20000)
    opt_level = options.get("opt_level", 3)

    if scheduler == "auto_scheduler":
        # pyrefly: ignore [missing-import]
        from tvm import auto_scheduler

        with (
            tempfile.NamedTemporaryFile() as log_file,
            auto_scheduler.ApplyHistoryBest(log_file),
            tvm.transform.PassContext(
                opt_level=opt_level, config={"relay.backend.use_auto_scheduler": True}
            ),
        ):
            lib = relay.build(mod, target=target, params=params)
    elif scheduler == "meta_schedule":
        # pyrefly: ignore [missing-import]
        from tvm import meta_schedule as ms

        with tempfile.TemporaryDirectory() as work_dir:
            if device.type != "cuda":
                # meta_schedule needs num-cores to be specified
                # here we use the maximum core count
                target = tvm.target.Target(
                    f"{llvm_target()} --num-cores {ms.utils.cpu_count(logical=False)}"
                )
            # TODO(shingjan): This could be replaced by tvm.contrib.torch.optimize_torch
            # once USE_PT_TVMDSOOP is updated and turned on by default in TVM.
            if trials <= 0:
                raise AssertionError(f"trials must be positive, got {trials}")
            database = ms.relay_integration.tune_relay(
                mod=mod,
                target=target,
                work_dir=work_dir,
                max_trials_global=trials,
                num_trials_per_iter=64,
                params=params,
                strategy="evolutionary",
                opt_level=opt_level,
            )
            lib = ms.relay_integration.compile_relay(
                database=database,
                mod=mod,
                target=target,
                params=params,
                opt_level=opt_level,
            )
    elif scheduler == "default" or not scheduler:
        # no autotuning
        with tvm.transform.PassContext(opt_level=opt_level):
            lib = relay.build(mod, target=target, params=params)
    else:
        raise NotImplementedError(
            "This tuning option is invalid/not implemented for torchdynamo's TVM-related backend. "
            "There are three available options: default, auto_scheduler and meta_schedule."
        )
    m = graph_executor.GraphModule(lib["default"](dev))

    def to_torch_tensor(nd_tensor: tvm.nd.array) -> torch.Tensor:
        """A helper function to transfer a NDArray to torch.tensor."""
        if nd_tensor.dtype == "bool":
            # DLPack does not support boolean so it can't be handled by
            # torch.utils.dlpack.from_pack. Workaround by going through
            # numpy, although this brings additional data copy overhead.
            return torch.from_numpy(nd_tensor.numpy())
        return torch.utils.dlpack.from_dlpack(nd_tensor.to_dlpack())

    def to_tvm_tensor(torch_tensor: torch.Tensor) -> tvm.nd.array:
        """A helper function to transfer a torch.tensor to NDArray."""
        if torch_tensor.dtype == torch.bool:
            # same reason as above, fallback to numpy conversion which
            # could introduce data copy overhead
            return tvm.nd.array(torch_tensor.cpu().numpy())
        return tvm.nd.from_dlpack(torch_tensor)

    # input info is fixed at compile time, so query it once instead of per call
    shape_info, _ = m.get_input_info()
    active_inputs = set(shape_info.keys())

    def exec_tvm(*i_args: torch.Tensor) -> list[torch.Tensor]:
        args = [a.contiguous() for a in i_args]
        for idx, arg in enumerate(args, 0):
            inp_name = f"inp_{idx}"
            if inp_name not in active_inputs:
                log.warning(
                    "input %s skipped as not found in tvm's runtime library",
                    inp_name,
                )
                continue
            if arg.requires_grad:
                arg = arg.detach()
            m.set_input(inp_name, to_tvm_tensor(arg))
        m.run()
        return [to_torch_tensor(m.get_output(i)) for i in range(m.get_num_outputs())]

    return exec_tvm


tvm_meta_schedule = functools.partial(
    tvm, options=MappingProxyType({"scheduler": "meta_schedule"})
)
tvm_auto_scheduler = functools.partial(
    tvm, options=MappingProxyType({"scheduler": "auto_scheduler"})
)


def has_tvm() -> bool:
    # avoid the heavy tvm import just to check availability
    return importlib.util.find_spec("tvm") is not None


@functools.cache
def llvm_target() -> str:
    if sys.platform == "linux":
        cpuinfo = Path("/proc/cpuinfo").read_text()
        if "avx512" in cpuinfo:
            return "llvm -mcpu=skylake-avx512"
        elif "avx2" in cpuinfo:
            return "llvm -mcpu=core-avx2"
    return "llvm"
