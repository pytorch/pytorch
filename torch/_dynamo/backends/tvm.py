# mypy: ignore-errors

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
"""

import functools
import importlib
import logging
import os
import sys
import tempfile
from types import MappingProxyType
from typing import Optional

import torch

from .common import device_from_inputs, fake_tensor_unsupported
from .registry import register_backend


log = logging.getLogger(__name__)


@register_backend
@fake_tensor_unsupported
def tvm(
    gm,
    example_inputs,
    *,
    options: Optional[MappingProxyType] = MappingProxyType(
        {"scheduler": None, "trials": 20000, "opt_level": 3}
    ),
):
    import tvm  # type: ignore[import]
    from tvm import relay  # type: ignore[import]
    from tvm.contrib import graph_executor  # type: ignore[import]

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
        from tvm import auto_scheduler

        log_file = tempfile.NamedTemporaryFile()

        if not os.path.exists(log_file):
            tasks, task_weights = auto_scheduler.extract_tasks(
                mod["main"], params, target
            )
            if len(tasks) != 0:
                tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
                if not os.path.exists(log_file):
                    assert trials > 0
                    tune_option = auto_scheduler.TuningOptions(
                        num_measure_trials=trials,
                        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
                        early_stopping=2000,
                    )
                    try:
                        tuner.tune(tune_option)
                    except Exception:
                        if os.path.exists(log_file):
                            os.unlink(log_file)
                        raise

        with auto_scheduler.ApplyHistoryBest(log_file):
            with tvm.transform.PassContext(
                opt_level=opt_level, config={"relay.backend.use_auto_scheduler": True}
            ):
                lib = relay.build(mod, target=target, params=params)
    elif scheduler == "meta_schedule":
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
            assert trials > 0
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

    def to_torch_tensor(nd_tensor):
        """A helper function to transfer a NDArray to torch.tensor."""
        if nd_tensor.dtype == "bool":
            # DLPack does not support boolean so it can't be handled by
            # torch.utils.dlpack.from_pack. Workaround by going through
            # numpy, although this brings additional data copy overhead.
            return torch.from_numpy(nd_tensor.numpy())
        return torch.utils.dlpack.from_dlpack(nd_tensor.to_dlpack())

    def to_tvm_tensor(torch_tensor):
        """A helper function to transfer a torch.tensor to NDArray."""
        if torch_tensor.dtype == torch.bool:
            # same reason as above, fallback to numpy conversion which
            # could introduce data copy overhead
            return tvm.nd.array(torch_tensor.cpu().numpy())
        return tvm.nd.from_dlpack(torch_tensor)

    def exec_tvm(*i_args):
        args = [a.contiguous() for a in i_args]
        shape_info, _ = m.get_input_info()
        active_inputs = {name for name, _ in shape_info.items()}
        for idx, arg in enumerate(args, 0):
            if arg.dim() != 0:
                if arg.requires_grad:
                    arg = arg.detach()
                inp_name = f"inp_{idx}"
                if inp_name not in active_inputs:
                    log.warning(
                        "input %s skipped as not found in tvm's runtime library",
                        inp_name,
                    )
                    continue
                m.set_input(
                    inp_name,
                    to_tvm_tensor(arg),
                )
        m.run()
        return [to_torch_tensor(m.get_output(i)) for i in range(m.get_num_outputs())]

    return exec_tvm


tvm_meta_schedule = functools.partial(tvm, scheduler="meta_schedule")
tvm_auto_scheduler = functools.partial(tvm, scheduler="auto_scheduler")


def has_tvm():
    try:
        importlib.import_module("tvm")
        return True
    except ImportError:
        return False


@functools.cache
def llvm_target():
    if sys.platform == "linux":
        cpuinfo = open("/proc/cpuinfo").read()
        if "avx512" in cpuinfo:
            return "llvm -mcpu=skylake-avx512"
        elif "avx2" in cpuinfo:
            return "llvm -mcpu=core-avx2"
    return "llvm"
