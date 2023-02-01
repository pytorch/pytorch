import functools
import importlib
import logging
import os
import tempfile

import torch
from .common import device_from_inputs, fake_tensor_unsupported

from .registry import register_backend

log = logging.getLogger(__name__)


@register_backend
@fake_tensor_unsupported
def tvm(gm, example_inputs, *, scheduler=None, trials=20000):
    import tvm  # type: ignore[import]
    from tvm import relay  # type: ignore[import]
    from tvm.contrib import graph_executor  # type: ignore[import]

    jit_mod = torch.jit.trace(gm, example_inputs)
    device = device_from_inputs(example_inputs)
    shape_list = [(f"inp_{idx}", i.shape) for idx, i in enumerate(example_inputs)]
    mod, params = relay.frontend.from_pytorch(jit_mod, shape_list)
    if device.type == "cuda":
        dev = tvm.cuda(device.index)
        target = tvm.target.cuda()
    else:
        dev = tvm.cpu(0)
        target = tvm.target.Target(llvm_target())

    if scheduler is None:
        scheduler = os.environ.get("TVM_SCHEDULER", None)

    if scheduler == "auto_scheduler":
        from tvm import auto_scheduler

        log_file = tempfile.NamedTemporaryFile()

        if not os.path.exists(log_file):
            tasks, task_weights = auto_scheduler.extract_tasks(
                mod["main"], params, target
            )
            for task in tasks:
                print(task.compute_dag)
            else:
                print("No tasks")
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
                opt_level=3, config={"relay.backend.use_auto_scheduler": True}
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
            database = ms.relay_integration.tune_relay(
                mod=mod,
                target=target,
                work_dir=work_dir,
                max_trials_global=20000,
                num_trials_per_iter=64,
                params=params,
                strategy="evolutionary",
            )
            lib = ms.relay_integration.compile_relay(
                database=database,
                mod=mod,
                target=target,
                params=params,
            )
    elif scheduler == "default" or not scheduler:
        # no autotuning
        with tvm.transform.PassContext(opt_level=10):
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
        for idx, arg in enumerate(args, 0):
            if arg.dim() != 0:
                if arg.requires_grad:
                    arg = arg.detach()
                m.set_input(
                    f"inp_{idx}",
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


@functools.lru_cache(None)
def llvm_target():
    if "avx512" in open("/proc/cpuinfo").read():
        return "llvm -mcpu=skylake-avx512"
    return "llvm -mcpu=core-avx2"
