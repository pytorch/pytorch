import copy
import functools
import io
import logging
import os
import subprocess
import tempfile

import numpy as np

import torch

from ..utils import identity
from .subgraph import SubGraph

log = logging.getLogger(__name__)
BACKENDS = dict()
_NP_DTYPE = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.longlong,
    torch.bool: np.bool_,
}


def register_backend(fn):
    @functools.wraps(fn)
    def inner(gm, example_inputs, **kwargs):
        return fn(gm, example_inputs, **kwargs)

    BACKENDS[fn.__name__] = inner
    return inner


def create_backend(fn):
    @functools.wraps(fn)
    def inner(model, example_inputs=None, **kwargs):
        if model is None:
            return None

        if not isinstance(model, SubGraph):
            with tempfile.TemporaryDirectory() as tmp:
                return inner(SubGraph(model, example_inputs, tmp), **kwargs)
        else:
            assert example_inputs is None

        try:
            return fn(model, **kwargs)
        except KeyboardInterrupt:
            raise
        except Exception:
            log.exception(f"{fn.__name__} error")
            return None

    BACKENDS[fn.__name__] = inner
    return inner


@create_backend
def eager(subgraph):
    return subgraph.model


@create_backend
def ts(subgraph):
    return subgraph.scripted


def reload_jit_model(subgraph, opt_fn=identity):
    tmp = io.BytesIO()
    torch.jit.save(subgraph.scripted, tmp)
    tmp.seek(0)
    model = torch.jit.load(tmp)
    model = opt_fn(model)
    # populate cache
    for _ in range(3):
        model(*subgraph.example_inputs)
    return model


def reload_jit_model_ofi(subgraph):
    return reload_jit_model(subgraph, torch.jit.optimize_for_inference)


@create_backend
def nnc(subgraph):
    with torch.jit.fuser("fuser1"):
        return reload_jit_model(subgraph)


@create_backend
def nnc_ofi(subgraph):
    with torch.jit.fuser("fuser1"):
        return reload_jit_model_ofi(subgraph)


@create_backend
def ts_nvfuser(subgraph):
    with torch.jit.fuser("fuser2"):
        return reload_jit_model(subgraph)


@create_backend
def ts_nvfuser_ofi(subgraph):
    with torch.jit.fuser("fuser2"):
        return reload_jit_model_ofi(subgraph)


@create_backend
def onednn(subgraph):
    with torch.jit.fuser("fuser3"):
        return reload_jit_model(subgraph)


@create_backend
def ofi(subgraph):
    return torch.jit.optimize_for_inference(subgraph.scripted)


@create_backend
def static_runtime(subgraph):
    scripted = subgraph.scripted
    if hasattr(scripted, "_c"):
        static_module = torch._C._jit_to_static_module(scripted._c)
    else:
        static_module = torch._C._jit_to_static_module(scripted.graph)
    return subgraph.wrap_returns(static_module)


def onnxrt_common(subgraph, provider, onnx_filename=None):
    import onnxruntime

    assert provider in onnxruntime.get_available_providers()
    session = onnxruntime.InferenceSession(
        onnx_filename or subgraph.onnx_filename, providers=[provider]
    )
    input_names = subgraph.input_names
    output_names = subgraph.output_names
    create_outputs = subgraph.empty_outputs_factory()
    is_cpu = subgraph.is_cpu

    def _call(*args):
        binding = session.io_binding()
        args = [a.contiguous() for a in args]
        for name, value in zip(input_names, args):
            dev = value.device
            binding.bind_input(
                name,
                dev.type,
                dev.index or 0,
                _NP_DTYPE[value.dtype],
                value.size(),
                value.data_ptr(),
            )
        outputs = create_outputs()
        for name, value in zip(output_names, outputs):
            dev = value.device
            binding.bind_output(
                name,
                dev.type,
                dev.index or 0,
                _NP_DTYPE[value.dtype],
                value.size(),
                value.data_ptr(),
            )
        session.run_with_iobinding(binding)
        if is_cpu:
            binding.copy_outputs_to_cpu()
        return outputs

    return subgraph.wrap_returns(_call)


@create_backend
def onnxrt_cpu(subgraph):
    return onnxrt_common(subgraph, provider="CPUExecutionProvider")


@create_backend
def onnxrt_cuda(subgraph):
    return onnxrt_common(subgraph, provider="CUDAExecutionProvider")


@create_backend
def onnx2tensorrt(subgraph):
    if subgraph.will_tensorrt_barf():
        # TensorRT fails violently with an abort() on this
        return None

    return onnxrt_common(subgraph, provider="TensorrtExecutionProvider")


@create_backend
def onnxrt_cpu_numpy(subgraph, provider="CPUExecutionProvider"):
    """Alternate version that integrates via numpy"""
    import onnxruntime

    assert provider in onnxruntime.get_available_providers()
    ort_session = onnxruntime.InferenceSession(
        subgraph.onnx_filename, providers=[provider]
    )

    def to_numpy(x):
        try:
            return x.numpy()
        except RuntimeError:
            return x.detach().numpy()

    def _call(*args):
        res = ort_session.run(
            None, {f"i{i}": to_numpy(arg) for i, arg in enumerate(args)}
        )
        res = [torch.from_numpy(x) for x in res]
        return res

    return subgraph.wrap_returns(_call)


@create_backend
def onnxrt(subgraph):
    if subgraph.is_cuda:
        return onnxrt_cuda(subgraph)
    else:
        return onnxrt_cpu(subgraph)


@functools.lru_cache(None)
def _init_tensorflow():
    import tensorflow as tf

    # prevent tensorflow from eating all the GPU memory
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    return tf


@create_backend
def onnx2tf(subgraph):
    import onnx
    from onnx_tf.backend import prepare

    tf = _init_tensorflow()
    filename = subgraph.filename("tensorflow")
    input_names = subgraph.input_names
    output_names = subgraph.output_names
    device = "/CPU:0" if subgraph.is_cpu else f"/GPU:{subgraph.device_index}"
    with tf.device(device):
        if not os.path.exists(filename):
            prepare(onnx.load(subgraph.onnx_filename)).export_graph(filename)
        tf_module = tf.saved_model.load(filename)
        tf_module = tf.function(tf_module, jit_compile=True)

    def run(*args):
        args = [a.contiguous() for a in args]
        with tf.device(device):
            outs = tf_module(
                **{
                    name: tf.experimental.dlpack.from_dlpack(
                        torch.utils.dlpack.to_dlpack(args[idx])
                    )
                    for idx, name in enumerate(input_names)
                }
            )
            return [
                torch.utils.dlpack.from_dlpack(
                    tf.experimental.dlpack.to_dlpack(outs[name])
                )
                for name in output_names
            ]

    return subgraph.wrap_returns(run)


@create_backend
def taso(subgraph):
    taso_filename = subgraph.filename("taso")
    subprocess.check_call(
        [
            os.path.expanduser("~/conda/envs/taso/bin/python"),
            "-c",
            "import taso,onnx; onnx.save(taso.export_onnx(taso.optimize("
            f"taso.load_onnx('{subgraph.onnx_filename}'))), '{taso_filename}')",
        ]
    )
    return onnxrt_common(
        subgraph, provider="CUDAExecutionProvider", onnx_filename=taso_filename
    )


@create_backend
def ipex(subgraph, **kwargs):
    import intel_extension_for_pytorch as ipex

    inputs = subgraph.example_inputs
    model = subgraph.model
    with torch.no_grad():
        model.eval()
        if kwargs["datatype"] == "bf16":
            model = ipex.optimize(model, dtype=torch.bfloat16)
        else:
            model = ipex.optimize(model, dtype=torch.float32)
        try:
            traced_model = torch.jit.trace(model, inputs).eval()
            traced_model = torch.jit.freeze(traced_model)
            return traced_model
        except Exception:
            log.warning("JIT trace failed during the 'ipex' optimize process.")
            return model


def _raise_timeout(signum, frame):
    raise TimeoutError()


@create_backend
def fx2trt(subgraph, **kwargs):
    if subgraph.will_tensorrt_barf():
        # TensorRT fails violently with an abort() on this
        return None

    from torch_tensorrt.fx.fx2trt import InputTensorSpec, TRTInterpreter
    from torch_tensorrt.fx.passes.lower_basic_pass import transform_setitem
    from torch_tensorrt.fx.tools.trt_splitter import TRTSplitter, TRTSplitterSetting
    from torch_tensorrt.fx.tracer.acc_tracer import acc_tracer
    from torch_tensorrt.fx.trt_module import TRTModule
    from torch_tensorrt.fx.utils import LowerPrecision

    from .normalize import normalize_ir

    try:
        model = subgraph.model
        inputs = subgraph.example_inputs
        # normalize
        model = normalize_ir(model, inputs)
        # pass rewrite
        model = transform_setitem(model, inputs)
        acc_model = acc_tracer.trace(model, inputs)
        # Split out unsupported ops
        splitter_setting = TRTSplitterSetting()
        splitter_setting.use_implicit_batch_dim = False
        splitter = TRTSplitter(acc_model, inputs, settings=splitter_setting)
        splitter.node_support_preview()
        split_mod = splitter()
        num_piece = 0
        for name, _ in split_mod.named_children():
            print(f"graph is split into {name}")
            num_piece += 1

        # if the graph module is split into pieces larger than 8, we consider its perf
        # is not good and fall back to non-TRT
        if num_piece > 8:
            print(
                f"The graph module is split into {num_piece} which is large than the \
                threshold=8. Fall back to non-TRT module."
            )
            return None

        if "fp16_mode" in kwargs and kwargs["fp16_mode"]:
            precision = LowerPrecision.FP16
        else:
            precision = LowerPrecision.FP32

        def get_submod_inputs(mod, submod, inputs):
            acc_inputs = None

            def get_input(self, inputs):
                nonlocal acc_inputs
                acc_inputs = inputs

            handle = submod.register_forward_pre_hook(get_input)
            mod(*inputs)
            handle.remove()
            return acc_inputs

        for name, _ in split_mod.named_children():
            if "_run_on_acc" in name:
                submod = getattr(split_mod, name)
                # print("acc=",submod.code)
                # Get submodule inputs for fx2trt
                acc_inputs = get_submod_inputs(split_mod, submod, inputs)

                # fx2trt replacement
                interp = TRTInterpreter(
                    submod,
                    InputTensorSpec.from_tensors(acc_inputs),
                    explicit_batch_dimension=True,
                )
                r = interp.run(
                    max_workspace_size=20 << 30,
                    lower_precision=precision,
                    # profiling_verbosity=trt.ProfilingVerbosity.DETAILED, #For profile
                )
                # For profile
                # from fx2trt_oss.fx.tools.trt_profiler_sorted import profile_trt_module
                # profile_trt_module("", trt_mod, acc_inputs)
                trt_mod = TRTModule(*r)

                setattr(split_mod, name, trt_mod)
            else:
                submod = getattr(split_mod, name)
                # print("gpu=",submod.code)
        return subgraph.wrap_returns(split_mod)
    except Exception:
        log.exception("FX2TRT conversion error")
        return None


@create_backend
def torch2trt(subgraph):
    if subgraph.will_tensorrt_barf():
        # TensorRT fails violently with an abort() on this
        return None

    from torch2trt import torch2trt

    inputs = subgraph.example_inputs
    trt_mod = torch2trt(
        subgraph.model,
        inputs,
        max_batch_size=len(inputs[0]),
        strict_type_constraints=True,
    )
    return subgraph.wrap_returns(trt_mod)


@create_backend
def tensorrt(subgraph):
    if subgraph.will_tensorrt_barf():
        # TensorRT fails violently with an abort() on this
        return None

    model = onnx2tensorrt(subgraph)
    if model is None:
        model = torch2trt(subgraph)
    return model


@create_backend
def onnx2tensorrt_alt(subgraph):
    if subgraph.will_tensorrt_barf():
        # TensorRT fails violently with an abort() on this
        return None

    import tensorrt as trt

    from torch.fx.experimental.fx2trt.trt_module import TRTModule

    inputs = subgraph.example_inputs

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    assert isinstance(inputs, (list, tuple))
    inputs = tuple(inputs)
    input_names = subgraph.input_names
    output_names = subgraph.output_names
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    success = parser.parse(open(subgraph.onnx_filename, "rb").read())
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))
    assert success

    config.max_workspace_size = 1 << 25
    config.set_flag(trt.BuilderFlag.STRICT_TYPES)
    builder.max_batch_size = len(inputs[0])

    engine = builder.build_engine(network, config)
    assert engine

    trt_mod = TRTModule(engine, input_names, output_names)
    return subgraph.wrap_returns(trt_mod)


@create_backend
def cudagraphs(subgraph):
    model = subgraph.model
    inputs = subgraph.example_inputs
    assert subgraph.is_cuda
    return subgraph.wrap_returns(cudagraphs_inner(model, inputs))


@create_backend
def cudagraphs_ts(subgraph):
    assert subgraph.is_cuda
    model = subgraph.scripted
    inputs = subgraph.example_inputs

    # warmup
    for _ in range(3):
        model(*inputs)

    return subgraph.wrap_returns(cudagraphs_inner(model, inputs))


@create_backend
def cudagraphs_ts_ofi(subgraph):
    assert subgraph.is_cuda
    model = torch.jit.optimize_for_inference(torch.jit.freeze(subgraph.scripted))
    inputs = subgraph.example_inputs

    # warmup
    for _ in range(3):
        model(*inputs)

    return subgraph.wrap_returns(cudagraphs_inner(model, inputs))


def cudagraphs_inner(model, inputs, copy_outputs=True):
    assert isinstance(inputs, (list, tuple))
    static_inputs = [torch.zeros_like(x) for x in inputs]

    # warmup
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        model(*inputs)
    stream.synchronize()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    # record
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        static_outputs = model(*static_inputs)
    if not isinstance(static_outputs, (list, tuple)):
        static_outputs = (static_outputs,)

    def run(*new_inputs):
        assert len(static_inputs) == len(new_inputs)
        for dst, src in zip(static_inputs, new_inputs):
            dst.copy_(src)
        graph.replay()
        if copy_outputs:
            return [x.clone() for x in static_outputs]
        else:
            return static_outputs

    return run


@create_backend
def aot_autograd(subgraph, **kwargs):
    def _wrapped_bw_compiler(*args, **kwargs):
        # stop TorchDynamo from trying to compile our generated backwards pass
        return disable(disable(bw_compiler)(*args, **kwargs))

    bw_compiler = kwargs.get("bw_compiler") or kwargs["fw_compiler"]
    kwargs["bw_compiler"] = _wrapped_bw_compiler

    from functorch.compile import aot_module_simplified

    from .. import disable

    return aot_module_simplified(subgraph.model, **kwargs)


def tvm_compile(jit_mod, example_inputs, log_file=None, **kwargs):
    if jit_mod is None:
        return None
    try:
        return tvm_compile_inner(jit_mod, example_inputs, None, log_file, **kwargs)
    except Exception as e:
        if log_file and os.path.exists(log_file):
            os.unlink(log_file)
        if isinstance(e, KeyboardInterrupt):
            raise
        log.exception("tvm error")
        return None


@create_backend
def tvm(subgraph):
    return subgraph.wrap_returns(
        tvm_compile_inner(
            subgraph.scripted,
            subgraph.example_inputs,
            tuning_option=None,
            cuda=subgraph.is_cuda,
        )
    )


@create_backend
def ansor(subgraph):
    """
    WARNING: this backend takes hours or days to train and
    often produces a slower result than the default schedule.
    """
    return subgraph.wrap_returns(
        tvm_compile_inner(
            subgraph.scripted,
            subgraph.example_inputs,
            tuning_option="auto_scheduler",
            log_file=subgraph.filename("ansor"),
            cuda=subgraph.is_cuda,
        )
    )


@create_backend
def tvm_meta_schedule(subgraph):
    return subgraph.wrap_returns(
        tvm_compile_inner(
            subgraph.scripted,
            subgraph.example_inputs,
            tuning_option="meta_schedule",
            trials=20000,
            cuda=subgraph.is_cuda,
        )
    )


@functools.lru_cache(None)
def llvm_target():
    if "avx512" in open("/proc/cpuinfo").read():
        return "llvm -mcpu=skylake-avx512"
    return "llvm -mcpu=core-avx2"


def tvm_compile_inner(
    jit_mod, example_inputs, tuning_option=None, log_file=None, trials=20000, cuda=False
):
    try:
        import tvm
        from tvm import relay
        from tvm.contrib import graph_executor

        shape_list = [(f"inp_{idx}", i.shape) for idx, i in enumerate(example_inputs)]
        mod, params = relay.frontend.from_pytorch(jit_mod, shape_list)
        if cuda:
            dev = tvm.cuda(0)
            target = tvm.target.cuda()
        else:
            dev = tvm.cpu(0)
            target = tvm.target.Target(llvm_target())

        if tuning_option == "auto_scheduler":
            from tvm import auto_scheduler

            if log_file is None:
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
        elif tuning_option == "meta_schedule":
            from os import path as osp

            from tvm import meta_schedule as ms

            with tempfile.TemporaryDirectory() as work_dir:
                if log_file is not None:
                    assert osp.isdir(
                        log_file
                    ), "TVM's meta_schedule requires a directory for storing log files."
                    work_dir = log_file
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

        elif tuning_option is None:
            # no autotuning (for debugging)
            with tvm.transform.PassContext(opt_level=10):
                lib = relay.build(mod, target=target, params=params)
        else:
            raise NotImplementedError(
                "This tuning option is invalid/not implemented for torchdynamo's TVM-related backend. "
                "There are three available options including None, auto_scheduler and meta_schedule."
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

        def exec_tvm(*args):
            args = [a.contiguous() for a in args]
            for idx, arg in enumerate(args, 0):
                if arg.dim() != 0:
                    if arg.requires_grad:
                        arg = arg.detach()
                    m.set_input(
                        f"inp_{idx}",
                        tvm.nd.array(arg.numpy(), dev),
                    )
            m.run()
            return [
                to_torch_tensor(m.get_output(i)) for i in range(m.get_num_outputs())
            ]

        return exec_tvm
    except Exception:
        log.exception("tvm error")
        return jit_mod  # explicit fall back to eager


@functools.lru_cache(None)
def _init_ltc():
    try:
        import torch._lazy.extract_compiled_graph
        from torch._lazy.ts_backend import init as init_ts_backend

        # hopefully changing this line to sth like _ltc_init_xla_backend in future
        # will enable XLA
        init_ts_backend()

        return torch._lazy
    except ModuleNotFoundError as e:
        print(f"ltc backend fails. Can not import {e.name}")
        raise


def ltc_reuse_graph(gm: torch.fx.GraphModule, example_inputs):
    ltc = _init_ltc()
    return ltc.extract_compiled_graph.extract_compiled_graph(gm, example_inputs)


def ltc_trivial(gm: torch.fx.GraphModule, example_inputs):
    ltc = _init_ltc()
    lazy_model = copy.deepcopy(gm).to(device="lazy")
    ltc.extract_compiled_graph.force_lazy_device(lazy_model)

    def ltc_model(*inputs):
        orig_device = inputs[0].device if len(inputs) > 0 else "cuda"
        lazy_inputs = tuple(inp.to(device="lazy") for inp in inputs)

        lazy_out = lazy_model(*lazy_inputs)
        out = tuple(out.to(device=orig_device) for out in lazy_out)
        return out

    return ltc_model


@functools.lru_cache(None)
def _init_torchxla():
    global xm
    try:
        import torch_xla.core.xla_model as xm
    except ModuleNotFoundError as e:
        print(f"torchxla backend fails. Can not import {e.name}")
        raise


@create_backend
def torchxla_trivial(subgraph):
    _init_torchxla()

    xla_dev = xm.xla_device()

    xla_model = copy.deepcopy(subgraph.model).to(device=xla_dev)

    def xla_model_wrapper(*inputs):
        orig_device = inputs[0].device if len(inputs) > 0 else "cpu"
        xla_inputs = tuple(inp.to(device=xla_dev) for inp in inputs)

        xla_out = xla_model(*xla_inputs)
        result = tuple(out.to(device=orig_device) for out in xla_out)
        return result

    return xla_model_wrapper


@create_backend
def torchxla_trace_once(subgraph):
    import torch._dynamo.optimizations.torchxla_integration as integration

    model = subgraph.model
    example_inputs = subgraph.example_inputs
    return integration.extract_compiled_graph(model, example_inputs)


def ipex_fp32(gm: torch.fx.GraphModule, example_inputs):
    kwargs_ipex = {"datatype": "fp32"}
    return BACKENDS["ipex"](gm, example_inputs, **kwargs_ipex)


def ipex_bf16(gm: torch.fx.GraphModule, example_inputs):
    kwargs_ipex = {"datatype": "bf16"}
    return BACKENDS["ipex"](gm, example_inputs, **kwargs_ipex)


def fx2trt_compiler_fp16(gm: torch.fx.GraphModule, example_inputs):
    kwargs_fx2trt = {"fp16_mode": True}
    trt_compiled = BACKENDS["fx2trt"](gm, example_inputs, **kwargs_fx2trt)
    if trt_compiled is not None:
        return trt_compiled
    else:
        print(
            "FX2TRT conversion failed on the subgraph. Return GraphModule forward instead"
        )
        return gm.forward


def fx2trt_compiler(gm: torch.fx.GraphModule, example_inputs):
    kwargs_fx2trt = {"fp16_mode": False}
    trt_compiled = BACKENDS["fx2trt"](gm, example_inputs, **kwargs_fx2trt)
    if trt_compiled is not None:
        return trt_compiled
    else:
        print(
            "FX2TRT conversion failed on the subgraph. Return GraphModule forward instead"
        )
        return gm.forward
