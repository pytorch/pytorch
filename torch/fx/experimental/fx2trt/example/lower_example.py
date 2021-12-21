from dataclasses import replace, field, dataclass
from copy import deepcopy

import torch
import typing as t
from torch.fx.experimental.fx2trt import LowerSetting
from torch.fx.experimental.fx2trt.lower import Lowerer
import torchvision


"""
The purpose of this example is to demostrate the onverall flow of lowering a PyTorch model
to TensorRT conveniently with lower.py.
"""
def lower_to_trt(
    module: torch.nn.Module,
    input,
    max_batch_size: int = 2048,
    max_workspace_size=1 << 25,
    explicit_batch_dimension=False,
    fp16_mode=True,
    enable_fuse=True,
    verbose_log=False,
    timing_cache_prefix="",
    save_timing_cache=False,
    cuda_graph_batch_size=-1,
) -> torch.nn.Module:
    """
    Takes in original module, input and lowering setting, run lowering workflow to turn module
    into lowered module, or so called TRTModule.

    Args:
    module: Original module for lowering.
    input: Input for module.
    max_batch_size: Maximum batch size (must be >= 1 to be set, 0 means not set)
    max_workspace_size: Maximum size of workspace given to TensorRT.
    explicit_batch_dimension: Use explicit batch dimension in TensorRT if set True, otherwise use implicit batch dimension.
    fp16_mode: fp16 config given to TRTModule.
    enable_fuse: Enable pass fusion during lowering if set to true. l=Lowering will try to find pattern defined
    in torch.fx.experimental.fx2trt.passes from original module, and replace with optimized pass before apply lowering.
    verbose_log: Enable verbose log for TensorRT if set True.
    timing_cache_prefix: Timing cache file name for timing cache used by fx2trt.
    save_timing_cache: Update timing cache with current timing cache data if set to True.
    cuda_graph_batch_size: Cuda graph batch size, default to be -1.

    Returns:
    A torch.nn.Module lowered by TensorRT.
    """
    lower_setting = LowerSetting(
        max_batch_size=max_batch_size,
        max_workspace_size=max_workspace_size,
        explicit_batch_dimension=explicit_batch_dimension,
        fp16_mode=fp16_mode,
        enable_fuse=enable_fuse,
        verbose_log=verbose_log,
        timing_cache_prefix=timing_cache_prefix,
        save_timing_cache=save_timing_cache,
    )
    lowerer = Lowerer.create(lower_setting=lower_setting)
    return lowerer(module, input)


@dataclass
class Configuration:
    """
    Specify the configuration used for fx2trt lowering and benchmark.

    To extend, add a new configuration field to this class, and modify the
    lowering or benchmark behavior in `run_configuration_benchmark()`
    correspondingly.

    It automatically prints all its values thanks to being a dataclass.
    """

    # number of inferences to run
    batch_iter: int

    # Input batch size
    batch_size: int

    # Friendly name of the configuration
    name: str = ""

    # Whether to apply TRT lowering to the model before benchmarking
    trt: bool = False

    # Whether to apply engine holder to the lowered model
    jit: bool = False

    # Whether to enable FP16 mode for TRT lowering
    fp16: bool = False

    # Relative tolerance for accuracy check after lowering. -1 means do not
    # check accuracy.
    accuracy_rtol: float = -1  # disable


@dataclass
class Result:
    """Holds and computes the benchmark results.

    Holds raw essential benchmark result values like duration.
    Also computes results that can be derived from the raw essential values
    (QPS), in the form of auto properties.

    """

    module: torch.nn.Module = field(repr=False)
    input: t.Any = field(repr=False)
    conf: Configuration
    time_sec: float
    accuracy_res: t.Optional[bool] = None

    @property
    def time_per_iter_ms(self) -> float:
        return self.time_sec * 1.0e3

    @property
    def qps(self) -> float:
        return self.conf.batch_size / self.time_sec

    def format(self) -> str:
        return (
            f"== Benchmark Result for: {self.conf}\n"
            f"BS: {self.conf.batch_size}, "
            f"Time per iter: {self.time_per_iter_ms:.2f}ms, "
            f"QPS: {self.qps:.2f}, "
            f"Accuracy: {self.accuracy_res} (rtol={self.conf.accuracy_rtol})"
        )


@torch.inference_mode()
def benchmark(
    model,
    inputs,
    batch_iter: int,
    batch_size: int,
) -> None:
    """
    Run fx2trt lowering and benchmark the given model according to the
    specified benchmark configuration. Prints the benchmark result for each
    configuration at the end of the run.
    """

    model = model.cuda().eval()
    inputs = [x.cuda() for x in inputs]

    # benchmark base configuration
    conf = Configuration(batch_iter=batch_iter, batch_size=batch_size)

    configurations = [
        # Baseline
        replace(conf, name="CUDA Eager", trt=False),
        # FP32
        replace(conf, name="TRT FP32 Eager", trt=True, jit=False, fp16=False, accuracy_rtol=1e-3),
        # FP16
        replace(conf, name="TRT FP16 Eager", trt=True, jit=False, fp16=True, accuracy_rtol=1e-2),
    ]

    results = [
        run_configuration_benchmark(deepcopy(model), inputs, conf_) for conf_ in configurations
    ]

    for res in results:
        print(res.format())


def benchmark_torch_function(iters: int, f, *args) -> float:
    """Estimates the average time duration for a single inference call in second

    If the input is batched, then the estimation is for the batches inference call.

    Args:
        iters: number of inference iterations to run
        f: a function to perform a single inference call

    Returns:
        estimated average time duration in second for a single inference call
    """
    with torch.inference_mode():
        f(*args)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    print("== Start benchmark iterations")
    with torch.inference_mode():
        start_event.record()
        for _ in range(iters):
            f(*args)
        end_event.record()
    torch.cuda.synchronize()
    print("== End benchmark iterations")
    return (start_event.elapsed_time(end_event) * 1.0e-3) / iters


def run_configuration_benchmark(
    module,
    input,
    conf: Configuration,
) -> Result:
    """
    Runs `module` through lowering logic and benchmark the module before and
    after lowering.
    """
    print(f"=== Running benchmark for: {conf}", "green")
    time = -1.0

    if conf.fp16:
        module = module.half()
        input = [i.half() for i in input]

    if not conf.trt:
        # Run eager mode benchmark
        time = benchmark_torch_function(conf.batch_iter, lambda: module(*input))
    elif not conf.jit:
        # Run lowering eager mode benchmark
        lowered_module = lower_to_trt(module, input, max_batch_size=conf.batch_size, fp16_mode=conf.fp16)
        time = benchmark_torch_function(conf.batch_iter, lambda: lowered_module(*input))
    else:
        print("Lowering with JIT is not available!", "red")

    result = Result(
        module=module, input=input, conf=conf, time_sec=time
    )
    return result


if __name__ == "__main__":
    test_model = torchvision.models.resnet101()
    input = [torch.cuda.FloatTensor(1024, 3, 224, 224)]  # type: ignore[attr-defined]
    benchmark(test_model, input, 100, 1024)
