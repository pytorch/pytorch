import argparse
import itertools
from . import benchmark
import os
from . import tensor_engine

from . import attention      # noqa: F401
from . import broadcast      # noqa: F401
from . import concat         # noqa: F401
# from . import conv           # noqa: F401
from . import elementwise    # noqa: F401
from . import matmul         # noqa: F401
# from . import normalization  # noqa: F401
# from . import pooling        # noqa: F401
from . import reduction      # noqa: F401
from . import softmax        # noqa: F401
from . import rnn_eltwise    # noqa: F401
from . import swish          # noqa: F401


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""Benchmark operators in specific shapes.
Works only with Python3.\n A few examples:
  * benchmark.py: runs all the default configs with all the benchmarks.
  * benchmark.py reduce: runs all the default configs with all benchmark with a prefix 'reduce'
  * benchmark.py layernorm_fwd_cpu_128_32_128_128: run a particular benchmark in that config""",
    )
    parser.add_argument(
        "benchmark_names",
        type=str,
        default=None,
        nargs="*",
        help="name of the benchmark to run",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu,cuda",
        help="a comma separated list of device names",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="fwd,both",
        help="a comma separated list of running modes",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="a comma separated list of Data Types: {float32[default], float16}",
    )
    parser.add_argument(
        "--input-iter",
        type=str,
        default=None,
        help="a comma separated list of Tensor dimensions that includes a start, \
              stop, and increment that can be constant or a power of 2 \
              {start:stop:inc,start:stop:pow2}",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="pt",
        help="the underlying tensor engine. only pt for now",
    )
    parser.add_argument(
        "--jit-mode",
        "--jit_mode",
        type=str,
        default="trace",
        help="the jit mode to use: one of {trace, none}",
    )
    parser.add_argument(
        "--cuda-pointwise-loop-levels",
        "--cuda_pointwise_loop_levels",
        type=int,
        default=None,
        help="num of loop levesl for Cuda pointwise operations: 2 or 3",
    )
    parser.add_argument(
        "--cuda-pointwise-block-count",
        "--cuda_pointwise_block_count",
        type=int,
        default=None,
        help="num of block for Cuda pointwise operations",
    )
    parser.add_argument(
        "--cuda-pointwise-block-size",
        "--cuda_pointwise_block_size",
        type=int,
        default=None,
        help="num of blocks for Cuda pointwise operations",
    )
    parser.add_argument(
        "--cuda-fuser",
        "--cuda_fuser",
        type=str,
        default="te",
        help="The Cuda fuser backend to use: one of {te, nvf, old, none}",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="stdout",
        help="The output format of the benchmark run {stdout[default], json}",
    )
    parser.add_argument(
        "--print-ir",
        action='store_true',
        help="Print the IR graph of the Fusion.",
    )
    parser.add_argument(
        "--print-kernel",
        action='store_true',
        help="Print generated kernel(s).",
    )
    parser.add_argument(
        "--no-dynamic-shape",
        action='store_true',
        help="Disable shape randomization in dynamic benchmarks.",
    )
    parser.add_argument(
        "--cpu-fusion",
        "--cpu_fusion",
        default=False,
        action='store_true',
        help="Enable CPU fusion.",
    )
    parser.add_argument(
        "--cat-wo-conditionals",
        "--cat_wo_conditionals",
        default=False,
        action='store_true',
        help="Enable CAT wo conditionals.",
    )

    args = parser.parse_args()

    if args.cuda_fuser == "te":
        import torch
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_set_texpr_fuser_enabled(True)
        torch._C._jit_override_can_fuse_on_gpu(True)
        torch._C._get_graph_executor_optimize(True)
    elif args.cuda_fuser == "old":
        import torch
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_override_can_fuse_on_gpu(True)
    elif args.cuda_fuser == "nvf":
        import torch
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(True)
        torch._C._get_graph_executor_optimize(True)
    else :
        raise ValueError("Undefined fuser: {}".format(args.cuda_fuser))

    if args.cpu_fusion:
        import torch
        torch._C._jit_override_can_fuse_on_cpu(True)
    else:
        import torch
        torch._C._jit_override_can_fuse_on_cpu(False)

    if args.cat_wo_conditionals:
        import torch
        torch._C._jit_cat_wo_conditionals(True)
    else:
        import torch
        torch._C._jit_cat_wo_conditionals(False)

    def set_global_threads(num_threads):
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_NUM_THREADS"] = str(num_threads)
        os.environ["TVM_NUM_THREADS"] = str(num_threads)
        os.environ["NNC_NUM_THREADS"] = str(num_threads)

    devices = args.device.split(",")
    # accept 'gpu' as an alternative as the 'cuda' device
    devices = ["cuda" if device == "gpu" else device for device in devices]
    cpu_count = 0
    for index, device in enumerate(devices):
        if device.startswith("cpu"):
            cpu_count += 1
            if cpu_count > 1:
                raise ValueError(
                    "more than one CPU device is not allowed: %d" % (cpu_count)
                )
            if device == "cpu":
                continue
            num_threads_str = device[3:]
            try:
                # see if the device is in 'cpu1' or 'cpu4' format
                num_threads = int(num_threads_str)
                set_global_threads(num_threads)
                devices[index] = "cpu"
            except ValueError:
                continue

    modes = args.mode.split(",")

    datatypes = args.dtype.split(",")
    for index, dtype in enumerate(datatypes):
        datatypes[index] = getattr(torch, dtype)
        if not datatypes[index] :
            raise AttributeError("DataType: {} is not valid!".format(dtype))

    tensor_engine.set_engine_mode(args.engine)

    def run_default_configs(bench_cls, allow_skip=True):
        for mode, device, dtype, config in itertools.product(
            modes, devices, datatypes, bench_cls.default_configs()
        ):
            bench = bench_cls(mode, device, dtype, *config)
            bench.output_type = args.output
            bench.jit_mode = args.jit_mode
            if not bench.is_supported():
                if allow_skip:
                    continue
                else:
                    raise ValueError(
                        "attempted to run an unsupported benchmark: %s" % (bench.desc())
                    )
            bench.run(args)

    def run_with_input_iter(bench_cls, input_iter, allow_skip=True):
        tensor_dim_specs = input_iter.split(',')
        tensor_dim_specs = [dim.split(':') for dim in tensor_dim_specs]

        configs = []
        for start, stop, inc in tensor_dim_specs:
            dim_list = []
            if inc == 'pow2' :
                curr = int(start)
                while curr <= int(stop) :
                    dim_list.append(curr)
                    curr <<= 1
            elif inc == 'pow2+1' :
                curr = int(start)
                while curr <= int(stop) :
                    dim_list.append(curr)
                    curr -= 1
                    curr <<= 1
                    curr += 1
            else :
                dim_list = list(range(int(start), int(stop) + int(inc), int(inc)))
            configs.append(dim_list)
        configs = itertools.product(*configs)

        for mode, device, dtype, config in itertools.product(
            modes, devices, datatypes, list(configs)
        ):
            bench = bench_cls(mode, device, dtype, *config)
            bench.output_type = args.output
            bench.jit_mode = args.jit_mode
            if not bench.is_supported():
                if allow_skip:
                    continue
                else:
                    raise ValueError(
                        "attempted to run an unsupported benchmark: %s" % (bench.desc())
                    )
            bench.run(args)

    benchmark_classes = benchmark.benchmark_classes
    if not args.benchmark_names:
        # by default, run all the benchmarks
        for benchmark_cls in benchmark_classes:
            run_default_configs(benchmark_cls, allow_skip=True)
    else:
        for name in args.benchmark_names:
            # if the name is the prefix of a benchmark class, run all the benchmarks for that class
            match_class_name = False
            for bench_cls in benchmark_classes:
                if name in bench_cls.module():
                    match_class_name = True
                    if (args.input_iter is not None) and bench_cls.input_iterable() :
                        run_with_input_iter(bench_cls, args.input_iter, allow_skip=True)
                    else :
                        if args.input_iter is not None :
                            print("WARNING: Incompatible benchmark class called with input_iter arg: {}".format(name))
                        run_default_configs(bench_cls, allow_skip=True)

            if match_class_name:
                continue

            # if not a class module, parse the config and call it that way
            match_class_name = False
            for bench_cls in benchmark_classes:
                cls_module = bench_cls.module()
                if name.startswith(cls_module):
                    match_class_name = True
                    if name[len(cls_module)] != "_":
                        raise ValueError("invalid name: %s" % (name))
                    config_str = name[(len(cls_module) + 1) :]
                    config = config_str.split("_")
                    if len(config) < 2:
                        raise ValueError("invalid config: %s" % config)
                    mode, device = config[0:2]
                    # TODO: make sure virtual devices such as 'cpu1' and 'cpu4' are supported.
                    if mode not in ["fwd", "both"]:
                        raise ValueError("invalid mode: %s" % (mode))
                    for i, entry in enumerate(config):
                        try:
                            value = int(entry)
                            config[i] = value
                        except ValueError:
                            pass
                    # TODO: output dtype in the config and  parse it back from the str
                    bench = bench_cls(config[0], config[1], torch.float32, *config[2:])
                    bench.jit_mode = args.jit_mode
                    bench.output_type = args.output
                    bench.run(args)

            if not match_class_name:
                available_classes = ", ".join(
                    [bench_cls.module() for bench_cls in benchmark_classes]
                )
                raise ValueError(
                    "invalid name: %s\nAvailable benchmark classes:\n%s"
                    % (name, available_classes)
                )


if __name__ == "__main__":
    main()
