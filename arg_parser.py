
import argparse
import torch
diff_branch_default = "DIFF-BRANCH-DEFAULT"

    
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filter", "-k", action="append", help="filter benchmarks with regexp"
    )
    parser.add_argument(
        "--exclude", "-x", action="append", help="filter benchmarks with regexp"
    )
    parser.add_argument(
        "--exclude-exact", action="append", help="filter benchmarks with exact match"
    )
    parser.add_argument(
        "--total-partitions",
        type=int,
        default=1,
        choices=range(1, 10),
        help="Total number of partitions we want to divide the benchmark suite into",
    )
    parser.add_argument(
        "--partition-id",
        type=int,
        default=0,
        help="ID of the benchmark suite partition to be run. Used to divide CI tasks",
    )
    parser.add_argument(
        "--devices", "--device", "-d", action="append", help="cpu or cuda"
    )
    parser.add_argument("--device-index", help="CUDA device index")
    parser.add_argument(
        "--repeat", "-n", type=int, default=30, help="number of timing runs"
    )
    iterations_per_run_help = """
        Run this may iterations for each time measurement. This is mainly used for
        XLA training. We want to run multiple iterations per measurement so the
        tracing and computation for different iteartions can overlap with each
        other. This makes sure we have an accurate xla baseline.
    """
    parser.add_argument(
        "--iterations-per-run", type=int, default=1, help=iterations_per_run_help
    )
    parser.add_argument(
        "--randomize-input",
        action="store_true",
        help="Whether to randomize the input values. Dimensions will be kept the same.",
    )
    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        help="number of threads to use for eager and inductor",
    )
    parser.add_argument(
        "--nopython", action="store_true", help="Turn graph breaks into errors"
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="run models that are in the global SKIP list",
    )
    parser.add_argument(
        "--prims-nvfuser", action="store_true", help="user prims + nvfuser backend"
    )
    parser.add_argument(
        "--dump-raw-metrics",
        action="store_true",
        help="dump raw timing metrics from speedup experiment",
    )
    parser.add_argument(
        "--log-operator-inputs",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--channels-last",
        action="store_true",
        default=False,
        help="use channels last format",
    )
    parser.add_argument(
        "--batch-size", "--batch_size", type=int, help="batch size for benchmarking"
    )
    parser.add_argument(
        "--iterations", type=int, default=2, help="how many iterations to run"
    )
    parser.add_argument(
        "--batch-size-file", type=str, help="String to load batch size from"
    )
    parser.add_argument("--cosine", action="store_true", help="use cosine similarity")
    parser.add_argument(
        "--cpp-wrapper", action="store_true", help="turn on cpp/cuda wrapper codegen"
    )
    parser.add_argument(
        "--freezing", action="store_true", help="turn on freezing", default=False
    )
    parser.add_argument(
        "--ci", action="store_true", help="Flag to tell that its a CI run"
    )
    parser.add_argument(
        "--dynamic-ci-skips-only",
        action="store_true",
        help=(
            "Run only the models that would have been skipped in CI "
            "if dynamic-shapes, compared to running without dynamic-shapes.  "
            "This is useful for checking if more models are now "
            "successfully passing with dynamic shapes.  "
            "Implies --dynamic-shapes and --ci"
        ),
    )
    parser.add_argument(
        "--dashboard", action="store_true", help="Flag to tell that its a Dashboard run"
    )
    parser.add_argument(
        "--skip-fp64-check", action="store_true", help="skip accuracy check using fp64"
    )
    parser.add_argument(
        "--fast", "-f", action="store_true", help="skip slow benchmarks"
    )
    parser.add_argument(
        "--only",
        help="""Run just one model from torchbench. Or
        specify the path and class name of the model in format like:
        --only=path:<MODEL_FILE_PATH>,class:<CLASS_NAME>

        Due to the fact that dynamo changes current working directory,
        the path should be an absolute path.

        The class should have a method get_example_inputs to return the inputs
        for the model. An example looks like
        ```
        class LinearModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

            def get_example_inputs(self):
                return (torch.randn(2, 10),)
        ```
    """,
    )
    parser.add_argument(
        "--multiprocess",
        action="store_true",
        help="Create n processes based on the number of devices (distributed use case).",
    )
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Wraps model in DDP before running it, and uses dynamo DDPOptmizer (graph breaks) by default.",
    )
    parser.add_argument(
        "--fsdp",
        action="store_true",
        help="""Wraps model in FSDP before running it. Disables cudagraphs by default.
        Doesn't recursively wrap, mainly useful for checking dynamo UnspecNNModule compatibility
    """,
    )
    parser.add_argument(
        "--no-optimize-ddp",
        action="store_true",
        help="Disables dynamo DDPOptimizer (graph breaks). (Applies only when using --ddp benchmark mode).",
    )
    parser.add_argument(
        "--distributed-master-port",
        default="6789",
        help="Port to bind for for torch.distributed.  Use the default unless it's conflicting with another user",
    )
    parser.add_argument(
        "--dynamic-shapes",
        action="store_true",
        help="Runs a dynamic shapes version of the benchmark, if available.",
    )
    parser.add_argument(
        "--dynamic-batch-only",
        action="store_true",
        help="Only assume batch dimension is dynamic.  Implies --dynamic-shapes",
    )
    parser.add_argument(
        "--specialize-int", action="store_true", help="Run with specialize_int=True."
    )
    parser.add_argument(
        "--use-eval-mode",
        action="store_true",
        help="sets model.eval() to reduce randomness",
    )
    parser.add_argument(
        "--skip-accuracy-check",
        action="store_true",
        help="keeps running even when accuracy fails",
    )
    parser.add_argument(
        "--generate-aot-autograd-stats",
        action="store_true",
        help="Generates AOT Autograd stats like how mnay graphs are sent to AOT",
    )
    parser.add_argument(
        "--inductor-settings",
        action="store_true",
        help="Use same settings as --inductor for baseline comparisons",
    )
    parser.add_argument(
        "--suppress-errors",
        action="store_true",
        help="Suppress errors instead of raising them",
    )
    parser.add_argument(
        "--output",
        help="Overrides the output filename",
    )
    parser.add_argument(
        "--output-directory",
        help="Overrides the directory to place output files.",
    )
    parser.add_argument(
        "--baseline",
        help="Compare with a prior --output",
    )
    parser.add_argument(
        "--part",
        default=None,
        help="Specify the part of the model to run.",
    )
    parser.add_argument(
        "--export-profiler-trace",
        action="store_true",
        help="exports trace of kineto profiler",
    )
    parser.add_argument(
        "--profiler-trace-name",
        "--profiler_trace_name",
        help="Overwrites exported trace name",
    )
    parser.add_argument(
        "--diff-branch",
        default=diff_branch_default,
        help="delta current branch against given branch.",
    )
    parser.add_argument(
        "--tag", default=None, help="Specify a tag to be included in csv files."
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="print some graph/op statistics during the run, similar to .explain()",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="print graph counter stats",
    )
    parser.add_argument(
        "--print-memory",
        action="store_true",
        help="print extra memory statistics",
    )
    parser.add_argument(
        "--print-dataframe-summary",
        action="store_true",
        help="print dataframe result used for calculating accuracy",
    )
    parser.add_argument(
        "--cold-start-latency",
        "--cold_start_latency",
        action="store_true",
        help="Use a fresh triton cachedir when running each model, to force cold-start compile.",
    )
    parser.add_argument(
        "--disable-cudagraphs",
        action="store_true",
        help="Disables cudagraphs for Inductor",
    )
    parser.add_argument(
        "--disable-split-reductions",
        action="store_true",
        help="Disables split reductions for Inductor",
    )
    parser.add_argument(
        "--disable-persistent-reductions",
        action="store_true",
        help="Disables split reductions for Inductor",
    )
    parser.add_argument(
        "--disable-divisible-by-16",
        action="store_true",
        help="Disables divisible by 16 hint to Triton for Inductor",
    )
    parser.add_argument(
        "--inductor-compile-mode",
        default=None,
        help="torch.compile mode argument for inductor runs.",
    )
    parser.add_argument(
        "--print-graph-breaks",
        action="store_true",
        help="Show a warning whenever graph break",
    )
    parser.add_argument(
        "--log-graph-breaks",
        action="store_true",
        help="log graph breaks in a file",
    )
    parser.add_argument(
        "--trace-on-xla",
        action="store_true",
        help="Whether to trace the model on XLA or on eager device",
    )
    parser.add_argument(
        "--xla-tolerance",
        type=float,
        default=1e-2,
        help="XLA needs a loose tolerance to pass the correctness check",
    )
    parser.add_argument(
        "--collect-outputs",
        action="store_true",
        help="""Whether to collect outputs for training. Set this to true if we
        want to verify the numerical correctness of graidents. But that may
        cause time measurement not accurate""",
    )
    parser.add_argument(
        "--enable-activation-checkpointing",
        action="store_true",
        help="Enables activation checkpointing for HF models",
    )
    parser.add_argument("--timing", action="store_true", help="Emits phase timing")

    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print n/k models message between each model run.",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=2000,
        help="timeout (second) for benchmarking.",
    )

    parser.add_argument(
        "--per_process_memory_fraction",
        type=float,
        default=1,
        help="Set per-process GPU memory fraction (limit) for reducing usable size and reproducing OOMs",
    )

    parser.add_argument(
        "--no-translation-validation",
        action="store_true",
        help="Disable translation validation for accuracy builds.",
    )

    parser.add_argument(
        "--minify",
        action="store_true",
        help="Enable minification when failure is below tolerance. Save repro script for each model.",
    )

    group_fuser = parser.add_mutually_exclusive_group()
    # --nvfuser is now the default, keep the option to not break scripts
    group_fuser.add_argument("--nvfuser", action="store_true", help=argparse.SUPPRESS)
    group_fuser.add_argument("--nnc", action="store_true", help="enable NNC for GPUs")

    group_prec = parser.add_mutually_exclusive_group()
    group_prec.add_argument("--float16", action="store_true", help="cast model to fp16")
    group_prec.add_argument(
        "--bfloat16", action="store_true", help="cast model to bf16"
    )
    group_prec.add_argument("--float32", action="store_true", help="cast model to fp32")
    group_prec.add_argument(
        "--amp", action="store_true", help="use automatic mixed precision"
    )

    group_printout = parser.add_mutually_exclusive_group()
    group_printout.add_argument(
        "--verbose", "-v", action="store_true", help="enable verbose debug printouts"
    )
    group_printout.add_argument(
        "--quiet", "-q", action="store_true", help="suppress debug printouts"
    )

    group = parser.add_mutually_exclusive_group()
    from common import coverage_experiment
    group.add_argument(
        "--coverage", action="store_true", help="(default) " + coverage_experiment.__doc__

    )
    
    from common import overhead_experiment

    group.add_argument(
        "--overhead", action="store_true", help= overhead_experiment.__doc__
    )
    group.add_argument(
        "--speedup-dynamo-ts",
        action="store_true",
        help="TorchDynamo frontend with torchscript backend",
    )
    
    from common import speedup_experiment_fx2trt

    group.add_argument(
        "--speedup-fx2trt", action="store_true", help=speedup_experiment_fx2trt.__doc__
    )
    group.add_argument(
        "--speedup-fx2trt-fp16",
        action="store_true",
        help=help(speedup_experiment_fx2trt),
    )
    group.add_argument(
        "--print-fx",
        action="store_true",
        help="Print fx traces captured from model",
    )
    group.add_argument(
        "--print-aten-ops",
        action="store_true",
        help="Print traces of aten ops captured by AOT autograd",
    )
    group.add_argument(
        "--inductor",
        action="store_true",
        help="Measure speedup with TorchInductor",
    )
    group.add_argument(
        "--export",
        action="store_true",
        help="Measure pass rate with export",
    )
    group.add_argument(
        "--export-aot-inductor",
        action="store_true",
        help="Measure pass rate with Export+AOTInductor",
    )
    group.add_argument(
        "--xla", action="store_true", help="Compare TorchXLA to eager PyTorch"
    )
    group.add_argument(
        "--torchscript-onnx",
        "--torchscript_onnx",
        action="store_true",
        help="Measure speedup with TorchScript ONNX, i.e. `torch.onnx.export`",
    )
    group.add_argument(
        "--dynamo-onnx",
        "--dynamo_onnx",
        action="store_true",
        help="Measure speedup with Dynamo ONNX, i.e. `torch.onnx.dynamo_export`",
    )
    group.add_argument(
        "--backend",
        choices=torch._dynamo.list_backends(exclude_tags=None),
        help="measure speedup with a given backend",
    )
    from common import null_experiment
    group.add_argument("--nothing", action="store_true", help=null_experiment.__doc__)
    group.add_argument(
        "--log-conv-args",
        action="store_true",
        help="Dump convolution input/weight/bias's shape/stride/dtype and other options to json",
    )
    group.add_argument(
        "--recompile-profiler",
        "--recompile_profiler",
        action="store_true",
        help="Run the dynamo recompilation profiler on each model.",
    )
    group.add_argument(
        "--find-batch-sizes",
        action="store_true",
        help="finds the largest batch size that could fit on GPUs",
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--accuracy",
        action="store_true",
        help="Checks accuracy with small batch size and eval mode",
    )
    mode_group.add_argument(
        "--performance", action="store_true", help="Measures performance speedup"
    )
    mode_group.add_argument(
        "--tolerance",
        action="store_true",
        help="extracts the tolerance for each model with small batch size and eval mode",
    )
    run_mode_group = parser.add_mutually_exclusive_group(required=True)
    run_mode_group.add_argument(
        "--training",
        action="store_true",
        help="Performs training",
    )
    run_mode_group.add_argument(
        "--inference", action="store_true", help="Performs inference"
    )
    return parser.parse_args(args)