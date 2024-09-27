import warnings
from collections import defaultdict
from contextlib import nullcontext

import click
import operators
from operators import BaseOperator
from utils.common import BenchmarkConfig, Device, dtype_mapping, Phase
from utils.metrics import get_execution_time, MetricResult, Metrics

import torch


# mapping from operator name to the input list.
# We use the same input list for different variants of the same operator.
# {operator_name: input_list}
input_mapping = {}


# Create operator instances from desired operator names
def create_operator_instances(
    operator_names: list[str],
    name_to_variant_list: dict[str, list[BaseOperator]],
    benchmark_config: BenchmarkConfig,
    skip_variants: list[str],
) -> list[BaseOperator]:
    operator_instances = []
    for operator_name in operator_names:
        variant_classes = name_to_variant_list.get(operator_name, [])
        if not variant_classes:
            warnings.warn(f"Operator {operator_name} not found")
            continue
        for VariantClass in variant_classes:
            if VariantClass.variant in skip_variants:
                continue
            operator_instances.append(VariantClass(benchmark_config))
    return operator_instances


def benchmark_operator(operator: BaseOperator, benchmark_config: BenchmarkConfig):
    print(f"Benchmarking {operator.full_name}")
    phase = benchmark_config.phase
    max_samples = benchmark_config.max_samples
    repeat = benchmark_config.repeat
    device = benchmark_config.device
    metrics = benchmark_config.metrics
    num_samples = min(
        max_samples, len(operator.get_inputs(input_mapping, benchmark_config))
    )

    metric_result = MetricResult()
    metric_result.op_name = operator.name
    metric_result.op_variantant = operator.variant
    profiler_context = (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            profile_memory=False,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                f"{benchmark_config.profile_folder}/operator_{operator.full_name}",
                use_gzip=True,
            ),
        )
        if benchmark_config.profile
        else nullcontext()
    )
    with profiler_context:
        for i in range(num_samples):
            input = operator.get_inputs(input_mapping, benchmark_config)[i]
            input = operator.prepare_input_and_functions(input, phase)
            if phase == Phase.FORWARD:
                phase_fn = operator.forward
            elif phase == Phase.BACKWARD:
                phase_fn = operator.backward
            else:
                phase_fn = operator.full
            metric_result.input.append(input)
            execution_time = []
            record_sample_context = (
                torch.profiler.record_function(f"sample_{i}")
                if benchmark_config.profile
                else nullcontext()
            )

            with record_sample_context:
                for repeat_idx in range(repeat):

                    def fn():
                        return phase_fn(input)

                    record_repeat_context = (
                        torch.profiler.record_function(f"repeat_{repeat_idx}")
                        if benchmark_config.profile
                        else nullcontext()
                    )
                    with record_repeat_context:
                        if Metrics.EXECUTION_TIME in metrics:
                            execution_time.append(
                                get_execution_time(
                                    fn,
                                    grad_to_none=None,
                                    device=device,
                                )
                            )
            metric_result.execution_time.append(execution_time)
    return metric_result


@click.command()
@click.option("--op", help="operator overload to benchmark. split by ','.")
@click.option(
    "--dtype",
    help="dtype to benchmark. [bfloat16, float16, float32]",
    default="bfloat16",
)
@click.option(
    "--max-samples",
    help="max samples per op. each operator may have different inputs. this is the number of inputs to sample.",
    default=15,
)
@click.option(
    "--device",
    help=f"device to benchmark, {[device.value.lower() for device in Device]}. ",
    default=Device.CUDA.value,
)
@click.option(
    "--phase",
    help=f"phase to benchmark. {[phase.value.lower() for phase in Phase]}. ",
    default="forward",
)
@click.option("--repeat", help="repeat", default=5)
@click.option(
    "--metrics",
    help=f"metrics to benchmark. {[metric.value.lower() for metric in Metrics]}. split by ','",
    default=Metrics.EXECUTION_TIME.value,
)
@click.option(
    "--skip-variants",
    help="variants to be skipped, [liger, baseline, inductor]. split by ','",
    default="",
)
@click.option("--profile", help="profile", is_flag=True, default=False)
@click.option(
    "--profile-folder",
    help="set profile folder",
    default="./log",
)
def run_benchmarks(
    op,
    dtype,
    max_samples,
    device,
    phase,
    repeat,
    metrics,
    skip_variants,
    profile,
    profile_folder,
):
    global input_mapping
    # Reset input mapping to avoid OOM and mismatch in different unit tests
    input_mapping = {}
    # process arguments and generate benchmark config
    dtype = dtype_mapping.get(dtype)
    metrics = [
        Metrics[metric.strip().upper()]
        for metric in metrics.split(",")
        if metric.strip().upper() in Metrics.__members__
    ]
    device = Device[device.upper()]
    if device != Device.CUDA and Metrics.GPU_PEAK_MEM in metrics:
        print(f"{Metrics.GPU_PEAK_MEM.value} is only supported on cuda")
        metrics.remove(Metrics.GPU_PEAK_MEM)
    phase = Phase[phase.upper()]
    benchmark_config = BenchmarkConfig(
        device=device,
        dtype=dtype,
        phase=phase,
        max_samples=max_samples,
        repeat=repeat,
        metrics=metrics,
        profile=profile,
        profile_folder=profile_folder,
    )

    # This is a list of classes, not instances
    operator_class_list: list[BaseOperator] = operators.list_operators()
    name_to_variant_list = defaultdict(list)
    for OperatorClass in operator_class_list:
        name_to_variant_list[OperatorClass.name].append(OperatorClass)
    desired_op_names = None
    if op is not None:
        desired_op_names = op.split(",")
    else:
        desired_op_names = name_to_variant_list.keys()

    skip_variants = skip_variants.split(",")
    skip_variants = [
        variant.lower().strip() for variant in skip_variants if variant.strip()
    ]

    operator_metric_results = {}

    operator_instances = create_operator_instances(
        desired_op_names, name_to_variant_list, benchmark_config, skip_variants
    )
    for Operator in operator_instances:
        metric_result = benchmark_operator(Operator, benchmark_config)
        operator_metric_results[f"{Operator.name}.{Operator.variant}"] = metric_result

    for metric_result in operator_metric_results.values():
        print(metric_result)


if __name__ == "__main__":
    run_benchmarks()
