import operators
from operators import BaseOperator
from utils.common import BenchmarkConfig
import click
import torch
QUANTILES = [0.2, 0.5, 0.8]


def benchmark_operator(OperatorClass: BaseOperator, device, dtype, phase, max_samples, repeat, single_run):
    print(f"Benchmarking {OperatorClass.name} {OperatorClass.variant}")
    benchmark_config = BenchmarkConfig(
        device=device,
        dtype=dtype,
        phase=phase,
        max_samples=max_samples,
        repeat=repeat,
        single_run=single_run,
    )
    operator = OperatorClass(benchmark_config)
    operator.generate_inputs(benchmark_config)
    if phase == "forward":
        phase_fn = operator.forward
    elif phase == "backward":
        phase_fn = operator.backward
    else:
        phase_fn = operator.full
    if single_run:
        input_count = 1
    else:
        input_count = len(operator.get_inputs())

    durations = []

    from triton.testing import do_bench
    for sample in range(max_samples):
        for _ in range(repeat):
            for i in range(input_count):
                input, target = operator.get_inputs()[i]

                def fn():  # Adding a blank line before the nested function definition
                    return phase_fn(input, target)
                durations.append(do_bench(fn, quantiles=QUANTILES))
    print(durations)


@click.command()
@click.option("--op", help="operator overload to benchmark. split by ','.")
@click.option("--dtype", help="dtype to benchmark. [bfloat16, float16, float32]", default="bfloat16")
@click.option("--max-samples", help="max samples per op", default=5)
@click.option("--device", help="device to benchmark", default="cuda")
@click.option("--phase", help="phase to benchmark", default="forward")
@click.option("--repeat", help="repeat", default=3)
@click.option("--single-run", help="run with the first input size", default=False)
def run_benchmarks(
    op,
    dtype,
    max_samples,
    device,
    phase,
    repeat,
    single_run,
):
    # This is a list of classes, not instances
    operators_list: list[BaseOperator] = operators.list_operators()
    desired_op_names = None
    if op is not None:
        desired_op_names = op.split(",")
    else:
        desired_op_names = [operator.name for operator in operators_list]

    dtype_mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_mapping.get(dtype, torch.float32)  # Default to float32 if not found

    for operator in operators_list:
        if operator.name in desired_op_names:
            benchmark_operator(operator, device, dtype, phase, max_samples, repeat, single_run)


if __name__ == "__main__":
    run_benchmarks()
