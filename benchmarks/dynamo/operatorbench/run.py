import operators
from operators import BaseOperator
import click

QUANTILES = [0.2, 0.5, 0.8]

def benchmark_operator(OperatorClass: BaseOperator, device, phase, repeat, single_run):
    print(f"Benchmarking {OperatorClass.name} {OperatorClass.variant}")
    operator = OperatorClass()
    operator.generate_inputs()
    if phase == "forward":
        phase_fn = operator.forward
    elif phase == "backward":
        phase_fn = operator.backward
    else:
        phase_fn = operator.full
    if single_run:
        run_count = 1
    else:
        run_count = len(operator.get_inputs())
    
    durations = []

    from triton.testing import do_bench
    for _ in range(repeat):
        for i in range(run_count):
            input, target = operator.get_inputs()[i]

            def fn():  # Adding a blank line before the nested function definition
                return phase_fn(input, target)

            durations.append(do_bench(fn, quantiles=QUANTILES))
    print(durations)

@click.command()
@click.option("--op", help="operator overload to benchmark")
@click.option("--dtype", help="dtype to benchmark")
@click.option("--max-samples", help="max samples per op", default=15)
@click.option("--device", help="device to benchmark", default="cuda")
@click.option("--phase", help="phase to benchmark", default="forward")
@click.option("--repeat", help="repeat", default=2)
@click.option("--single-run", help="single run", default=False)
def run_benchmarks(
    op,
    dtype,
    max_samples,
    device,
    phase,
    repeat,
    single_run,
):
    operators_list = operators.list_operators()
    for operator in operators_list:
        benchmark_operator(operator, device, phase, repeat, single_run)


if __name__ == "__main__":
    run_benchmarks()
