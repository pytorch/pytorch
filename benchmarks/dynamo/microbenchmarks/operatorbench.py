#!/usr/bin/env python3
import click
import numpy as np
import torch
from operator_inp_utils import OperatorInputsLoader

from torch._dynamo.backends.cudagraphs import cudagraphs_inner
from torch._dynamo.testing import same
from torch._inductor.compile_fx import compile_fx
from torch._inductor.decomposition import decompositions
from torch._inductor.lowering import fallbacks, lowerings
from torch._inductor.utils import gen_gm_and_inputs


aten = torch.ops.aten


def compute_speedups(
    operator, models, example_inputs, repeats, accuracy_checking=False, device="cuda"
):
    expected = models[0](*example_inputs)
    if accuracy_checking:
        for model in models[1:]:
            actual = model(*example_inputs)
            # change to assert later
            try:
                same(actual, expected, cos_similarity=True, equal_nan=True)
            except AssertionError as e:
                print(e)
                print(f"Accuracy check failed: {operator}")
                print((expected[0] - actual[0]).abs().max())

    timings = np.zeros((repeats, len(models)), np.float64)
    for rep in range(repeats):
        # interleave the runs to handle frequency scaling and load changes
        for m, model in enumerate(models):
            if device == "cuda":
                import triton

                # do_bench() clears L2 cache to hide the latency of CPU launch time
                # along with cuda synchronization
                timings[rep, m] = triton.testing.do_bench(
                    lambda: model(*example_inputs)
                )
            else:
                from torch._inductor.utils import timed

                timings[rep, m] = timed(model, example_inputs)
    return np.median(timings, axis=0)


def strip_overloads(gm):
    """
    Modifies the target of graph nodes in :attr:`gm` to strip overloads.
    Args:
        gm(fx.GraphModule): The input Fx graph module to be modified
    """
    for node in gm.graph.nodes:
        if isinstance(node.target, torch._ops.OpOverload):
            node.target = node.target.overloadpacket
    gm.recompile()


def convert_to_jit(gm, gm_args):
    strip_overloads(gm)
    try:
        return torch.jit.script(gm)
    except Exception:
        pass
    return torch.jit.trace(gm, gm_args)


def microbenchmark(
    operator, args, kwargs, dtype, accuracy_checking, repeats, measure_nvfuser, device
):
    gm, gm_args = gen_gm_and_inputs(operator, args, kwargs)
    torch.jit._builtins._register_builtin(
        torch.ops.aten.convolution_backward.default, "aten::convolution_backward"
    )
    if device == "cuda":
        cudagraphs_eager = cudagraphs_inner(gm, gm_args, copy_outputs=False)
        compiled_fn = compile_fx(gm, gm_args)
        compiled = [cudagraphs_eager, compiled_fn]
    else:
        compiled_fn = compile_fx(gm, gm_args)
        compiled = [gm, compiled_fn]
    if measure_nvfuser:
        g = convert_to_jit(gm, gm_args)
        cudagraphs_jit = cudagraphs_inner(g, gm_args, copy_outputs=False)
        compiled += [cudagraphs_jit]
    if accuracy_checking:
        repeats = 1

    medians = compute_speedups(
        operator, compiled, gm_args, repeats, accuracy_checking, device
    )
    return medians


def skip_operator(operator):
    nyi_strings = (
        "aten.gather.default",
        "nll_loss",
        "aten.index",
        "aten.scatter_",
        "masked_fill_.Scalar",
    )

    if any(nyi_string in str(operator) for nyi_string in nyi_strings):
        # maybe disable aten.native_layer_norm.default
        # TODO - inputs cannot be randomly initialized, causes cyda failures
        print(f"Skipping {operator}, input generator nyi")
        return True

    # not covered by other non-compute operator heuristics
    if operator == torch.ops.aten._unsafe_view.default:
        print(f"Skipping {operator}, non compute operator")
        return True

    # some of inductor registered to the OpOverload, some registered to OpOverloadPacket
    op_impls = [operator]
    if isinstance(operator, torch._ops.OpOverload):
        op_impls.append(operator.overloadpacket)

    if any(op in fallbacks for op in op_impls):
        print(f"Skipping {operator}, no inductor impl")
        return True

    if all(op not in decompositions and op not in lowerings for op in op_impls):
        print(f"Skipping {operator}, no inductor impl")
        return True

    if "convolution" in str(operator):
        return True

    return False


@click.command()
@click.option(
    "--suite",
    help="suite to load inps from: options: timm, huggingface, torchbench",
    default="torchbench",
)
@click.option("--op", help="operator overload to benchmark")
@click.option("--dtype", help="dtype to benchmark")
@click.option("--max-samples", help="max samples per op", default=15)
@click.option("--accuracy-checking", help="check accuracy", default=False)
@click.option(
    "--repeats", help="how many times to repeat for perf measurement", default=3
)
@click.option(
    "--measure-nvfuser", help="default we only measure inductor", default=False
)
@click.option("--device", help="cpu or cuda", default="cuda")
@click.option("--inp-file", help="use custom input file instead of suite", default=None)
@click.option("--start-idx", help="specify start index of samples", default=0)
def benchmark(
    suite,
    op,
    dtype,
    max_samples,
    accuracy_checking,
    repeats,
    measure_nvfuser,
    device,
    inp_file,
    start_idx,
):
    if inp_file is not None:
        loader = OperatorInputsLoader(inp_file)
    else:
        assert suite in ("timm", "huggingface", "torchbench"), f"got {suite}"
        if suite == "timm":
            loader = OperatorInputsLoader.get_timm_loader()
        elif suite == "huggingface":
            loader = OperatorInputsLoader.get_huggingface_loader()
        else:
            loader = OperatorInputsLoader.get_torchbench_loader()

    assert dtype in ("float16", "float32"), f"got {dtype}"

    if op == "all":
        filename = f"timings_{suite}_{op.replace('.', '_')}{dtype}.txt"
        f = open(filename, "a")

    dtype = torch.float16 if dtype == "float16" else torch.float32

    if op == "all":
        ops = loader.get_all_ops()
    else:
        ops = [eval(op)]

    max_samples = max_samples + start_idx
    for operator in ops:
        if skip_operator(operator):
            continue

        print(f"Running {operator}")
        inp_gen = loader.get_inputs_for_operator(operator, dtype=dtype, device=device)
        timings = []

        for i in range(min(max_samples, 1000000)):
            try:
                inps = next(inp_gen)
                if inps is None:
                    break
                if i < start_idx:
                    continue
                print(f"Iter {i}")
                args, kwargs = inps
            except StopIteration:
                break
            try:
                # aten, nvfuser, inductor
                timings.append(
                    microbenchmark(
                        operator,
                        args,
                        kwargs,
                        dtype,
                        accuracy_checking,
                        repeats,
                        measure_nvfuser,
                        device,
                    )
                )
            except Exception as e:
                print(f"error {operator}")
                print(e)
                # comment out this line to avoid blocking other tests
                # raise e

        if not timings:
            continue

        timings = torch.tensor(timings).T
        q = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)
        output = f"{operator}:\nInductor Speedups : {(torch.quantile(timings[0] / timings[1], q)).tolist()}\n"
        if measure_nvfuser:
            output += f"NVFUSER Speedups :{(torch.quantile(timings[0] / timings[2], q)).tolist()}\n"
        if op == "all":
            f.write(output)
        print(output)

    if op == "all":
        f.close()


if __name__ == "__main__":
    benchmark()
