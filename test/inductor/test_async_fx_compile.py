import argparse
import time

import torch
from torch._functorch import config as functorch_config
from torch._inductor import config
from torch._inductor.compile_fx import FxCompileMode
import torch._inductor


parser = argparse.ArgumentParser()
parser.add_argument(
    "--backend",
    default="inductor",
    choices=("none", "inductor", "eager"),
)
parser.add_argument("--size", type=int, default=100)
parser.add_argument("--instrument", action=argparse.BooleanOptionalAction)
parser.add_argument("--model", default="add", choices=("mm", "add"))
parser.add_argument("--header", default=True, action=argparse.BooleanOptionalAction)
# 20,000 = 227s
# 5280 = 60s
parser.add_argument("--count", type=int, default=5280)
parser.add_argument("--repeat", type=int, default=10)
parser.add_argument("--mode", default="normal", choices=tuple(x.lower() for x in FxCompileMode.__members__.keys()))
args = parser.parse_args()


def model_mm(x, y):
    grid = [x, y] * 512
    while len(grid) > 1:
        gridout = []
        for a, b in zip(grid[0::2], grid[1::2]):
            gridout.append(torch.matmul(a, b))
        grid = gridout

    return grid[0]


def model_add(x, y):
    out = x
    for i in range(args.count):
        out = torch.add(out, y)
    return out


def run_test(f):
    if args.backend != "none":
        f = torch.compile(f, fullgraph=True, backend=args.backend)

    for i in range(args.repeat):
        x = torch.randn(args.size, args.size)
        y = torch.randn(args.size, args.size)

        start = time.time()
        result = f(x, y)
        end = time.time()
        print("result:", repr(result).replace("\n", "").replace(" ", "")[:100])
        if args.repeat == 1:
            print(f"{end - start:0.1f}s")
        else:
            print(f"iteration {i:02}: {end - start:0.1f}s")


def main():
    # Turn off local caches
    config.autotune_local_cache = False
    config.fx_graph_cache = False
    functorch_config.enable_autograd_cache = False
    functorch_config.strict_autograd_cache = True

    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float32)

    if args.header:
        print(
            f"MODEL: {args.model}, BACKEND: {args.backend}, SIZE: {args.size}, COUNT: {args.count}, MODE: {args.mode}"
        )

    if args.model == "mm":
        f = model_mm
    elif args.model == "add":
        f = model_add
    else:
        raise NotImplementedError

    mode = FxCompileMode.__members__[args.mode.upper()]
    torch._inductor.compile_fx.compile_mode = mode

    profiler = None
    if args.instrument:
        import pyinstrument

        profiler = pyinstrument.Profiler()
        profiler.start()

    run_test(f)

    if profiler:
        session = profiler.stop()
        renderer = pyinstrument.renderers.SpeedscopeRenderer(show_all=True)
        with open("runner.ss", "w") as f:
            f.write(renderer.render(session))
        print("<output written to runner.ss>")


if __name__ == "__main__":
    main()
