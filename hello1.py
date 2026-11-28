# A small model with one branch on a global and one on tensor data, run through
# the torch._rust CFG executor, then shown as its CFG with the side effects (torch
# ops, Python calls) recorded at each instruction and the type of each value.
#
#   python hello1.py

import contextlib
import io

import torch
import torch._rust as rust


g0 = True


def submodel(a, b, c):
    return a * b + c


def model(a, b, c):
    if g0:
        e = submodel(a, b, c)
    else:
        e = submodel(b, c, a)

    if e[0].item() >= 0:
        print("  >= 0")
    else:
        print("  < 0")

    return e + 1


def accumulate(x, w):
    y = x * w
    y = y + x
    z = y * y
    return z + y


def run(label, a, b, c):
    print(f"\n=== {label} ===")
    out = rust.call_with_cfg_execution(model, a, b, c)
    with contextlib.redirect_stdout(io.StringIO()):
        expected = model(a, b, c)
    torch.testing.assert_close(out, expected)
    print("  result:", out)


if __name__ == "__main__":
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])

    # e = a * b + c = [-6, 10, 28]
    run("g0=True, e[0] < 0", a, b, torch.tensor([-10.0, 0.0, 10.0]))
    # e = a * b + c = [14, 20, 28]
    run("g0=True, e[0] >= 0", a, b, torch.tensor([10.0, 10.0, 10.0]))

    g0 = False
    # e = b * c + a = [-39, 2, 63]
    run("g0=False, e[0] < 0", a, b, torch.tensor([-10.0, 0.0, 10.0]))
    # e = b * c + a = [41, 52, 63]
    run("g0=False, e[0] >= 0", a, b, torch.tensor([10.0, 10.0, 10.0]))

    print("\n=== model CFG with side effects and types ===")
    print(rust.cfg_observed(model))

    print("\n=== submodel CFG with side effects and types ===")
    print(rust.cfg_observed(submodel))

    # Optimize submodel's entry block: it splits into glue that loads a, b and
    # c, a pure block whose version runs aten ops (guarded on its params, with
    # the original code as its origin), and glue that returns. Then run once
    # more through the optimized CFG.
    rust.optimize_block(submodel, 0)
    run("after optimizing submodel bb0", a, b, torch.tensor([10.0, 10.0, 10.0]))

    print("\n=== submodel CFG after optimizing bb0 ===")
    print(rust.cfg_observed(submodel))

    # A block that reuses its locals: optimizing splits it into glue that loads
    # x and w, a pure block of aten ops guarded on its inputs, and glue that
    # returns. The stores to y and z are gone: only the landing blocks a raise
    # unwinds to write them.
    x, w = torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])
    rust.call_with_cfg_execution(accumulate, x, w)
    rust.optimize_block(accumulate, 0)
    torch.testing.assert_close(
        rust.call_with_cfg_execution(accumulate, x, w), accumulate(x, w)
    )
    print("\n=== accumulate CFG after optimizing bb0 ===")
    print(rust.cfg_observed(accumulate))
