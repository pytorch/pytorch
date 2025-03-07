import torch


@torch.library.custom_op("foo::bar", device_types="cpu", mutates_args=())
def bar(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    e: torch.Tensor,
    f: torch.Tensor,
    g: torch.Tensor,
    h: torch.Tensor,
    i: torch.Tensor,
    j: torch.Tensor,
    k: torch.Tensor,
    l: torch.Tensor,
    m: torch.Tensor,
    n: torch.Tensor,
) -> torch.Tensor:
    return b.clone()


@torch.library.custom_op("foo::baz", device_types="cpu", mutates_args=(["a"]))
def baz(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    e: torch.Tensor,
    f: torch.Tensor,
    g: torch.Tensor,
    h: torch.Tensor,
    i: torch.Tensor,
    j: torch.Tensor,
    k: torch.Tensor,
    l: torch.Tensor,
    m: torch.Tensor,
    n: torch.Tensor,
) -> torch.Tensor:
    return b.clone()


def barbaz(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    e: torch.Tensor,
    f: torch.Tensor,
    g: torch.Tensor,
    h: torch.Tensor,
    i: torch.Tensor,
    j: torch.Tensor,
    k: torch.Tensor,
    l: torch.Tensor,
    m: torch.Tensor,
    n: torch.Tensor,
) -> torch.Tensor:
    return b.clone()


foo_lib = torch.library.Library("foo", "FRAGMENT")


def direct_register_custom_op(op_name, op_func, mutates_args):
    schema_str = torch.library.infer_schema(op_func, mutates_args=mutates_args)
    foo_lib.define(op_name + schema_str)
    foo_lib.impl(op_name, op_func, "CPU")


direct_register_custom_op("foo::bar_op", barbaz, mutates_args=())
direct_register_custom_op("foo::baz_op", barbaz, mutates_args=(["a"]))

a = torch.rand([1, 1], device="cpu")
b = torch.rand([1, 1], device="cpu")
c = torch.rand([1, 1], device="cpu")
d = torch.rand([1, 1], device="cpu")
e = torch.rand([1, 1], device="cpu")
f = torch.rand([1, 1], device="cpu")
g = torch.rand([1, 1], device="cpu")
h = torch.rand([1, 1], device="cpu")
i = torch.rand([1, 1], device="cpu")
j = torch.rand([1, 1], device="cpu")
k = torch.rand([1, 1], device="cpu")
l = torch.rand([1, 1], device="cpu")
m = torch.rand([1, 1], device="cpu")
n = torch.rand([1, 1], device="cpu")

import os
import time


n_warmup = 3
n_bench = 1


def do_bench(fn, name):
    fn(n_warmup)

    start_time = time.time()
    fn(n_bench)
    dur_us = (time.time() - start_time) * 1000_000
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
        ],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        fn(n_bench)
    profile_path = f"/home/ivankobzarev/task_custom_ops_perf/{name}.json"
    if os.path.exists(profile_path):
        os.remove(profile_path)

    prof.export_chrome_trace(profile_path)
    print(f"DO_BENCH {name}: {dur_us} us PROFILE:{profile_path}")
    return dur_us


def test():
    def mutate(num):
        for z in range(num):
            o = torch.ops.foo.baz(a, b, c, d, e, f, g, h, i, j, k, l, m, n)

    def no_mutate(num):
        for z in range(num):
            o = torch.ops.foo.bar(a, b, c, d, e, f, g, h, i, j, k, l, m, n)

    def direct_mutate(num):
        for z in range(num):
            o = torch.ops.foo.baz_op(a, b, c, d, e, f, g, h, i, j, k, l, m, n)

    def direct_no_mutate(num):
        for z in range(num):
            o = torch.ops.foo.bar_op(a, b, c, d, e, f, g, h, i, j, k, l, m, n)

    import os

    sfx = os.getenv("SFX", "")
    mutate_time = do_bench(mutate, f"mutate{sfx}")
    no_mutate_time = do_bench(no_mutate, f"no_mutate{sfx}")
    direct_mutate_time = do_bench(direct_mutate, f"direct_mutate{sfx}")
    direct_no_mutate_time = do_bench(direct_no_mutate, f"direct_no_mutate{sfx}")


test()
