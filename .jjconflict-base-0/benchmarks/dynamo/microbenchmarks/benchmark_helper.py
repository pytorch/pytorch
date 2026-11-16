from torch.utils.benchmark import Timer


def time_with_torch_timer(fn, args, kwargs=None, iters=100):
    kwargs = kwargs or {}
    env = {"args": args, "kwargs": kwargs, "fn": fn}
    fn_call = "fn(*args, **kwargs)"

    # Measure end-to-end time
    timer = Timer(stmt=f"{fn_call}", globals=env)
    tt = timer.timeit(iters)

    return tt
