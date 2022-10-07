from torch.utils.benchmark import Timer


def time_with_torch_timer(fn, args, kwargs={}, iters=100):
    env = {"args": args, "kwargs": kwargs, "fn": fn}
    fn_call = "fn(*args, **kwargs)"

    # Measure end-to-end time
    timer = Timer(stmt=f"{fn_call}", globals=env)
    tt = timer.timeit(iters)

    return tt
