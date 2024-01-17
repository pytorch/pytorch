def dummy_user_function_to_inline_gm(gm, args):
    return gm(*args)


def dummy_accumulate_grad_(t1, t2):
    if t1.grad is None:
        t1.grad = t2
    else:
        t1.grad += t2
