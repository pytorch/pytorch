# mypy: ignore-errors

from torch._dynamo import is_win32, register_backend


@register_backend
def inductor(*args, **kwargs):
    if is_win32():
        raise RuntimeError("Windows not yet supported for inductor")

    # do import here to avoid loading inductor into memory when it is not used
    from torch._inductor.compile_fx import compile_fx

    return compile_fx(*args, **kwargs)
