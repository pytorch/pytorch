


from torch._dynamo import register_backend


@register_backend
def inductor(*args, **kwargs):
    # do import here to avoid loading inductor into memory when it is not used
    from torch._inductor.compile_fx import compile_fx

    return compile_fx(*args, **kwargs)
