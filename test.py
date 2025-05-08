import inspect
def f(a) -> int:
    return 0

ret_annotate = inspect.signature(f).return_annotation
breakpoint()
