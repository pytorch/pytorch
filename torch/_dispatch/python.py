import torch

def has_key(op, k):
    return torch._C._dispatch_has_kernel_for_dispatch_key(op.name, k)

def resolve_key(op, k):
    if has_key(op, k):
        return k
    return k

# The Python dispatcher
def python_dispatcher(op, ks, args, kwargs):
    g = torch._C._DisablePythonDispatcher()
    print(op, ks, args, kwargs)
    del g
    k = str(ks.highestPriorityTypeId()).split('.')[1]
    k = resolve_key(op, k)
    return op._op_dk(k, *args, **kwargs)

torch._C._set_python_dispatcher(python_dispatcher)
