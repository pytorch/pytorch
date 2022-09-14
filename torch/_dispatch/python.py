import torch
from contextlib import contextmanager

__all__ = ['enable_python_dispatcher', 'no_python_dispatcher']

DispatchKey = torch._C.DispatchKey  # type: ignore[attr-defined]

def has_key(op, k):
    return (
        torch._C._dispatch_has_kernel_for_dispatch_key(op.name(), k)
        or k in op.py_kernels
    )

is_included_in_alias = torch._C._dispatch_is_included_in_alias

# Equivalent to computeDispatchTableEntryWithDebug
# TODO: memoize this or something
def resolve_key(op: torch._ops.PyOperatorABC, k: DispatchKey):  # type: ignore[valid-type]
    # 1. (Direct) operator registration
    if has_key(op, k):
        return k
    # 2.1 Use CompositeExplicitAutogradNonFunctional kernel if available
    cand = DispatchKey.CompositeExplicitAutogradNonFunctional
    if (k == DispatchKey.Undefined or is_included_in_alias(k, cand)) and has_key(op, cand):
        return cand
    # 2.2 Use CompositeExplicitAutograd kernel if available
    cand = DispatchKey.CompositeExplicitAutograd
    if (k == DispatchKey.Undefined or is_included_in_alias(k, cand)) and has_key(op, cand):
        return cand
    has_backend_kernel = (
        torch._C._dispatch_has_kernel_for_any_dispatch_key(op.name(), torch._C._dispatch_get_backend_keyset_from_autograd(k))
        or has_key(op, DispatchKey.CompositeExplicitAutograd)
    )
    # 2.3. Use CompositeImplicitAutograd kernel if available
    cand = DispatchKey.CompositeImplicitAutogradNestedTensor
    if (
        (k != DispatchKey.Undefined and is_included_in_alias(k, cand))  # type: ignore[attr-defined]
            and has_key(op, cand) and not has_backend_kernel):
        return cand
    cand = DispatchKey.CompositeImplicitAutograd
    if (k == DispatchKey.Undefined or is_included_in_alias(k, cand)) and has_key(op, cand):
        if (
            k == DispatchKey.AutogradOther
            and torch._C._dispatch_has_kernel_for_any_dispatch_key(op.name(), torch._C._dispatch_autogradother_backends)  # type: ignore[attr-defined] # noqa: B950
        ):
            raise RuntimeError("ambiguous autogradother kernel")
        elif not has_backend_kernel:
            return cand
    # 2.4. For autograd backend keys, use kernel from DispatchKey::Autograd if available
    cand = DispatchKey.Autograd
    if is_included_in_alias(k, cand) and has_key(op, cand):
        return cand
    # Backend fallback
    if torch._C._dispatch_has_backend_fallback(k):
        # The dispatch key itself will implicitly route to backend fallback.
        # This is probably not great for the pure Python implementation.
        return k
    raise RuntimeError("could not find kernel")

@contextmanager
def no_python_dispatcher():
    g = torch._C._DisablePythonDispatcher()
    try:
        yield
    finally:
        del g

@contextmanager
def enable_python_dispatcher():
    g = torch._C._EnablePythonDispatcher()
    try:
        yield
    finally:
        del g

# The Python dispatcher
def python_dispatcher(op, ks, args, kwargs):
    """
    with no_python_dispatcher():
        print(op, ks, args, kwargs)
    """
    k = resolve_key(op, ks.highestPriorityTypeId())
    source = f'torch.ops.{op}.dispatch(k, *args, **kwargs)'
    filename = f'{op}[{torch._C._dispatch_key_name(k)}]'
    compiled = compile(source, filename, 'eval')  # TODO: maybe cache?
    return eval(compiled, {'torch': torch, 'k': k, 'args': args, 'kwargs': kwargs})

torch._C._set_python_dispatcher(python_dispatcher)
