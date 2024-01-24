import torch.fx
from .. import config


def dummy_user_function_to_inline_gm(gm, args):
    return gm(*args)


def dummy_user_function_to_inline_wrapped_gm(wrapped_gm, args, kwargs):
    return wrapped_gm(args, kwargs)


def dummy_accumulate_grad_(t1, t2):
    if t1.grad is None:
        t1.grad = t2
    else:
        t1.grad += t2


def should_decompose_torch_op(fn):
    from torch._dynamo import compiled_autograd

    # TODO(JackCaoG): we need a better way to tell if a torch function should we decompose
    allowed_torch_fn = fn.__name__ not in ["_make_grads"]
    definanilly_not_composite_kernel = type(
        fn
    ) == torch._ops.OpOverload and not torch._C._dispatch_has_kernel_for_dispatch_key(
        fn.name(), torch._C.DispatchKey.CompositeImplicitAutograd
    )

    # only decompoization torch ops for forward
    in_compiled_backward = compiled_autograd.compiled_autograd_enabled
    return (
        config.use_single_step_graph
        and allowed_torch_fn
        and not definanilly_not_composite_kernel
        and not in_compiled_backward
    )


def decompose_and_inline_function_with_makefx(tx, fn, args, kwargs):
    from functorch import make_fx

    from torch._dispatch.python import enable_python_dispatcher
    from ..utils import get_fake_value
    from .base import MutableLocal
    from .builder import SourcelessBuilder
    from .dicts import ConstDictVariable
    from .lists import BaseListVariable

    # convert he arguments from VariableTracker to fake tensors + constants again
    fake_value_args = []
    for arg in args:
        if type(arg.as_proxy()) is torch.fx.proxy.Proxy:
            fake_value_args.append(get_fake_value(arg.as_proxy().node, tx))
        else:
            # mostly handle tuple and scalar
            fake_value_args.append(arg.as_proxy())

    fake_value_kwargs = {}
    for key, value in kwargs.items():
        if type(value.as_proxy()) is torch.fx.proxy.Proxy:
            fake_value_kwargs[key] = get_fake_value(value.as_proxy().node, tx)
        else:
            # mostly handle tuple and scalar
            fake_value_kwargs[key] = value.as_proxy()

    # Wrap the function before calling make_fx to avoid make_fx modify the kwargs's key.
    def wrapper_fn(fn):
        def inner(arg, kwargs):
            return fn(*arg, **kwargs)

        return inner

    wrapped_fn = wrapper_fn(fn)

    with enable_python_dispatcher():
        fx_g = make_fx(wrapped_fn, pre_dispatch=True)(
            fake_value_args, fake_value_kwargs
        )

    print("\nfx code")
    print(fx_g.code)

    # now inline this fx graph and return the output
    user_fn_variable_with_kwargs = SourcelessBuilder()(
        tx, dummy_user_function_to_inline_wrapped_gm
    )
    gm_variable = SourcelessBuilder()(tx, fx_g)
    cls = BaseListVariable.cls_for(list)
    input_args_variable = cls(
        args,
        mutable_local=MutableLocal(),
    )

    input_kwargs_variable = ConstDictVariable(
        kwargs,
        dict,
        mutable_local=MutableLocal(),
    )
    res = tx.inline_user_function_return(
        user_fn_variable_with_kwargs,
        (gm_variable, input_args_variable, input_kwargs_variable),
        {},
    )
    return res
