import torch.fx
from .. import config


def dummy_user_function_to_inline_gm(gm, args):
    return gm(*args)


def dummy_user_function_to_inline_gm_with_kwargs(gm, args, kwargs):
    return gm(*args, **kwargs)


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


def decompose_and_inline_function_with_makefx(tx, op, args, kwargs):
    import inspect

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

    def get_full_args_with_default_value(func, args):
        try:
            signature = inspect.signature(func)
        except ValueError as ve:
            # no signature found, just return an empty dict
            return args

        keys = signature.parameters.keys()
        # function takes `args` and `kwargs`, below method won't be able to
        # tell if it has any default arguments. Return args directly.
        if len(keys) == 2 and "args" in keys and "kwargs" in keys:
            return args

        current_i = 0
        res = []
        for _, v in signature.parameters.items():
            if v.default is inspect.Parameter.empty or current_i < len(args):
                # this arg does not have default value or caller provided the args.
                assert current_i < len(args)
                res.append(args[current_i])
                current_i += 1
            else:
                # use default value
                res.append(v.default)

        # All args should be used at this point.
        assert current_i == len(args)
        return res

    # make_fx requires caller to pass into all args including
    # those with default value. Note that we don't need to provide the exact
    # value even if they are provided as kwargs.
    complete_fake_value_args = get_full_args_with_default_value(op, fake_value_args)
    with enable_python_dispatcher():
        fx_g = make_fx(op, pre_dispatch=True)(*complete_fake_value_args)

    print("\nfx code")
    print(fx_g.code)

    # now inline this fx graph and return the output
    # question: will there be a loop? How do I tell if op is CompositeImplicitAutograd
    user_fn_variable = SourcelessBuilder()(tx, dummy_user_function_to_inline_gm)
    user_fn_variable_with_kwargs = SourcelessBuilder()(
        tx, dummy_user_function_to_inline_gm_with_kwargs
    )
    gm_variable = SourcelessBuilder()(tx, fx_g)
    cls = BaseListVariable.cls_for(list)
    input_args_variable = cls(
        args,
        mutable_local=MutableLocal(),
    )
    if len(kwargs) > 0:
        updated_kwargs = {}
        # make_fx will modify the key for kwargs, try to reverse engineering the
        # key mapping.
        for k, v in kwargs.items():
            updated_kwargs[k + "_1"] = v
        input_kwargs_variable = ConstDictVariable(
            updated_kwargs,
            dict,
            mutable_local=MutableLocal(),
        )
        res = tx.inline_user_function_return(
            user_fn_variable_with_kwargs,
            (gm_variable, input_args_variable, input_kwargs_variable),
            {},
        )
    else:
        res = tx.inline_user_function_return(
            user_fn_variable, (gm_variable, input_args_variable), {}
        )
    return res
