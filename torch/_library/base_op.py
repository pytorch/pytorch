import torch
import functools
import torch.utils._pytree as pytree
import inspect

def define_op(qualname, *, lib=None):
    return functools.partial(construct_op, qualname, lib=lib)


def construct_op(qualname, base_op, *, lib):
    members = dict(inspect.getmembers(base_op))
    default_members = dict(inspect.getmembers(object))
    user_members = {m for m in members if m not in default_members}

    allowed_members = set({"impl_default", "abstract", "schema"})

    device_impls = set({}) 
    for key in user_members:
        if key.startswith("__"):
            continue
        if key in allowed_members:
            continue

        # Allow "impl_{device}"
        impl = "impl_"
        if key.startswith(impl):
            device_type = key[len(impl):]
            # validates the device_type
            _ = torch.device(device_type)
            device_impls.add(device_type)
            continue

        raise ValueError(key)

    # get schema, define the op
    schema = members['schema']
    if schema == NOT_PROVIDED:
        raise RuntimeError(
            f"define_op(\"{qualname}\"): You must provide a schema class "
            f"attribute on the subclassed FunctionalBaseOp")
    if not torch._library.utils.is_functional_schema(qualname + schema):
        raise RuntimeError(
            f"{qualname}: FunctionalBaseOp can only be used to create "
            f"functional operators. Got non-function schema "
            f"specified in {base_op.__name__}: {schema}")

    assert schema != NOT_PROVIDED
    torch.library.define(qualname, schema, lib=lib)

    # register abstract impl
    abstract_impl = members['abstract']
    torch.library.impl_abstract(qualname, abstract_impl, lib=lib)

    # register device impls
    torch.library.impl(qualname, "default", members['impl_default'], lib=lib)
    for device_type in device_impls:
        impl = members[f"impl_{device_type}"]
        torch.library.impl(qualname, device_type, impl, lib=lib)

    # add autograd kernel that errors out if any inputs require grad.
    op = torch._library.utils.lookup_op(qualname)

    def autograd_impl(*args):
        if torch.is_grad_enabled() and pytree.tree_any_only(torch.Tensor, lambda x: x.requires_grad, args):
            raise RuntimeError(
                "Operator \"{qualname}\" received some inputs that require "
                "grad but does not have an autograd kernel registered. If "
                "you did not wish to "
                "compute gradients for this op, please call it under a "
                "torch.no_grad context manager or don't pass inputs with "
                "requires_grad=True. If you want to add autograd support: "
                "we do not yet have a way to directly register autograd kernels "
                "for FunctionalBaseOp; please instead put the call to the "
                "op inside of a torch.autograd.Function.")
        with torch._C._AutoDispatchBelowAutograd():
            return op(*args)

    torch.library.impl(qualname, "Autograd", autograd_impl, lib=lib)

    return base_op


NOT_PROVIDED = "not_provided"


class FunctionalBaseOp:
    schema: str = NOT_PROVIDED

    @staticmethod
    def impl_default(*args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def abstract(*args, **kwargs):
        raise NotImplementedError()


