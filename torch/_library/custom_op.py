import torch
from torch._custom_ops import infer_schema
from typing import *


class NewMeta(type):
    def __new__(cls, name, bases, attrs):
        # Custom behavior here
        if name == "FunctionalOp":
            return super().__new__(cls, name, bases, attrs)

        import torch

        namespace = mangle_module(__name__)
        lib = torch.library.Library(namespace, "FRAGMENT")
        schema = infer_schema(attrs["forward"])
        lib.define(f"{namespace}::{name}{schema}")
        forward_op = getattr(getattr(torch.ops, namespace), name).default
        attrs["_opoverload"] = forward_op

        # TODO(rzou): handle BackendSelect if no Tensor inputs
        lib.impl(name, attrs["forward"], "CompositeExplicitAutograd")

        torch.library.impl_abstract(attrs["forward_abstract"], lib=lib)

        # TODO
        # assert attrs['composite_backward']
        schema, num_grad_out = backward_schema(attrs["backward"])
        lib.define(f"{namespace}::{name}Backward{schema}")

        def call_backward(*args):
            grad_outs = args[:num_grad_out]
            saved = args[num_grad_out:]
            return attrs["backward"](grad_outs, saved)

        lib.impl(f"{name}Backward", call_backward, "CompositeExplicitAutograd")
        backward_op = getattr(getattr(torch.ops, namespace), f"{name}Backward").default
        torch.library.impl_abstract(attrs["backward_abstract"], lib=lib)
        # TODO(rzou): ban double backwards

        autograd_kernel = torch._custom_op.autograd.construct_autograd_kernel2(
            forward_op, attrs["save_for_backward"], backward_op
        )
        lib.impl(name, autograd_kernel, "Autograd")

        attrs["_lib"] = lib
        return super().__new__(cls, name, bases, attrs)


def mangle_module(module):
    no_underscores = module.replace(".", "_")
    return "_py_" + no_underscores


def backward_schema(backward):
    import inspect
    import typing

    sig = inspect.signature(backward)
    grad_output = sig.parameters["grad_output"]
    saved = sig.parameters["saved"]

    def parse_annotations(name, tupled):
        assert tupled.annotation is not inspect.Parameter.empty
        assert tupled.default is inspect.Parameter.empty
        assert tupled.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        # TODO(rzou): allow more than just Tensor
        assert (
            tupled.annotation is torch.Tensor
            or typing.get_origin(tupled.annotation) is tuple
        )
        if tupled.annotation is torch.Tensor:
            return [
                f"{torch._custom_op.impl.SUPPORTED_PARAM_TYPES[tupled.annotation]} {name}"
            ]
        else:
            args = typing.get_args(tupled.annotation)
            partials = []
            for i, arg in enumerate(args):
                assert arg in torch._custom_op.impl.SUPPORTED_PARAM_TYPES.keys()
                schema_type = torch._custom_op.impl.SUPPORTED_PARAM_TYPES[arg]
                partials.append(f"{schema_type} {name}_{i}")
            return partials

    grad_outputs = parse_annotations("grad", grad_output)
    saved = parse_annotations("saved", saved)
    inp = ", ".join(grad_outputs + saved)

    def error_fn(msg):
        raise AssertionError(x)

    ret = torch._custom_op.impl.parse_return(sig.return_annotation, error_fn)
    return f"({inp}) ->{ret}", len(grad_outputs)

    assert saved.annotation is not None
    assert saved.annotation is inspect.Paramaeter.POSITIONAL_OR_KEYWORD

    assert typing.get_origin(grad_output.annotation) is tuple
    assert typing.get_origin(saved.annotation) is tuple


class FunctionalOp(metaclass=NewMeta):
    composite_backward: bool = False

    @classmethod
    def apply(cls, *args):
        return cls._opoverload(*args)

    @staticmethod
    def forward(*args):
        raise NotImplementedError("forward")

    @staticmethod
    def forward_abstract(*args):
        raise NotImplementedError("forward_abstract")

    @staticmethod
    def save_for_backward(args, output):
        raise NotImplementedError("save_for_backward")

    @staticmethod
    # If composite_backward is False, then:
    # - grad_output is a Tuple of things understood by the Dispatcher
    # - saved is a Tuple of things understood by the Dispatcher
    def backward(grad_output, saved):
        raise NotImplementedError("backward")

    @staticmethod
    def backward_abstract(grad_output, saved):
        raise NotImplementedError("backward_abstract")


from torch import Tensor


class Mul(FunctionalOp):
    def forward(x: Tensor, y: Tensor) -> Tensor:
        return x * y

    def forward_abstract(x: Tensor) -> Tensor:
        return torch.empty_like(x)

    def save_for_backward(args, output) -> Tuple[Tensor, Tensor]:
        return args

    def backward(
        grad_output: Tuple[Tensor], saved: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        return grad_output[0] * saved[1], grad_output[0] * saved[0]

    def backward_abstract(
        grad_output: Tuple[Tensor], saved: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        return grad_output[0] * saved[1], grad_output[0] * saved[0]


from functorch import make_fx

x = torch.randn(3)


def f(x):
    return Mul.apply(x, x)


result = make_fx(f, tracing_mode="fake")(x)
print(result.code)


x = torch.randn([], requires_grad=True)
y = torch.randn([], requires_grad=True)
z = Mul.apply(x, y)
z.backward()
assert torch.allclose(x.grad, y)
assert torch.allclose(y.grad, x)


def f(x, y):
    z = Mul.apply(x, y)
    return torch.autograd.grad(z, (x, y))


gm = make_fx(f)(x, y)
print(gm.code)
