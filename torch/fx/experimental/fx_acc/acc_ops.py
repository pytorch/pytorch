# encoding: utf-8
import operator
import warnings

import torch  # isort:skip
from typing import Sequence, List, cast

import torch.fx.experimental.fx_acc.acc_utils as acc_utils
import torch.nn as nn
from torch.fx.experimental.fx_acc.acc_normalizer import (
    register_acc_op,
    register_acc_op_mapping,
    register_custom_acc_mapper_fn,
)
from torch.fx.experimental.fx_acc.acc_op_properties import (
    AccOpProperty,
    register_acc_op_properties,
)
from torch.fx.passes.shape_prop import _extract_tensor_metadata, TensorMetadata

this_arg_is_optional = True
move_to_qparams = True
dont_move_to_qparams = False


@register_acc_op_mapping(op_and_target=("call_function", nn.functional.linear))
@register_acc_op
def linear(*, input, weight, bias):
    return nn.functional.linear(input=input, weight=weight, bias=bias)


@register_acc_op_properties(AccOpProperty.quantized)
@register_acc_op
def quantized_linear(*, input, weight, bias, acc_out_ty=None):
    assert acc_out_ty is not None
    qparams = TensorMetadata(*acc_out_ty).qparams
    return nn.quantized.functional.linear(
        input,
        weight,
        bias,
        qparams["scale"],
        qparams["zero_point"],
    )


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op_mapping(
    op_and_target=("call_method", "flatten"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("start_dim", "start_dim", this_arg_is_optional),
        ("end_dim", "end_dim", this_arg_is_optional),
    ],
)
@register_acc_op_mapping(op_and_target=("call_function", torch.flatten))
@register_acc_op
def flatten(*, input, start_dim=0, end_dim=-1):
    return torch.flatten(input=input, start_dim=start_dim, end_dim=end_dim)


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op_mapping(
    op_and_target=("call_method", "squeeze"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
    ],
)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.squeeze),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
    ],
)
@register_acc_op
def squeeze(*, input, dim=None):
    if dim is None:
        return input.squeeze()
    return input.squeeze(dim=dim)


@register_acc_op_mapping(op_and_target=("call_function", nn.functional.max_pool2d))
@register_acc_op
def max_pool2d(
    *, input, kernel_size, stride, padding, dilation, ceil_mode, return_indices
):
    return nn.functional.max_pool2d(
        input=input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=return_indices,
    )


@register_acc_op_mapping(
    op_and_target=("call_function", nn.functional.adaptive_avg_pool2d)
)
@register_acc_op
def adaptive_avg_pool2d(*, input, output_size):
    return nn.functional.adaptive_avg_pool2d(input=input, output_size=output_size)


@register_acc_op_mapping(op_and_target=("call_function", nn.functional.avg_pool2d))
@register_acc_op
def avg_pool2d(
    *,
    input,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
):
    return nn.functional.avg_pool2d(
        input=input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.sign))
@register_acc_op
def sign(*, input):
    return torch.sign(input)


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op
def size(*, input):
    return input.size()


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", getattr),
    arg_replacement_tuples=[],
)
def custom_getattr_mapper(node: torch.fx.Node, _: nn.Module) -> torch.fx.Node:
    """
    Custom function for mapping a call_function getattr to other ops. Currently only
    supports loading a getattr called on a torch.Tensor with attr name "shape", which is
    supported by mapping it to acc_ops.size().
    """
    # Have to use args here since getattr forces positional args.
    input_obj = node.args[0]
    attr_name = node.args[1]
    assert isinstance(input_obj, torch.fx.Node)
    assert (
        input_obj.meta["type"] == torch.Tensor
    ), f"Expected torch.Tensor type for {input_obj.meta['type']}"
    assert (
        attr_name == "shape"
    ), f"Only supporting shape getattr for now, not {attr_name}"
    with node.graph.inserting_before(node):
        size_node = node.graph.call_function(size, kwargs={"input": input_obj})
        size_node.meta = node.meta.copy()
        return size_node


@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "size"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
    ],
)
def tensor_size_mapper(node: torch.fx.Node, _: nn.Module) -> torch.fx.Node:
    """
    Mapping from Tensor.size() to acc_ops.size. We map size() to acc_ops.size directly
    and map size(dim) to acc_ops.size + acc_ops.getitem.
    """

    with node.graph.inserting_before(node):
        size_node = node.graph.call_function(
            size, kwargs={"input": node.kwargs["input"]}
        )

        if "dim" not in node.kwargs:
            size_node.meta = node.meta.copy()
            return size_node

        size_node.meta["type"] = torch.Size
        getitem_node = node.graph.call_function(
            getitem, kwargs={"input": size_node, "idx": node.kwargs["dim"]}
        )
        getitem_node.meta = node.meta.copy()
        return getitem_node


@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op_mapping(op_and_target=("call_function", operator.add))
@register_acc_op_mapping(op_and_target=("call_method", "add"))
@register_acc_op
def add(*, input, other):
    return input + other


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_method", "unsqueeze"))
@register_acc_op_mapping(op_and_target=("call_function", torch.unsqueeze))
@register_acc_op
def unsqueeze(*, input, dim):
    return torch.unsqueeze(input=input, dim=dim)


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_method", "tile"))
@register_acc_op_mapping(op_and_target=("call_function", torch.tile))
@register_acc_op
def tile(*, input, dims):
    return torch.tile(input=input, dims=dims)


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.stack),
    arg_replacement_tuples=[
        ("tensors", "tensors"),
        ("dim", "dim"),
    ],
)
def stack_mapper(node: torch.fx.Node, _: nn.Module) -> torch.fx.Node:
    """
    Map torch.stack to unsqueeze + cat.
    """
    with node.graph.inserting_before(node):
        inputs = node.kwargs["tensors"]
        unsqueeze_nodes = []
        assert isinstance(inputs, Sequence)
        for i, t in enumerate(inputs):
            new_node = node.graph.create_node(
                "call_function",
                unsqueeze,
                kwargs={"input": t, "dim": node.kwargs["dim"]},
                name=f"{node.name}_unsqueeze_{i}",
            )
            new_node.meta["type"] = torch.Tensor
            unsqueeze_nodes.append(new_node)
        cat_node = node.graph.create_node(
            "call_function",
            cat,
            kwargs={"tensors": unsqueeze_nodes, "dim": node.kwargs["dim"]},
        )
        cat_node.meta = node.meta.copy()
        return cat_node


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.clamp))
@register_acc_op_mapping(op_and_target=("call_method", "clamp"))
@register_acc_op
def clamp(*, input, min=None, max=None):
    return torch.clamp(input=input, min=min, max=max)


@register_acc_op_mapping(op_and_target=("call_function", torch.cat))
@register_acc_op
def cat(*, tensors, dim):
    return torch.cat(tensors=tensors, dim=dim)


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.transpose),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim0", "dim0"),
        ("dim1", "dim1"),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "transpose"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim0", "dim0"),
        ("dim1", "dim1"),
    ],
)
def transpose_mapper(node: torch.fx.Node, _: nn.Module) -> torch.fx.Node:
    # Get the dim-permutation/shuffle
    shape_as_list = node.meta["tensor_meta"].shape
    ranks = len(shape_as_list)
    shuffle = list(i for i in range(ranks))
    dim0 = cast(int, node.kwargs["dim0"])
    dim1 = cast(int, node.kwargs["dim1"])
    shuffle[dim0] = dim1
    shuffle[dim1] = dim0

    # Create the new acc_ops.permute node. Update all uses of the transpose
    # node and then delete the transpose node.
    with node.graph.inserting_after(node):
        permute_node = node.graph.call_function(
            the_function=permute,
            kwargs={
                "input": node.kwargs.get("input"),
                "permutation": shuffle,
            },
        )
        permute_node.meta = node.meta.copy()
        node.replace_all_uses_with(permute_node)

    permute_node.graph.erase_node(node)
    return permute_node


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_method", "contiguous"))
@register_acc_op
def contiguous(*, input):
    return input.contiguous()


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.nn.functional.softmax))
@register_acc_op
def softmax(*, input, dim, dtype):
    """
    _stacklevel are ignored here.
    """
    return torch.nn.functional.softmax(input=input, dim=dim, dtype=dtype)


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.addmm),
    arg_replacement_tuples=[
        ("input", "input"),
        ("mat1", "mat1"),
        ("mat2", "mat2"),
        ("beta", "beta"),
        ("alpha", "alpha"),
    ],
)
def addmm_mapper(node: torch.fx.Node, _: nn.Module) -> torch.fx.Node:
    """
    Mapping from torch.addmm to acc_ops.mm -> acc_ops.add, if alpha or beta is not 1
    then we also insert acc_ops.mul to the right place.
    """
    with node.graph.inserting_before(node):
        mm_kwargs = {"input": node.kwargs["mat1"], "other": node.kwargs["mat2"]}
        mm_node = node.graph.create_node(
            "call_function", matmul, kwargs=mm_kwargs, name=f"{node.name}_mm"
        )
        mm_node.meta = node.meta.copy()

        if node.kwargs["alpha"] != 1:
            mul_kwargs = {"input": mm_node, "other": node.kwargs["alpha"]}
            mm_node = node.graph.create_node(
                "call_function", mul, kwargs=mul_kwargs, name=f"{mm_node.name}_mul"
            )
        mm_node.meta = node.meta.copy()

        input_node = node.kwargs["input"]
        if node.kwargs["beta"] != 1:
            mul_kwargs = {"input": input_node, "other": node.kwargs["beta"]}
            new_input_node = node.graph.create_node(
                "call_function", mul, kwargs=mul_kwargs, name=f"{node.name}_input_mul"
            )
            assert isinstance(input_node, torch.fx.Node)
            new_input_node.meta = input_node.meta.copy()
            input_node = new_input_node

        add_kwargs = {"input": mm_node, "other": input_node}
        add_node = node.graph.create_node(
            "call_function", add, kwargs=add_kwargs, name=f"{node.name}_add"
        )
        add_node.meta = node.meta.copy()
        return add_node


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.t),
    arg_replacement_tuples=[
        ("input", "input"),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "t"),
    arg_replacement_tuples=[
        ("input", "input"),
    ],
)
def t_mapper(node: torch.fx.Node, _: nn.Module):
    ranks = len(node.meta["tensor_meta"].shape)
    shuffle = [1, 0] if (ranks > 1) else [0]

    with node.graph.inserting_before(node):
        new_node = node.graph.create_node(
            "call_function",
            permute,
            kwargs={"input": node.kwargs["input"], "permutation": shuffle},
        )
        new_node.meta = node.meta.copy()
        return new_node


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op_mapping(
    op_and_target=("call_method", "permute"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("*", "permutation"),
    ],
)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.permute),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dims", "permutation"),
    ],
)
@register_acc_op
def permute(*, input, permutation):
    return input.permute(*permutation)


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.square),
    arg_replacement_tuples=[
        ("input", "input"),
    ],
)
def square_mapper(node: torch.fx.Node, _: nn.Module) -> torch.fx.Node:
    input_node = node.kwargs["input"]
    with node.graph.inserting_before(node):
        new_node = node.graph.call_function(
            mul, kwargs={"input": input_node, "other": input_node}
        )
        new_node.meta = node.meta.copy()
        return new_node


@register_acc_op_mapping(
    op_and_target=("call_function", torch.bmm),
    arg_replacement_tuples=[
        ("input", "input"),
        ("mat2", "other"),
    ],
)
@register_acc_op_mapping(op_and_target=("call_function", torch.matmul))
@register_acc_op
def matmul(*, input, other):
    return torch.matmul(input=input, other=other)


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", nn.functional.dropout),
    arg_replacement_tuples=[("input", "input")],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "detach"), arg_replacement_tuples=[("input", "input")]
)
def dropout_mapper(node: torch.fx.Node, mod: nn.Module):
    """
    Remove dropout node and directly map its input to output.
    """
    return node.kwargs["input"]


try:
    from torchvision.ops import stochastic_depth
except Exception as e:
    warnings.warn(f"Unable to import torchvision related libraries.: {e}")
else:

    @register_custom_acc_mapper_fn(
        op_and_target=("call_function", stochastic_depth),
        arg_replacement_tuples=[("input", "input")],
    )
    def stochastic_depth_mapper(node: torch.fx.Node, mod: nn.Module):
        """
        Remove dropout node and directly map its input to output.
        """
        return node.kwargs["input"]


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(
    op_and_target=("call_function", nn.functional.hardtanh),
)
@register_acc_op
def hardtanh(*, input, min_val=-1.0, max_val=1.0):
    return nn.functional.hardtanh(input=input, min_val=min_val, max_val=max_val)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", nn.functional.hardsigmoid))
@register_acc_op
def hardsigmoid(*, input):
    return nn.functional.hardsigmoid(input)


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", nn.functional.silu),
    arg_replacement_tuples=[
        ("input", "input"),
    ],
)
def silu(node: torch.fx.Node, _: nn.Module) -> torch.fx.Node:
    input_node = node.kwargs["input"]
    with node.graph.inserting_before(node):
        sigmoid_node = node.graph.call_function(sigmoid, kwargs={"input": input_node})
        sigmoid_node.meta = node.meta.copy()
        new_node = node.graph.call_function(
            mul, kwargs={"input": sigmoid_node, "other": input_node}
        )
        new_node.meta = node.meta.copy()
        return new_node


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", nn.functional.hardswish),
    arg_replacement_tuples=[
        ("input", "input"),
    ],
)
def hardswish_mapper(node: torch.fx.Node, _: nn.Module) -> torch.fx.Node:
    input_node = node.kwargs["input"]
    with node.graph.inserting_before(node):
        new_sigmoid_node = node.graph.call_function(
            hardsigmoid, kwargs={"input": input_node}
        )
        new_sigmoid_node.meta = node.meta.copy()
        new_node = node.graph.call_function(
            mul, kwargs={"input": new_sigmoid_node, "other": input_node}
        )
        new_node.meta = node.meta.copy()
        return new_node


@register_acc_op_properties(AccOpProperty.quantized)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.ops.quantized.add),
    arg_replacement_tuples=[
        ("qa", "input"),
        ("qb", "other"),
        ("scale", "scale"),
        ("zero_point", "zero_point"),
    ],
    kwargs_to_move_to_acc_out_ty=[
        ("scale", "scale", move_to_qparams),
        ("zero_point", "zero_point", move_to_qparams),
    ],
)
@register_acc_op
def quantized_add(*, input, other, acc_out_ty=None):
    assert acc_out_ty is not None
    qparams = TensorMetadata(*acc_out_ty).qparams
    return torch.ops.quantized.add(
        input,
        other,
        qparams["scale"],
        qparams["zero_point"],
    )


@register_acc_op_properties(AccOpProperty.quantized)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.ops.quantized.mul),
    arg_replacement_tuples=[
        ("qa", "input"),
        ("qb", "other"),
        ("scale", "scale"),
        ("zero_point", "zero_point"),
    ],
    kwargs_to_move_to_acc_out_ty=[
        ("scale", "scale", move_to_qparams),
        ("zero_point", "zero_point", move_to_qparams),
    ],
)
@register_acc_op
def quantized_mul(*, input, other, acc_out_ty=None):
    assert acc_out_ty is not None
    qparams = TensorMetadata(*acc_out_ty).qparams
    return torch.ops.quantized.mul(
        input,
        other,
        qparams["scale"],
        qparams["zero_point"],
    )


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_properties(AccOpProperty.quantized)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.quantize_per_tensor),
    arg_replacement_tuples=[
        ("input", "input"),
        ("scale", "scale"),
        ("zero_point", "zero_point"),
        ("dtype", "dtype"),
    ],
    kwargs_to_move_to_acc_out_ty=[
        ("scale", "scale", move_to_qparams),
        ("zero_point", "zero_point", move_to_qparams),
        ("dtype", "dtype", dont_move_to_qparams),
    ],
)
@register_acc_op
def quantize_per_tensor(*, input, acc_out_ty=None):
    assert acc_out_ty is not None
    qparams = TensorMetadata(*acc_out_ty).qparams
    dtype = TensorMetadata(*acc_out_ty).dtype
    return torch.quantize_per_tensor(
        input, qparams["scale"], qparams["zero_point"], dtype
    )


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.quantize_per_channel),
    arg_replacement_tuples=[
        ("input", "input"),
        ("scales", "scales"),
        ("zero_points", "zero_points"),
        ("axis", "axis"),
        ("dtype", "dtype"),
    ],
    kwargs_to_move_to_acc_out_ty=[
        ("scales", "scale", move_to_qparams),
        ("zero_points", "zero_point", move_to_qparams),
        ("axis", "axis", move_to_qparams),
        ("dtype", "dtype", dont_move_to_qparams),
    ],
)
@register_acc_op
def quantize_per_channel(*, input, acc_out_ty=None):
    assert acc_out_ty is not None
    qparams = TensorMetadata(*acc_out_ty).qparams
    dtype = TensorMetadata(*acc_out_ty).dtype
    return torch.quantize_per_channel(
        input,
        torch.tensor(qparams["scale"]),
        torch.tensor(qparams["zero_point"]),
        qparams["axis"],
        dtype,
    )  # type: ignore[call-overload]


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_method", "dequantize"))
@register_acc_op_mapping(op_and_target=("call_function", torch.dequantize))
@register_acc_op
def dequantize(*, input):
    return torch.dequantize(input)


@register_acc_op_properties(
    AccOpProperty.pointwise, AccOpProperty.unary, AccOpProperty.quantized
)
@register_acc_op
def rescale_quantize_per_tensor(*, input, acc_out_ty=None):
    assert acc_out_ty is not None
    d = dequantize(input=input)
    return quantize_per_tensor(input=d, acc_out_ty=acc_out_ty)


@register_acc_op_properties(AccOpProperty.unary, AccOpProperty.quantized)
@register_acc_op
def rescale_quantize_per_channel(*, input, acc_out_ty=None):
    assert acc_out_ty is not None
    d = dequantize(input=input)
    return quantize_per_channel(input=d, acc_out_ty=acc_out_ty)


@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op_mapping(op_and_target=("call_function", operator.sub))
@register_acc_op
def sub(*, input, other):
    return input - other


@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op_mapping(op_and_target=("call_function", torch.mul))
@register_acc_op_mapping(op_and_target=("call_function", operator.mul))
@register_acc_op_mapping(op_and_target=("call_method", "mul"))
@register_acc_op
def mul(*, input, other):
    return input * other


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.div),
    arg_replacement_tuples=[
        ("input", "input"),
        ("other", "other"),
        ("rounding_mode", "rounding_mode", this_arg_is_optional),
    ],
)
def div_mapper(node: torch.fx.Node, mod: torch.fx.GraphModule) -> torch.fx.Node:
    with node.graph.inserting_before(node):
        div_kwargs = dict(node.kwargs)
        if "rounding_mode" not in div_kwargs or div_kwargs["rounding_mode"] is None:
            div_node = node.graph.call_function(
                div, kwargs={"input": div_kwargs["input"], "other": div_kwargs["other"]}
            )
        elif div_kwargs["rounding_mode"] == "trunc":
            div_node = node.graph.call_function(
                trunc_div,
                kwargs={"input": div_kwargs["input"], "other": div_kwargs["other"]},
            )
        elif div_kwargs["rounding_mode"] == "floor":
            div_node = node.graph.call_function(
                floor_div,
                kwargs={"input": div_kwargs["input"], "other": div_kwargs["other"]},
            )
        else:
            raise RuntimeError(
                f"Unhandled div rounding mode {div_kwargs['rounding_mode']}"
            )
        div_node.meta = node.meta.copy()
        return div_node


@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op_mapping(op_and_target=("call_function", operator.truediv))
@register_acc_op
def div(*, input, other):
    return input / other


@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op_mapping(op_and_target=("call_function", operator.floordiv))
@register_acc_op
def floor_div(*, input, other):
    # This is temp fix because currently operator.floor_div for tensors would
    # traslate into torch.floor_divide which would throw an error. After it's
    # fixed we can stick to `input // other`.
    if isinstance(input, torch.Tensor) or isinstance(other, torch.Tensor):
        return torch.div(input, other, rounding_mode="floor")
    return input // other


# torch.floor_divide rounds result toward zero, rather than -Inf.
# https://github.com/pytorch/pytorch/issues/43874
@register_acc_op_mapping(op_and_target=("call_function", torch.floor_divide))
@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op
def trunc_div(*, input, other):
    return torch.div(input, other, rounding_mode="trunc")


@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op_mapping(op_and_target=("call_function", torch.pow))
@register_acc_op
def pow(*, input, exponent):
    return torch.pow(input, exponent)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", nn.functional.relu))
@register_acc_op_mapping(
    op_and_target=("call_function", torch.relu),
    arg_replacement_tuples=[("input", "input")],
)
@register_acc_op_mapping(
    op_and_target=("call_method", "relu"),
    arg_replacement_tuples=[("input", "input")],
)
@register_acc_op
def relu(*, input, inplace=False):
    return nn.functional.relu(input=input, inplace=inplace)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.nn.functional.leaky_relu)
)
@register_acc_op
def leaky_relu(*, input, negative_slope=0.01, inplace=False):
    return nn.functional.leaky_relu(
        input=input, negative_slope=negative_slope, inplace=inplace
    )


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.nn.functional.elu))
@register_acc_op
def elu(*, input, alpha=1.0, inplace=False):
    return nn.functional.elu(input=input, alpha=alpha, inplace=inplace)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.nn.functional.selu))
@register_acc_op
def selu(*, input, inplace=False):
    return nn.functional.selu(input=input, inplace=inplace)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.nn.functional.softsign))
@register_acc_op
def softsign(*, input):
    return nn.functional.softsign(input=input)


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.log1p),
    arg_replacement_tuples=[
        ("input", "input"),
    ],
)
def torch_log1p_mapper(node: torch.fx.Node, _: torch.nn.Module) -> torch.fx.Node:
    with node.graph.inserting_before(node):
        add_kwargs = {"input": node.kwargs["input"], "other": 1.0}
        add_node = node.graph.call_function(add, kwargs=add_kwargs)
        add_node.meta = node.meta.copy()
        log_kwargs = {"input": add_node}
        log_node = node.graph.call_function(log, kwargs=log_kwargs)
        log_node.meta = node.meta.copy()
        return log_node


def reduce_op_mapper(
    node: torch.fx.Node, mod: torch.fx.GraphModule, func
) -> torch.fx.Node:
    with node.graph.inserting_before(node):
        kwargs = dict(node.kwargs)
        if "dim" in kwargs and isinstance(kwargs["dim"], int):
            kwargs["dim"] = (kwargs["dim"],)
        new_node = node.graph.call_function(func, kwargs=kwargs)
        new_node.meta = node.meta.copy()
        return new_node


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op
def sum(*, input, dim=None, keepdim=False, dtype=None):
    if dim is not None:
        return torch.sum(input, dim=dim, keepdim=keepdim, dtype=dtype)
    else:
        return input.sum(dtype=dtype)


@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "sum"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
        ("dtype", "dtype", this_arg_is_optional),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.sum),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
        ("dtype", "dtype", this_arg_is_optional),
    ],
)
def sum_mapper(node: torch.fx.Node, mod: torch.fx.GraphModule) -> torch.fx.Node:
    return reduce_op_mapper(node, mod, sum)


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op
def mean(*, input, dim=None, keepdim=False, dtype=None):
    if dim is not None:
        return torch.mean(input, dim=dim, keepdim=keepdim, dtype=dtype)
    else:
        return input.mean(dtype=dtype)


@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "mean"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
        ("dtype", "dtype", this_arg_is_optional),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.mean),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
        ("dtype", "dtype", this_arg_is_optional),
    ],
)
def mean_mapper(node, mod):
    return reduce_op_mapper(node, mod, mean)


@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "max"),
    arg_replacement_tuples=[
        ("input", "input"),
        (("dim", "other"), "dim_or_other", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.max),
    arg_replacement_tuples=[
        ("input", "input"),
        (("dim", "other"), "dim_or_other", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "min"),
    arg_replacement_tuples=[
        ("input", "input"),
        (("dim", "other"), "dim_or_other", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.min),
    arg_replacement_tuples=[
        ("input", "input"),
        (("dim", "other"), "dim_or_other", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
    ],
)
def add_maximum_minimum_mapper(
    node: torch.fx.Node, mod: torch.fx.GraphModule
) -> torch.fx.Node:
    # there are effectively three versions of torch.max / torch.min
    # full reduce: torch.max(input) -> Tensor
    # dimensional reduce: torch.max(input, dim, keepdim=False, *, out=None) -> (Tensor, LongTensor)
    # elementwise: torch.max(input, other, *, out=None) -> Tensor

    # the mapper function is remapping for both min and max situations
    # this helper function makes the choices available clearer and provides an easier way
    # to lookup the right function
    def target_map(op, target):
        if (op, target) in (("call_method", "max"), ("call_function", torch.max)):
            return dict(
                full_reduce=max_full_reduce,
                dim_reduce=max_dim_reduce,
                elementwise=maximum,
            )
        elif (op, target) in (("call_method", "min"), ("call_function", torch.min)):
            return dict(
                full_reduce=min_full_reduce,
                dim_reduce=min_dim_reduce,
                elementwise=minimum,
            )

    with node.graph.inserting_before(node):
        new_targets = target_map(node.op, node.target)
        max_kwargs = dict()
        max_kwargs["input"] = node.kwargs["input"]
        if ("dim_or_other" not in node.kwargs) or (node.kwargs["dim_or_other"] is None):
            nt = new_targets["full_reduce"]
            max_node = node.graph.call_function(nt, kwargs=max_kwargs)
        elif isinstance(node.kwargs["dim_or_other"], int):
            nt = new_targets["dim_reduce"]
            dim = node.kwargs["dim_or_other"]
            max_kwargs["dim"] = dim
            max_kwargs["keepdim"] = node.kwargs.get("keepdim", False)
            max_node = node.graph.call_function(nt, kwargs=max_kwargs)
        else:
            other = node.kwargs["dim_or_other"]
            assert isinstance(other, torch.fx.Node)
            # Lowering path for when provided "other", where we do elem-wise max
            nt = new_targets["elementwise"]
            max_kwargs["other"] = other
            max_node = node.graph.call_function(nt, kwargs=max_kwargs)
        max_node.meta = node.meta.copy()
        return max_node


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op
def max_full_reduce(*, input):
    return torch.max(input=input)


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op
def max_dim_reduce(*, input, dim=None, keepdim=False):
    return torch.max(input=input, dim=dim, keepdim=keepdim)


@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op_mapping(op_and_target=("call_function", torch.maximum))
@register_acc_op_mapping(op_and_target=("call_method", "maximum"))
@register_acc_op
def maximum(*, input, other):
    return torch.maximum(input=input, other=other)


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op
def min_full_reduce(*, input):
    return torch.min(input=input)


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op
def min_dim_reduce(*, input, dim=None, keepdim=False):
    return torch.min(input, dim=dim, keepdim=keepdim)


@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op_mapping(op_and_target=("call_function", torch.minimum))
@register_acc_op_mapping(op_and_target=("call_method", "minimum"))
@register_acc_op
def minimum(*, input, other):
    return torch.minimum(input=input, other=other)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.sigmoid))
@register_acc_op_mapping(op_and_target=("call_method", "sigmoid"))
@register_acc_op
def sigmoid(*, input):
    return torch.sigmoid(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.sinh))
@register_acc_op
def sinh(*, input):
    return torch.sinh(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.cosh))
@register_acc_op
def cosh(*, input):
    return torch.cosh(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.tanh))
@register_acc_op_mapping(op_and_target=("call_method", "tanh"))
@register_acc_op
def tanh(*, input):
    return torch.tanh(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.asin))
@register_acc_op
def asin(*, input):
    return torch.asin(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.acos))
@register_acc_op
def acos(*, input):
    return torch.acos(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.atan))
@register_acc_op
def atan(*, input):
    return torch.atan(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.exp))
@register_acc_op
def exp(*, input):
    return torch.exp(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.log))
@register_acc_op
def log(*, input):
    return torch.log(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.sqrt))
@register_acc_op
def sqrt(*, input):
    return torch.sqrt(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.reciprocal))
@register_acc_op
def reciprocal(*, input):
    return torch.reciprocal(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.abs))
@register_acc_op
def abs(*, input):
    return torch.abs(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.neg))
@register_acc_op
def neg(*, input):
    return torch.neg(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.floor))
@register_acc_op
def floor(*, input):
    return torch.floor(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.ceil))
@register_acc_op
def ceil(*, input):
    return torch.ceil(input=input)


@register_acc_op_mapping(op_and_target=("call_function", torch.nn.functional.pad))
@register_acc_op
def pad(*, input, pad, mode, value):
    return torch.nn.functional.pad(input=input, pad=pad, mode=mode, value=value)


@register_acc_op_mapping(op_and_target=("call_function", torch.conv2d))
@register_acc_op
def conv2d(*, input, weight, bias, stride, padding, dilation, groups):
    return nn.functional.conv2d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


@register_acc_op_properties(AccOpProperty.quantized)
@register_acc_op
def quantized_conv2d(
    *,
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    padding_mode,
    acc_out_ty,
):
    qparams = TensorMetadata(*acc_out_ty).qparams
    return torch.nn.quantized.functional.conv2d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        padding_mode=padding_mode,
        scale=qparams["scale"],
        zero_point=qparams["zero_point"],
    )


@register_acc_op_mapping(op_and_target=("call_function", nn.functional.batch_norm))
@register_acc_op
def batch_norm(
    *, input, running_mean, running_var, weight, bias, training, momentum, eps
):
    return nn.functional.batch_norm(
        input=input,
        running_mean=running_mean,
        running_var=running_var,
        weight=weight,
        bias=bias,
        training=training,
        momentum=momentum,
        eps=eps,
    )


@register_acc_op_mapping(op_and_target=("call_function", nn.functional.layer_norm))
@register_acc_op
def layer_norm(*, input, normalized_shape, weight, bias, eps):
    return nn.functional.layer_norm(
        input=input,
        normalized_shape=normalized_shape,
        weight=weight,
        bias=bias,
        eps=eps,
    )


def argmin_max_mapper_impl(node: torch.fx.Node, largest: bool) -> torch.fx.Node:
    """
    Map torch.argmin or torch.argmax to acc_ops.flatten (depend on dim) + acc_ops.topk
    + acc_ops.getitem + acc_ops.squeeze (depends on keepdim).
    """
    input_node = node.kwargs["input"]
    dim = node.kwargs["dim"]
    keepdim = node.kwargs["keepdim"]

    if dim is None and keepdim:
        raise RuntimeError(
            "We currently don't support argmin/argmax with dim=None and keepdim=True"
        )

    with node.graph.inserting_before(node):
        if dim is None:
            flatten_kwargs = {
                "input": node.kwargs["input"],
                "start_dim": 0,
                "end_dim": -1,
            }
            flatten_node = node.graph.call_function(flatten, kwargs=flatten_kwargs)
            flatten_node.meta["type"] = torch.Tensor
            input_node = flatten_node
            dim = -1

        topk_kwargs = {
            "input": input_node,
            "k": 1,
            "dim": dim,
            "largest": largest,
            "sorted": False,
        }
        topk_node = node.graph.call_function(topk, kwargs=topk_kwargs)
        # It's actually more like NamedTuple but tuple here should be fine.
        topk_node.meta["type"] = tuple

        getitem_kwargs = {"input": topk_node, "idx": 1}
        getitem_node = node.graph.call_function(getitem, kwargs=getitem_kwargs)
        getitem_node.meta["type"] = torch.Tensor
        output_node = getitem_node

        if not keepdim:
            squeeze_kwargs = {"input": getitem_node, "dim": dim}
            output_node = node.graph.call_function(squeeze, kwargs=squeeze_kwargs)

        output_node.meta = node.meta.copy()
        return output_node


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.argmin),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim"),
        ("keepdim", "keepdim"),
    ],
)
def torch_argmin_mapper(node: torch.fx.Node, _: torch.nn.Module) -> torch.fx.Node:
    """
    Map torch.argmin to acc_ops.flatten (depend on dim) + acc_ops.topk + acc_ops.getitem
    + acc_ops.squeeze (depends on keepdim).
    """
    return argmin_max_mapper_impl(node, largest=False)


@register_acc_op_mapping(op_and_target=("call_function", torch.linalg.norm))
@register_acc_op
def linalg_norm(*, input, ord, dim, keepdim):
    return torch.linalg.norm(input=input, ord=ord, dim=dim, keepdim=keepdim)


@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "split"),
    arg_replacement_tuples=[
        ("tensor", "input"),
        ("split_size_or_sections", "split_size_or_sections"),
        ("dim", "dim"),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "split_with_sizes"),
    arg_replacement_tuples=[
        ("tensor", "input"),
        ("split_sizes", "split_size_or_sections"),
        ("dim", "dim"),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.split),
    arg_replacement_tuples=[
        ("tensor", "input"),
        ("split_size_or_sections", "split_size_or_sections"),
        ("dim", "dim"),
    ],
)
def torch_split_mapper(node: torch.fx.Node, mod: nn.Module) -> torch.fx.Node:
    """
    If split_size_or_sections is sections, map the node to slice_tensors
    + tuple_construct. Otherwise, if split_size_or_sections is split_size,
    map the node to acc_ops.split.
    """
    split_size_or_sections = node.kwargs["split_size_or_sections"]
    with node.graph.inserting_before(node):
        if isinstance(split_size_or_sections, int):
            new_kwargs = {
                "input": node.kwargs["input"],
                "split_size": split_size_or_sections,
                "dim": node.kwargs["dim"],
            }
            new_node = node.graph.call_function(split, kwargs=new_kwargs)
            new_node.meta = node.meta.copy()
            return new_node

        assert isinstance(split_size_or_sections, Sequence)
        start = 0
        slice_nodes = []
        for i in split_size_or_sections:
            assert isinstance(i, int)
            new_kwargs = {
                "input": node.kwargs["input"],
                "dim": node.kwargs["dim"],
                "start": start,
                "stop": start + i,
                "step": 1,
            }
            new_node = node.graph.call_function(slice_tensor, kwargs=new_kwargs)
            new_node.meta["type"] = torch.Tensor
            slice_nodes.append(new_node)
            start += i

        new_node = node.graph.call_function(
            tuple_construct, kwargs={"tensors": tuple(slice_nodes)}
        )
        new_node.meta = node.meta.copy()
        return new_node


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op
def split(*, input, split_size, dim):
    return torch.split(input, split_size, dim)


@register_acc_op
def tuple_construct(*, tensors):
    return tuple(tensors)


@register_acc_op_properties(AccOpProperty.quantized)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.ops.quantized.batch_norm2d),
    arg_replacement_tuples=[
        ("input", "input"),
        ("weight", "weight"),
        ("bias", "bias"),
        ("running_mean", "running_mean"),
        ("running_var", "running_var"),
        ("eps", "eps"),
        ("scale", "scale"),
        ("zero_point", "zero_point"),
    ],
    kwargs_to_move_to_acc_out_ty=[
        ("scale", "scale", move_to_qparams),
        ("zero_point", "zero_point", move_to_qparams),
    ],
)
@register_acc_op
def quantized_batch_norm2d(
    *, input, running_mean, running_var, weight, bias, eps, acc_out_ty
):
    qparams = TensorMetadata(*acc_out_ty).qparams
    return torch.ops.quantized.batch_norm2d(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        eps,
        qparams["scale"],
        qparams["zero_point"],
    )


@register_acc_op_mapping(op_and_target=("call_function", nn.functional.embedding_bag))
@register_acc_op
def embedding_bag(
    *,
    input,
    weight,
    offsets,
    max_norm,
    norm_type,
    scale_grad_by_freq,
    mode,
    sparse,
    per_sample_weights,
    include_last_offset,
    padding_idx,
):
    return nn.functional.embedding_bag(
        input=input,
        weight=weight,
        offsets=offsets,
        max_norm=max_norm,
        norm_type=norm_type,
        scale_grad_by_freq=scale_grad_by_freq,
        mode=mode,
        sparse=sparse,
        per_sample_weights=per_sample_weights,
        include_last_offset=include_last_offset,
        padding_idx=padding_idx,
    )


@register_acc_op_mapping(
    op_and_target=(
        "call_function",
        torch.ops.quantized.embedding_bag_byte_rowwise_offsets,
    )
)
@register_acc_op
def embedding_bag_byte_rowwise_offsets(
    *,
    weight,
    indices,
    offsets,
    scale_grad_by_freq,
    mode,
    pruned_weights,
    per_sample_weights,
    compressed_indices_mapping,
    include_last_offset,
):
    return torch.ops.quantized.embedding_bag_byte_rowwise_offsets(
        weight=weight,
        indices=indices,
        offsets=offsets,
        scale_grad_by_freq=scale_grad_by_freq,
        mode=mode,
        pruned_weights=pruned_weights,
        per_sample_weights=per_sample_weights,
        compressed_indices_mapping=compressed_indices_mapping,
        include_last_offset=include_last_offset,
    )


@register_acc_op_mapping(
    op_and_target=(
        "call_function",
        torch.ops.quantized.embedding_bag_4bit_rowwise_offsets,
    )
)
@register_acc_op
def embedding_bag_4bit_rowwise_offsets(
    *,
    weight,
    indices,
    offsets,
    scale_grad_by_freq,
    mode,
    pruned_weights,
    per_sample_weights,
    compressed_indices_mapping,
    include_last_offset,
):
    return torch.ops.quantized.embedding_bag_4bit_rowwise_offsets(
        weight=weight,
        indices=indices,
        offsets=offsets,
        scale_grad_by_freq=scale_grad_by_freq,
        mode=mode,
        pruned_weights=pruned_weights,
        per_sample_weights=per_sample_weights,
        compressed_indices_mapping=compressed_indices_mapping,
        include_last_offset=include_last_offset,
    )


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.sin))
@register_acc_op
def sin(*, input):
    return torch.sin(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.cos))
@register_acc_op
def cos(*, input):
    return torch.cos(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.tan))
@register_acc_op
def tan(*, input):
    return torch.tan(input=input)


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.topk))
@register_acc_op
def topk(*, input, k, dim, largest, sorted):
    return torch.topk(input=input, k=k, dim=dim, largest=largest, sorted=sorted)


@register_acc_op_mapping(op_and_target=("call_function", operator.getitem))
@register_acc_op
def getitem(*, input, idx):
    return input[idx]


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op
def slice_tensor(*, input, dim, start, stop, step):
    slc = slice(start, stop, step)
    if dim >= 0:
        slices: List[slice] = [slice(None, None, None) for _ in range(dim)]
        slices.append(slc)
    else:
        slices = [Ellipsis, slc]  # type: ignore[list-item]
        slices.extend([slice(None, None, None) for _ in range(-dim - 1)])

    return input[tuple(slices)]


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.narrow),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim"),
        ("start", "start"),
        ("length", "length"),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "narrow"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim"),
        ("start", "start"),
        ("length", "length"),
    ],
)
def custom_narrow_mapper(node: torch.fx.Node, mod: nn.Module) -> torch.fx.Node:
    assert isinstance(node.kwargs["start"], int) and isinstance(
        node.kwargs["length"], int
    )
    kwargs = {
        "input": node.kwargs["input"],
        "dim": node.kwargs["dim"],
        "start": node.kwargs["start"],
        "stop": node.kwargs["start"] + node.kwargs["length"],
        "step": 1,
    }
    with node.graph.inserting_before(node):
        new_node = node.graph.call_function(slice_tensor, kwargs=kwargs)
    new_node.meta = node.meta.copy()
    return new_node


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.reshape),
    arg_replacement_tuples=[
        ("input", "input"),
        ("shape", "shape"),
    ],
    kwargs_to_move_to_acc_out_ty=[("shape", "shape")],
)
@register_acc_op_mapping(
    op_and_target=("call_method", "view"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("*", "shape"),
    ],
    kwargs_to_move_to_acc_out_ty=[("shape", "shape")],
)
@register_acc_op
def reshape(*, input, acc_out_ty=None):
    assert acc_out_ty is not None
    return input.reshape(TensorMetadata(*acc_out_ty).shape)


@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "reshape"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("*", "shape"),
    ],
)
def custom_tensor_reshape_mapper(node: torch.fx.Node, _: nn.Module) -> torch.fx.Node:
    """
    For Tensor.reshape node, args could be (input, 1, 2, 3) or (input, (1, 2, 3)).
    Here we do some special handling with the `shape` arg in order to map it to
    acc_ops.reshape. It also handles the case when `shape` is a list instead of
    tuple.
    """
    input_node = node.kwargs["input"]
    shape = node.kwargs["shape"]

    assert isinstance(shape, Sequence)
    if isinstance(shape[0], (tuple, list)):  # type: ignore[index]
        shape = shape[0]  # type: ignore[index]

    with node.graph.inserting_before(node):
        new_node = node.graph.call_function(
            reshape,
            kwargs={
                "input": input_node,
                "acc_out_ty": acc_utils.build_raw_tensor_meta(shape=shape),
            },
        )
        new_node.meta = node.meta.copy()
        return new_node


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op
def to_dtype(input, acc_out_ty=None):
    assert acc_out_ty is not None
    return input.to(dtype=TensorMetadata(*acc_out_ty).dtype)


@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "to"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dtype", "dtype"),
    ],
)
def custom_tensor_to_mapper(node: torch.fx.Node, _: nn.Module):
    dest_dtype = node.kwargs["dtype"]
    mem_format = node.kwargs.get("memory_format")
    device = node.kwargs.get("device")
    assert dest_dtype is not None
    assert mem_format is None or mem_format == torch.preserve_format
    assert device is None

    new_kwargs = {
        "input": node.kwargs["input"],
        "acc_out_ty": acc_utils.build_raw_tensor_meta(dtype=dest_dtype),
    }

    with node.graph.inserting_before(node):
        new_node = node.graph.create_node(
            "call_function", to_dtype, kwargs=new_kwargs, name=node.name
        )
        new_node.meta = node.meta
        return new_node


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.add),
    # Note that we may have aliases for inputs here due to issues with deterministically
    # knowing the correct target that will be resolved by pytorch.
    arg_replacement_tuples=[
        (("input", "a"), "input"),
        (("other", "b"), "other"),
        ("alpha", "alpha", this_arg_is_optional),
    ],
)
def custom_torch_add_mapper(node: torch.fx.Node, mod: nn.Module) -> torch.fx.Node:
    """
    Add custom mapping for torch.add because it has an `alpha` parameter which scales
    the `other` input, and we want to make that mul a separate node.
    """
    with node.graph.inserting_before(node):
        # If alpha is in kwargs check if we need to add a mul, and use correct kwargs.
        if "alpha" in node.kwargs:
            # Add mul node only if it has a numerical impact, i.e. alpha != 1.0.
            if node.kwargs["alpha"] != 1.0:
                other_node = node.graph.create_node(
                    "call_function",
                    mul,
                    kwargs={
                        "input": node.kwargs["other"],
                        "other": node.kwargs["alpha"],
                    },
                    name=node.name + "_mul_alpha",
                )
                other_node.meta = node.meta
            else:
                other_node = node.kwargs["other"]
            add_kwargs = {"input": node.kwargs["input"], "other": other_node}
        else:
            add_kwargs = node.kwargs

        new_node = node.graph.create_node(
            "call_function", add, kwargs=add_kwargs, name=node.name
        )
        new_node.meta = node.meta
        return new_node


@register_custom_acc_mapper_fn(
    op_and_target=("call_module", nn.quantized.Linear),
    arg_replacement_tuples=[
        ("input", "input"),
    ],
)
def packed_quantized_linear_mapper(
    node: torch.fx.Node, mod: nn.Module
) -> torch.fx.Node:
    """
    Mapping from quantized_linear module to acc_op.linear. We unpack weight and bias
    in this mapper and pass them directly to linear node.
    """
    assert isinstance(node.target, str)
    linear_module = dict(mod.named_modules())[node.target]
    prefix = node.target.replace(".", "_")
    weight_name = f"{prefix}_weight"
    bias_name = f"{prefix}_bias"

    # Store weight and bias in the main module
    mod.register_buffer(weight_name, linear_module.weight())
    if linear_module.bias() is not None:
        mod.register_buffer(bias_name, linear_module.bias())

    with node.graph.inserting_before(node):
        # Insert get_attr nodes for weight and bias
        get_weight = node.graph.get_attr(weight_name)
        get_weight.meta["tensor_meta"] = _extract_tensor_metadata(
            linear_module.weight()
        )

        get_bias = None
        if linear_module.bias() is not None:
            get_bias = node.graph.get_attr(bias_name)
            get_bias.meta["tensor_meta"] = _extract_tensor_metadata(
                linear_module.bias()
            )

        qparams = {"scale": linear_module.scale, "zero_point": linear_module.zero_point}
        # Create kwargs for acc_op.quantized_linear
        kwargs = {
            "input": node.kwargs["input"],
            "weight": get_weight,
            "bias": get_bias,
            "acc_out_ty": acc_utils.build_raw_tensor_meta(qparams=qparams),
        }

        new_node = node.graph.call_function(quantized_linear, kwargs=kwargs)
        new_node.meta = node.meta
        return new_node


@register_custom_acc_mapper_fn(
    op_and_target=("call_module", nn.quantized.Conv2d),
    arg_replacement_tuples=[
        ("input", "input"),
    ],
)
def packed_quantized_conv2d_mapper(
    node: torch.fx.Node, mod: nn.Module
) -> torch.fx.Node:
    """
    Mapping from quantzed Conv2d module to acc_op.conv. We unpack all the parameters
    in this mapper and pass them directly to conv2d node.
    """
    assert isinstance(node.target, str)
    conv_module = dict(mod.named_modules())[node.target]
    prefix = node.target.replace(".", "_")
    weight_name = f"{prefix}_weight"
    bias_name = f"{prefix}_bias"

    # Store weight and bias in the main module
    mod.register_buffer(weight_name, conv_module.weight())
    if conv_module.bias() is not None:
        mod.register_buffer(bias_name, conv_module.bias())

    with node.graph.inserting_before(node):
        # Insert get_attr nodes for weight and bias
        get_weight = node.graph.get_attr(weight_name)
        get_weight.meta["tensor_meta"] = _extract_tensor_metadata(conv_module.weight())

        get_bias = None
        if conv_module.bias() is not None:
            get_bias = node.graph.get_attr(bias_name)
            get_bias.meta["tensor_meta"] = _extract_tensor_metadata(conv_module.bias())

        qparams = {"scale": conv_module.scale, "zero_point": conv_module.zero_point}

        # Create kwargs for acc_op.conv
        kwargs = {
            "input": node.kwargs["input"],
            "weight": get_weight,
            "bias": get_bias,
            "stride": conv_module.stride,
            "padding": conv_module.padding,
            "dilation": conv_module.dilation,
            "groups": conv_module.groups,
            "padding_mode": conv_module.padding_mode,
            "acc_out_ty": acc_utils.build_raw_tensor_meta(qparams=qparams),
        }

        new_node = node.graph.call_function(quantized_conv2d, kwargs=kwargs)
        new_node.meta = node.meta
        return new_node


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.ops.quantized.add_relu),
    arg_replacement_tuples=[
        ("input", "input"),
        ("other", "other"),
        ("scale", "scale"),
        ("zero_point", "zero_point"),
    ],
)
def add_relu_unfuse_mapper(
    node: torch.fx.Node, mod: torch.fx.GraphModule
) -> torch.fx.Node:
    with node.graph.inserting_before(node):
        qparams = {
            "scale": node.kwargs["scale"],
            "zero_point": node.kwargs["zero_point"],
        }
        add_kwargs = {
            "input": node.kwargs["input"],
            "other": node.kwargs["other"],
            "acc_out_ty": acc_utils.build_raw_tensor_meta(qparams=qparams),
        }
        add_node = node.graph.call_function(quantized_add, kwargs=add_kwargs)
        add_node.meta = node.meta.copy()

        relu_node = node.graph.call_function(
            relu, kwargs={"input": add_node, "inplace": False}
        )
        relu_node.meta = node.meta
        return relu_node


@register_custom_acc_mapper_fn(
    op_and_target=("call_module", nn.intrinsic.quantized.ConvReLU2d),
    arg_replacement_tuples=[
        ("input", "input"),
    ],
)
def packed_quantized_convrelu2d_mapper(
    node: torch.fx.Node, mod: nn.Module
) -> torch.fx.Node:
    """
    Mapping from quantized ConvReLU2d module to acc_op.relu. We use packed_quantized_conv2d_mapper to unpack all the parameters
    in this mapper and pass the returned conv2d node directly to relu node.
    """

    with node.graph.inserting_before(node):
        # conv2d op
        conv2d_node = packed_quantized_conv2d_mapper(node, mod)

        # relu op
        relu_node = node.graph.call_function(
            relu, kwargs={"input": conv2d_node, "inplace": False}
        )
        relu_node.meta = node.meta
        return relu_node


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.nn.functional.gelu))
@register_acc_op_mapping(op_and_target=("call_method", "gelu"))
@register_acc_op
def gelu(*, input):
    return torch.nn.functional.gelu(input=input)


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.cumsum))
@register_acc_op_mapping(op_and_target=("call_method", "cumsum"))
@register_acc_op
def cumsum(*, input, dim, dtype=None):
    return torch.cumsum(input=input, dim=dim, dtype=dtype)


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.chunk))
@register_acc_op_mapping(op_and_target=("call_method", "chunk"))
@register_acc_op
def chunk(*, input, chunks, dim=0):
    return torch.chunk(input=input, chunks=chunks, dim=dim)
