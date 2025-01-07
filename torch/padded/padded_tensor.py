import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import return_and_correct_aliasing


def log(*msgs):
    msg = " ".join([str(x) for x in msgs])
    print(msg)


def slice_nd(input, start_idxs, end_idxs):
    log("Slicing tensor with shape %s to %s" % (input.shape, end_idxs))

    # Slice a tensor along multiple dimensions
    # This is a generalization of torch.slice, which only supports slicing along one dimension
    assert len(start_idxs) == len(end_idxs)

    for dim_idx, (start_idx, end_idx) in enumerate(zip(start_idxs, end_idxs)):
        if start_idx is not None and end_idx is not None:
            if end_idx != input.shape[dim_idx]:
                assert start_idx >= 0
                assert end_idx <= input.shape[dim_idx]

                if not start_idx < end_idx:
                    raise ValueError(
                        f"Invalid slice indices: {start_idx}:{end_idx} for dimension {dim_idx}"
                    )

                input = torch.ops.aten.slice(input, dim_idx, start_idx, end_idx)

    # return input
    return input.clone()


def padded_to_tensor(args, kwargs):
    if kwargs is None:
        kwargs = {}
    tensor_args, tensor_kwargs = pytree.tree_map_only(
        PaddedTensor, lambda x: x.tensor, (args, kwargs)
    )
    tensor_args = list(tensor_args)

    return tensor_args, tensor_kwargs


class SlicingOp:
    def __init__(self) -> None:
        pass

    def infer_shape(self, args, kwargs):
        raise NotImplementedError

    def modify_args(self, args, kwargs):
        def fn(padded_tensor):
            tensor = slice_nd(
                padded_tensor.tensor,
                [0] * len(padded_tensor.orig_shape),
                padded_tensor.orig_shape,
            )

            return tensor

        if kwargs is None:
            kwargs = {}
        tensor_args, tensor_kwargs = pytree.tree_map_only(
            PaddedTensor, fn, (args, kwargs)
        )
        tensor_args = list(tensor_args)

        return args, kwargs, tensor_args, tensor_kwargs

    def modify_results(self, results):
        return results


class NonSlicingOp(SlicingOp):
    def __init__(self) -> None:
        super().__init__()

    def infer_shape(self, args, kwargs):
        raise NotImplementedError

    def modify_args(self, args, kwargs):
        tensor_args, tensor_kwargs = padded_to_tensor(args, kwargs)

        return args, kwargs, tensor_args, tensor_kwargs

    def modify_results(self, results):
        return results


class OnesLikeOp(NonSlicingOp):
    def __init__(self) -> None:
        super().__init__()

    def infer_shape(self, args, kwargs):
        input_shape = args[0].orig_shape
        return [input_shape]


def infer_minus_1_shape(input_shape, output_shape):
    input_shape_prod = math.prod(input_shape)
    output_shape_prod = math.prod(output_shape) * -1

    for idx, output_dim in enumerate(output_shape):
        if output_dim == -1:
            output_shape[idx] = input_shape_prod // output_shape_prod
            break
    return output_shape


class ViewOp(SlicingOp):
    def __init__(self) -> None:
        super().__init__()

    def infer_shape(self, args, kwargs):
        def find_mapping(input_shape, output_shape):
            mapping = []
            input_index = 0

            for output_dim in output_shape:
                current_mapping = []

                while True:
                    if (
                        input_index >= len(input_shape)
                        or output_dim < input_shape[input_index]
                    ):
                        break

                    current_mapping.append(input_index)
                    output_dim //= input_shape[input_index]
                    input_index += 1
                mapping.append(current_mapping)

            return mapping

        def apply_mapping(input_shape, mapping):
            output_shape = []

            for current_mapping in mapping:
                output_dim = 1
                for index in current_mapping:
                    output_dim *= input_shape[index]

                output_shape.append(output_dim)

            return output_shape

        orig_input_shape = args[0].orig_shape
        input_shape = args[0].shape
        output_shape = list(args[1])

        # If the shapes are compatible, we can just return the orig output shape.
        if math.prod(orig_input_shape) == math.prod(output_shape):
            return [torch.Size(output_shape)]

        # Does the output shape contain -1? If so, we need to infer the value of -1
        if -1 in output_shape:
            input_shape_prod = math.prod(input_shape)
            output_shape_prod = math.prod(output_shape) * -1

            for idx, output_dim in enumerate(output_shape):
                if output_dim == -1:
                    output_shape[idx] = input_shape_prod // output_shape_prod
                    break
            return [torch.Size(output_shape)]

        # Then apply this mapping to the orig input shape, to find the orig output shape.
        # E.g. input_shape = [32, 32, 32], output_shape = [1024, 32]
        # The mapping is: [[0, 1], [2]]
        mapping = find_mapping(input_shape, output_shape)
        orig_output_shape = apply_mapping(orig_input_shape, mapping)

        return [torch.Size(orig_output_shape)]

    def modify_args(self, args, kwargs):
        tensor_args, tensor_kwargs = padded_to_tensor(args, kwargs)
        inp, shape = tensor_args

        if len(args[0].shape) > len(args[1]):
            prefix = strip_common_suffix(args[0].orig_shape, args[1])
            args[0].view_shape_stack.append(prefix)
            log("Adding to view stack %s" % (prefix))

        # If the shapes are not compatible, we need to slice the input tensor to the orig shape
        if len(args[0].shape) < len(args[1]):
            log("Applying view stack from %s" % (shape))
            log("View stack", args[0].view_shape_stack)

            if len(args[0].view_shape_stack) > 0:
                b = args[0].view_shape_stack[-1]
            else:
                # TODO: Remove this hard-coded hack.
                b = [992]

            # Find the first non-1 dim
            idx = 0
            for dim in shape:
                if dim != 1:
                    break
                idx += 1

            shape[idx : idx + len(b)] = b

            if -1 in shape:
                shape = infer_minus_1_shape(args[0].orig_shape, shape)

            log("Result", shape)
        else:
            shape = self.infer_shape(args, kwargs)[0]

        # if math.prod(inp.shape) != math.prod(shape):
        inp = slice_nd(inp, [0] * len(args[0].orig_shape), args[0].orig_shape)

        args = (args[0], shape)

        return args, kwargs, (inp, shape), tensor_kwargs


class ViewAsRealOp(SlicingOp):
    def __init__(self) -> None:
        super().__init__()

    def infer_shape(self, args, kwargs):
        input_shape = args[0].orig_shape
        return [input_shape + (2,)]


class UnsqueezeOp(NonSlicingOp):
    def __init__(self) -> None:
        super().__init__()

    def infer_shape(self, args, kwargs):
        input_shape = args[0].orig_shape
        dim = args[1]

        if dim < 0:
            dim += len(input_shape) + 1

        return [input_shape[:dim] + (1,) + input_shape[dim:]]


class PolarOp(NonSlicingOp):
    def __init__(self) -> None:
        super().__init__()

    def infer_shape(self, args, kwargs):
        input_shape = args[0].orig_shape
        return [input_shape]


class TransposeOp(SlicingOp):
    def __init__(self) -> None:
        super().__init__()

    def infer_shape(self, args, kwargs):
        input_shape = args[0].orig_shape
        dim0 = args[1]
        dim1 = args[2]

        if dim0 < 0:
            dim0 += len(input_shape)
        if dim1 < 0:
            dim1 += len(input_shape)

        # Exchange dim0 and dim1
        input_shape = list(input_shape)
        input_shape[dim0], input_shape[dim1] = input_shape[dim1], input_shape[dim0]

        return [torch.Size(input_shape)]


class ExpandOp(SlicingOp):
    def __init__(self) -> None:
        super().__init__()

    def infer_shape(self, args, kwargs):
        input_shape = args[0].orig_shape
        shape = args[1]

        return [torch.Size(shape)]

    # def modify_args(self, args, kwargs):
    #    padded_args, padded_kwargs = padded_to_tensor(args, kwargs)

    #    inp, shape = padded_args

    #    pa = slice_nd(inp, [0] * len(shape), shape)

    #    return args, kwargs, (pa, shape), padded_kwargs


class ElementwiseUnaryOp(NonSlicingOp):
    def __init__(self) -> None:
        super().__init__()

    def infer_shape(self, args, kwargs):
        input_shape = args[0].orig_shape
        return [input_shape]


class ElementwiseBinaryOp(SlicingOp):
    def __init__(self) -> None:
        super().__init__()

    def infer_shape(self, args, kwargs):
        # Broadcasting
        lhs_shape = args[0].orig_shape if type(args[0]) is PaddedTensor else [1]
        rhs_shape = args[1].orig_shape if type(args[1]) is PaddedTensor else [1]

        new_shape = []
        for idx in range(max(len(lhs_shape), len(rhs_shape))):
            lhs_dim = lhs_shape[-idx - 1] if idx < len(lhs_shape) else 1
            rhs_dim = rhs_shape[-idx - 1] if idx < len(rhs_shape) else 1
            new_shape.append(max(lhs_dim, rhs_dim))

        return [torch.Size(reversed(new_shape))]

    # def modify_args(self, args, kwargs):
    #    tensor_args, tensor_kwargs = padded_to_tensor(args, kwargs)

    #    def is_broadcastable(shape1, shape2):
    #        # Broadcastable means that at each dimension, either the shapes are equal or one of them is 1
    #        num_unequal_and_1 = 0
    #        for s1, s2 in zip(shape1, shape2):
    #            if s1 != s2 and s2 == 1:
    #                num_unequal_and_1 += 1

    #        return num_unequal_and_1 > 0

    #    def get_broadcast_dims(shape1, shape2):
    #        # Get the broadcast dims for two shapes
    #        # This are the dism where the two shapes are not equal and one of them is 1
    #        broadcast_dims = []
    #        for dim_idx, (s1, s2) in reversed(list(enumerate(zip(shape1, shape2)))):
    #            if s1 != s2 and s1 == 1:
    #                broadcast_dims.append((dim_idx, 0))
    #            if s1 != s2 and s2 == 1:
    #                broadcast_dims.append((dim_idx, 1))
    #        return broadcast_dims

    #    # Handle broadcasting. Broadcasting will fail on padded tensors,
    #    # so we need to slice the padded tensor to a broadcastable shape.
    #    if type(args[0]) is PaddedTensor and type(args[1]) is PaddedTensor:
    #        if is_broadcastable(args[0].orig_shape, args[1].orig_shape):
    #            broadcast_dims = get_broadcast_dims(
    #                args[0].orig_shape, args[1].orig_shape
    #            )
    #            # For each broadcast dim, slice the padded tensor to the correct shape.
    #            for broadcast_dim, arg_idx in broadcast_dims:
    #                other_arg_idx = abs(arg_idx - 1)
    #                tensor_args[arg_idx] = torch.ops.aten.slice(
    #                    tensor_args[arg_idx], broadcast_dim, 0, 1
    #                )

    #    return args, kwargs, tensor_args, tensor_kwargs


class MatmulOp(NonSlicingOp):
    def __init__(self) -> None:
        super().__init__()

    def infer_shape(self, args, kwargs):
        return [torch.Size([args[0].orig_shape[0], args[1].orig_shape[1]])]


class BmmOp(NonSlicingOp):
    def __init__(self) -> None:
        super().__init__()

    def infer_shape(self, args, kwargs):
        b1, n1, m1 = args[0].orig_shape
        b2, m2, p2 = args[1].orig_shape

        assert b1 == b2
        assert m1 == m2

        return [torch.Size([b1, n1, p2])]


class MeanOp(SlicingOp):
    def __init__(self) -> None:
        super().__init__()

    def infer_shape(self, args, kwargs):
        input_shape = args[0].orig_shape
        dims = args[1]
        assert len(dims) == 1
        dim = dims[0]

        if dim < 0:
            dim += len(input_shape)

        # Eliminate last dim
        return [torch.Size(list(input_shape[:dim]) + [1])]


class ScaledDotProductAttentionOp(SlicingOp):
    def __init__(self) -> None:
        super().__init__()

    def infer_shape(self, args, kwargs):
        input_shape = args[0].orig_shape

        attn_shape = input_shape[:-1]
        return [input_shape, attn_shape]

    # def modify_args(self, args, kwargs):
    #    def fn(padded_tensor):
    #        tensor = slice_nd(
    #            padded_tensor.tensor,
    #            [0] * len(padded_tensor.orig_shape),
    #            padded_tensor.orig_shape,
    #        )

    #        return tensor

    #    if kwargs is None:
    #        kwargs = {}
    #    tensor_args, tensor_kwargs = pytree.tree_map_only(
    #        PaddedTensor, fn, (args, kwargs)
    #    )
    #    tensor_args = list(tensor_args)

    #    return args, kwargs, tensor_args, tensor_kwargs


class IndexOp(NonSlicingOp):
    def __init__(self) -> None:
        super().__init__()

    def infer_shape(self, args, kwargs):
        input_shape = args[0].orig_shape
        input_shape_mod = list(input_shape)
        dims = args[1]

        for dim_idx, dim in enumerate(dims):
            if dim is None:
                continue
            elif type(dim) is torch.Tensor or type(dim) is PaddedTensor:
                input_shape_mod[dim_idx] = dim.orig_shape[0]
            else:
                raise NotImplementedError(f"Encountered unsupported type: {type(dim)}")

        return [torch.Size(input_shape_mod)]


class SelectOp(NonSlicingOp):
    def __init__(self) -> None:
        super().__init__()

    def infer_shape(self, args, kwargs):
        input_shape = args[0].orig_shape
        dim = args[1]
        index = args[2]

        if dim < 0:
            dim += len(input_shape)
        if index < 0:
            index += input_shape[dim]

        return [input_shape[:dim] + input_shape[dim + 1 :]]


class IndexPutOp(SlicingOp):
    def __init__(self) -> None:
        super().__init__()

    def infer_shape(self, args, kwargs):
        input_shape = args[0].orig_shape
        return [torch.Size(input_shape)]

    # def modify_args(self, args, kwargs):
    #    tensor_args, tensor_kwargs = padded_to_tensor(args, kwargs)

    #    # Slice out the padded indices and values, so they fit the input tensor
    #    inp, indices, values = args
    #    padded_inp, padded_indices, padded_values = tensor_args

    #    depadded_indices = [
    #        x if x is None else torch.arange(x.orig_shape[0]).int() for x in indices
    #    ]
    #    depadded_values = slice_nd(
    #        padded_values, [0] * len(values.orig_shape), values.orig_shape
    #    )

    #    return [padded_inp, depadded_indices, depadded_values], tensor_kwargs


class SplitWithSizesOp(SlicingOp):
    def __init__(self) -> None:
        super().__init__()

    def infer_shape(self, args, kwargs):
        input_shape = args[0].orig_shape
        indices_or_sections = args[1]
        dim = args[2]

        if dim < 0:
            dim += len(input_shape)

        return [
            list(input_shape[:dim])
            + [indices_or_sections[i]]
            + list(input_shape[dim + 1 :])
            for i in range(len(indices_or_sections))
        ]

    # def modify_args(self, args, kwargs):
    #    _, _, dim = args

    #    if dim < 0:
    #        dim += len(args[0].orig_shape)

    #    tensor_args, tensor_kwargs = padded_to_tensor(args, kwargs)

    #    # Slice the input tensor to the correct shape
    #    tensor_args[0] = torch.ops.aten.slice(
    #        tensor_args[0], dim, 0, sum(tensor_args[1])
    #    )

    #    return args, kwargs, tensor_args, tensor_kwargs


class CatOp(SlicingOp):
    def __init__(self) -> None:
        super().__init__()

    def infer_shape(self, args, kwargs):
        input_shape = args[0][0].orig_shape
        return [input_shape]


class StackOp(NonSlicingOp):
    def __init__(self) -> None:
        super().__init__()

    def infer_shape(self, args, kwargs):
        input = args[0]
        dim = args[1]

        if dim < 0:
            dim += len(input[0].orig_shape) + 1

        return [input[0].orig_shape[:dim] + (len(input),) + input[0].orig_shape[dim:]]


class DetachOp(NonSlicingOp):
    def __init__(self) -> None:
        super().__init__()

    def infer_shape(self, args, kwargs):
        input_shape = args[0].orig_shape
        return [input_shape]


class EmbeddingOp(NonSlicingOp):
    def __init__(self) -> None:
        super().__init__()

    def infer_shape(self, args, kwargs):
        # Embedding is a special case, where we don't do any padding
        input_shape = args[0].orig_shape
        indices = args[1]

        out_shape = list(indices.orig_shape) + list(input_shape)[1:]

        return [torch.Size(out_shape)]


class NoOp(NonSlicingOp):
    def __init__(self) -> None:
        super().__init__()

    def infer_shape(self, args, kwargs):
        input_shape = args[0].orig_shape
        return [input_shape]

    def modify_results(self, results):
        return results


class OpDatabase:
    def __init__(self):
        self.ops = {
            # Tensor creation and manipulation
            "ones_like": OnesLikeOp(),
            "view": ViewOp(),
            "_unsafe_view": ViewOp(),
            "view_as_real": ViewOp(),
            "unsqueeze": UnsqueezeOp(),
            "polar": PolarOp(),
            "transpose": TransposeOp(),
            "expand": ExpandOp(),
            "clone": ElementwiseUnaryOp(),
            # Elementwise operations
            "where": ElementwiseUnaryOp(),
            "tril": ElementwiseUnaryOp(),
            "sin": ElementwiseUnaryOp(),
            "rsqrt": ElementwiseUnaryOp(),
            "silu": ElementwiseUnaryOp(),
            # Elementwise binary operations
            "add": ElementwiseBinaryOp(),
            "sub": ElementwiseBinaryOp(),
            "mul": ElementwiseBinaryOp(),
            # Contraction / Reduction operations
            "mm": MatmulOp(),
            "bmm": BmmOp(),
            "mean": MeanOp(),
            "_scaled_dot_product_flash_attention": ScaledDotProductAttentionOp(),
            # Indexing operations
            "index": IndexOp(),
            "select": SelectOp(),
            "index_put_": IndexPutOp(),
            # Splitting / Stacking
            "split_with_sizes": SplitWithSizesOp(),
            "cat": CatOp(),
            "stack": StackOp(),
            # Other
            "detach": DetachOp(),
            "embedding": EmbeddingOp(),
            "slice": NoOp(),
            "unbind": NoOp(),
            "_to_copy": NoOp(),
        }

    def get_op(self, opname):
        if opname in self.ops:
            return self.ops[opname]
        else:
            raise NotImplementedError(f"Op '{opname}' not supported")

            # log("WARNING: Op", opname, "not supported. Using NoOp")
            # return NoOp()


OP_DATABASE = OpDatabase()


def log_function_with_shapes(func, args, tensor_args, out=None, orig_shape_out=None):
    def to_shape_str(arg):
        if isinstance(arg, torch.Tensor):
            return [i for i in arg.shape]
        else:
            return arg

    func_name_str = str(func)

    arg_shapes = []
    for arg in args:
        arg_shapes.append(str(pytree.tree_map(to_shape_str, arg)))

    arg_shapes_str = "[" + ", ".join(arg_shapes) + "]"

    out_shape_str = str(pytree.tree_map(to_shape_str, out)) if out is not None else ""

    out_str = "{0:40} P: {1:60} {2:20}".format(
        func_name_str, arg_shapes_str, out_shape_str
    )
    log(out_str)

    def to_orig_shape_str(arg):
        if isinstance(arg, PaddedTensor):
            return [i for i in arg.orig_shape]
        elif isinstance(arg, torch.Tensor):
            return "Tensor"
        else:
            return arg

    arg_shapes = []
    for arg in args:
        arg_shapes.append(str(pytree.tree_map(to_orig_shape_str, arg)))

    arg_shapes_str = "[" + ", ".join(arg_shapes) + "]"

    out_shape_str = (
        str(pytree.tree_map(to_shape_str, orig_shape_out))
        if orig_shape_out is not None
        else ""
    )

    out_str = "{0:40} U: {1:60} {2:20}".format("", arg_shapes_str, out_shape_str)
    log(out_str)


def get_strides(shape: torch.Size) -> List[int]:
    if len(shape) == 0:
        return []

    strides = [1]
    for i in range(len(shape) - 1, 0, -1):
        strides.append(strides[-1] * shape[i])
    return strides[::-1]


def get_padded_shape(shape: torch.Size, multipliers: Dict[int, int]) -> torch.Size:
    padded_shape = list(shape)
    for dim, multiplier in multipliers.items():
        if dim >= len(padded_shape):
            continue
        padded_shape[dim] = (
            (padded_shape[dim] + multiplier - 1) // multiplier * multiplier
        )
    return torch.Size(padded_shape)


def get_pad(shape: torch.Size, multipliers: Dict[int, int]) -> Tuple[int, ...]:
    pad = [0] * (len(shape) * 2)
    for dim, multiplier in multipliers.items():
        if dim >= len(shape):
            continue
        pad[2 * dim] = (shape[dim] + multiplier - 1) // multiplier * multiplier - shape[
            dim
        ]
        pad[2 * dim + 1] = 0
    return tuple(pad[::-1])


def get_multipliers(args):
    for arg in pytree.tree_leaves(args):
        if type(arg) is PaddedTensor:
            return arg.multipliers
    return {n: 1 for n in range(10)}


# def strip_common_suffix(list_1, list_2):
#    """
#    Remove common suffix from two lists.
#    Args:
#        list_1 (list): The first list.
#        list_2 (list): The second list.
#    Returns:
#        tuple: Two lists with common suffix removed.
#    """
#    list_1, list_2 = list(list_1), list(list_2)
#    while list_1 and list_2 and list_1[-1] == list_2[-1]:
#        list_1.pop()
#        list_2.pop()
#    return list_1, list_2


def strip_common_suffix(list1, list2):
    list1, list2 = list(list1), list(list2)

    def f(list1, list2):
        i, j = len(list1) - 1, len(list2) - 1

        prod = 1
        while i >= 0 and j >= 0:
            if prod == list1[i]:
                i -= 1
                prod = 1
            else:
                prod *= list2[j]
                j -= 1

        return list1[: i + 1]

    cand1 = f(list1, list2)
    cand2 = f(list2, list1)

    return cand1

    # if len(cand1) > len(cand2):
    #    return cand2
    # else:
    #    return cand1


class PaddedTensor(torch.Tensor):
    @staticmethod
    def __new__(
        cls,
        tensor: torch.Tensor,
        multipliers: Optional[Dict[int, int]],
        orig_shape: Optional[torch.Size] = None,
        view_shape_stack: Optional[list] = None,
    ):
        assert type(multipliers) is dict

        # TODO: change ori_shape as torch.Tensor
        if multipliers is None:
            multipliers = {}

        padded_shape = get_padded_shape(tensor.shape, multipliers)
        kwargs = {}
        # TODO: Improve kwargs. Support different strides, storage_offset, etc.
        kwargs["strides"] = get_strides(padded_shape)
        kwargs["storage_offset"] = 0
        kwargs["device"] = tensor.device
        kwargs["layout"] = tensor.layout
        kwargs["requires_grad"] = tensor.requires_grad
        kwargs["dtype"] = tensor.dtype
        out = torch.Tensor._make_wrapper_subclass(cls, padded_shape, **kwargs)

        log(
            "Creating padded tensor with shape",
            list(out.shape),
            "orig_shape",
            list(orig_shape) if orig_shape is not None else None,
            "multipliers",
            multipliers,
        )

        return out

    def __init__(
        self,
        tensor: torch.Tensor,
        multipliers: Optional[Dict[int, int]],
        orig_shape: Optional[torch.Size] = None,
        view_shape_stack: Optional[list] = None,
    ):
        if multipliers is None:
            multipliers = {}
        self.multipliers = multipliers
        self.orig_shape = tensor.shape if orig_shape is None else orig_shape
        self.view_shape_stack = view_shape_stack if view_shape_stack is not None else []
        if tensor.shape != self.shape:
            pad = get_pad(tensor.shape, multipliers)
            self.tensor = F.pad(input=tensor, pad=pad, mode="constant", value=0)
        else:
            self.tensor = tensor

    def __repr__(self):
        return f"PaddedTensor(shape:{self.tensor.shape}, orig_shape:{self.orig_shape})"

    def __tensor_flatten__(self):
        return ["tensor"], {
            "multipliers": self.multipliers,
            "orig_shape": self.orig_shape,
            "view_shape_stack": self.view_shape_stack,
        }

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        return PaddedTensor(
            inner_tensors["tensor"],
            meta["multipliers"],
            meta["orig_shape"],
            meta["view_shape_stack"],
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        log()
        log("Dispatching", func._opname)
        log("-" * 40)

        op = OP_DATABASE.get_op(func._opname)
        multipliers = get_multipliers(args)

        # Convert args and kwargs to padded tensors
        args_new = []
        for arg in args:
            if type(arg) is torch.Tensor or type(arg) is torch.nn.Parameter:
                args_new.append(PaddedTensor(arg, multipliers))
                log(
                    "Encountered tensor with shape",
                    arg.shape,
                    "and converted to padded tensor",
                )
            else:
                args_new.append(arg)
        args = tuple(args_new)

        # Infer shape
        orig_shape_out = op.infer_shape(args, kwargs)
        # log_function_with_shapes(func, args, args, orig_shape_out)

        # Modify args
        args, kwargs, tensor_args, tensor_kwargs = op.modify_args(args, kwargs)
        if "view" in func._opname:
            orig_shape_out = [torch.Size(args[1])]

        # Run function
        out = func(*tensor_args, **tensor_kwargs)

        log_function_with_shapes(func, args, tensor_args, out, orig_shape_out)

        # Modify results
        out = op.modify_results(out)

        # Merge arguments view_shape_stacks
        view_shape_stack = []
        for arg in args:
            if type(arg) is PaddedTensor:
                view_shape_stack += arg.view_shape_stack

        out_flat, spec = pytree.tree_flatten(out)

        out_flat_padded = []
        for idx, t in enumerate(out_flat):
            if type(t) is torch.Tensor and idx < len(orig_shape_out):
                s = orig_shape_out[idx]
                out_flat_padded.append(
                    PaddedTensor(t, multipliers, s, view_shape_stack)
                )
            else:
                out_flat_padded.append(t)

        out = pytree.tree_unflatten(out_flat_padded, spec)
        return return_and_correct_aliasing(func, args, kwargs, out)
