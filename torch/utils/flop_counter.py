import torch
import torch.nn as nn
from torch.utils._pytree import tree_map, tree_flatten
from typing import List, Any
from numbers import Number
from collections import defaultdict
from torch.utils._python_dispatch import TorchDispatchMode

aten = torch.ops.aten

def get_shape(i):
    if isinstance(i, torch.Tensor):
        return i.shape
    return i

def prod(x):
    res = 1
    for i in x:
        res *= i
    return res

def mm_flop(input_shapes: List[Any], _ = None) -> Number:
    """
    Count flops for matmul.
    """
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two matrices.
    assert len(input_shapes) == 2, input_shapes
    a_shape, b_shape = input_shapes
    m, k = a_shape
    k2, n = b_shape
    assert k == k2
    # NB(chilli): Should be 2 * k - 1 technically for FLOPs.
    return m * n * 2 * k

def addmm_flop(input_shapes: List[Any], _ = None) -> Number:
    """
    Count flops for addmm
    """
    assert len(input_shapes) == 3
    _, a_shape, b_shape = input_shapes
    mm_flops = mm_flop([a_shape, b_shape])
    batch_size, _ = a_shape
    _, output_dim = b_shape
    return mm_flops + batch_size * output_dim

def bmm_flop(input_shapes: List[Any], _ = None) -> Number:
    """
    Count flops for the bmm operation.
    """
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two tensor.
    assert len(input_shapes) == 2, len(input_shapes)
    a_shape, b_shape = input_shapes
    b, m, k = a_shape
    b2, k2, n = b_shape
    assert b == b2
    assert k == k2
    # NB(chilli): Should be 2 * k - 1 technically for FLOPs.
    flop = b * m * n * 2 * k
    return flop

def baddbmm_flop(input_shapes: List[Any], _ = None) -> Number:
    """
    Count flops for the baddbmm operation.
    """
    # Inputs should be a list of length 3.
    # Inputs contains the shapes of three tensors.
    assert len(input_shapes) == 3, len(input_shapes)
    _, a_shape, b_shape = input_shapes
    bmm_flops = bmm_flop([a_shape, b_shape])
    b, m, _ = a_shape
    _, _, n = b_shape
    flop = bmm_flops + b * m * n
    return flop


def conv_flop_count(
    x_shape: List[int],
    w_shape: List[int],
    out_shape: List[int],
    transposed: bool = False,
) -> Number:
    """
    Count flops for convolution. Note only multiplication is
    counted. Computation for addition and bias is ignored.
    Flops for a transposed convolution are calculated as
    flops = (x_shape[2:] * prod(w_shape) * batch_size).
    Args:
        x_shape (list(int)): The input shape before convolution.
        w_shape (list(int)): The filter shape.
        out_shape (list(int)): The output shape after convolution.
        transposed (bool): is the convolution transposed
    Returns:
        int: the number of flops
    """
    batch_size = x_shape[0]
    conv_shape = (x_shape if transposed else out_shape)[2:]
    c_out, c_in, *dims = w_shape

    # NB(chilli): I don't think this properly accounts for padding :think:
    # NB(chilli): Should be 2 * c_in - 1 technically for FLOPs.
    flop = batch_size * prod(conv_shape) * c_out * prod(dims) * 2 * c_in
    return flop

def conv_flop(input_shapes: List[Any], output_shapes: List[Any]):
    """
    Count flops for convolution.
    """
    x_shape, w_shape = input_shapes[:2]
    out_shape = output_shapes[0]
    transposed = input_shapes[6]

    return conv_flop_count(x_shape, w_shape, out_shape, transposed=transposed)

def transpose_shape(shape):
    return [shape[1], shape[0]] + list(shape[2:])

def conv_backward_flop(input_shapes: List[Any], output_shapes: List[Any]):
    grad_out_shape, x_shape, w_shape = input_shapes[:3]
    output_mask = input_shapes[-1]
    fwd_transposed = input_shapes[7]
    flop_count = 0

    if output_mask[0]:
        grad_input_shape = get_shape(output_shapes[0])
        flop_count += conv_flop_count(grad_out_shape, w_shape, grad_input_shape, not fwd_transposed)
    if output_mask[1]:
        grad_weight_shape = get_shape(output_shapes[1])
        flop_count += conv_flop_count(transpose_shape(x_shape), grad_out_shape, grad_weight_shape, fwd_transposed)

    return flop_count


flop_mapping = {
    aten.mm: mm_flop,
    aten.addmm: addmm_flop,
    aten.bmm: bmm_flop,
    aten.baddbmm: baddbmm_flop,
    aten.convolution: conv_flop,
    aten._convolution: conv_flop,
    aten.convolution_backward: conv_backward_flop,
}

def normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x


# Thanks BingChat!
def convert_num_to_suffix_str(number):
  # Define the suffixes for different orders of magnitude
  suffixes = ["", "K", "M", "B", "T"]
  # Find the index of the appropriate suffix based on the number of digits
  index = max(0, min(len(suffixes) - 1, (len(str(number)) - 1) // 3))
  # Divide the number by 1000^index and format it to two decimal places
  value = "{:.2f}".format(number / (1000 ** index))
  # Return the value and the suffix as a string
  return value + suffixes[index]

class FlopCounterMode(TorchDispatchMode):
    def __init__(self, mod=None, depth=None, display=True):
        self.flop_counts = defaultdict(lambda: defaultdict(int))
        self.depth = depth
        self.parents = ["Global"]
        self.display = display
        if mod is not None:
            for name, module in dict(mod.named_modules()).items():
                module.register_forward_pre_hook(self.enter_module(name))
                module.register_forward_hook(self.exit_module(name))

    def enter_module(self, name):
        def f(module, inputs):
            self.parents.append(name)
            inputs = normalize_tuple(inputs)
            out = self.create_backwards_pop(name)(*inputs)
            return out

        return f

    def exit_module(self, name):
        def f(module, inputs, outputs):
            assert(self.parents[-1] == name)
            self.parents.pop()
            outputs = normalize_tuple(outputs)
            return self.create_backwards_push(name)(*outputs)
        return f

    def create_backwards_push(self, name):
        class PushState(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                args = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args)
                if len(args) == 1:
                    return args[0]
                return args

            @staticmethod
            def backward(ctx, *grad_outs):
                self.parents.append(name)
                return grad_outs

        return PushState.apply

    def create_backwards_pop(self, name):
        class PopState(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                args = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args)
                if len(args) == 1:
                    return args[0]
                return args

            @staticmethod
            def backward(ctx, *grad_outs):
                assert(self.parents[-1] == name)
                self.parents.pop()
                return grad_outs

        return PopState.apply


    def get_table(self, depth=None):
        if depth is None:
            depth = self.depth
        if depth is None:
            depth = 999999
        import tabulate
        tabulate.PRESERVE_WHITESPACE = True
        header = ["Module", "FLOPS"]
        values = []

        for mod in self.flop_counts.keys():
            total_flops = sum(self.flop_counts[mod].values())
            mod_str = mod
            mod_depth = mod.count(".")

            if mod_str == "":
                continue
            elif mod_str != "Global":
                mod_depth += 1
            if mod_depth > depth:
                continue

            padding = " " * mod_str.count(".")
            values.append([padding + mod_str, total_flops])
            for k, v in self.flop_counts[mod].items():
                values.append([padding + " - " + str(k), v])
        if len(values) == 0:
            values = [["Global", 0]]
        for value in values:
            value[1] = convert_num_to_suffix_str(value[1])
        return tabulate.tabulate(values, headers=header, colalign=("left", "right"))

    def __enter__(self):
        self.flop_counts.clear()
        super().__enter__()

    def __exit__(self, *args):
        if self.display:
            print(self.get_table(self.depth))
        super().__exit__(*args)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs else {}
        out = func(*args, **kwargs)
        func_packet = func._overloadpacket
        if func_packet in flop_mapping:
            flop_count = flop_mapping[func_packet](tree_map(get_shape, args), tree_map(get_shape, normalize_tuple(out)))
            for par in self.parents:
                self.flop_counts[par][func_packet] += flop_count

        return out
