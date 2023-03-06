import torch
from torch.utils._pytree import tree_map
from typing import List, Any, Dict, Optional, Union
from collections import defaultdict
from torch.utils._python_dispatch import TorchDispatchMode
from math import prod

__all__ = ["FlopCounterMode"]

aten = torch.ops.aten

def get_shape(i):
    if isinstance(i, torch.Tensor):
        return i.shape
    return i

def mm_flop(input_shapes: List[Any], _=None) -> int:
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

def addmm_flop(input_shapes: List[Any], _=None) -> int:
    """
    Count flops for addmm
    """
    assert len(input_shapes) == 3
    _, a_shape, b_shape = input_shapes
    return mm_flop([a_shape, b_shape])

def bmm_flop(input_shapes: List[Any], _=None) -> int:
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

def baddbmm_flop(input_shapes: List[Any], _=None) -> int:
    """
    Count flops for the baddbmm operation.
    """
    # Inputs should be a list of length 3.
    # Inputs contains the shapes of three tensors.
    assert len(input_shapes) == 3, len(input_shapes)
    _, a_shape, b_shape = input_shapes
    return bmm_flop([a_shape, b_shape])


def conv_flop_count(
    x_shape: List[int],
    w_shape: List[int],
    out_shape: List[int],
    transposed: bool = False,
) -> int:
    """
    Count flops for convolution. Note only multiplication is
    counted. Computation for bias are ignored.
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

def conv_flop(input_shapes: List[Any], output_shapes: List[Any]) -> int:
    """
    Count flops for convolution.
    """
    x_shape, w_shape = input_shapes[:2]
    out_shape = output_shapes[0]
    transposed = input_shapes[6]

    return conv_flop_count(x_shape, w_shape, out_shape, transposed=transposed)

def transpose_shape(shape):
    return [shape[1], shape[0]] + list(shape[2:])

def conv_backward_flop(input_shapes: List[Any], output_shapes: List[Any]) -> int:
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

def sdpa_flop_count(query_shape, key_shape):
    """
    Count flops for self-attention.
    """
    b, h, s, d = query_shape
    b2, h2, s2, d2 = key_shape
    assert b == b2 and h == h2 and d == d2
    total_flops = 0
    # [b, h, s, d] @ [b, h, d, s2] -> [b, h, s, s2]
    total_flops += bmm_flop([(b * h, s, d), (b * h, d, s2)])
    # [b, h, s, s2] @ [b, h, s2, d] -> [b, h, s2, d]
    total_flops += bmm_flop([(b * h, s, s2), (b * h, s2, d)])
    return total_flops



def sdpa_flop(input_shapes: List[Any], _=None) -> int:
    """
    Count flops for self-attention.
    """
    # NB: We aren't accounting for causal attention here
    query_shape, key_shape, value_shape, *_ = input_shapes
    return sdpa_flop_count(query_shape, key_shape)

flop_mapping = {
    aten.mm: mm_flop,
    aten.addmm: addmm_flop,
    aten.bmm: bmm_flop,
    aten.baddbmm: baddbmm_flop,
    aten.convolution: conv_flop,
    aten._convolution: conv_flop,
    aten.convolution_backward: conv_backward_flop,
    aten._scaled_dot_product_efficient_attention: sdpa_flop,
    aten._scaled_dot_product_flash_attention: sdpa_flop,
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
    """
    ``FlopCounterMode`` is a context manager that counts the number of
    flops within its context. It does this using a ``TorchDispatchMode``.

    It also supports hierarchical output by passing a module (or list of modules) to FlopCounterMode on construction.

    Example usage

    .. code-block:: python

        mod = ...
        flop_counter = FlopCounterMode(mod)
        with flop_counter:
            mod.sum().backward()

    """
    def __init__(
            self,
            mods: Optional[Union[torch.nn.Module, List[torch.nn.Module]]] = None,
            depth: int = 2,
            display: bool = True,
            custom_mapping: Dict[Any, Any] = None):
        self.flop_counts: Dict[str, Dict[Any, int]] = defaultdict(lambda: defaultdict(int))
        self.depth = depth
        self.parents = ["Global"]
        self.display = display
        if custom_mapping is None:
            custom_mapping = {}
        if isinstance(mods, torch.nn.Module):
            mods = [mods]
        self.mods = mods
        if mods is not None:
            for mod in mods:
                prefix = type(mod).__name__
                for name, module in dict(mod.named_modules()).items():
                    if name == "":
                        name = prefix
                    else:
                        name = ".".join([prefix, name])
                    module.register_forward_pre_hook(self._enter_module(name))
                    module.register_forward_hook(self._exit_module(name))
        self.flop_mapping = {**flop_mapping, **custom_mapping}

    def _enter_module(self, name):
        def f(module, inputs):
            inputs = normalize_tuple(inputs)
            out = self._create_pre_module(name)(*inputs)
            return out

        return f

    def _exit_module(self, name):
        def f(module, inputs, outputs):
            outputs = normalize_tuple(outputs)
            return self._create_post_module(name)(*outputs)
        return f

    def _create_post_module(self, name):
        class PushState(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                assert(self.parents[-1] == name)
                self.parents.pop()
                args = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args)
                if len(args) == 1:
                    return args[0]
                return args

            @staticmethod
            def backward(ctx, *grad_outs):
                self.parents.append(name)
                return grad_outs

        return PushState.apply

    def _create_pre_module(self, name):
        class PopState(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                self.parents.append(name)
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

    def get_flop_counts(self) -> Dict[str, Dict[Any, int]]:
        """Returns the flop counts as a dictionary of dictionaries. The outer
        dictionary is keyed by module name, and the inner dictionary is keyed by
        operation name.

        Returns:
            Dict[str, Dict[Any, int]]: The flop counts as a dictionary.
        """
        return dict(self.flop_counts)

    def get_table(self, depth=None):
        if depth is None:
            depth = self.depth
        if depth is None:
            depth = 999999

        import tabulate
        tabulate.PRESERVE_WHITESPACE = True
        header = ["Module", "FLOPS"]
        values = []
        max_flops = 0

        def process_mod(mod_name, depth):
            nonlocal max_flops

            total_flops = sum(self.flop_counts[mod_name].values())
            max_flops = max(total_flops, max_flops)

            padding = " " * depth
            values = []
            values.append([padding + mod_name, convert_num_to_suffix_str(total_flops)])
            for k, v in self.flop_counts[mod_name].items():
                values.append([padding + " - " + str(k), convert_num_to_suffix_str(v)])
            return values

        for mod in self.flop_counts.keys():
            if mod == 'Global':
                continue
            mod_depth = mod.count(".") + 1
            if mod_depth > depth:
                continue

            cur_values = process_mod(mod, mod_depth - 1)
            for value in cur_values:
                values.append(value)

        # We do a bit of messing around here to only output the "Global" value
        # if there are any FLOPs in there that aren't already fully contained by
        # a module.
        if 'Global' in self.flop_counts and sum(self.flop_counts['Global'].values()) > max_flops:
            for idx, value in enumerate(values):
                values[idx][0] = " " + values[idx][0]

            values = process_mod('Global', 0) + values

        if len(values) == 0:
            values = [["Global", "0"]]

        return tabulate.tabulate(values, headers=header, colalign=("left", "right"))

    def __enter__(self):
        self.flop_counts.clear()
        super().__enter__()
        return self

    def __exit__(self, *args):
        if self.display:
            print(self.get_table(self.depth))
        super().__exit__(*args)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs else {}
        out = func(*args, **kwargs)
        func_packet = func._overloadpacket
        if func_packet in self.flop_mapping:
            flop_count_func = self.flop_mapping[func_packet]
            args_shape, out_shape = tree_map(get_shape, (args, normalize_tuple(out)))
            flop_count = flop_count_func(args_shape, out_shape)  # type: ignore[operator]
            for par in self.parents:
                self.flop_counts[par][func_packet] += flop_count

        return out
