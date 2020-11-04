import torch.fx
from torch.fx.experimental.shape_prop import ShapeProp
from typing import Any, List
from torch.fx.node import map_arg
import copy
import time

class CPUNodeCost:
    def __init__(self, op_kind : str):
        self.op_kind = op_kind
        self.ops = self.bytes_read = self.bytes_written = 0

    ops : int
    bytes_read : int
    bytes_written : int

    @property
    def bytes_total(self):
        return self.bytes_read + self.bytes_written

    def __add__(self, other : 'CPUNodeCost'):
        sum_cost = copy.copy(self)
        sum_cost.ops += other.ops
        sum_cost.bytes_read += other.bytes_read
        sum_cost.bytes_written += other.bytes_written
        return sum_cost

    def __str__(self):
        return f'{self.op_kind}[ops={self.ops}, bytes_read={self.bytes_read}, bytes_written={self.bytes_written}, bytes_total={self.bytes_total}]'


class CalculateCPUCost:
    def __init__(self, mod):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())
        self.dtype_to_element_size_cache = {}

    def calculate(self):
        total_cpu_cost = CPUNodeCost('<total>')
        for node in self.graph.nodes:
            getattr(self, node.op)(node)
            if hasattr(node, 'cpu_cost'):
                total_cpu_cost += node.cpu_cost
        self.mod.total_cpu_cost = total_cpu_cost
        return total_cpu_cost

    def placeholder(self, node : torch.fx.Node):
        # Reads from placeholders will be accounted for at their respective
        # usages. No need to account for it here.
        pass

    def output(self, node : torch.fx.Node):
        # Writes to outputs will be accounted for at their respective
        # nodes. No need to account for it here.
        pass

    def call_module(self, node : torch.fx.Node):
        callee = self.modules[node.target]
        lookup_key = torch.typename(callee).replace('.', '_')

        if not hasattr(self, lookup_key):
            raise RuntimeError(f'Module {torch.typename(callee)} currently not supported in CPU cost analysis.')

        getattr(self, lookup_key)(node, callee)

    def call_function(self, node : torch.fx.Node):
        fn_name = torch.typename(node.target).replace(
            '_VariableFunctionsClass', 'torch')
        lookup_key = fn_name.replace('.', '_')

        if not hasattr(self, lookup_key):
            raise RuntimeError(f'Function {fn_name} currently not supported in CPU cost analysis.')

        getattr(self, lookup_key)(node)

    def dtype_to_element_size(self, dtype):
        if dtype in self.dtype_to_element_size_cache:
            return self.dtype_to_element_size_cache[dtype]

        element_size = torch.tensor([], dtype=dtype).element_size()
        self.dtype_to_element_size_cache[dtype] = element_size
        return element_size

    # Functions
    def torch_relu(self, node : torch.fx.Node):
        cpu_cost = CPUNodeCost('torch.relu')

        # Account for bytes read at input
        assert len(node.args) == 1
        input_val, *_ = node.args
        assert isinstance(input_val, torch.fx.Node)
        assert hasattr(input_val, 'shape') and hasattr(input_val, 'dtype')
        cpu_cost.bytes_read += input_val.shape.numel() * self.dtype_to_element_size(input_val.dtype)

        # Account for bytes written at output
        assert hasattr(node, 'shape') and hasattr(node, 'dtype')
        cpu_cost.bytes_written += node.shape.numel() * self.dtype_to_element_size(node.dtype)

        # FLOPS is 1-to-1 with number of elements processed
        cpu_cost.ops += node.shape.numel()

        node.cpu_cost = cpu_cost

    def _operator_add(self, node : torch.fx.Node):
        cpu_cost = CPUNodeCost('__add__')

        assert len(node.args) == 2

        # Bytes read at input tensors
        for arg in node.args:
            assert isinstance(arg, torch.fx.Node)
            assert hasattr(arg, 'shape') and hasattr(arg, 'dtype')
            cpu_cost.bytes_read += arg.shape.numel() * self.dtype_to_element_size(arg.dtype)

        # Account for bytes written at output
        assert hasattr(node, 'shape') and hasattr(node, 'dtype')
        cpu_cost.bytes_written += node.shape.numel() * self.dtype_to_element_size(node.dtype)

        # Flops
        # Each output element is the result of (#args - 1) additions
        cpu_cost.ops += node.shape.numel() * (len(node.args) - 1)

        node.cpu_cost = cpu_cost

    def torch_flatten(self, node : torch.fx.Node):
        cpu_cost = CPUNodeCost('__add__')

        # Assuming this is only a viewing operation. TODO: handle contiguous if
        # we see it.

        node.cpu_cost = cpu_cost

    # Modules
    def torch_nn_modules_linear_Linear(self, node : torch.fx.Node, mod : torch.nn.Module):
        cpu_cost = CPUNodeCost(torch.typename(mod))

        assert len(node.args) == 1
        input_val, *_ = node.args
        # Account for bytes read at input
        assert isinstance(input_val, torch.fx.Node)
        assert hasattr(input_val, 'shape') and hasattr(input_val, 'dtype')
        cpu_cost.bytes_read += input_val.shape.numel() * self.dtype_to_element_size(input_val.dtype)

        # Account for bytes read at weight and bias
        weight, bias = mod.weight, mod.bias
        cpu_cost.bytes_read += weight.shape.numel() * weight.element_size()
        cpu_cost.bytes_read += bias.shape.numel() * bias.element_size()

        # Account for bytes written at output
        assert hasattr(node, 'shape') and hasattr(node, 'dtype')
        cpu_cost.bytes_written += node.shape.numel() * self.dtype_to_element_size(node.dtype)

        # FLOPS = M * K * N
        cpu_cost.ops = input_val.shape.numel() * weight.size(0)

        node.cpu_cost = cpu_cost

    def torch_nn_modules_conv_Conv2d(self, node : torch.fx.Node, mod : torch.nn.Module):
        if mod.dilation != (1, 1):
            raise RuntimeError('Dilated convolution currently not supported')
        if mod.groups != 1:
            raise RuntimeError('Grouped convolution currently not supported')

        cpu_cost = CPUNodeCost(torch.typename(mod))

        # Bytes read for input value
        assert len(node.args) == 1
        input_val, *_ = node.args
        assert isinstance(input_val, torch.fx.Node)
        assert hasattr(input_val, 'shape') and hasattr(input_val, 'dtype')
        cpu_cost.bytes_read += input_val.shape.numel() * self.dtype_to_element_size(input_val.dtype)

        # Bytes read for weights and biases
        cpu_cost.bytes_read += mod.weight.numel() * mod.weight.element_size()
        if mod.bias is not None:
            cpu_cost.bytes_read += mod.bias.numel() * mod.bias.element_size()

        # Bytes written at output
        assert hasattr(node, 'shape') and hasattr(node, 'dtype')
        cpu_cost.bytes_written += node.shape.numel() * self.dtype_to_element_size(node.dtype)

        # Flops
        N, C_out, H_out, W_out = node.shape
        _, C_in, H_k, W_k = mod.weight.shape

        # TODO: verify
        # TODO: bias
        cpu_cost.ops = N * (C_out * (C_in * (H_out * W_out) * (2 * (H_k * W_k) - 1)))

        node.cpu_cost = cpu_cost

    def torch_nn_modules_batchnorm_BatchNorm2d(self, node : torch.fx.Node, mod : torch.nn.Module):
        cpu_cost = CPUNodeCost(torch.typename(mod))

        if mod.training:
            # TODO
            pass
        else:
            # TODO
            pass

        node.cpu_cost = cpu_cost

    def torch_nn_modules_activation_ReLU(self, node : torch.fx.Node, mod : torch.nn.Module):
        self.torch_relu(node)

    def torch_nn_modules_pooling_MaxPool2d(self, node : torch.fx.Node, mod : torch.nn.Module):
        cpu_cost = CPUNodeCost(torch.typename(mod))

        # Bytes read for input value
        assert len(node.args) == 1
        input_val, *_ = node.args
        assert isinstance(input_val, torch.fx.Node)
        assert hasattr(input_val, 'shape') and hasattr(input_val, 'dtype')
        cpu_cost.bytes_read += input_val.shape.numel() * self.dtype_to_element_size(input_val.dtype)

        # Bytes written at output
        assert hasattr(node, 'shape') and hasattr(node, 'dtype')
        cpu_cost.bytes_written += node.shape.numel() * self.dtype_to_element_size(node.dtype)

        # FLOPS
        N, C_out, H_out, W_out = node.shape
        kernel_size = mod.kernel_size

        cpu_cost.ops = N * (H_out * W_out * (kernel_size * kernel_size))

        node.cpu_cost = cpu_cost

    def torch_nn_modules_pooling_AdaptiveAvgPool2d(self, node : torch.fx.Node, mod : torch.nn.Module):
        cpu_cost = CPUNodeCost(torch.typename(mod))

        # Bytes read for input value
        assert len(node.args) == 1
        input_val, *_ = node.args
        assert isinstance(input_val, torch.fx.Node)
        assert hasattr(input_val, 'shape') and hasattr(input_val, 'dtype')
        cpu_cost.bytes_read += input_val.shape.numel() * self.dtype_to_element_size(input_val.dtype)

        # Bytes written at output
        assert hasattr(node, 'shape') and hasattr(node, 'dtype')
        cpu_cost.bytes_written += node.shape.numel() * self.dtype_to_element_size(node.dtype)

        # FLOPS
        # TODO

        node.cpu_cost = cpu_cost

class ProfiledRunInfo:
    analytical_cost : CPUNodeCost
    runtimes_sec : List[float]

    def __init__(self, cost):
        self.analytical_cost = cost
        self.runtimes_sec = []

    @property
    def average_runtime_sec(self) -> float:
        return sum(self.runtimes_sec) / len(self.runtimes_sec)

    @property
    def gops(self) -> float:
        return self.analytical_cost.ops / self.average_runtime_sec / 1e9

    @property
    def gbps(self) -> float:
        return self.analytical_cost.bytes_total / self.average_runtime_sec / 2**30

    def __str__(self) -> str:
        return f'{self.analytical_cost.op_kind}[Runtime avg (s) = {self.average_runtime_sec}, avg GOps/s = {self.gops}, avg GB/s = {self.gbps}]'


class ProfilingModule:
    def __init__(self, gm : torch.fx.GraphModule):
        super().__init__()
        self.gm = gm
        self.modules = dict(self.gm.named_modules())

    def __call__(self, *args):
        args_iter = iter(args)
        env : Dict[str, Node] = {}

        def load_arg(a):
            return torch.fx.node.map_arg(a, lambda n: env[n.name])

        def fetch_attr(target : str):
            target_atoms = target.split('.')
            attr_itr = self.gm
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
            return attr_itr

        network_start_time = time.time()

        for node in self.gm.graph.nodes:
            op_start_time = time.time()
            if node.op == 'placeholder':
                result = next(args_iter)
            elif node.op == 'get_attr':
                result = fetch_attr(node.target)
            elif node.op == 'call_function':
                result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
            elif node.op == 'call_method':
                self_obj, *args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)
            elif node.op == 'call_module':
                result = self.modules[node.target](*load_arg(node.args), **load_arg(node.kwargs))
            elif node.op == 'output':
                network_end_time = time.time()
                network_elapsed_sec = network_end_time - network_start_time
                if hasattr(self.gm, 'total_cpu_cost'):
                    if not hasattr(self, 'profiled_info'):
                        self.profiled_info = ProfiledRunInfo(self.gm.total_cpu_cost)
                    self.profiled_info.runtimes_sec.append(network_elapsed_sec)
                return load_arg(node.args[0])

            op_end_time = time.time()
            elapsed_sec = op_end_time - op_start_time
            if hasattr(node, 'cpu_cost'):
                cpu_cost = node.cpu_cost
                if not hasattr(node, 'profiled_info'):
                    node.profiled_info = ProfiledRunInfo(cpu_cost)
                node.profiled_info.runtimes_sec.append(elapsed_sec)

            env[node.name] = result

        return None


def estimate_cpu_cost(m : torch.nn.Module, args : List[Any], tracer_class=torch.fx.Tracer) -> ProfilingModule:
    traced_module = torch.fx.GraphModule(m, tracer_class().trace(m))
    ShapeProp(traced_module).propagate(*args)
    cost = CalculateCPUCost(traced_module).calculate()

    return cost, ProfilingModule(traced_module)
