# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
# This file is a model zoo for testing torch.distributed.pipelining.
import torch
from torch.autograd import Function
from torch.distributed.pipelining import pipe_split, SplitPoint


class ExampleCode(torch.nn.Module):
    def __init__(self, d_hid, splits=2):
        if not (splits <= 8):
            raise AssertionError(f"Expected splits <= 8, got {splits}")
        super().__init__()
        self.splits = splits
        self.mm_param0 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param1 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.cval = torch.nn.Buffer(torch.randn((d_hid,), requires_grad=False))
        self.lin0 = torch.nn.Linear(d_hid, d_hid)
        self.lin1 = torch.nn.Linear(d_hid, d_hid)
        self.lin2 = torch.nn.Linear(d_hid, d_hid)
        self.lin3 = torch.nn.Linear(d_hid, d_hid)
        self.lin4 = torch.nn.Linear(d_hid, d_hid)
        self.lin5 = torch.nn.Linear(d_hid, d_hid)
        self.lin6 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x):
        x = torch.mm(x, self.mm_param0)
        x = torch.relu(x)
        # try passing a value that doesn't require_grad across skip boundaries
        a_constant = self.cval.clone()
        x = self.lin0(x)
        pipe_split()
        x = torch.relu(x) + a_constant
        x = torch.mm(x, self.mm_param1)
        if self.splits > 2:
            pipe_split()
            x = self.lin1(x)
            x = torch.relu(x)
        if self.splits > 3:
            pipe_split()
            x = self.lin2(x)
            x = torch.relu(x)
        if self.splits > 4:
            pipe_split()
            x = self.lin3(x)
            x = torch.relu(x)
        if self.splits > 5:
            pipe_split()
            x = self.lin4(x)
            x = torch.relu(x)
        if self.splits > 6:
            pipe_split()
            x = self.lin5(x)
            x = torch.relu(x)
        if self.splits > 7:
            pipe_split()
            x = self.lin6(x)
            x = torch.relu(x)
        return x


class ModelWithKwargs(torch.nn.Module):
    DEFAULT_DHID = 512
    DEFAULT_BATCH_SIZE = 256

    def __init__(self, d_hid: int = DEFAULT_DHID, splits=2):
        if not (splits <= 8):
            raise AssertionError(f"Expected splits <= 8, got {splits}")
        super().__init__()
        self.splits = splits
        self.mm_param0 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param1 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.lin0 = torch.nn.Linear(d_hid, d_hid)
        self.lin1 = torch.nn.Linear(d_hid, d_hid)
        self.lin2 = torch.nn.Linear(d_hid, d_hid)
        self.lin3 = torch.nn.Linear(d_hid, d_hid)
        self.lin4 = torch.nn.Linear(d_hid, d_hid)
        self.lin5 = torch.nn.Linear(d_hid, d_hid)
        self.lin6 = torch.nn.Linear(d_hid, d_hid)
        self.lin7 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x, y=torch.zeros(DEFAULT_BATCH_SIZE, DEFAULT_DHID)):
        x = torch.mm(x, self.mm_param0)
        x = x + y
        x = self.lin0(x)
        x = torch.relu(x)
        pipe_split()
        x = torch.mm(x, self.mm_param1)
        x = self.lin1(x)
        x = torch.relu(x)
        if self.splits > 2:
            pipe_split()
            x = self.lin2(x)
            x = torch.relu(x)
        if self.splits > 3:
            pipe_split()
            x = self.lin3(x)
            x = torch.relu(x)
        if self.splits > 4:
            pipe_split()
            x = self.lin4(x)
            x = torch.relu(x)
        if self.splits > 5:
            pipe_split()
            x = self.lin5(x)
            x = torch.relu(x)
        if self.splits > 6:
            pipe_split()
            x = self.lin6(x)
            x = torch.relu(x)
        if self.splits > 7:
            pipe_split()
            x = self.lin7(x)
            x = torch.relu(x)
        return x


class ModelWithParamAlias(torch.nn.Module):
    default_dhid = 512
    default_batch_size = 256

    def __init__(self, d_hid: int = default_dhid):
        super().__init__()
        self.mm_param1 = self.mm_param0 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.lin1 = self.lin0 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x, y):
        x = torch.mm(x, self.mm_param0)
        x = x + y
        x = self.lin0(x)
        x = torch.relu(x)
        pipe_split()
        x = torch.mm(x, self.mm_param1)
        x = self.lin1(x)
        x = torch.relu(x)
        return x


# MLP Layer
class MLPModule(torch.nn.Module):
    def __init__(self, d_hid: int):
        super().__init__()
        self.net1 = torch.nn.Linear(d_hid, d_hid)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x):
        x = self.net1(x)
        x = self.relu(x)
        x = self.net2(x)
        return x


class MLPKWargModule(torch.nn.Module):
    def __init__(self, d_hid: int, layer_num):
        super().__init__()
        self.net1 = torch.nn.Linear(d_hid, d_hid)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(d_hid, d_hid)
        self.layer_num = layer_num

    def forward(self, x, unused_kwarg: torch.Tensor = torch.zeros(1)):
        x = self.net1(x)
        x = self.relu(x)
        x = self.net2(x)
        # Test when only 1 module has extra outputs
        # TODO: handle this case later
        # if self.layer_num == 0:
        #     return x, unused_kwarg
        # else:
        #     return x
        return x


# Multi-MLP model
class MultiMLP(torch.nn.Module):
    def __init__(self, d_hid: int, n_layers: int = 2):
        super().__init__()
        self.layers = torch.nn.ModuleList([MLPModule(d_hid) for _ in range(n_layers)])
        # For testing purpose only, this should be defined by user
        self.split_spec = {
            f"layers.{i}": SplitPoint.BEGINNING for i in range(1, n_layers)
        }

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TwoInputOutputOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor):
        return input, weight

    @staticmethod
    def backward(ctx, grad_input, grad_weight):
        return grad_input, grad_weight


# Model with multi-output intermediates
class MultiInterMediateModel(torch.nn.Module):
    def __init__(self, weight_shape: list[int]):
        super().__init__()
        self.shape = weight_shape
        self.w = torch.nn.Parameter(torch.randn(*weight_shape))

    def forward(self, x):
        a, b = torch.split(x, self.shape, dim=1)
        a, w = TwoInputOutputOp.apply(a, self.w)
        a = torch.matmul(a, w)
        return a * b


# Multi-MLP with kwargs model
class MultiMLPKwargs(torch.nn.Module):
    def __init__(self, d_hid: int, n_layers: int = 2):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [MLPKWargModule(d_hid, i) for i in range(n_layers)]
        )
        # For testing purpose only, this should be defined by user
        self.split_spec = {
            f"layers.{i}": SplitPoint.BEGINNING for i in range(1, n_layers)
        }

    def forward(self, x, unused_kwarg: torch.Tensor = torch.zeros(1)):
        for layer in self.layers:
            # TODO: handle this case later
            # if layer.layer_num == 0:
            #     x, _ = layer(x, unused_kwarg)
            # else:
            #     x = layer(x)
            x = layer(x)
        return x


class CustomLinearDx(Function):
    @staticmethod
    def forward(ctx, input_val, weight, bias, module, layer_idx):
        ctx.save_for_backward(input_val, weight, bias)
        ctx.module = module
        ctx.layer_idx = layer_idx
        return input_val.mm(weight.t()) + bias

    @staticmethod
    def backward(ctx, grad_output):
        input_val, weight, _ = ctx.saved_tensors
        grad_input = grad_output.mm(weight)
        ctx.module.cached_context[ctx.layer_idx].append(grad_output.clone())
        ctx.module.cached_context[str(ctx.layer_idx) + "_input"].append(
            input_val.clone()
        )
        return grad_input, None, None, None, None


class CustomLinearDxDw(Function):
    @staticmethod
    def forward(ctx, input_val, weight, bias):
        ctx.save_for_backward(input_val, weight, bias)
        return input_val.mm(weight.t()) + bias

    @staticmethod
    def backward(ctx, grad_output):
        input_val, weight, _ = ctx.saved_tensors
        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(input_val)
        grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, grad_bias


class MLPModuleWithDw(torch.nn.Module):
    def __init__(self, d_hid: int):
        super().__init__()
        self.fc1_weight = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.fc1_bias = torch.nn.Parameter(torch.randn(d_hid))
        self.fc2_weight = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.fc2_bias = torch.nn.Parameter(torch.randn(d_hid))

        torch.nn.init.uniform_(self.fc1_weight, -0.001, 0.001)
        torch.nn.init.uniform_(self.fc2_weight, -0.001, 0.001)
        torch.nn.init.uniform_(self.fc1_bias, -0.001, 0.001)
        torch.nn.init.uniform_(self.fc2_bias, -0.001, 0.001)

        self.cached_context = {}
        self.cached_context["fc1"] = []
        self.cached_context["fc2"] = []
        self.cached_context["fc1_input"] = []
        self.cached_context["fc2_input"] = []

        self.use_custom_logic = False

    def forward(self, x):
        if not self.use_custom_logic:
            self.hidden = CustomLinearDxDw.apply(x, self.fc1_weight, self.fc1_bias)
            self.hidden = torch.nn.functional.relu(self.hidden)
            output = CustomLinearDxDw.apply(self.hidden, self.fc2_weight, self.fc2_bias)
            return output

        self.hidden = CustomLinearDx.apply(
            x, self.fc1_weight, self.fc1_bias, self, "fc1"
        )
        self.hidden = torch.nn.functional.relu(self.hidden)
        output = CustomLinearDx.apply(
            self.hidden, self.fc2_weight, self.fc2_bias, self, "fc2"
        )
        return output

    def compute_dW(self):
        grad_output_fc1 = self.cached_context["fc1"].pop(0)
        grad_output_fc2 = self.cached_context["fc2"].pop(0)
        cached_input_fc1 = self.cached_context["fc1_input"].pop(0)
        cached_input_fc2 = self.cached_context["fc2_input"].pop(0)

        dW2 = grad_output_fc2.t().mm(cached_input_fc2)
        db2 = grad_output_fc2.sum(0)

        dW1 = grad_output_fc1.t().mm(cached_input_fc1)
        db1 = grad_output_fc1.sum(0)

        if self.fc1_weight.grad is not None:
            self.fc1_weight.grad += dW1
            self.fc1_bias.grad += db1
            self.fc2_weight.grad += dW2
            self.fc2_bias.grad += db2
        else:
            self.fc1_weight.grad = dW1
            self.fc1_bias.grad = db1
            self.fc2_weight.grad = dW2
            self.fc2_bias.grad = db2

    def toggle(self):
        self.use_custom_logic = not self.use_custom_logic


# Multi-MLP model With Dw
class MultiMLPWithDw(torch.nn.Module):
    def __init__(self, d_hid: int, n_layers: int = 2):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [MLPModuleWithDw(d_hid) for _ in range(n_layers)]
        )
        # For testing purpose only, this should be defined by user
        self.split_spec = {
            f"layers.{i}": SplitPoint.BEGINNING for i in range(1, n_layers)
        }
        self.use_custom_logic = False

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def toggle(self):
        self.use_custom_logic = not self.use_custom_logic
        for layer in self.layers:
            layer.toggle()

    def compute_dW(self):
        if not self.use_custom_logic:
            raise RuntimeError("Need to call toggle() to enable custom backward and dW")

        for i in reversed(range(len(self.layers))):
            self.layers[i].compute_dW()
