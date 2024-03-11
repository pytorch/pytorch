# Owner(s): ["module: dynamo"]
import torch
import functorch
import torch._dynamo.testing

from functools import partial
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.backends.common import aot_autograd
from torch._higher_order_ops.wrap import *

class ToyModelBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([
                        torch.nn.Sequential(
                            torch.nn.Linear(10, 15),
                            torch.nn.ReLU()),
                        torch.nn.Sequential(
                            torch.nn.Linear(15, 20),
                            torch.nn.ReLU()),
                        torch.nn.Sequential(
                            torch.nn.Linear(20, 15),
                            torch.nn.ReLU())
                      ])
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        out = x
        out = self.layers[0](out)
        out = self.layers[1](out)
        out = self.layers[2](out)
        out = self.softmax(out)
        return torch.reshape(out, (out.size(0),out.size(1)))

class ToyModelOuterHint(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([
                        torch.nn.Sequential(
                            torch.nn.Linear(10, 15),
                            torch.nn.ReLU()),
                        torch.nn.Sequential(
                            torch.nn.Linear(15, 20),
                            torch.nn.ReLU()),
                        torch.nn.Sequential(
                            torch.nn.Linear(20, 15),
                            torch.nn.ReLU())
                      ])
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):

        def outer_hint_function(input, hint):
            out = input
            out = self.layers[0](out)
            out = self.layers[1](out)
            out = self.layers[2](out)
            return self.softmax(out)

        out = torch.ops.higher_order.hinted_context(outer_hint_function, x, hint='{"outer_hint": "True"}')

        return torch.reshape(out, (out.size(0),out.size(1)))

class ToyModelNestedHint(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([
                        torch.nn.Sequential(
                            torch.nn.Linear(10, 15),
                            torch.nn.ReLU()),
                        torch.nn.Sequential(
                            torch.nn.Linear(15, 20),
                            torch.nn.ReLU()),
                        torch.nn.Sequential(
                            torch.nn.Linear(20, 15),
                            torch.nn.ReLU())
                      ])
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        def inner_hint_function(input, block, hint):
            return self.layers[block](input)

        def outer_hint_function(input, hint):
            out = input
            out = torch.ops.higher_order.hinted_context(inner_hint_function, out, 0, hint='{"inner_hint": "1"}')
            out = torch.ops.higher_order.hinted_context(inner_hint_function, out, 1, hint='{"inner_hint": "2"}')
            out = torch.ops.higher_order.hinted_context(inner_hint_function, out, 2, hint='{"inner_hint": "3"}')
            return self.softmax(out)

        out = torch.ops.higher_order.hinted_context(outer_hint_function, x, hint='{"outer_hint": "True"}')

        return torch.reshape(out, (out.size(0),out.size(1)))

class LinearConstant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, const1, const2):
        ctx.const1 = const1
        ctx.const2 = const2
        return tensor * const1 + const2

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.const1 + ctx.const2, None, None

class ToyModelAutogradOverrideWithoutHints(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([
                        torch.nn.Sequential(
                            torch.nn.Linear(10, 15),
                            torch.nn.ReLU()),
                        torch.nn.Sequential(
                            torch.nn.Linear(15, 20),
                            torch.nn.ReLU()),
                        torch.nn.Sequential(
                            torch.nn.Linear(20, 15),
                            torch.nn.ReLU())
                      ])
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        out = x
        out = self.layers[0](out)
        out = self.layers[1](out)
        out = self.layers[2](out)
        out = self.softmax(out)

        out = LinearConstant.apply(out, 1.05, 0.05)

        return torch.reshape(out, (out.size(0),out.size(1)))

class LinearConstantHinted(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, const1, const2):
        ctx.const1 = const1
        ctx.const2 = const2
        def forward_hinted(tensor, const1, const2, hint):
            return tensor * const1 + const2
        return torch.ops.higher_order.hinted_context(forward_hinted, tensor, const1, const2, hint='{"fwd_custom_linear": "True"}')

    @staticmethod
    def backward(ctx, grad_output):
        def backward_hinted(grad_output, const1, const2, hint):
            return grad_output * const1 + const2
        return torch.ops.higher_order.hinted_context(backward_hinted, grad_output, ctx.const1, ctx.const2, hint='{"bwd_custom_linear": "True"}'), None, None

class ToyModelAutogradOverrideWithHints(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([
                        torch.nn.Sequential(
                            torch.nn.Linear(10, 15),
                            torch.nn.ReLU()),
                        torch.nn.Sequential(
                            torch.nn.Linear(15, 20),
                            torch.nn.ReLU()),
                        torch.nn.Sequential(
                            torch.nn.Linear(20, 15),
                            torch.nn.ReLU())
                      ])
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        def inner_hint_function(input, block, hint):
            return self.layers[block](input)

        def outer_hint_function(input, hint):
            out = input
            out = torch.ops.higher_order.hinted_context(inner_hint_function, out, 0, hint='{"inner_hint": "1"}')
            out = torch.ops.higher_order.hinted_context(inner_hint_function, out, 1, hint='{"inner_hint": "2"}')
            out = torch.ops.higher_order.hinted_context(inner_hint_function, out, 2, hint='{"inner_hint": "3"}')
            return self.softmax(out)

        out = torch.ops.higher_order.hinted_context(outer_hint_function, x, hint='{"outer_hint": "True"}')

        out = LinearConstantHinted.apply(out, 1.05, 0.05)

        return torch.reshape(out, (out.size(0),out.size(1)))

class LinearConstantNestedHinted(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, const1, const2):
        ctx.const1 = const1
        ctx.const2 = const2
        def forward_mul(tensor, const1, hint):
            return tensor * const1
        def forward_add(tensor, const2, hint):
            return tensor + const2
        def forward_hinted(tensor, const1, const2, hint):
            out = torch.ops.higher_order.hinted_context(forward_mul, tensor, const1, hint='{"part": "mul"}')
            out = torch.ops.higher_order.hinted_context(forward_add, out, const2, hint='{"part": "add"}')
            return out
        return torch.ops.higher_order.hinted_context(forward_hinted, tensor, const1, const2, hint='{"fwd_custom_linear": "True"}')

    @staticmethod
    def backward(ctx, grad_output):
        def backward_mul(tensor, const1, hint):
            return tensor * const1
        def backward_add(tensor, const2, hint):
            return tensor + const2
        def backward_hinted(grad_output, const1, const2, hint):
            out = torch.ops.higher_order.hinted_context(backward_mul, grad_output, const1, hint='{"part": "mul"}')
            out = torch.ops.higher_order.hinted_context(backward_add, out, const2, hint='{"part": "add"}')
            return out
        return torch.ops.higher_order.hinted_context(backward_hinted, grad_output, ctx.const1, ctx.const2, hint='{"bwd_custom_linear": "True"}'), None, None

class ToyModelAutogradOverrideWithNestedHints(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([
                        torch.nn.Sequential(
                            torch.nn.Linear(10, 15),
                            torch.nn.ReLU()),
                        torch.nn.Sequential(
                            torch.nn.Linear(15, 20),
                            torch.nn.ReLU()),
                        torch.nn.Sequential(
                            torch.nn.Linear(20, 15),
                            torch.nn.ReLU())
                      ])
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        def inner_hint_function(input, block, hint):
            return self.layers[block](input)

        def outer_hint_function(input, hint):
            out = input
            out = torch.ops.higher_order.hinted_context(inner_hint_function, out, 0, hint='{"inner_hint": "1"}')
            out = torch.ops.higher_order.hinted_context(inner_hint_function, out, 1, hint='{"inner_hint": "2"}')
            out = torch.ops.higher_order.hinted_context(inner_hint_function, out, 2, hint='{"inner_hint": "3"}')
            return self.softmax(out)

        out = torch.ops.higher_order.hinted_context(outer_hint_function, x, hint='{"outer_hint": "True"}')

        out = LinearConstantNestedHinted.apply(out, 1.05, 0.05)

        return torch.reshape(out, (out.size(0),out.size(1)))

def _inner_compile(graph_module, example_inputs, is_fwd):
    print("#### _inner_compile BEGIN", "FWD" if is_fwd else "BWD")
    graph_module.print_readable(True)
    for node in graph_module.graph.nodes:
        if node.op != "placeholder" and node.op != "output": 
            print("\nNODE:", node)
            print("hints:", node.meta["context_hints"] if "context_hints" in node.meta else None)

    print("#### _inner_compile END")
    return functorch.compile.make_boxed_func(graph_module.forward)

def sample_backend(graph_module, example_inputs):
    return aot_autograd(
        fw_compiler=partial(_inner_compile, is_fwd=True),
        bw_compiler=partial(_inner_compile, is_fwd=False),
    )(graph_module, example_inputs)

class ContextHintsTests(TestCase):
    def test_basic(self):
        torch._dynamo.reset()

        model = ToyModelBase().to("cpu")
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

        model = torch.compile(model, backend=sample_backend)

        x = torch.rand(4, 10).to("cpu")
        y = torch.ones(4, dtype=torch.long).to("cpu")

        def iteration(x, y):
            result = model(x)
            loss = criterion(result, y)
            loss.backward()
            optimizer.step()
            return loss

        loss1 = iteration(x, y)
        loss2 = iteration(x, y)
        loss3 = iteration(x, y)
        loss4 = iteration(x, y)

        print("Loss1:", loss1)
        print("Loss2:", loss2)
        print("Loss3:", loss3)
        print("Loss4:", loss4)

        self.assertTrue(loss1 > loss2)
        self.assertTrue(loss2 > loss3)
        self.assertTrue(loss3 > loss4)
        self.assertTrue(loss1 > loss2)

    def test_outer_hint(self):
        torch._dynamo.reset()

        model = ToyModelOuterHint().to("cpu")
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

        model = torch.compile(model, backend=sample_backend)

        x = torch.rand(4, 10).to("cpu")
        y = torch.ones(4, dtype=torch.long).to("cpu")

        def iteration(x, y):
            result = model(x)
            loss = criterion(result, y)
            loss.backward()
            optimizer.step()
            return loss

        loss1 = iteration(x, y)
        loss2 = iteration(x, y)
        loss3 = iteration(x, y)
        loss4 = iteration(x, y)

        print("Loss1:", loss1)
        print("Loss2:", loss2)
        print("Loss3:", loss3)
        print("Loss4:", loss4)

        self.assertTrue(loss1 > loss2)
        self.assertTrue(loss2 > loss3)
        self.assertTrue(loss3 > loss4)
        self.assertTrue(loss1 > loss2)

    def test_nested_hint(self):
        torch._dynamo.reset()

        model = ToyModelNestedHint().to("cpu")
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

        model = torch.compile(model, backend=sample_backend)

        x = torch.rand(4, 10).to("cpu")
        y = torch.ones(4, dtype=torch.long).to("cpu")

        def iteration(x, y):
            result = model(x)
            loss = criterion(result, y)
            loss.backward()
            optimizer.step()
            return loss

        loss1 = iteration(x, y)
        loss2 = iteration(x, y)
        loss3 = iteration(x, y)
        loss4 = iteration(x, y)

        print("Loss1:", loss1)
        print("Loss2:", loss2)
        print("Loss3:", loss3)
        print("Loss4:", loss4)

        self.assertTrue(loss1 > loss2)
        self.assertTrue(loss2 > loss3)
        self.assertTrue(loss3 > loss4)
        self.assertTrue(loss1 > loss2)

    def test_nested_autograd_nohint(self):
        torch._dynamo.reset()

        model = ToyModelAutogradOverrideWithoutHints().to("cpu")
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

        model = torch.compile(model, backend=sample_backend)

        x = torch.rand(4, 10).to("cpu")
        y = torch.ones(4, dtype=torch.long).to("cpu")

        def iteration(x, y):
            result = model(x)
            loss = criterion(result, y)
            loss.backward()
            optimizer.step()
            return loss

        loss1 = iteration(x, y)
        loss2 = iteration(x, y)
        loss3 = iteration(x, y)
        loss4 = iteration(x, y)

        print("Loss1:", loss1)
        print("Loss2:", loss2)
        print("Loss3:", loss3)
        print("Loss4:", loss4)

        self.assertTrue(loss1 > loss2)
        self.assertTrue(loss2 > loss3)
        self.assertTrue(loss3 > loss4)
        self.assertTrue(loss1 > loss2)

    def test_nested_autograd_hint(self):
        torch._dynamo.reset()

        model = ToyModelAutogradOverrideWithHints().to("cpu")
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

        model = torch.compile(model, backend=sample_backend)

        x = torch.rand(4, 10).to("cpu")
        y = torch.ones(4, dtype=torch.long).to("cpu")

        def iteration(x, y):
            result = model(x)
            loss = criterion(result, y)
            loss.backward()
            optimizer.step()
            return loss

        loss1 = iteration(x, y)
        loss2 = iteration(x, y)
        loss3 = iteration(x, y)
        loss4 = iteration(x, y)

        print("Loss1:", loss1)
        print("Loss2:", loss2)
        print("Loss3:", loss3)
        print("Loss4:", loss4)

        self.assertTrue(loss1 > loss2)
        self.assertTrue(loss2 > loss3)
        self.assertTrue(loss3 > loss4)
        self.assertTrue(loss1 > loss2)

    def test_nested_autograd_nestedhint(self):
        torch._dynamo.reset()

        model = ToyModelAutogradOverrideWithNestedHints().to("cpu")
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

        model = torch.compile(model, backend=sample_backend)

        x = torch.rand(4, 10).to("cpu")
        y = torch.ones(4, dtype=torch.long).to("cpu")

        def iteration(x, y):
            result = model(x)
            loss = criterion(result, y)
            loss.backward()
            optimizer.step()
            return loss

        loss1 = iteration(x, y)
        loss2 = iteration(x, y)
        loss3 = iteration(x, y)
        loss4 = iteration(x, y)

        print("Loss1:", loss1)
        print("Loss2:", loss2)
        print("Loss3:", loss3)
        print("Loss4:", loss4)

        self.assertTrue(loss1 > loss2)
        self.assertTrue(loss2 > loss3)
        self.assertTrue(loss3 > loss4)
        self.assertTrue(loss1 > loss2)

if __name__ == "__main__":
    torch.manual_seed(0xBADC0FEE)
    torch.use_deterministic_algorithms(True)

    run_tests()
