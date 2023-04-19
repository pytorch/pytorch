# Owner(s): ["module: dynamo"]
import contextlib
from copy import deepcopy

import torch

import torch._dynamo
import torch._dynamo.backends.ipex
import torch._dynamo.test_case
from torch._dynamo.testing import same


import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import PythonKeyTracer, ProxyTorchDispatchMode
from torch.fx import Graph, GraphModule

# Limitations:
#   - initialization cannot refer to external tensors
#   - parameters are these weird ProxyTensors, should have a custom class for
#     these placeholders
#   - DCE is likely not sound, needs to be implemented more carefully by
#     understanding aliasing relationships
#   - only top level module is rematerialized
#   - we lose parameter-ness and requires_grad-ness
#   - no version counter safety to guard against input mutation


def get_deferred_modes():
    fx_tracer = PythonKeyTracer()
    fx_tracer.graph = Graph(fx_tracer)
    fx_tracer.root = torch.nn.Module()
    fx_tracer.tensor_attrs = {}
    fake_tensor_mode = FakeTensorMode(allow_fallback_kernels=True)
    proxy_mode = ProxyTorchDispatchMode(fx_tracer, tracing_mode="real")
    return fake_tensor_mode, proxy_mode, fx_tracer

@contextlib.contextmanager
def deferred_init(f, *args, **kwargs):
    fake_tensor_mode, proxy_mode, fx_tracer = get_deferred_modes()
    with fake_tensor_mode, proxy_mode:
        r = f(*args, **kwargs)
        r._deferred = fx_tracer
        yield r

def materialize_module(m):
    # TODO: handle children

    outputs = []

    def mark_for_materialize(tensors):
        for k, t in tensors.items():
            if t is None:
                continue
            outputs.append(t.proxy.node)

    mark_for_materialize(m._parameters)
    mark_for_materialize(m._buffers)

    m._deferred.graph.output(outputs)
    m._deferred.graph.eliminate_dead_code()  # hmmm
    recomp = GraphModule(m._deferred.root, m._deferred.graph)
    results = recomp()
    results_iter = iter(results)

    def replace_results(tensors):
        for k, t in tensors.items():
            if t is None:
                continue
            tensors[k] = next(results_iter)

    replace_results(m._parameters)
    replace_results(m._buffers)

    del m._deferred


class Seq(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class TestCompileTrainStep(torch._dynamo.test_case.TestCase):
    """
    The Basic Idea
    1) dynamo stashes optimizer on graphmodule in special key
    2) optimizer.step() just sticks a call in graph without trying to arg-proxy anything
    3) inside train_step_compiler, we reparameterize the optimizer

    WIP/Issues
    - handle an optimizer with actual states
    - dynamo asserts empty backward tape before trace
    - train_step backend asserts full_graph mode was used
    - handle more than one optimizer (e.g. for different submodules)
    """

    def test_no_optimizer(self):
        def train_step(model, inputs):
            out = model(*inputs)
            loss = out.sum()
            loss.backward()
            return loss

        model = Seq()
        model.apply(init_weights)
        inputs = [torch.randn((128, 10))]

        correct_loss = train_step(model, inputs)

        opt_train_step = torch.compile(train_step, backend="train_step_eager")
        opt_loss = opt_train_step(model, inputs)

        self.assertTrue(same(correct_loss, opt_loss))

    def test_sgd_optimizer(self):
        model = Seq()
        model.apply(init_weights)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        def train_step(model, optimizer, inputs):
            out = model(*inputs)
            loss = out.sum()

            # dynamo special case lets this pass through as a call in the FX graph
            # it gets traced out in the train_step backend, before being functionalized
            loss.backward()

            # dynamo tracks the optimizer and smuggles it on the graphmodule as an attr
            # train_step backend can reparametrize it with the fake parameter tensors it already
            # uses for module tracing
            optimizer.step()

            model.zero_grad()
            return loss

        # copy the model/optimizer up front so we don't have to reset them between eager/compile runs
        opt_model = deepcopy(model)
        opt_optimizer = deepcopy(optimizer)
        inputs = [torch.randn((128, 10))]

        correct_loss = train_step(model, optimizer, inputs)
        correct_params = {
            name: param.clone().detach() for name, param in model.named_parameters()
        }

        opt_train_step = torch.compile(
            train_step, backend="train_step_eager", fullgraph=True
        )
        opt_loss = opt_train_step(opt_model, opt_optimizer, inputs)
        opt_params = {
            name: param.clone().detach() for name, param in opt_model.named_parameters()
        }

        self.assertTrue(same(correct_loss, opt_loss))
        for name in correct_params:
            self.assertTrue(name in opt_params)
            self.assertTrue(same(correct_params[name], opt_params[name]))

            # Note: the train_step compiler never sets .grad on the original param objects due to how it traces,
            # so we insist that the user puts .zero_grad in the train_step so there is no discrepancy between running
            # under eager or under compile
            self.assertTrue(correct_params[name].grad is None)
            self.assertTrue(opt_params[name].grad is None)

    def test_adam_optimizer(self):
        model = Seq().cuda()
        model.apply(init_weights)

        optimizer = torch.optim.Adam(model.parameters(), capturable=True)

        def train_step(model, optimizer, inputs):
            out = model(*inputs)
            loss = out.sum()

            # dynamo special case lets this pass through as a call in the FX graph
            # it gets traced out in the train_step backend, before being functionalized
            loss.backward()

            # dynamo tracks the optimizer and smuggles it on the graphmodule as an attr
            # train_step backend can reparametrize it with the fake parameter tensors it already
            # uses for module tracing
            optimizer.step()

            model.zero_grad()
            return loss

        # copy the model/optimizer up front so we don't have to reset them between eager/compile runs
        opt_model = deepcopy(model)
        opt_optimizer = deepcopy(optimizer)
        inputs = [torch.randn((128, 10)).cuda()]

        opt_train_step = torch.compile(
            train_step, backend="train_step_eager", fullgraph=True
        )
        for step in range(10):
            correct_loss = train_step(model, optimizer, inputs)
            opt_loss = opt_train_step(opt_model, opt_optimizer, inputs)
            self.assertEqual(correct_loss, opt_loss)

        correct_params = {
            name: param.clone().detach() for name, param in model.named_parameters()
        }

        opt_params = {
            name: param.clone().detach() for name, param in opt_model.named_parameters()
        }

        self.assertTrue(same(correct_loss, opt_loss))
        for name in correct_params:
            self.assertTrue(name in opt_params)
            self.assertTrue(same(correct_params[name], opt_params[name]))

            # Note: the train_step compiler never sets .grad on the original param objects due to how it traces,
            # so we insist that the user puts .zero_grad in the train_step so there is no discrepancy between running
            # under eager or under compile
            self.assertTrue(correct_params[name].grad is None)
            self.assertTrue(opt_params[name].grad is None)


    def test_deferred_smoke(self):
        # currently test_sgd and smoke both fail with the same error:
        # RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
        # paste: https://www.internalfb.com/phabricator/paste/view/P682652292
        def train_step(model, optimizer, inputs):
            out = model(*inputs)
            loss = out.sum()
            loss.backward()
            optimizer.step()
            model.zero_grad()
            return loss

        fake_tensor_mode, proxy_mode, fx_tracer = get_deferred_modes()

        with fake_tensor_mode, proxy_mode:
            deferred_model = Seq()
            deferred_model.apply(init_weights)
            deferred_model._deferred = fx_tracer
            opt_optimizer = torch.optim.Adam(deferred_model.parameters(), capturable=True)
        
        # materialize_module(m)
        inputs = [torch.randn((128, 10))]
        opt_train_step = torch.compile(
            train_step, backend="train_step_eager", fullgraph=True
        )

        loss = []
        for step in range(10):
            opt_loss = opt_train_step(deferred_model, opt_optimizer, inputs)
            loss.append(opt_loss)
            if step > 0:
                # in practice, this model loss goes 684, 458, 264, 125, ... so this check should not be too noisy
                self.assertTrue(loss[-2] > loss[-1])


    def test_smoke(self):
        # currently test_sgd and smoke both fail with the same error:
        # RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
        # paste: https://www.internalfb.com/phabricator/paste/view/P682652292
        def train_step(model, optimizer, inputs):
            out = model(*inputs)
            loss = out.sum()
            loss.backward()
            optimizer.step()
            model.zero_grad()
            return loss

        opt_model = Seq()
        opt_model.apply(init_weights)
        opt_optimizer = torch.optim.SGD(opt_model.parameters(), lr=0.01, momentum=0.9)
        inputs = [torch.randn((128, 10))]
        opt_train_step = torch.compile(
            train_step, backend="train_step_eager", fullgraph=True
        )

        loss = []
        for step in range(10):
            opt_loss = opt_train_step(opt_model, opt_optimizer, inputs)
            loss.append(opt_loss)
            if step > 0:
                # in practice, this model loss goes 684, 458, 264, 125, ... so this check should not be too noisy
                self.assertTrue(loss[-2] > loss[-1])


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
