# Owner(s): ["module: nn"]
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
    skipIfTorchDynamo,
    IS_WINDOWS
)
from torch.testing._internal.common_nn import NNTestCase, _create_basic_net

import torch
import torch.nn as nn

from functools import partial
from typing import Any, Dict, List, Tuple
import gc
import unittest
from copy import deepcopy
from tempfile import NamedTemporaryFile
import weakref
import pickle
from collections import OrderedDict
import math
import warnings


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seq1 = nn.Sequential(*[nn.Linear(10, 10) for _ in range(2)])
        self.seq2 = nn.Sequential(*[nn.Linear(10, 10) for _ in range(2)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq2(self.seq1(x))


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net1 = Net()
        self.net2 = Net()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net2(self.net1(x))


def forward_hook(
    self: TestCase,
    fired_hooks: List[int],
    expected_module: nn.Module,
    hook_id: int,
    module: nn.Module,
    inp: Tuple[torch.Tensor],
    out: torch.Tensor,
) -> None:
    fired_hooks.append(hook_id)
    self.assertEqual(id(module), id(expected_module))
    self.assertEqual(len(inp), 1)


def forward_pre_hook(
    self: TestCase,
    fired_hooks: List[int],
    expected_module: nn.Module,
    hook_id: int,
    module: nn.Module,
    inp: Tuple[torch.Tensor],
) -> None:
    fired_hooks.append(hook_id)
    self.assertEqual(id(module), id(expected_module))
    self.assertEqual(len(inp), 1)


def full_backward_hook(
    self: TestCase,
    fired_hooks: List[int],
    expected_module: nn.Module,
    hook_id: int,
    module: nn.Module,
    grad_input: Tuple[torch.Tensor],
    grad_output: Tuple[torch.Tensor],
) -> None:
    fired_hooks.append(hook_id)
    self.assertEqual(id(module), id(expected_module))
    self.assertEqual(len(grad_input), 1)
    self.assertEqual(len(grad_output), 1)


def full_backward_pre_hook(
    self: TestCase,
    fired_hooks: List[int],
    expected_module: nn.Module,
    hook_id: int,
    module: nn.Module,
    grad_input: Tuple[torch.Tensor],
) -> None:
    fired_hooks.append(hook_id)
    self.assertEqual(id(module), id(expected_module))
    self.assertEqual(len(grad_input), 1)


class KwargModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net1 = Net()
        self.net2 = Net()

    def forward(
        self, x: torch.Tensor, bias: torch.Tensor = None
    ) -> torch.Tensor:
        if bias is not None:
            x = x + bias
        return x

    def internal_forward_hook(
        self,
        module: nn.Module,
        args: Tuple[torch.Tensor],
        kwargs: Dict[str, Any],
        out: torch.Tensor,
    ):
        return out + kwargs["bias"]


class FailsInForwardModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net1 = Net()

    def forward(self, x: torch.Tensor, fail: bool = True) -> torch.Tensor:
        if fail:
            raise RuntimeError("failing in forward")
        return self.net1(x)

def kwarg_forward_pre_hook(
    self: TestCase,
    fired_hooks: List[int],
    expected_module: nn.Module,
    hook_id: int,
    module: nn.Module,
    args: Tuple[torch.Tensor],
    kwargs: Dict[str, Any],
) -> Tuple[Any, Any]:
    fired_hooks.append(hook_id)
    self.assertEqual(id(module), id(expected_module))
    self.assertEqual(len(args), 1)
    kwargs["bias"] = 2 * kwargs["bias"]
    return args, kwargs


def kwarg_forward_hook(
    self: TestCase,
    fired_hooks: List[int],
    expected_module: nn.Module,
    hook_id: int,
    module: nn.Module,
    args: Tuple[torch.Tensor],
    kwargs: Dict[str, Any],
    out: torch.Tensor,
) -> Any:
    fired_hooks.append(hook_id)
    self.assertEqual(id(module), id(expected_module))
    self.assertEqual(len(args), 1)

    out = out + kwargs["bias"]
    return out


class DummyContextManager:
    def __init__(self, inp):
        self.input = inp

    def __enter__(self, *args, **kwargs):
        self.input.append(2)

    def __exit__(self, *args, **kwargs):
        self.input.append(-1)


class TestModuleHooks(TestCase):
    @skipIfTorchDynamo("Dynamo does not yet capture hooks")
    def test_forward_hooks(self):
        fired_hooks: List[int] = []
        model = ToyModel()
        x = torch.randn(10, 10)
        hook = partial(forward_hook, self, fired_hooks, model.net1.seq2)
        model.net1.seq2.register_forward_hook(partial(hook, 0))
        model.net1.seq2.register_forward_hook(partial(hook, 1), prepend=True)
        model.net1.seq2.register_forward_hook(partial(hook, 2))
        model.net1.seq2.register_forward_hook(partial(hook, 3))
        model.net1.seq2.register_forward_hook(partial(hook, 4), prepend=True)
        expected = [4, 1, 0, 2, 3]

        self.assertEqual(fired_hooks, [])
        out = model(x)
        self.assertEqual(fired_hooks, expected)
        out.sum().backward()
        self.assertEqual(fired_hooks, expected)
        model(x).sum().backward()
        self.assertEqual(fired_hooks, expected + expected)

    @skipIfTorchDynamo("Dynamo does not yet capture hooks")
    def test_forward_pre_hooks(self):
        fired_hooks: List[int] = []
        model = ToyModel()
        x = torch.randn(10, 10)
        hook = partial(forward_pre_hook, self, fired_hooks, model.net2.seq1)
        model.net2.seq1.register_forward_pre_hook(
            partial(hook, 0), prepend=True
        )
        model.net2.seq1.register_forward_pre_hook(partial(hook, 1))
        model.net2.seq1.register_forward_pre_hook(partial(hook, 2))
        model.net2.seq1.register_forward_pre_hook(partial(hook, 3))
        model.net2.seq1.register_forward_pre_hook(
            partial(hook, 4), prepend=True
        )
        expected = [4, 0, 1, 2, 3]

        self.assertEqual(fired_hooks, [])
        out = model(x)
        self.assertEqual(fired_hooks, expected)
        out.sum().backward()
        self.assertEqual(fired_hooks, expected)
        model(x).sum().backward()
        self.assertEqual(fired_hooks, expected + expected)

    @skipIfTorchDynamo("Dynamo does not yet capture hooks")
    def test_full_backward_hooks(self):
        fired_hooks: List[int] = []
        model = ToyModel()
        x = torch.randn(10, 10)
        hook = partial(full_backward_hook, self, fired_hooks, model.net1)
        model.net1.register_full_backward_hook(partial(hook, 0))
        model.net1.register_full_backward_hook(partial(hook, 1))
        model.net1.register_full_backward_hook(partial(hook, 2))
        model.net1.register_full_backward_hook(partial(hook, 3), prepend=True)
        model.net1.register_full_backward_hook(partial(hook, 4), prepend=True)
        expected = [4, 3, 0, 1, 2]

        self.assertEqual(fired_hooks, [])
        out = model(x)
        self.assertEqual(fired_hooks, [])
        out.sum().backward()
        self.assertEqual(fired_hooks, expected)
        model(x).sum().backward()
        self.assertEqual(fired_hooks, expected + expected)

    @skipIfTorchDynamo("Dynamo does not yet capture hooks")
    def test_full_backward_pre_hooks(self):
        fired_hooks: List[int] = []
        model = ToyModel()
        x = torch.randn(10, 10)
        hook = partial(full_backward_pre_hook, self, fired_hooks, model.net1)
        model.net1.register_full_backward_pre_hook(
            partial(hook, 0), prepend=True
        )
        model.net1.register_full_backward_pre_hook(
            partial(hook, 1), prepend=True
        )
        model.net1.register_full_backward_pre_hook(partial(hook, 2))
        model.net1.register_full_backward_pre_hook(partial(hook, 3))
        model.net1.register_full_backward_pre_hook(partial(hook, 4))
        expected = [1, 0, 2, 3, 4]

        self.assertEqual(fired_hooks, [])
        out = model(x)
        self.assertEqual(fired_hooks, [])
        out.sum().backward()
        self.assertEqual(fired_hooks, expected)
        model(x).sum().backward()
        self.assertEqual(fired_hooks, expected + expected)

        # Backward pre hook can affect subsequent gradient computation
        a = torch.ones(2, requires_grad=True)
        model = nn.Linear(2, 2)

        def fn(_unused_module, grad_output):
            return (grad_output[0] * 0,)

        model.register_full_backward_pre_hook(fn)

        out = model(a)
        out.sum().backward()
        self.assertEqual(a.grad, torch.zeros_like(a))

    @skipIfTorchDynamo("Dynamo does not yet capture hooks")
    def test_mixed_hooks(self):
        fired_hooks: List[int] = []
        model = ToyModel()
        x = torch.randn(10, 10)
        model.register_forward_pre_hook(
            partial(forward_pre_hook, self, fired_hooks, model, 0)
        )
        model.register_forward_hook(
            partial(forward_hook, self, fired_hooks, model, 1)
        )
        model.register_full_backward_pre_hook(
            partial(full_backward_pre_hook, self, fired_hooks, model, 2)
        )
        model.register_full_backward_hook(
            partial(full_backward_hook, self, fired_hooks, model, 3)
        )

        self.assertEqual(fired_hooks, [])
        out = model(x)
        self.assertEqual(fired_hooks, [0, 1])
        out.sum().backward()
        self.assertEqual(fired_hooks, [0, 1, 2, 3])
        model(x).sum().backward()
        self.assertEqual(fired_hooks, [0, 1, 2, 3, 0, 1, 2, 3])

    @skipIfTorchDynamo("Dynamo does not yet capture hooks")
    def test_kwarg_hooks(self):
        # 1. test forward pre hook
        fired_hooks: List[int] = []
        x: torch.Tensor = torch.ones(10, 10)
        bias: torch.Tensor = torch.ones(10, 10)
        model = KwargModel()
        model.register_forward_pre_hook(
            partial(kwarg_forward_pre_hook, self, fired_hooks, model, 0),
            with_kwargs=True,
        )

        # forward-pre: bias' = bias * 2
        # So, out = x + bias * 2
        self.assertEqual(fired_hooks, [])
        out = model(x, bias=bias)
        self.assertEqual(fired_hooks, [0])
        self.assertEqual(out, x + 2 * bias, rtol=0, atol=1e-5)

        # 2. test forward pre and forward hooks
        fired_hooks: List[int] = []
        x: torch.Tensor = torch.ones(10, 10)
        bias: torch.Tensor = torch.ones(10, 10)
        model = KwargModel()
        model.register_forward_hook(
            partial(kwarg_forward_hook, self, fired_hooks, model, 1),
            with_kwargs=True,
        )
        model.register_forward_pre_hook(
            partial(kwarg_forward_pre_hook, self, fired_hooks, model, 0),
            with_kwargs=True,
        )

        # forward-pre: bias' = bias * 2
        # forward: out = x + bias'
        # forward-post: out = out + bias'
        # So, out = x + bias * 4
        self.assertEqual(fired_hooks, [])
        out = model(x, bias=bias)
        self.assertEqual(fired_hooks, [0, 1])
        self.assertEqual(out, x + 4 * bias, rtol=0, atol=1e-5)

        # 3. test nn.Module member method as forward-post hook
        x: torch.Tensor = torch.ones(10, 10)
        bias: torch.Tensor = torch.ones(10, 10)
        model = KwargModel()
        model.register_forward_hook(
            model.internal_forward_hook, with_kwargs=True
        )

        # forward: out = x + bias
        # forward-post: out = out + bias
        # So, out = x + bias * 2
        out = model(x, bias=bias)
        self.assertEqual(out, x + 2 * bias, rtol=0, atol=1e-5)


    @skipIfTorchDynamo("Dynamo does not yet capture hooks")
    def test_remove_kwarg_hooks(self):
        # test forward pre and forward hooks
        fired_hooks: List[int] = []
        x: torch.Tensor = torch.ones(10, 10)
        bias: torch.Tensor = torch.ones(10, 10)
        model = KwargModel()
        forward_hook_handle = model.register_forward_hook(
            partial(kwarg_forward_hook, self, fired_hooks, model, 1),
            with_kwargs=True,
        )
        forward_pre_hook_handle = model.register_forward_pre_hook(
            partial(kwarg_forward_pre_hook, self, fired_hooks, model, 0),
            with_kwargs=True,
        )

        # forward-pre: bias' = bias * 2
        # forward: out = x + bias'
        # forward-post: out = out + bias'
        # So, out = x + bias * 4
        self.assertEqual(fired_hooks, [])
        out = model(x, bias=bias)
        self.assertEqual(fired_hooks, [0, 1])
        self.assertEqual(out, x + 4 * bias, rtol=0, atol=1e-5)

        # forward-pre: bias' = bias * 2
        # forward: out = x + bias'
        # So, out = x + bias * 2
        forward_hook_handle.remove()
        out = model(x, bias=bias)
        self.assertEqual(fired_hooks, [0, 1, 0])
        self.assertEqual(out, x + 2 * bias, rtol=0, atol=1e-5)
        self.assertFalse(
            forward_hook_handle.id in model._forward_hooks_with_kwargs
        )

        # forward: out = x + bias
        # So, out = x + bias
        forward_pre_hook_handle.remove()
        out = model(x, bias=bias)
        self.assertEqual(fired_hooks, [0, 1, 0])
        self.assertEqual(out, x + bias, rtol=0, atol=1e-5)
        self.assertFalse(
            forward_pre_hook_handle.id in model._forward_pre_hooks_with_kwargs
        )

    @skipIfTorchDynamo("Dynamo does not yet capture hooks")
    def test_always_called_forward_hooks(self):
        x: torch.Tensor = torch.ones(10, 10)
        model = FailsInForwardModel()
        stack = []
        ctx = None

        def setup_context():
            nonlocal ctx
            ctx = DummyContextManager(stack)

        def ctx_setup_hook(m, i):
            setup_context()
            ctx.__enter__()

        def ctx_setup_failure_hook(m, i):
            setup_context()
            ctx.__enter__()
            raise RuntimeError("failing in ctx setup")

        def ctx_shutdown_hook(m, i, o):
            ctx.__exit__()

        def ctx_shutdown_failure_hook(m, i, o):
            ctx.__exit__()
            raise RuntimeError("failing in ctx shutdown")

        def throw_hook(m, i, o):
            raise RuntimeError("failing in throw")

        forward_pre_hook_handle = model.register_forward_pre_hook(ctx_setup_hook)
        forward_hook_handle = model.register_forward_hook(ctx_shutdown_hook, always_call=True)
        self.assertTrue(len(model._forward_hooks_always_called) == 1)

        # make sure always_called forward hook runs when model.forward raises RuntimeError
        with self.assertRaisesRegex(RuntimeError, "failing in forward"):
            model(x)
        self.assertEqual(stack, [2, -1])

        # make sure that always_called forward hook does not run twice if there is no error
        model(x, fail=False)
        self.assertEqual(stack, [2, -1, 2, -1])

        # make sure always_called forward hook runs when forward pre hook raises RuntimeError
        forward_pre_hook_handle.remove()
        model.register_forward_pre_hook(ctx_setup_failure_hook)

        with self.assertRaisesRegex(RuntimeError, "failing in ctx setup"):
            model(x, fail=False)
        self.assertEqual(stack, [2, -1, 2, -1, 2, -1])

        # make sure always_called hook runs when another always_called forward hook raises an error
        forward_hook_handle2 = model.register_forward_hook(throw_hook,
                                                           prepend=True,
                                                           always_call=True)

        # error raised should not be error of the forced hook
        with self.assertRaisesRegex(RuntimeError, "failing in ctx setup"):
            model(x, fail=False)
        self.assertEqual(stack, [2, -1, 2, -1, 2, -1, 2, -1])

        # make sure that always called forward hooks are properly removed
        forward_hook_handle.remove()
        forward_hook_handle2.remove()
        self.assertTrue(len(model._forward_hooks_always_called) == 0)

        # make sure that always called forward hook is not run twice if it fails while running
        forward_hook_handle3 = model.register_forward_hook(ctx_shutdown_failure_hook, always_call=True)
        with self.assertRaisesRegex(RuntimeError, "failing in ctx setup"):
            model(x, fail=False)
        self.assertEqual(stack, [2, -1, 2, -1, 2, -1, 2, -1, 2, -1])

        forward_hook_handle3.remove()

        global_forward_hook_handle = nn.modules.module.register_module_forward_hook(ctx_shutdown_hook, always_call=True)
        self.assertTrue(len(nn.modules.module._global_forward_hooks_always_called) == 1)
        # make sure global forward hook runs when forward pre hook raises RuntimeError
        with self.assertRaisesRegex(RuntimeError, "failing in ctx setup"):
            model(x, fail=False)
        self.assertEqual(stack, [2, -1, 2, -1, 2, -1, 2, -1, 2, -1, 2, -1])

        # make sure forced global forward hook is properly removed
        global_forward_hook_handle.remove()
        self.assertTrue(len(nn.modules.module._global_forward_hooks_always_called) == 0)
        with self.assertRaisesRegex(RuntimeError, "failing in ctx setup"):
            model(x)
        self.assertEqual(stack, [2, -1, 2, -1, 2, -1, 2, -1, 2, -1, 2, -1, 2])

    @skipIfTorchDynamo("Dynamo does not yet capture hooks")
    def test_bw_hook_warning_for_non_tensor_or_tuple(self):
        # Test to verify that backward hook raises warning
        # if result is not a Tensor or tuple of Tensors.
        counter = {'forward': 0, 'backward': 0}

        def fw_pre_hook(module: nn.Module, _inputs):
            counter['forward'] += 1

        def fw_hook(module: nn.Module, _inputs, _outputs):
            counter['forward'] += 1

        def bw_hook(module: nn.Module, _inputs, _outputs):
            counter['backward'] += 1

        class TestModule(nn.Module):
            def forward(self, dict):
                inp = dict['x']
                x = torch.nn.functional.softmax(inp, dim=0)
                return {'x': x}

        x = torch.ones(2, requires_grad=True)
        model = TestModule()
        model.register_forward_pre_hook(fw_pre_hook)
        model.register_forward_hook(fw_hook)
        model.register_full_backward_pre_hook(bw_hook)
        model.register_full_backward_hook(bw_hook)

        with warnings.catch_warnings(record=True) as w:
            y = model({'x': x})['x']
            loss = y.sum()
            loss.backward()

        self.assertEqual(counter['forward'], 2)
        self.assertEqual(counter['backward'], 0)
        self.assertEqual(len(w), 1)
        self.assertTrue("should be a Tensor or a tuple of Tensors" in str(w[0].message))


def _hook_to_pickle(*args, **kwargs):
    pass

class TestStateDictHooks(TestCase):

    def test_load_state_dict_pre_hook(self):

        m = nn.Linear(10, 10)
        m_state_dict = m.state_dict()

        m_load = nn.Linear(10, 10)

        hook_called = 0

        def hook_without_module(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            self.assertEqual(m_state_dict, state_dict)
            nonlocal hook_called
            hook_called += 1

        def hook_with_module(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            self.assertEqual(m_state_dict, state_dict)
            self.assertTrue(m_load is module)
            nonlocal hook_called
            hook_called += 1

        hook_called = 0
        m_load._register_load_state_dict_pre_hook(hook_without_module)
        m_load.load_state_dict(m_state_dict)
        self.assertEqual(1, hook_called)

        hook_called = 0
        m_load._register_load_state_dict_pre_hook(hook_with_module, True)
        m_load.load_state_dict(m_state_dict)
        self.assertEqual(2, hook_called)

    def test_no_extra_ref_to_module(self):
        try:
            gc.disable()
            m = nn.Linear(10, 10)

            m._register_load_state_dict_pre_hook(_hook_to_pickle, True)
            weak_m = weakref.ref(m)
            del m

            self.assertEqual(weak_m(), None)
        finally:
            gc.enable()

    def test_pickled_hook(self):
        m = nn.Linear(10, 10)
        m._register_load_state_dict_pre_hook(_hook_to_pickle, True)
        pickle.loads(pickle.dumps(m))

    def test_load_state_dict_module_pre_hook(self):
        hook_called = 0

        # Test with module instance method as hook
        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.foo = torch.nn.Parameter(torch.rand(10))

            def my_pre_load_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
                assert [] == error_msgs
                assert [] == unexpected_keys
                assert [] == missing_keys
                assert strict
                nonlocal hook_called
                hook_called += 1

            def my_pre_load_hook_with_module(
                self,
                module,
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            ):
                assert [] == error_msgs
                assert [] == unexpected_keys
                assert [] == missing_keys
                assert strict
                assert self is module
                nonlocal hook_called
                hook_called += 1

        # Test that hooks registered on a submodule are also called
        # appropriately, i.e. with the submodule as module argument in
        # my_pre_load_hook_with_module.
        class MyModuleContainer(nn.Module):
            def __init__(self, mod):
                super().__init__()
                self.mod = mod

        for ctor in [MyModuleContainer, lambda x: x]:
            m = ctor(MyModule())
            state_dict = m.state_dict()
            if isinstance(m, MyModuleContainer):
                mod = m.mod
            else:
                mod = m

            hook_called = 0
            mod._register_load_state_dict_pre_hook(
                mod.my_pre_load_hook
            )
            m.load_state_dict(state_dict)
            self.assertEqual(1, hook_called)

            hook_called = 0
            mod._register_load_state_dict_pre_hook(
                mod.my_pre_load_hook_with_module, True
            )
            m.load_state_dict(state_dict)
            self.assertEqual(2, hook_called)

    def test_load_state_dict_post_hook(self):
        hook_called = 0

        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.foo = torch.nn.Parameter(torch.rand(10))

            def my_post_load_hook(self, module, incompatible_keys):
                assert module is self
                nonlocal hook_called
                incompatible_keys.missing_keys.append("foo")
                incompatible_keys.unexpected_keys.append("bar")
                hook_called += 1

        nested = MyModule()
        wrapped = nn.ModuleList([nested])
        handle = nested.register_load_state_dict_post_hook(
            nested.my_post_load_hook,
        )
        # Hook must be called even if it is wrapped
        ret = wrapped.load_state_dict(wrapped.state_dict(), strict=False)
        self.assertEqual(hook_called, 1)
        # Ensure that the hook modified missing_keys and unexpected_keys
        missing = ret.missing_keys
        unexpected = ret.unexpected_keys
        self.assertEqual(missing, ["foo"])
        self.assertEqual(unexpected, ["bar"])
        # When called with strict=True, the error raised should mention the
        # missing and unexpected keys the hook added.
        with self.assertRaisesRegex(RuntimeError, "foo.*\n.*bar"):
            wrapped.load_state_dict(wrapped.state_dict(), strict=True)
        self.assertEqual(hook_called, 2)
        # Removing the hook via handle.remove() should cause it not to
        # fire anymore.
        handle.remove()
        # Hook did not run so it should not have added any keys
        ret = wrapped.load_state_dict(wrapped.state_dict(), strict=False)
        self.assertEqual(ret.missing_keys, [])
        self.assertEqual(ret.unexpected_keys, [])
        # hook_called should not have been incremented
        self.assertEqual(hook_called, 2)

        def load_hook_clear_incompatible(module, incompatible_keys):
            incompatible_keys.missing_keys.clear()
            incompatible_keys.unexpected_keys.clear()

        nested.register_load_state_dict_post_hook(load_hook_clear_incompatible)
        state_dict = wrapped.state_dict()
        state_dict["extra"] = torch.ones(1)
        # load state_dict with strict=True should not throw.
        ret = wrapped.load_state_dict(state_dict, strict=True)
        # explicitly ensure that the post hook clearned out incompatible_keys
        self.assertEqual([], ret.missing_keys)
        self.assertEqual([], ret.unexpected_keys)

    @unittest.skipIf(IS_WINDOWS, "Tempfile permission issue on windows")
    def test_load_state_dict_post_hook_backward_compatibility(self):
        def my_post_load_hook(mod, _):
            nonlocal called
            called = True

        for m in [nn.Softmin(10), nn.Softmax(10), nn.LogSoftmax(10)]:
            called = False
            sd = deepcopy(m.state_dict())
            self.assertTrue(hasattr(m, '_load_state_dict_post_hooks'))
            # Simulate an older model that did not have this attr
            delattr(m, '_load_state_dict_post_hooks')
            # Save and load, and ensure that load_state_dict works (without proper
            # BC we would run into errors because this attribute would be expected).
            # In particular, Softmax runs into the issue described here:
            # https://github.com/pytorch/pytorch/issues/77280
            with NamedTemporaryFile() as f:
                # Note that torch.save / torch.load is not recommended to save/load
                # modules.
                torch.save(m, f.name)
                m = torch.load(f.name)
                m.load_state_dict(sd)
                self.assertFalse(called)

            # Ensure hooks can be registered and called.
            m.register_load_state_dict_post_hook(my_post_load_hook)
            m.load_state_dict(sd)
            self.assertTrue(called)


class TestModuleGlobalHooks(TestCase):

    def tearDown(self):
        nn.modules.module._global_backward_hooks = OrderedDict()
        nn.modules.module._global_forward_hooks = OrderedDict()
        nn.modules.module._global_forward_pre_hooks = OrderedDict()

    @skipIfTorchDynamo("TorchDynamo does not work well with hooks")
    def test_module_global_hooks(self):
        module = nn.Sigmoid

        module_1 = module()
        module_2 = module()
        module_3 = module()

        input = torch.ones(5, 5, requires_grad=True)

        counter = {
            'forwards': 0,
            'backwards': 0
        }

        def fw_hook(inc, h_module, input, output):
            self.assertIsInstance(input, tuple)
            self.assertTrue(isinstance(output, torch.Tensor))
            self.assertTrue(isinstance(h_module, module))
            self.assertEqual(input[0], torch.ones(5, 5))
            self.assertEqual(output, torch.empty(5, 5).fill_(1 / (1 + 1 / math.e)))
            counter['forwards'] += inc

        def bw_hook(inc, h_module, grad_input, grad_output):
            self.assertIsInstance(grad_input, tuple)
            self.assertIsInstance(grad_output, tuple)
            self.assertTrue(isinstance(h_module, module))
            self.assertEqual(grad_output[0], torch.ones(5, 5) * 2)
            counter['backwards'] += inc

        test_fwd = nn.modules.module.register_module_forward_hook(lambda *args: fw_hook(1, *args))

        module_1(input)
        module_2(input)
        module_3(input)
        self.assertEqual(counter['forwards'], 3)
        self.assertEqual(counter['backwards'], 0)

        test_bwd = nn.modules.module.register_module_backward_hook(
            lambda *args: bw_hook(1, *args))

        output_1 = module_1(input)
        output_2 = module_2(input)
        output_3 = module_3(input)
        self.assertEqual(counter['forwards'], 6)
        self.assertEqual(counter['backwards'], 0)

        output_1.backward(torch.ones(5, 5) * 2, retain_graph=True)
        output_2.backward(torch.ones(5, 5) * 2, retain_graph=False)
        output_3.backward(torch.ones(5, 5) * 2, retain_graph=False)
        self.assertEqual(counter['forwards'], 6)
        self.assertEqual(counter['backwards'], 3)

        output_1.backward(torch.ones(5, 5) * 2, retain_graph=True)
        self.assertEqual(counter['forwards'], 6)
        self.assertEqual(counter['backwards'], 4)

        test2_fwd = nn.modules.module.register_module_forward_hook(lambda *args: fw_hook(2, *args))

        output = module_1(input)
        output = module_2(input)
        output = module_3(input)
        self.assertEqual(counter['forwards'], 15)
        self.assertEqual(counter['backwards'], 4)

        test2_bwd = nn.modules.module.register_module_backward_hook(lambda *args: bw_hook(2, *args))

        module_1(input).backward(torch.ones(5, 5) * 2)
        self.assertEqual(counter['forwards'], 18)
        self.assertEqual(counter['backwards'], 7)

        test2_bwd.remove()

        module_2(input).backward(torch.ones(5, 5) * 2)
        self.assertEqual(counter['forwards'], 21)
        self.assertEqual(counter['backwards'], 8)

        test2_fwd.remove()

        module_3(input).backward(torch.ones(5, 5) * 2)
        self.assertEqual(counter['forwards'], 22)
        self.assertEqual(counter['backwards'], 9)

        test_fwd.remove()
        test_bwd.remove()

    def test_module_global_hook_invalid_outputs(self):
        module = nn.Sigmoid()
        input = torch.randn(5, 5, requires_grad=True)

        def bw_fail1(self, grad_input, grad_output):
            return grad_input[:-1]

        def bw_fail2(self, grad_input, grad_output):
            return grad_input + (torch.randn(2, 2),)

        with nn.modules.module.register_module_backward_hook(bw_fail1):
            with self.assertRaisesRegex(RuntimeError, 'got 0, but expected 1'):
                module(input).sum().backward()

        with nn.modules.module.register_module_backward_hook(bw_fail2):
            with self.assertRaisesRegex(RuntimeError, 'got 2, but expected 1'):
                module(input).sum().backward()

    @skipIfTorchDynamo("https://github.com/pytorch/torchdynamo/issues/847")
    def test_module_backward_global_hook_writeable(self):
        module = nn.Sigmoid()
        input = torch.randn(5, 5, requires_grad=True)
        sig_x = torch.sigmoid(input)

        def bw_hook(module, grad_input, grad_output):
            for grad in grad_input:
                self.assertTrue(isinstance(grad, torch.Tensor))
            for grad in grad_output:
                self.assertTrue(isinstance(grad, torch.Tensor))
            return tuple(gi * 2 for gi in grad_input)

        nn.modules.module.register_module_backward_hook(bw_hook)
        module(input).backward(torch.ones(5, 5))
        expected_grad = sig_x * (1 - sig_x) * 2
        self.assertEqual(input.grad, expected_grad)

    @skipIfTorchDynamo("TorchDynamo does not work well with hooks")
    def test_module_global_forward_preforward_hook_writeable(self):
        module = nn.Sigmoid()
        input = torch.randn(5, 5, requires_grad=True)
        sig_x = torch.sigmoid(input)

        def forward_pre_hook(m, input):
            return torch.nn.functional.relu(input[0])

        def forward_hook(m, input, output):
            return -output

        nn.modules.module.register_module_forward_pre_hook(forward_pre_hook)
        nn.modules.module.register_module_forward_hook(forward_hook)
        output = module(input)
        expected_res = -torch.sigmoid(torch.nn.functional.relu(input))
        self.assertEqual(output, expected_res)
        output.backward(torch.ones(5, 5) * 2, retain_graph=True)
        mask = (input > 0)
        expected_grad = -sig_x * (1 - sig_x) * 2 * mask
        self.assertEqual(input.grad, expected_grad)

    @skipIfTorchDynamo("TorchDynamo does not work well with hooks")
    def test_module_forward_preforward_hook_removable(self):
        """
        This test is to test when multiple pre-forward hook functions can be
        registered successfully and used correctly, if the handle can be removable
        during the pre-forward hook function call.
        """
        module = nn.Sigmoid()

        def removable_hook(m, input):
            nonlocal handle
            handle.remove()
            return input

        def removable_hook_2(m, input):
            nonlocal handle_2
            handle_2.remove()
            return input

        handle = module.register_forward_pre_hook(removable_hook)
        handle_2 = module.register_forward_pre_hook(removable_hook_2)

        # make sure hook register is successful
        self.assertEqual(len(handle.hooks_dict_ref()), 2)
        self.assertEqual(len(handle_2.hooks_dict_ref()), 2)

        input = torch.randn(2, 2)
        output = module(input)
        self.assertEqual(torch.sigmoid(input), output)

        # make sure hook removal is successful
        self.assertFalse(handle.id in handle.hooks_dict_ref())
        self.assertFalse(handle_2.id in handle.hooks_dict_ref())
        self.assertEqual(len(handle.hooks_dict_ref()), 0)
        self.assertEqual(len(handle_2.hooks_dict_ref()), 0)

    @skipIfTorchDynamo("TorchDynamo does not work well with hooks")
    def test_module_forward_forward_hook_removable(self):
        """
        This test is to test when multiple forward hook functions can be registered
        successfully and used correctly, if the handle can be removable during the
        forward hook function call.
        """
        module = nn.Sigmoid()

        def removable_hook(m, input, output):
            nonlocal handle
            handle.remove()
            return output

        def removable_hook_2(m, input, output):
            nonlocal handle_2
            handle_2.remove()
            return output

        handle = module.register_forward_hook(removable_hook)
        handle_2 = module.register_forward_hook(removable_hook_2)

        # make sure hook register is successful
        self.assertEqual(len(handle.hooks_dict_ref()), 2)
        self.assertEqual(len(handle_2.hooks_dict_ref()), 2)

        input = torch.randn(2, 2)
        output = module(input)
        self.assertEqual(torch.sigmoid(input), output)

        # make sure hook removal is successful
        self.assertFalse(handle.id in handle.hooks_dict_ref())
        self.assertFalse(handle_2.id in handle.hooks_dict_ref())
        self.assertEqual(len(handle.hooks_dict_ref()), 0)
        self.assertEqual(len(handle_2.hooks_dict_ref()), 0)

    @skipIfTorchDynamo("TorchDynamo does not work well with hooks")
    def test_global_and_local_hooks_order(self):
        module = nn.Sigmoid()

        global_forward_pre_called = False
        local_forward_pre_called = False
        global_forward_called = False
        local_forward_called = False
        global_backward_called = False
        local_backward_called = False

        def global_forward_pre_hook(m, input):
            nonlocal global_forward_pre_called
            self.assertTrue(not local_forward_pre_called)
            global_forward_pre_called = True
            return input

        def local_forward_pre_hook(m, input):
            nonlocal local_forward_pre_called
            self.assertTrue(global_forward_pre_called)
            local_forward_pre_called = True
            return input

        def global_forward_hook(m, input, output):
            nonlocal global_forward_called
            self.assertTrue(not local_forward_called)
            global_forward_called = True
            return output

        def local_forward_hook(m, input, output):
            nonlocal local_forward_called
            self.assertTrue(global_forward_called)
            local_forward_called = True
            return output

        def global_backward_hook(m, input, output):
            nonlocal global_backward_called
            self.assertTrue(not local_backward_called)
            global_backward_called = True
            return input

        def local_backward_hook(m, input, output):
            nonlocal local_backward_called
            self.assertTrue(global_backward_called)
            local_backward_called = True
            return input

        input = torch.randn(5, 5, requires_grad=True)
        nn.modules.module.register_module_forward_pre_hook(global_forward_pre_hook)
        module.register_forward_pre_hook(local_forward_pre_hook)
        nn.modules.module.register_module_forward_hook(global_forward_hook)
        module.register_forward_hook(local_forward_hook)
        nn.modules.module.register_module_backward_hook(global_backward_hook)
        module.register_backward_hook(local_backward_hook)

        output = module(input)
        self.assertTrue(local_forward_called and local_forward_pre_called and global_forward_called and global_forward_pre_called)

        output.backward(torch.ones(5, 5), retain_graph=True)
        self.assertTrue(local_backward_called and global_backward_called)


class TestModuleHookNN(NNTestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    def _test_hooks(self, backward_register_fn):
        module = nn.Sigmoid()
        input = torch.ones(5, 5, requires_grad=True)

        counter = {
            'forwards': 0,
            'backwards': 0
        }

        def fw_hook(inc, h_module, input, output):
            self.assertIsInstance(input, tuple)
            self.assertTrue(isinstance(output, torch.Tensor))
            self.assertTrue(h_module is module)
            self.assertEqual(input[0], torch.ones(5, 5))
            self.assertEqual(output, torch.empty(5, 5).fill_(1 / (1 + 1 / math.e)))
            counter['forwards'] += inc

        def bw_hook(inc, h_module, grad_input, grad_output):
            self.assertIsInstance(grad_input, tuple)
            self.assertIsInstance(grad_output, tuple)
            self.assertTrue(h_module is module)
            self.assertEqual(grad_output[0], torch.ones(5, 5) * 2)
            counter['backwards'] += inc

        # backward_pre_hook expects callback with only `module` and `grad_output`
        # as arguments.
        def bw_pre_hook(inc, h_module, grad_output):
            self.assertIsInstance(grad_output, tuple)
            self.assertTrue(h_module is module)
            self.assertEqual(grad_output[0], torch.ones(5, 5) * 2)
            counter['backwards'] += inc

        test_fwd = module.register_forward_hook(lambda *args: fw_hook(1, *args))

        module(input)
        module(input)
        self.assertEqual(counter['forwards'], 2)
        self.assertEqual(counter['backwards'], 0)

        bw_hook_fn = bw_pre_hook if backward_register_fn == 'register_full_backward_pre_hook' else bw_hook
        test_bwd = getattr(module, backward_register_fn)(
            lambda *args: bw_hook_fn(1, *args))

        output = module(input)
        self.assertEqual(counter['forwards'], 3)
        self.assertEqual(counter['backwards'], 0)

        output.backward(torch.ones(5, 5) * 2, retain_graph=True)
        self.assertEqual(counter['forwards'], 3)
        self.assertEqual(counter['backwards'], 1)

        output.backward(torch.ones(5, 5) * 2, retain_graph=True)
        self.assertEqual(counter['forwards'], 3)
        self.assertEqual(counter['backwards'], 2)

        test2_fwd = module.register_forward_hook(lambda *args: fw_hook(2, *args))

        output = module(input)
        self.assertEqual(counter['forwards'], 6)
        self.assertEqual(counter['backwards'], 2)

        test2_bwd = getattr(module, backward_register_fn)(lambda *args: bw_hook_fn(2, *args))

        module(input).backward(torch.ones(5, 5) * 2)
        self.assertEqual(counter['forwards'], 9)
        self.assertEqual(counter['backwards'], 5)

        test2_bwd.remove()

        module(input).backward(torch.ones(5, 5) * 2)
        self.assertEqual(counter['forwards'], 12)
        self.assertEqual(counter['backwards'], 6)

        test2_fwd.remove()

        module(input).backward(torch.ones(5, 5) * 2)
        self.assertEqual(counter['forwards'], 13)
        self.assertEqual(counter['backwards'], 7)

        test_fwd.remove()
        test_bwd.remove()

    @skipIfTorchDynamo("TorchDynamo does not work well with hooks")
    def test_hooks(self):
        self._test_hooks("register_backward_hook")
        self._test_hooks("register_full_backward_hook")
        self._test_hooks("register_full_backward_pre_hook")

    def test_hook_cpp(self):
        bn = nn.BatchNorm1d(5)

        def hook(module, grad_inputs, grad_outputs):
            self.assertEqual(len(grad_inputs), 1)
            self.assertEqual(len(grad_outputs), 1)
            self.assertEqual(module, bn)

        bn.register_full_backward_hook(hook)
        output = bn(torch.randn(5, 5, requires_grad=True))
        output.sum().backward()

    @skipIfTorchDynamo("TorchDynamo does not work well with hooks")
    def test_backward_hooks_interaction(self):
        # Test to make sure that the grad_outputs
        # updated by full_backward_pre_hook are received by
        # the full_backward_hook
        module = torch.nn.Sigmoid()

        cnt = {'backward_cnt': 0}

        def bw_pre_hook(m, grad_output):
            cnt['backward_cnt'] += 1
            return (grad_output[0] * 0.5, )

        def bw_hook(m, grad_in, grad_output):
            self.assertEqual(torch.full_like(grad_output[0], 0.5), grad_output[0])
            cnt['backward_cnt'] += 1
            return grad_output

        module.register_full_backward_pre_hook(bw_pre_hook)
        module.register_full_backward_hook(bw_hook)

        t = torch.ones(1, 2, requires_grad=True)
        module(t).sum().backward()
        self.assertEqual(cnt['backward_cnt'], 2)

    def test_hook_invalid_outputs(self):
        module = nn.Sigmoid()
        input = torch.randn(5, 5, requires_grad=True)

        def bw_fail1(self, grad_input, grad_output):
            return grad_input[:-1]

        def bw_fail2(self, grad_input, grad_output):
            return grad_input + (torch.randn(2, 2),)

        with module.register_backward_hook(bw_fail1):
            with self.assertRaisesRegex(RuntimeError, 'got 0, but expected 1'):
                module(input).sum().backward()

        with module.register_backward_hook(bw_fail2):
            with self.assertRaisesRegex(RuntimeError, 'got 2, but expected 1'):
                module(input).sum().backward()

        def bw_pre_fail1(self, grad_output):
            return ()

        def bw_pre_fail2(self, grad_output):
            return grad_output + (torch.randn(2, 2),)

        with module.register_full_backward_pre_hook(bw_pre_fail1):
            with self.assertRaisesRegex(RuntimeError, 'got 0, but expected 1'):
                module(input).sum().backward()

        with module.register_full_backward_pre_hook(bw_pre_fail2):
            with self.assertRaisesRegex(RuntimeError, 'got 2, but expected 1'):
                module(input).sum().backward()

    def test_hook_requires_grad(self):
        test_self = self

        class MyModule(nn.Module):
            def forward(self, arg1, arg2, arg3):
                test_self.assertTrue(arg1.requires_grad)
                test_self.assertFalse(arg2.requires_grad)
                test_self.assertTrue(arg3.requires_grad)
                return arg1.sum() + arg2.sum() + arg3.sum()

        inp = torch.rand(2, requires_grad=True)
        mod = MyModule()

        mod(inp, inp.detach(), inp)
        # Ensure that requires grad is properly propagated
        mod.register_full_backward_hook(lambda mod, gI, gO: None)
        mod(inp, inp.detach(), inp)

    @skipIfTorchDynamo("TorchDynamo does not work well with hooks")
    def test_hook_no_requires_grad(self):
        mod = nn.Linear(2, 3)

        inp = torch.rand(1, 2)

        return_val = "None"
        hook_called = [0]

        def hook(mod, grad_input, grad_output):
            hook_called[0] += 1
            for gI in grad_input:
                self.assertIsNone(gI)
            for gO in grad_output:
                self.assertEqual(gO.size(), (1, 3))

            if return_val == "grad_input":
                return grad_input
            elif return_val == "invalid":
                # If the inputs were requiring gradients, this would be
                # a valid return
                return inp
            elif return_val == "None":
                return None
            else:
                raise RuntimeError("Invalid return_val string")

        mod.register_full_backward_hook(hook)

        # This should run and trigger the hook properly
        mod(inp).sum().backward()
        self.assertEqual(hook_called[0], 1)

        return_val = "grad_input"

        mod(inp).sum().backward()
        self.assertEqual(hook_called[0], 2)

        return_val = "invalid"
        with self.assertRaisesRegex(RuntimeError, "where no input requires gradient"):
            mod(inp).sum().backward()

    def test_hook_last_arg_requires_grad(self):
        mod = nn.L1Loss()
        inp = torch.rand(1, requires_grad=True)
        mod.register_full_backward_hook(lambda m, gI, gO: None)

        try:
            mod(inp.detach(), inp)
        except Exception as ex:
            self.fail("Unexpected exception: %s" % ex)

    def test_hook_extra_input(self):
        class MyModule(nn.Module):
            def forward(self, non_tensor, tensor):
                return tensor.clone(), non_tensor

        inp = torch.rand(2, requires_grad=True)
        mod = MyModule()

        def hook(mod, grad_input, grad_output):
            self.assertIsNone(grad_input[0])
            self.assertIsInstance(grad_input[1], torch.Tensor)

            self.assertIsInstance(grad_output[0], torch.Tensor)
            self.assertIsNone(grad_output[1])

        mod.register_full_backward_hook(hook)
        out, _ = mod(True, inp)
        out.sum().backward()

    def test_hook_inplace(self):
        class MyModule(nn.Module):
            def forward(self, inp, do_inplace):
                self.inp = inp
                if do_inplace:
                    inp += 1
                return inp.clone()

        hook_called = [0]

        def hook(mod, grad_input, grad_output):
            hook_called[0] += 1

        def hook_pre(mod, grad_output):
            hook_called[0] += 1

        inp = torch.rand(10, requires_grad=True)
        mod = MyModule()
        for hook_fn, register_fn in [(hook, mod.register_full_backward_hook),
                                     (hook_pre, mod.register_full_backward_pre_hook)]:
            hook_called[0] = 0
            with register_fn(hook_fn):
                # No inplace should work
                mod(inp, False).sum().backward()
                self.assertEqual(hook_called[0], 1)

                # Input inplace error should throw an error
                with self.assertRaisesRegex(RuntimeError, "Output 0 of BackwardHookFunctionBackward is "
                                            "a view and is being modified inplace."):
                    mod(inp.clone(), True)

                # Input inplace error should throw an error if we try to re-use the view after they have
                # been modified
                local_inp = inp.clone()
                out = mod(local_inp, False)
                local_inp[0] *= 1
                with self.assertRaisesRegex(RuntimeError, "Output 0 of BackwardHookFunctionBackward is "
                                            "a view and its base or another view"):
                    # Any operation involving the view will fail here
                    mod.inp + 2

                # Output inplace error should throw an error
                out = mod(inp, False)
                with self.assertRaisesRegex(RuntimeError, "BackwardHookFunctionBackward is a view "
                                            "and is being modified inplace."):
                    out += 1

    def test_hook_non_full_warning(self):
        def noop(*args):
            pass

        a = torch.rand(2, requires_grad=True)
        b = torch.rand(2, requires_grad=True)

        # Check invalid input container
        class MyModule(nn.Module):
            def forward(self, l):
                return l[0].clone(), l[1].clone()

        m = MyModule()
        m.register_backward_hook(noop)

        with self.assertWarnsRegex(UserWarning, "does not take as input a single Tensor or a tuple of Tensors"):
            m([a, b])

        # Check invalid output container
        class MyModule(nn.Module):
            def forward(self, a, b):
                return [a.clone(), b.clone()]

        m = MyModule()
        m.register_backward_hook(noop)

        with self.assertWarnsRegex(UserWarning, "does not return a single Tensor or a tuple of Tensors"):
            m(a, b)

        # Check invalid output from different Nodes
        class MyModule(nn.Module):
            def forward(self, a, b):
                return a.clone(), b.clone()

        m = MyModule()
        m.register_backward_hook(noop)

        with self.assertWarnsRegex(UserWarning, "outputs are generated by different autograd Nodes"):
            m(a, b)

        # Check invalid forward with multiple Nodes
        class MyModule(nn.Module):
            def forward(self, a):
                return a.clone().clone()

        m = MyModule()
        m.register_backward_hook(noop)

        with self.assertWarnsRegex(UserWarning, "the forward contains multiple autograd Nodes"):
            m(a)

    def test_hook_backward_size(self):
        # Make module with multiple operations in forward
        # And different size for input and outputs
        class MyModule(nn.Module):
            def forward(self, arg1, arg2):
                tmp = arg1.sum() * arg2
                tmp = tmp + arg2.sum() * arg1.sum()
                tmp = tmp.sum().view(1)
                tmp = tmp.expand(8).contiguous()
                return tmp

        module = MyModule()
        inp1 = torch.randn(5, 5, requires_grad=True)
        inp2 = torch.randn(10, 10, requires_grad=True)

        def bw_hook(module, grad_input, grad_output):
            self.assertEqual(len(grad_input), 2)
            self.assertEqual(grad_input[0].size(), torch.Size([5, 5]))
            self.assertEqual(grad_input[1].size(), torch.Size([10, 10]))
            self.assertEqual(len(grad_output), 1)
            self.assertEqual(grad_output[0].size(), torch.Size([8]))

        with module.register_full_backward_hook(bw_hook):
            module(inp1, inp2).sum().backward()

    @skipIfTorchDynamo("TorchDynamo does not work well with hooks")
    def test_hook_backward_writeable(self):
        module = nn.Sigmoid()
        input = torch.randn(5, 5, requires_grad=True)
        sig_x = torch.nn.functional.sigmoid(input)

        def bw_hook(module, grad_input, grad_output):
            for grad in grad_input:
                self.assertTrue(isinstance(grad, torch.Tensor))
            for grad in grad_output:
                self.assertTrue(isinstance(grad, torch.Tensor))
            return tuple(gi * 2 for gi in grad_input)

        module.register_backward_hook(bw_hook)
        module(input).backward(torch.ones(5, 5))
        expected_grad = sig_x * (1 - sig_x) * 2
        self.assertEqual(input.grad, expected_grad)

    @skipIfTorchDynamo("TorchDynamo does not work well with hooks")
    def test_hook_forward_preforward_writable(self):
        module = nn.Sigmoid()
        input = torch.randn(5, 5, requires_grad=True)
        sig_x = torch.nn.functional.sigmoid(input)

        def forward_pre_hook(m, input):
            return torch.nn.functional.relu(input[0])

        def forward_hook(m, input, output):
            return -output

        module.register_forward_pre_hook(forward_pre_hook)
        module.register_forward_hook(forward_hook)
        output = module(input)
        expected_res = -torch.nn.functional.sigmoid(torch.nn.functional.relu(input))
        self.assertEqual(output, expected_res)
        output.backward(torch.ones(5, 5) * 2, retain_graph=True)
        mask = (input > 0)
        expected_grad = -sig_x * (1 - sig_x) * 2 * mask
        self.assertEqual(input.grad, expected_grad)

    def test_hook_buffer_registration(self):
        for return_buffer in (True, False):
            def buffer_registration_hook(module, name, buffer):
                buffer.registered = True
                if return_buffer:
                    return buffer
            handle = torch.nn.modules.module.register_module_buffer_registration_hook(
                buffer_registration_hook
            )
            try:
                l, n, s = _create_basic_net()
                for b in s.buffers():
                    self.assertTrue(getattr(b, "registered", False))
            finally:
                handle.remove()

    def test_hook_submodule_registration(self):
        for return_submodule in (True, False):
            def module_registration_hook(module, name, submodule):
                module.registered = True
                submodule.registered = True
                if return_submodule:
                    return submodule
            handle = torch.nn.modules.module.register_module_module_registration_hook(
                module_registration_hook
            )
            try:
                l, n, s = _create_basic_net()
                for m in s.modules():
                    self.assertTrue(getattr(m, "registered", False))
            finally:
                handle.remove()

    def test_hook_parameter_registration(self):
        for return_parameter in (True, False):
            def parameter_registration_hook(module, name, parameter):
                parameter.registered = True
                if return_parameter:
                    return parameter
            handle = torch.nn.modules.module.register_module_parameter_registration_hook(
                parameter_registration_hook
            )
            try:
                l, n, s = _create_basic_net()
                for p in s.parameters():
                    self.assertTrue(getattr(p, "registered", False))
            finally:
                handle.remove()


if __name__ == "__main__":
    run_tests()
