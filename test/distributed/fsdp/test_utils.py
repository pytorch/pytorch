# Owner(s): ["oncall: distributed"]

import random
import sys
import unittest
from collections import OrderedDict
from dataclasses import dataclass
from enum import auto, Enum
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch import distributed as dist
from torch.distributed.fsdp._utils import _apply_to_tensors
from torch.distributed.fsdp._wrap_utils import _get_fully_sharded_module_to_states
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.utils import _replace_by_prefix
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
)

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestUtils(TestCase):
    @parametrize(
        "devices", [["cpu"], ["cuda"], subtest(["cpu", "cuda"], name="cpu_cuda")]
    )
    def test_apply_to_tensors(self, devices):
        if "cuda" in devices and (
            not torch.cuda.is_available() or torch.cuda.device_count() < 1
        ):
            raise unittest.SkipTest("Skipped due to lack of GPU")

        expected = 0

        def get_a_tensor():
            """Return a random tensor on random device."""
            dev = random.choice(devices)
            shape = random.choice(((1), (2, 3), (4, 5, 6), (7, 8, 9, 10)))
            t = torch.rand(shape).to(dev)
            nonlocal expected
            expected += t.numel()
            return t

        @dataclass
        class SomeDataClass:
            some_key: str
            some_float: float
            some_tensor: List[torch.Tensor]

        # create a mixed bag of data.
        data = [1, "str"]
        data.append({"key1": get_a_tensor(), "key2": {1: get_a_tensor()}, "key3": 3})
        data.insert(0, set(["x", get_a_tensor(), get_a_tensor()]))
        data.append(([1], get_a_tensor(), (1), [get_a_tensor()], set((1, 2))))
        data.append({"abc": SomeDataClass("some_key", 1.0, [get_a_tensor()])})
        od = OrderedDict()
        od["k"] = "value"
        data.append(od)

        total = 0

        def fn(t):
            nonlocal total
            total += t.numel()
            return t

        new_data = _apply_to_tensors(fn, data)
        self.assertEqual(total, expected)
        for i, v in enumerate(data):
            self.assertEqual(type(new_data[i]), type(v))

    def test_replace_by_prefix(self):
        state_dict = {
            "layer.a": torch.tensor(1),
            "abc.layer.def": torch.tensor(2),
            "layer.b": torch.tensor(3),
        }
        original_state_dict = state_dict.copy()
        _replace_by_prefix(state_dict, "layer.", "module.layer.")
        assert state_dict == {
            "module.layer.a": torch.tensor(1),
            "abc.layer.def": torch.tensor(2),
            "module.layer.b": torch.tensor(3),
        }
        _replace_by_prefix(state_dict, "module.layer.", "layer.")
        assert state_dict == original_state_dict

    def test_packed_sequence(self):
        """Test to ensure RNN packed sequences are modified correctly."""
        rnn = nn.RNN(5, 5)

        x = torch.rand((5, 1, 5), dtype=torch.float)
        seq_length = torch.tensor([4], dtype=torch.int)

        def fill_fn(x):
            x.fill_(0)

        x = nn.utils.rnn.pack_padded_sequence(x, seq_length)
        x, h = rnn(x)
        x = _apply_to_tensors(fill_fn, x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        self.assertEqual(torch.sum(x), 0)


class TestGetSubmoduleToStates(TestCase):
    """Tests the function ``_get_fully_sharded_module_to_states()``."""

    class SharedParameterMode(Enum):
        """
        - ``PARENT_CHILD``: A parent submodule shares a parameter with a child
        submodule.
        - ``SIBLING``: Two sibling submodules share a parameter.
        """

        PARENT_CHILD = auto()
        SIBLING = auto()  # TODO: not yet supported

    class Model(nn.Module):
        """Nested model with buffers and a shared parameter."""

        def __init__(self, shared_parameter_mode) -> None:
            super().__init__()
            self.seq1 = nn.Sequential(
                nn.Linear(5, 5, bias=False),
                nn.Linear(5, 5, bias=False),
            )
            self.seq1.register_buffer("seq1_buffer", torch.randn((5,)))
            self.lin = nn.Linear(5, 5, bias=False)
            self.seq2 = nn.Sequential(
                nn.Sequential(nn.Linear(5, 5, bias=False)), nn.Linear(5, 5, bias=False)
            )
            if (
                shared_parameter_mode
                == TestGetSubmoduleToStates.SharedParameterMode.PARENT_CHILD
            ):
                self.seq2[0][0].weight = self.lin.weight
            elif (
                shared_parameter_mode
                == TestGetSubmoduleToStates.SharedParameterMode.SIBLING
            ):
                self.seq2[0][0].weight = self.seq1[0].weight
            self.seq2[1].register_buffer("seq2_1_buffer", torch.randn((5,)))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.seq2(self.lin(self.seq1(x)))  # equivalent to one matmul

    def test_get_fully_sharded_module_to_states_parent_child_sharing(self):
        """
        Tests the helper function ``_get_fully_sharded_module_states()`` that
        performs the pseudo-auto-wrapping for the non-wrapper path in the
        presence of parent-child shared parameters.

        NOTE: This test is hard coded against ``Model``.
        """
        model = self.Model(TestGetSubmoduleToStates.SharedParameterMode.PARENT_CHILD)
        (
            fully_sharded_module_to_states,
            shared_param_to_lca_module,
        ) = self._get_fully_sharded_module_to_states(model, (nn.Sequential,))

        # - Root module `model`
        self.assertIn(model, fully_sharded_module_to_states)
        root_states = fully_sharded_module_to_states[model]
        self.assertEqual(root_states.params, [model.lin.weight])
        self.assertEqual(root_states.buffers, [])
        # - `seq1`
        self.assertIn(model.seq1, fully_sharded_module_to_states)
        seq1_states = fully_sharded_module_to_states[model.seq1]
        self.assertEqual(
            seq1_states.params, [model.seq1[0].weight, model.seq1[1].weight]
        )
        self.assertEqual(seq1_states.buffers, [model.seq1.seq1_buffer])
        # - `seq2`
        self.assertIn(model.seq2, fully_sharded_module_to_states)
        seq2_states = fully_sharded_module_to_states[model.seq2]
        self.assertEqual(seq2_states.params, [model.seq2[1].weight])
        self.assertEqual(seq2_states.buffers, [model.seq2[1].seq2_1_buffer])
        # - `seq2[0]`
        self.assertIn(model.seq2[0], fully_sharded_module_to_states)
        seq2_0_states = fully_sharded_module_to_states[model.seq2[0]]
        self.assertEqual(seq2_0_states.params, [])
        self.assertEqual(seq2_0_states.buffers, [])

        self.assertEqual(len(shared_param_to_lca_module), 1)
        self.assertIn(model.seq2[0][0].weight, shared_param_to_lca_module)
        self.assertIn(model.lin.weight, shared_param_to_lca_module)  # same reference
        self.assertEqual(shared_param_to_lca_module[model.seq2[0][0].weight], model)

    def test_get_fully_sharded_module_to_states_sibling_sharing(self):
        """
        Tests the helper function ``_get_fully_sharded_module_states()`` that
        performs the pseudo-auto-wrapping for the non-wrapper path in the
        presence of sibling shared parameters.

        NOTE: This test is hard coded against ``Model``.
        """
        model = self.Model(TestGetSubmoduleToStates.SharedParameterMode.SIBLING)
        (
            fully_sharded_module_to_states,
            shared_param_to_lca_module,
        ) = self._get_fully_sharded_module_to_states(model, (nn.Sequential,))

        # - Root module `model`
        self.assertIn(model, fully_sharded_module_to_states)
        root_states = fully_sharded_module_to_states[model]
        self.assertEqual(
            set(root_states.params), {model.lin.weight, model.seq1[0].weight}
        )
        self.assertEqual(root_states.buffers, [])
        # - `seq1`
        self.assertIn(model.seq1, fully_sharded_module_to_states)
        seq1_states = fully_sharded_module_to_states[model.seq1]
        self.assertEqual(seq1_states.params, [model.seq1[1].weight])
        self.assertEqual(seq1_states.buffers, [model.seq1.seq1_buffer])
        # - `seq2`
        self.assertIn(model.seq2, fully_sharded_module_to_states)
        seq2_states = fully_sharded_module_to_states[model.seq2]
        self.assertEqual(seq2_states.params, [model.seq2[1].weight])
        self.assertEqual(seq2_states.buffers, [model.seq2[1].seq2_1_buffer])
        # - `seq2[0]`
        self.assertIn(model.seq2[0], fully_sharded_module_to_states)
        seq2_0_states = fully_sharded_module_to_states[model.seq2[0]]
        self.assertEqual(seq2_0_states.params, [])
        self.assertEqual(seq2_0_states.buffers, [])

        self.assertEqual(len(shared_param_to_lca_module), 1)
        self.assertIn(model.seq2[0][0].weight, shared_param_to_lca_module)
        self.assertIn(model.seq1[0].weight, shared_param_to_lca_module)
        self.assertEqual(shared_param_to_lca_module[model.seq2[0][0].weight], model)

    def _get_fully_sharded_module_to_states(
        self,
        model: nn.Module,
        module_classes: Tuple[nn.Module, ...],
    ) -> Tuple[Dict, Dict]:
        # Compute the mapping from fully sharded module to states according to
        # a logical module wrap policy
        auto_wrap_policy = ModuleWrapPolicy(set(module_classes))
        (
            fully_sharded_module_to_states,
            shared_param_to_lca_module,
        ) = _get_fully_sharded_module_to_states(model, auto_wrap_policy, set(), set())
        # Check the number of submodules with states in the mapping
        num_submodules_with_states = sum(
            isinstance(submodule, module_classes) for submodule in model.modules()
        )  # explicitly show how to compute the expected number
        if not isinstance(model, module_classes):
            num_submodules_with_states += 1  # always include the root
        assert num_submodules_with_states == 4, f"{num_submodules_with_states}"

        # Check the mapping's keys are expected
        fully_sharded_modules = set(fully_sharded_module_to_states.keys())
        expected_fully_sharded_modules = {
            module
            for module in model.modules()
            if isinstance(module, module_classes) or module is model
        }
        self.assertEqual(expected_fully_sharded_modules, fully_sharded_modules)
        return fully_sharded_module_to_states, shared_param_to_lca_module


instantiate_parametrized_tests(TestUtils)
instantiate_parametrized_tests(TestGetSubmoduleToStates)

if __name__ == "__main__":
    run_tests()
