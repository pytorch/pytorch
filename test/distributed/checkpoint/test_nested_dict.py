# Owner(s): ["oncall: distributed"]

import torch
from torch.distributed.checkpoint._nested_dict import (
    flatten_state_dict,
    unflatten_state_dict,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class TestFlattening(TestCase):
    def test_flattening_round_trip(self) -> None:
        state_dict = {
            "key0": 1,
            "key1": [1, 2],
            "key2": {1: 2, 2: 3},
            "key3": torch.tensor([1]),
            "key4": [[torch.tensor(2), "x"], [1, 2, 3], {"key6": [44]}],
        }

        flatten_dict, mapping = flatten_state_dict(state_dict)
        """
        flatten_dict:
            {
                'key0': 1,
                'key1': [1, 2],
                'key2': {1: 2, 2: 3},
                'key3': tensor([1]),
                'key4.0.0': tensor(2),
                'key4.0.1': 'x',
                'key4.1': [1, 2, 3],
                'key4.2': {'key6': [44]}
            }
        """
        restored = unflatten_state_dict(flatten_dict, mapping)

        self.assertEqual(state_dict, restored)

    def test_mapping(self) -> None:
        state_dict = {
            "k0": [1],
            "k2": [torch.tensor([1]), 99, [{"k3": torch.tensor(1)}]],
            "k3": ["x", 99, [{"k3": "y"}]],
        }

        flatten_dict, mapping = flatten_state_dict(state_dict)
        """
        flatten_dict:
        {'k0': [1], 'k2.0': tensor([1]), 'k2.1': 99, 'k2.2.0.k3': tensor(1), 'k3': ['x', 99, [{'k3': 'y'}]]}
        mapping:
        {'k0': ('k0',), 'k2.0': ('k2', 0), 'k2.1': ('k2', 1), 'k2.2.0.k3': ('k2', 2, 0, 'k3'), 'k3': ('k3',)}
        """

        self.assertEqual(("k0",), mapping["k0"])
        self.assertEqual(("k2", 0), mapping["k2.0"])
        self.assertEqual(("k2", 1), mapping["k2.1"])
        self.assertEqual(("k2", 2, 0, "k3"), mapping["k2.2.0.k3"])
        self.assertEqual(("k3",), mapping["k3"])


if __name__ == "__main__":
    run_tests()
