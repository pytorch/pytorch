# Owner(s): ["oncall: fx"]

import torch
from torch.fx.experimental.dynamism import track_dynamism_across_examples
from torch.testing._internal.common_utils import run_tests, TestCase

class TestDynamism(TestCase):
    def test_dynamic_tensor(self):
        ex1 = {"x": 1, "y": torch.ones(1, 1), "z": {0: torch.ones(1)}}
        ex2 = {"x": 2, "y": torch.ones(2, 1), "z": {0: torch.ones(2)}}
        ex3 = {"x": 3, "y": torch.ones(3, 1), "z": {0: torch.ones(3)}}
        ex4 = {"x": 4, "y": torch.ones(4, 1), "z": {0: torch.ones(4)}}
        ex5 = {"x": 5, "y": torch.ones(5, 1), "z": {0: torch.ones(5)}}
        examples = [ex1, ex2, ex3, ex4, ex5]

        result = track_dynamism_across_examples(examples)
        expected = {
            "x": {"['x']": (True,)},
            "y": {"['y']": (True, False)},
            "z": {"['z'][0]": (True,)}
        }
        assert result == expected, f"test_dynamic_tensor failed. Expected {expected}, got {result}"

if __name__ == "__main__":
    run_tests()
