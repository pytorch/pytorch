# Owner(s): ["oncall: fx"]

import torch
from torch.fx.experimental._dynamism import track_dynamism_across_examples
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
            "x": {"L['x']": (True,)},
            "y": {"L['y']": (True, False)},
            "z": {"L['z'][0]": (True,)},
        }
        self.assertEqual(result, expected)

    def test_dynamic_tensor_deeply_nested(self):
        ex1 = {"z": {"z": {"z": {"z": {0: torch.ones(1)}}}}}
        ex2 = {"z": {"z": {"z": {"z": {0: torch.ones(2)}}}}}
        ex3 = {"z": {"z": {"z": {"z": {0: torch.ones(3)}}}}}
        ex4 = {"z": {"z": {"z": {"z": {0: torch.ones(4)}}}}}
        ex5 = {"z": {"z": {"z": {"z": {0: torch.ones(5)}}}}}
        examples = [ex1, ex2, ex3, ex4, ex5]

        result = track_dynamism_across_examples(examples)
        expected = {
            "z": {
                "L['z']['z']['z']['z'][0]": (True,),
            },
        }
        self.assertEqual(result, expected)

    def test_mixed_dynamism(self):
        ex1 = {"a": torch.ones(1, 2), "b": [torch.ones(1), 3], "c": {"d": 42}}
        ex2 = {"a": torch.ones(2, 2), "b": [torch.ones(2), 4], "c": {"d": 42}}
        ex3 = {"a": torch.ones(3, 2), "b": [torch.ones(3), 5], "c": {"d": 42}}
        ex4 = {"a": torch.ones(4, 2), "b": [torch.ones(4), 6], "c": {"d": 42}}
        ex5 = {"a": torch.ones(5, 2), "b": [torch.ones(5), 7], "c": {"d": 42}}
        examples = [ex1, ex2, ex3, ex4, ex5]

        result = track_dynamism_across_examples(examples)
        expected = {
            "a": {"L['a']": (True, False)},
            "b": {"L['b'][0]": (True,), "L['b'][1]": (True,)},
            "c": {"L['c']['d']": (False,)},
        }
        self.assertEqual(result, expected)

    def test_nn_module(self):
        class Y(torch.nn.Module):
            def __init__(self, n_input, n_output):
                super().__init__()
                self.compress = torch.nn.Linear(n_input, n_output)
                self.x = n_input

            def forward(self, x):
                return self.compress(x) * self.x

        class M(torch.nn.Module):
            def __init__(self, n_input, n_output):
                self.n_input = n_input
                self.n_output = n_output
                super().__init__()
                self.y = Y(n_input, n_output)

            def forward(self, x):
                return self.y(x)

        model1 = M(3210, 30)
        model2 = M(3211, 30)

        result = track_dynamism_across_examples(
            [
                {"self": model1},
                {"self": model2},
            ]
        )
        expected = {
            "self": {
                "L['self']['_modules']['y']['_modules']['compress']['_parameters']['weight']": (
                    False,
                    True,
                ),
                "L['self']['_modules']['y']['_modules']['compress']['_parameters']['bias']": (
                    False,
                ),
                "L['self']['_modules']['y']['_modules']['compress']['bias']": (False,),
                "L['self']['_modules']['y']['_modules']['compress']['in_features']": (
                    True,
                ),
                "L['self']['_modules']['y']['_modules']['compress']['out_features']": (
                    False,
                ),
                "L['self']['_modules']['y']['_modules']['compress']['weight']": (
                    False,
                    True,
                ),
                "L['self']['_modules']['y']['x']": (True,),
                "L['self']['n_input']": (True,),
                "L['self']['n_output']": (False,),
            }
        }
        self.assertEqual(result, expected)


if __name__ == "__main__":
    run_tests()
