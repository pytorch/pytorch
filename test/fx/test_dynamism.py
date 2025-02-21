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
            "x": {"L['x']": (True,)},
            "y": {"L['y']": (True, False)},
            "z": {"L['z'][0]": (True,)},
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

            @torch.compile(delay=1)
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
        print(result)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    run_tests()
