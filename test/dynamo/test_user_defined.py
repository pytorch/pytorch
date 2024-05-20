# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo.testing

import torch._inductor.test_case
import torch.onnx.operators


class UserDefined:
    def __init__(self):
        self.a = 5

    def mutate_a(self):
        self.a = 10

    def add_c(self):
        self.c = 15

    def run(self, x):
        self.mutate_a()
        self.add_c()

        if "c" in self.__dict__:
            x = x * 2

        x = x * self.a

        del self.a

        if "a" in self.__dict__:
            x = x * 1000
        return x


class UserDefinedTests(torch._inductor.test_case.TestCase):
    def test_dunder_dict(self):
        udf = UserDefined()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return udf.run(x)

        x = torch.ones(4)
        self.assertEqual(fn(x), torch.ones(4) * 20)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
