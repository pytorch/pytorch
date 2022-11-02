# Owner(s): ["module: dynamo"]
import torch
import torch._dynamo.test_case
import torch._dynamo.testing

class MutationExportTests(torch._dynamo.test_case.TestCase):
    def check_failure_on_export(self, mod, *args):
        with self.assertRaises(AssertionError):
            torch._dynamo.export(mod, *args)

    def check_same_with_export(self, mod, arg):
        real_result = mod(arg)
        graph, _ = torch._dynamo.export(mod, arg)
        result = graph(arg)
        self.assertTrue(torch._dynamo.utils.same(result, real_result))

    def test_module_attribute_mutation_violation_positive_1(self):
        # Mutating attribute with a Tensor type
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.Tensor(3, 2)

            def forward(self, x):
                self.a = self.a.to(torch.float64)
                return x.sum() + self.a.sum()

        self.check_failure_on_export(Foo(), torch.Tensor(3, 2))

    def test_module_attribute_mutation_violation_positive_2(self):
        # Mutating attribute with a scalar type
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = 2

            def forward(self, x):
                self.a = self.a * 3
                return x.sum() + self.a

        self.check_failure_on_export(Foo(), torch.Tensor(3, 2))

    def test_module_attribute_mutation_violation_positive_3(self):
        # Setting a new attribute inside forward()
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.Tensor(3, 2)

            def forward(self, x):
                self.b = 2
                return x.sum() + self.a.sum() + self.b

        self.check_failure_on_export(Foo(), torch.Tensor(3, 2))

    def test_module_attribute_mutation_violation_negative_1(self):
        # Mutating attribute with a Tensor type inside __init__ but
        # not in forward()
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.Tensor(3, 2)

            def forward(self, x):
                return x.sum() + self.a.to(torch.float64).sum()

        self.check_same_with_export(Foo(), torch.Tensor(3, 2))

    def test_module_attribute_mutation_violation_negative_2(self):
        # Mutating attribute with a Tensor type inside __init__ twice
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.Tensor(3, 2)
                self.a = self.a.to(torch.float64)

            def forward(self, x):
                return x.sum() + self.a.sum()

        self.check_same_with_export(Foo(), torch.Tensor(3, 2))

    def test_module_attribute_mutation_violation_negative_3(self):
        # Mutating local variable inside forward()
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.Tensor(3, 2)

            def forward(self, x):
                b = 1
                b = b * 5
                return x.sum() + self.a.sum() + b

        self.check_same_with_export(Foo(), torch.Tensor(3, 2))


    def test_module_attribute_mutation_violation_negative_4(self):
        # Mutating attribute with a Tensor type
        # But not exporting but using eager mode as well as dynamo optimize mode
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.Tensor(3, 2)

            def forward(self, x):
                self.a = self.a.to(torch.float64)
                return x.sum() + self.a.sum()

        mod = Foo()
        arg = torch.Tensor(3, 2)
        real_result = mod(arg)
        opt_mod = torch._dynamo.optimize("eager", nopython=True)(mod)
        self.assertTrue(torch._dynamo.utils.same(opt_mod(arg), real_result))

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
