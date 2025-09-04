# Owner(s): ["module: dynamo"]
import unittest

import torch


class TestNestedConstWeightWeakref(unittest.TestCase):
    def test_nested_const_weight_weakref(self):
        class DummyModule(torch.nn.Module):
            def __init__(self):
                super(DummyModule, self).__init__()
                self.a = torch.nn.ModuleDict(
                    {
                        "b": torch.nn.ModuleDict(
                            {
                                "c": torch.nn.ModuleDict(
                                    {
                                        "d": torch.nn.ModuleDict(
                                            {"e": torch.nn.Linear(10, 10, bias=False)}
                                        )
                                    }
                                )
                            }
                        )
                    }
                )

            def forward(self, x):
                return self.a.b.c.d.e(x)

        model = DummyModule()
        opt_model = torch.compile(model)
        x = torch.randn(10, 10)
        opt_model(x)

        from torch._dynamo.eval_frame import _debug_get_cache_entry_list

        cache_entries = _debug_get_cache_entry_list(
            opt_model._torchdynamo_orig_callable.__code__
        )
        code = cache_entries[0].code

        from depyf import decompile

        self.assertRegex(
            decompile(code),
            r"""__compiled_fn_1_.*\(\n.*L_self_modules_a_modules_b_modules_c_modules_d_modules_e_parameters_weight_.*\n.*\(\), x\)""",
        )
        assert torch.allclose(model(x), opt_model(x))


if __name__ == "__main__":
    unittest.main()
