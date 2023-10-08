# Owner(s): ["module: dynamo"]
import unittest

import torch
import torch._dynamo.test_case
from torch._dynamo.utils import istype


class AllowInlineSkipTests(torch._dynamo.test_case.TestCase):
    def test_func_inlinelist(self):
        def fn(x):
            if istype(x, torch.Tensor):
                return x + 1
            else:
                return x - 1

        my_func_inlinelist = torch._dynamo.skipfiles.FUNC_INLINELIST.copy()
        my_func_inlinelist.add("torch._dynamo.utils.istype")

        def my_get_func_inlinelist():
            inlinelist = set()
            for f in my_func_inlinelist:
                inlinelist.add(eval(f).__code__)
            return inlinelist

        self.assertTrue(
            "torch._dynamo.utils" not in torch._dynamo.skipfiles.FILE_INLINELIST
        )
        self.assertTrue(
            "torch._dynamo" not in torch._dynamo.skipfiles.SUBMODULE_INLINELIST
        )

        with unittest.mock.patch(
            "torch._dynamo.skipfiles.get_func_inlinelist",
            my_get_func_inlinelist,
        ):
            x = torch.rand(3)
            opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
            ref = fn(x)
            res = opt_fn(x)
            self.assertEqual(ref, res)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
