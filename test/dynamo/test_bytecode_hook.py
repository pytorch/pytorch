# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo.test_case


class BytecodeHookTests(torch._dynamo.test_case.TestCase):
    def test_bytecode_hook(self):
        def fn(a, b):
            return a - b * 10

        def hook(code, out_code):
            print(code)
            print(out_code)
            return code

        torch._dynamo.reset()
        handle = torch._dynamo.convert_frame.register_bytecode_hook(hook)
        try:
            opt_fn = torch.compile(fn)
            for i in range(2, 12):
                opt_fn(torch.randn(i), torch.randn(i))
        finally:
            handle.remove()


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
