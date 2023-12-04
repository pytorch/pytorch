# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo.test_case


def target(x):
    # one argument (x)
    # two local variables (y, z)
    y = 2
    z = 1
    return x - y + z


def transformed_code1(x):
    # one argument (x)
    # one local variables (y,)
    # constant folding z = 1
    y = 2
    return x - y + 1


def transformed_code2(x):
    # one argument (x)
    # three local variables (y, z, another)
    # expand a constant: another = 0
    y = 2
    z = 1
    another = 0
    return x - y + z - another


class FrameInitTests(torch._dynamo.test_case.TestCase):
    def test_frame_init(self):
        code1 = transformed_code1.__code__
        code2 = transformed_code2.__code__

        def callback1(frame, cache_entry, frame_state):
            if frame.f_code.co_name == "target":
                return torch._dynamo.types.GuardedCode(code1, lambda f_locals: True)
            return None

        def callback2(frame, cache_entry, frame_state):
            if frame.f_code.co_name == "target":
                return torch._dynamo.types.GuardedCode(code2, lambda f_locals: True)
            return None

        torch._dynamo.reset()
        original = torch._dynamo.eval_frame.set_eval_frame(callback1)
        self.assertEqual(target(5), 4)
        torch._dynamo.eval_frame.set_eval_frame(original)

        torch._dynamo.reset()
        original = torch._dynamo.eval_frame.set_eval_frame(callback2)
        self.assertEqual(target(5), 4)
        torch._dynamo.eval_frame.set_eval_frame(original)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
