# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo.test_case
import torch._dynamo.testing


def fn_creator():
    var1 = 1

    def fn(x):
        x = x + 1
        var2 = 1
        torch._dynamo.graph_break()
        x = x + var1

        def inner_fn():
            return var2

        return x

    return fn


class ResumeFunctionTests(torch._dynamo.test_case.TestCase):
    def test_freevars(self):
        fn = fn_creator()
        opt_fn = torch.compile(fn, backend="eager")
        opt_fn(torch.randn(10))
        entries = [v for k, v in list(globals().items()) if k.startswith("__resume_at")]
        self.assertEqual(len(entries), 1)
        # When freevars are present, install_resume_function_global stores a
        # factory that closes over the code object (first closure cell).
        code = entries[0].__closure__[0].cell_contents
        # co_freevars of resume functions, are sorted concatenation of the original function's co_freevars and co_cellvars
        self.assertEqual(code.co_freevars, ("var1", "var2"))

    def _check_graph_break_in_generated_fn(self, body, *args):
        src = "\n".join(
            [
                "def fn(x, flag):",
                *body,
                "    torch._dynamo.graph_break()",
                "    return x + 1",
            ]
        )
        scope = {"torch": torch}
        exec(src, scope)
        fn = scope["fn"]
        cnt = torch._dynamo.testing.CompileCounter()
        self.assertEqual(torch.compile(fn, backend=cnt)(*args), fn(*args))
        self.assertEqual(cnt.frame_count, 2)

    def test_graph_break_after_many_with_blocks(self):
        body = ["    with torch.no_grad():\n        x = x + 1"] * 400
        self._check_graph_break_in_generated_fn(body, torch.zeros(2), True)

    def test_graph_break_after_many_branches_in_with_block(self):
        body = ["    with torch.no_grad():"]
        body += ["        if flag:\n            x = x + 1"] * 200
        body += ["        x = x * 2"]
        self._check_graph_break_in_generated_fn(body, torch.zeros(2), True)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
