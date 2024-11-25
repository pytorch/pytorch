# Owner(s): ["module: dynamo"]

import collections
import re
import sys
import time
from io import StringIO

import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.comptime import comptime


# Because we don't support free variables in comptime at the moment,
# we have to communicate via globals.  This also means these tests cannot
# be run in parallel in a single process (not that you'd... ever want
# to do that?)
FILE = None
SELF = None


class ComptimeTests(torch._dynamo.test_case.TestCase):
    def test_print_single(self):
        global FILE
        FILE = StringIO()
        cnt = torch._dynamo.testing.CompileCounter()

        def comptime_print(e):
            @comptime
            def _(ctx):
                ctx.print(ctx.get_local("e"), file=FILE)

        Employee = collections.namedtuple("Employee", ["name", "id"])

        class mylist(list):
            pass

        @torch.compile(backend=cnt, dynamic=True)
        def f(x):
            y = x * 2
            comptime_print(y)
            comptime_print(2)
            comptime_print([y, 2])
            comptime_print((y, 2))
            comptime_print({"foo": y})
            comptime_print(range(1, 3))
            comptime_print(Employee("foo", 2))
            comptime_print(mylist([1, 2]))
            comptime_print(collections.defaultdict(lambda: None))
            comptime_print(set())
            comptime_print({"a", "b"})
            comptime_print(x.size(0))
            return y + 3

        f(torch.randn(2))
        self.assertEqual(cnt.frame_count, 1)
        self.assertExpectedInline(
            FILE.getvalue().strip(),
            """\
FakeTensor(..., size=(s0,))
2
[FakeTensor(..., size=(s0,)), 2]
(FakeTensor(..., size=(s0,)), 2)
{'foo': FakeTensor(..., size=(s0,))}
range(1, 3, 1)
Employee(name='foo', id=2)
[1, 2]
defaultdict(NestedUserFunctionVariable(), {})
set()
{'a','b'}
s0""",
        )

    def test_print_graph(self):
        global FILE
        FILE = StringIO()
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def f(x):
            y = x * 2

            @comptime
            def _(ctx):
                ctx.print_graph(verbose=False, file=FILE)

            # Test the compact notation doesn't error or graph break;
            # you'll have to visually inspect to see that it printed
            comptime.print_graph()

            return y + 3

        f(torch.randn(2))
        self.assertEqual(cnt.frame_count, 1)
        self.assertExpectedInline(
            FILE.getvalue().strip(),
            """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    y = l_x_ * 2;  l_x_ = y = None""",
        )

    def test_print_disas(self):
        global FILE
        FILE = StringIO()
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def f(x):
            y = x * 2

            @comptime
            def _(ctx):
                ctx.print_disas(file=FILE)

            comptime.print_disas()

            return y + 3

        def munge_disas(s):
            re.sub(
                r"^(?: +\d+)?(?: +(-->)) \+\d+ ([A-Za-z0-9_]+)",
                "\1 \3",
                s,
                flags=re.MULTILINE,
            )

        f(torch.randn(2))
        self.assertEqual(cnt.frame_count, 1)
        out = FILE.getvalue()
        # Check that the instruction offset is working
        self.assertIn("-->", out)
        # Check that the bytecode resembles what we expect
        self.assertIn("STORE_FAST", out)
        if sys.version_info < (3, 11):
            self.assertIn("BINARY_MULTIPLY", out)
        else:
            self.assertIn("BINARY_OP", out)

    def test_print_value_stack(self):
        global FILE
        FILE = StringIO()
        cnt = torch._dynamo.testing.CompileCounter()

        def g(x):
            @comptime
            def _(ctx):
                ctx.print_value_stack(file=FILE, stacklevel=1)

            return x

        @torch.compile(backend=cnt)
        def f(x):
            y = x + g(x)

            return y + comptime.print_value_stack_and_return(y * 2)

        f(torch.randn(2))
        self.assertEqual(cnt.frame_count, 1)
        self.assertExpectedInline(
            FILE.getvalue(),
            """\
- FakeTensor(..., size=(2,))
""",
        )

    def test_print_locals(self):
        global FILE
        FILE = StringIO()
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def f(x):
            y = x * 2

            @comptime
            def _(ctx):
                ctx.print_locals(file=FILE)

            comptime.print_locals()

            return y + 3

        f(torch.randn(2))
        self.assertEqual(cnt.frame_count, 1)
        self.assertExpectedInline(
            FILE.getvalue(),
            """\
x = FakeTensor(..., size=(2,))
y = FakeTensor(..., size=(2,))
""",
        )

    # Just make sure it doesn't crash
    def test_print_direct(self):
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def f(x, z):
            y = x * 2
            lambda: z
            comptime.print(z)
            return y + 3

        f(torch.randn(2), torch.randn(2))

    def test_sleep(self):
        sleep_time = 5
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def f(x, z, should_sleep):
            if should_sleep:
                comptime.sleep(sleep_time)
            y = x * 2
            return y + 3

        start = time.time()
        f(torch.randn(2), torch.randn(2), False)
        total_no_sleep = time.time() - start

        start = time.time()
        f(torch.randn(2), torch.randn(2), True)
        total_with_sleep = time.time() - start

        self.assertTrue(total_with_sleep > sleep_time)
        # Hopefully this won't be flaky
        self.assertTrue(abs(total_with_sleep - sleep_time - total_no_sleep) < 3)

    # Just make sure it doesn't crash
    def test_get_local_closure_variable(self):
        global SELF
        SELF = self
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def f(x):
            z = 3

            def g():
                @comptime
                def _(ctx):
                    r = ctx.get_local("z")
                    SELF.assertEqual(repr(r), "3")

                comptime.print(z)
                return 2

            y = x * g()
            return y + 3

        f(torch.randn(2))

    def test_print_bt(self):
        global FILE
        FILE = StringIO()
        cnt = torch._dynamo.testing.CompileCounter()

        def g(x):
            @comptime
            def _(ctx):
                ctx.print_bt(file=FILE)

            comptime.print_bt()

            return x + 3

        @torch.compile(backend=cnt)
        def f(x):
            y = x * 2
            y = g(y)
            return y + 3

        def munge_filenames(s):
            return re.sub(r'File "[^"]+", line \d+', 'File "X", line X', s)

        f(torch.randn(2))
        self.assertEqual(cnt.frame_count, 1)
        bt = FILE.getvalue()
        self.assertIn("y = g(y)", bt)

    def test_print_guards(self):
        global FILE
        FILE = StringIO()
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def f(x):
            y = x * 2

            @comptime
            def _(ctx):
                ctx.print_guards(file=FILE)

            comptime.print_guards()

            return y + 3

        f(torch.randn(2))
        self.assertEqual(cnt.frame_count, 1)
        self.assertExpectedInline(
            re.sub(r"\s+$", "", FILE.getvalue().rstrip(), flags=re.MULTILINE),
            """\

        local "L['x']" TENSOR_MATCH
        {
            'guard_types': None,
            'code': None,
            'obj_weakref': None
            'guarded_class': None
        }
        global '' GRAD_MODE
        {
            'guard_types': None,
            'code': None,
            'obj_weakref': None
            'guarded_class': None
        }
        global '' DETERMINISTIC_ALGORITHMS
        {
            'guard_types': None,
            'code': None,
            'obj_weakref': None
            'guarded_class': None
        }
        global '' TORCH_FUNCTION_STATE
        {
            'guard_types': None,
            'code': None,
            'obj_weakref': None
            'guarded_class': None
        }
        global '' DEFAULT_DEVICE
        {
            'guard_types': None,
            'code': None,
            'obj_weakref': None
            'guarded_class': None
        }
        shape_env '' SHAPE_ENV
        {
            'guard_types': None,
            'code': None,
            'obj_weakref': None
            'guarded_class': None
        }""",
        )

    def test_graph_break(self):
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def f(x):
            y = x * 2

            @comptime
            def _(ctx):
                pass

            return y + 3

        f(torch.randn(2))
        self.assertEqual(cnt.frame_count, 1)
        cnt.frame_count = 0

        @torch.compile(backend=cnt)
        def g(x):
            y = x * 2

            @comptime
            def _(ctx):
                ctx.graph_break()

            y = y + 2

            comptime.graph_break()

            return y * 3

        g(torch.randn(2))
        self.assertEqual(cnt.frame_count, 3)

    def test_get_local(self):
        global SELF, FILE
        SELF = self
        FILE = StringIO()
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def f(x):
            y = x * 2
            lit = 2

            @comptime
            def _(ctx):
                y = ctx.get_local("y")
                SELF.assertEqual(y.as_fake().size(0), 2)
                SELF.assertEqual(y.size(0), 2)
                # Trigger a graph write (TODO: this is not so
                # useful right now as there's no way to make use
                # of the output proxy; maybe it's useful for inserting
                # side-effectful operations into the graph)
                y.as_proxy() + 4
                ctx.print_graph(verbose=False, file=FILE)
                SELF.assertIs(y.python_type(), torch.Tensor)
                lit = ctx.get_local("lit")
                SELF.assertEqual(lit.as_python_constant(), 2)

            return y + 3

        f(torch.randn(2))
        self.assertEqual(cnt.frame_count, 1)
        self.assertExpectedInline(
            FILE.getvalue().strip(),
            """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    y = l_x_ * 2;  l_x_ = None
    add = y + 4;  y = add = None""",
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
