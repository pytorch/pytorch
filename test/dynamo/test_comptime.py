import torch._dynamo.test_case
from torch._dynamo.eval_frame import comptime
import torch._dynamo.testing
from io import StringIO
import re

# Because we don't support free variables in comptime at the moment,
# we have to communicate via globals.  This also means these tests cannot
# be run in parallel in a single process (not that you'd... ever want
# to do that?)
FILE = None
SELF = None

class ComptimeTests(torch._dynamo.test_case.TestCase):
    def test_print_graph(self):
        global FILE
        FILE = StringIO()

        @torch._dynamo.optimize("eager")
        def f(x):
            y = x * 2

            @comptime
            def _(ctx):
                ctx.print_graph(verbose=False, file=FILE)

            return y + 3

        f(torch.randn(2))
        self.assertExpectedInline(FILE.getvalue().strip(), """\
def forward(self, x : torch.Tensor):
    mul = x * 2;  x = None""")

    def test_print_disas(self):
        global FILE
        FILE = StringIO()

        @torch._dynamo.optimize("eager")
        def f(x):
            y = x * 2

            @comptime
            def _(ctx):
                ctx.print_disas(file=FILE)

            return y + 3

        def munge_disas(s):
            re.sub(
                r'^(?: +\d+)?(?: +(-->)) \+\d+ ([A-Za-z0-9_]+)',
                '\1 \3',
                s,
                flags=re.MULTILINE
            )

        f(torch.randn(2))
        out = FILE.getvalue()
        # Check that the instruction offset is working
        self.assertIn("-->", out)
        # Check that the bytecode resembles what we expect
        self.assertIn("STORE_FAST", out)
        self.assertIn("BINARY_MULTIPLY", out)

    def test_print_locals(self):
        global FILE
        FILE = StringIO()

        @torch._dynamo.optimize("eager")
        def f(x):
            y = x * 2

            @comptime
            def _(ctx):
                ctx.print_locals(file=FILE)

            return y + 3

        f(torch.randn(2))
        self.assertExpectedInline(FILE.getvalue(), """\
x = TensorVariable()
y = TensorVariable()
""")

    def test_print_bt(self):
        global FILE
        FILE = StringIO()

        def g(x):
            @comptime
            def _(ctx):
                ctx.print_bt(file=FILE)

            return x + 3

        @torch._dynamo.optimize("eager")
        def f(x):
            y = x * 2
            y = g(y)
            return y + 3

        def munge_filenames(s):
            return re.sub(r'File "[^"]+", line \d+', 'File "X", line X', s)

        f(torch.randn(2))
        self.assertExpectedInline(munge_filenames(FILE.getvalue()), """\
  File "X", line X, in f
    y = g(y)
  File "X", line X, in g
    def _(ctx):

""")

    def test_print_guards(self):
        global FILE
        FILE = StringIO()

        @torch._dynamo.optimize("eager")
        def f(x):
            y = x * 2

            @comptime
            def _(ctx):
                ctx.print_guards(file=FILE)

            return y + 3

        f(torch.randn(2))
        self.assertExpectedInline(FILE.getvalue().rstrip(), """\
-
            local 'x' TENSOR_MATCH
            {
                'guard_types': None,
                'code': None,
                'obj_weakref': None
                'guarded_class': None
            }""")

    def test_graph_break(self):
        cnt = torch._dynamo.testing.CompileCounter()

        @torch._dynamo.optimize(cnt)
        def f(x):
            y = x * 2

            @comptime
            def _(ctx):
                pass

            return y + 3

        f(torch.randn(2))
        self.assertEqual(cnt.frame_count, 1)
        cnt.frame_count = 0

        @torch._dynamo.optimize(cnt)
        def g(x):
            y = x * 2

            @comptime
            def _(ctx):
                ctx.graph_break()

            return y + 3

        g(torch.randn(2))
        self.assertEqual(cnt.frame_count, 2)

    def test_get_local(self):
        global SELF, FILE
        SELF = self
        FILE = StringIO()

        @torch._dynamo.optimize("eager")
        def f(x):
            y = x * 2
            lit = 2

            @comptime
            def _(ctx):
                y = ctx.get_local('y')
                SELF.assertEqual(y.as_fake().size(0), 2)
                # Trigger a graph write (TODO: this is not so
                # useful right now as there's no way to make use
                # of the output proxy; maybe it's useful for inserting
                # side-effectful operations into the graph)
                y.as_proxy() + 4
                ctx.print_graph(verbose=False, file=FILE)
                SELF.assertIs(y.python_type(), torch.Tensor)
                lit = ctx.get_local('lit')
                SELF.assertEqual(lit.as_python_constant(), 2)

            return y + 3

        f(torch.randn(2))
        self.assertExpectedInline(FILE.getvalue().strip(), """\
def forward(self, x : torch.Tensor):
    mul = x * 2;  x = None
    add = mul + 4;  mul = None""")


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
