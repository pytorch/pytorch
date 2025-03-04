from __future__ import annotations

import unittest
from io import StringIO

from expecttest import TestCase
from tools.linter.adapters._linter import PythonFile
from tools.linter.adapters.perf_linter import PerfLinter


linty_code = """\
def linty_code() -> None:
    x = ",".join(str(x) for x in range(10))
    x = list(x)
    x = list(range(2))
    x = list(a.b())
    x = list(x + x)
    x = list(x or y)
    x = list()
    x = dict()
    x = list  (  )
    x = dict  (  )
    if x in (1, 2, 3):
        pass
    if x in [1, 2, 3]:
        pass
    if x in [1, 2, y]:
        pass
    if x in {1, OrderedSet, 3}:
        pass
    if node.op in ("placeholder", "get_attr", "output"):
        pass
"""


not_linty_code = """\
def not_linty_code() -> None:
    x = ",".join([str(x) for x in range(10)])
    x = [range(2)]
    x = dict
    x = dict(a=1)
    x = list
    x = any(x == 0 for x in range(10))
    x = all(x == 0 for x in range(10))
    x = next(x == 0 for x in range(10))
    x = next(
        (x == 0 for x in range(10))
    )
    self.x = (x == 0 for x in range(10))
    if x in (1, 2, y):
        pass
    if x in {1, True, False, None, ..., "x", 1.5, -1}:
        pass
    if x in [1] + y:
        pass
    res = x in [1, 2, 3]
"""


def lint_function(code: str) -> tuple[str, str]:
    buf = StringIO()
    linter = PerfLinter([])
    pf = PythonFile(
        PerfLinter.linter_name,
        contents=code,
    )
    replacement, results = linter._replace(pf)
    print(*linter._display(pf, results), sep="\n", file=buf)
    return buf.getvalue(), replacement


class TestSetLinter(TestCase):
    def test_linty(self) -> None:
        output, replacement = lint_function(linty_code)
        self.assertExpectedInline(
            replacement,
            """\
def linty_code() -> None:
    x = ",".join([str(x) for x in range(10)])
    x = [*x]
    x = [*range(2)]
    x = [*a.b()]
    x = [*(x + x)]
    x = [*(x or y)]
    x = []
    x = {}
    x =   [  ]
    x =   {  }
    if x in {1, 2, 3}:
        pass
    if x in {1, 2, 3}:
        pass
    if x in (1, 2, y):
        pass
    if x in (1, OrderedSet, 3):
        pass
    if node.op in {"placeholder", "get_attr", "output"}:
        pass
""",
        )
        self.assertExpectedInline(
            output,
            """\
None:2:17: Generators are slow! Use list comprehensions instead.
    1 | def linty_code() -> None:
    2 |     x = ",".join(str(x) for x in range(10))
                        ^
    3 |     x = list(x)
    4 |     x = list(range(2))
    5 |     x = list(a.b())

None:2:43: Generators are slow! Use list comprehensions instead.
    1 | def linty_code() -> None:
    2 |     x = ",".join(str(x) for x in range(10))
                                                  ^
    3 |     x = list(x)
    4 |     x = list(range(2))
    5 |     x = list(a.b())

None:3:9: list(x) is slow! Use [*x] instead.
    1 | def linty_code() -> None:
    2 |     x = ",".join(str(x) for x in range(10))
    3 |     x = list(x)
                ^^^^
    4 |     x = list(range(2))
    5 |     x = list(a.b())

None:3:13: list(x) is slow! Use [*x] instead.
    1 | def linty_code() -> None:
    2 |     x = ",".join(str(x) for x in range(10))
    3 |     x = list(x)
                    ^
    4 |     x = list(range(2))
    5 |     x = list(a.b())

None:3:15: list(x) is slow! Use [*x] instead.
    1 | def linty_code() -> None:
    2 |     x = ",".join(str(x) for x in range(10))
    3 |     x = list(x)
                      ^
    4 |     x = list(range(2))
    5 |     x = list(a.b())

None:4:9: list(x) is slow! Use [*x] instead.
    2 |     x = ",".join(str(x) for x in range(10))
    3 |     x = list(x)
    4 |     x = list(range(2))
                ^^^^
    5 |     x = list(a.b())
    6 |     x = list(x + x)

None:4:13: list(x) is slow! Use [*x] instead.
    2 |     x = ",".join(str(x) for x in range(10))
    3 |     x = list(x)
    4 |     x = list(range(2))
                    ^
    5 |     x = list(a.b())
    6 |     x = list(x + x)

None:4:22: list(x) is slow! Use [*x] instead.
    2 |     x = ",".join(str(x) for x in range(10))
    3 |     x = list(x)
    4 |     x = list(range(2))
                             ^
    5 |     x = list(a.b())
    6 |     x = list(x + x)

None:5:9: list(x) is slow! Use [*x] instead.
    3 |     x = list(x)
    4 |     x = list(range(2))
    5 |     x = list(a.b())
                ^^^^
    6 |     x = list(x + x)
    7 |     x = list(x or y)

None:5:13: list(x) is slow! Use [*x] instead.
    3 |     x = list(x)
    4 |     x = list(range(2))
    5 |     x = list(a.b())
                    ^
    6 |     x = list(x + x)
    7 |     x = list(x or y)

None:5:19: list(x) is slow! Use [*x] instead.
    3 |     x = list(x)
    4 |     x = list(range(2))
    5 |     x = list(a.b())
                          ^
    6 |     x = list(x + x)
    7 |     x = list(x or y)

None:6:9: list(x) is slow! Use [*x] instead.
    4 |     x = list(range(2))
    5 |     x = list(a.b())
    6 |     x = list(x + x)
                ^^^^
    7 |     x = list(x or y)
    8 |     x = list()

None:6:13: list(x) is slow! Use [*x] instead.
    4 |     x = list(range(2))
    5 |     x = list(a.b())
    6 |     x = list(x + x)
                    ^
    7 |     x = list(x or y)
    8 |     x = list()

None:6:19: list(x) is slow! Use [*x] instead.
    4 |     x = list(range(2))
    5 |     x = list(a.b())
    6 |     x = list(x + x)
                          ^
    7 |     x = list(x or y)
    8 |     x = list()

None:7:9: list(x) is slow! Use [*x] instead.
    5 |     x = list(a.b())
    6 |     x = list(x + x)
    7 |     x = list(x or y)
                ^^^^
    8 |     x = list()
    9 |     x = dict()

None:7:13: list(x) is slow! Use [*x] instead.
    5 |     x = list(a.b())
    6 |     x = list(x + x)
    7 |     x = list(x or y)
                    ^
    8 |     x = list()
    9 |     x = dict()

None:7:20: list(x) is slow! Use [*x] instead.
    5 |     x = list(a.b())
    6 |     x = list(x + x)
    7 |     x = list(x or y)
                           ^
    8 |     x = list()
    9 |     x = dict()

None:8:9: list()/dict() is slow! Use []/{} instead.
    6 |     x = list(x + x)
    7 |     x = list(x or y)
    8 |     x = list()
                ^^^^
    9 |     x = dict()
   10 |     x = list  (  )

None:8:13: list()/dict() is slow! Use []/{} instead.
    6 |     x = list(x + x)
    7 |     x = list(x or y)
    8 |     x = list()
                    ^
    9 |     x = dict()
   10 |     x = list  (  )

None:8:14: list()/dict() is slow! Use []/{} instead.
    6 |     x = list(x + x)
    7 |     x = list(x or y)
    8 |     x = list()
                     ^
    9 |     x = dict()
   10 |     x = list  (  )

None:9:9: list()/dict() is slow! Use []/{} instead.
    7 |     x = list(x or y)
    8 |     x = list()
    9 |     x = dict()
                ^^^^
   10 |     x = list  (  )
   11 |     x = dict  (  )

None:9:13: list()/dict() is slow! Use []/{} instead.
    7 |     x = list(x or y)
    8 |     x = list()
    9 |     x = dict()
                    ^
   10 |     x = list  (  )
   11 |     x = dict  (  )

None:9:14: list()/dict() is slow! Use []/{} instead.
    7 |     x = list(x or y)
    8 |     x = list()
    9 |     x = dict()
                     ^
   10 |     x = list  (  )
   11 |     x = dict  (  )

None:10:9: list()/dict() is slow! Use []/{} instead.
    8 |     x = list()
    9 |     x = dict()
   10 |     x = list  (  )
                ^^^^
   11 |     x = dict  (  )
   12 |     if x in (1, 2, 3):

None:10:15: list()/dict() is slow! Use []/{} instead.
    8 |     x = list()
    9 |     x = dict()
   10 |     x = list  (  )
                      ^
   11 |     x = dict  (  )
   12 |     if x in (1, 2, 3):

None:10:18: list()/dict() is slow! Use []/{} instead.
    8 |     x = list()
    9 |     x = dict()
   10 |     x = list  (  )
                         ^
   11 |     x = dict  (  )
   12 |     if x in (1, 2, 3):

None:11:9: list()/dict() is slow! Use []/{} instead.
    9 |     x = dict()
   10 |     x = list  (  )
   11 |     x = dict  (  )
                ^^^^
   12 |     if x in (1, 2, 3):
   13 |         pass

None:11:15: list()/dict() is slow! Use []/{} instead.
    9 |     x = dict()
   10 |     x = list  (  )
   11 |     x = dict  (  )
                      ^
   12 |     if x in (1, 2, 3):
   13 |         pass

None:11:18: list()/dict() is slow! Use []/{} instead.
    9 |     x = dict()
   10 |     x = list  (  )
   11 |     x = dict  (  )
                         ^
   12 |     if x in (1, 2, 3):
   13 |         pass

None:12:13: `in (...)` is slower than `in {...}` for constant sets, set becomes a code constant.
   10 |     x = list  (  )
   11 |     x = dict  (  )
   12 |     if x in (1, 2, 3):
                    ^
   13 |         pass
   14 |     if x in [1, 2, 3]:

None:12:21: `in (...)` is slower than `in {...}` for constant sets, set becomes a code constant.
   10 |     x = list  (  )
   11 |     x = dict  (  )
   12 |     if x in (1, 2, 3):
                            ^
   13 |         pass
   14 |     if x in [1, 2, 3]:

None:14:13: `in (...)` is slower than `in {...}` for constant sets, set becomes a code constant.
   12 |     if x in (1, 2, 3):
   13 |         pass
   14 |     if x in [1, 2, 3]:
                    ^
   15 |         pass
   16 |     if x in [1, 2, y]:

None:14:21: `in (...)` is slower than `in {...}` for constant sets, set becomes a code constant.
   12 |     if x in (1, 2, 3):
   13 |         pass
   14 |     if x in [1, 2, 3]:
                            ^
   15 |         pass
   16 |     if x in [1, 2, y]:

None:16:13: `in {...}` is slower than `in (...)` for non-constant sets, set must be built every time.
   14 |     if x in [1, 2, 3]:
   15 |         pass
   16 |     if x in [1, 2, y]:
                    ^
   17 |         pass
   18 |     if x in {1, OrderedSet, 3}:

None:16:21: `in {...}` is slower than `in (...)` for non-constant sets, set must be built every time.
   14 |     if x in [1, 2, 3]:
   15 |         pass
   16 |     if x in [1, 2, y]:
                            ^
   17 |         pass
   18 |     if x in {1, OrderedSet, 3}:

None:18:13: `in {...}` is slower than `in (...)` for non-constant sets, set must be built every time.
   16 |     if x in [1, 2, y]:
   17 |         pass
   18 |     if x in {1, OrderedSet, 3}:
                    ^
   19 |         pass
   20 |     if node.op in ("placeholder", "get_attr", "output"):

None:18:30: `in {...}` is slower than `in (...)` for non-constant sets, set must be built every time.
   16 |     if x in [1, 2, y]:
   17 |         pass
   18 |     if x in {1, OrderedSet, 3}:
                                     ^
   19 |         pass
   20 |     if node.op in ("placeholder", "get_attr", "output"):

None:20:19: `in (...)` is slower than `in {...}` for constant sets, set becomes a code constant.
   18 |     if x in {1, OrderedSet, 3}:
   19 |         pass
   20 |     if node.op in ("placeholder", "get_attr", "output"):
                          ^
   21 |         pass

None:20:55: `in (...)` is slower than `in {...}` for constant sets, set becomes a code constant.
   18 |     if x in {1, OrderedSet, 3}:
   19 |         pass
   20 |     if node.op in ("placeholder", "get_attr", "output"):
                                                              ^
   21 |         pass
""",
        )

    def test_not_linty(self) -> None:
        output, replacement = lint_function(not_linty_code)
        self.assertEqual(output.strip(), "")


if __name__ == "__main__":
    unittest.main()
