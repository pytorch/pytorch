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

None:10:15: list()/dict() is slow! Use []/{} instead.
    8 |     x = list()
    9 |     x = dict()
   10 |     x = list  (  )
                      ^
   11 |     x = dict  (  )

None:10:18: list()/dict() is slow! Use []/{} instead.
    8 |     x = list()
    9 |     x = dict()
   10 |     x = list  (  )
                         ^
   11 |     x = dict  (  )

None:11:9: list()/dict() is slow! Use []/{} instead.
    9 |     x = dict()
   10 |     x = list  (  )
   11 |     x = dict  (  )
                ^^^^

None:11:15: list()/dict() is slow! Use []/{} instead.
    9 |     x = dict()
   10 |     x = list  (  )
   11 |     x = dict  (  )
                      ^

None:11:18: list()/dict() is slow! Use []/{} instead.
    9 |     x = dict()
   10 |     x = list  (  )
   11 |     x = dict  (  )
                         ^
""",
        )

    def test_not_linty(self) -> None:
        output, replacement = lint_function(not_linty_code)
        self.assertFalse(output.strip())


if __name__ == "__main__":
    unittest.main()
