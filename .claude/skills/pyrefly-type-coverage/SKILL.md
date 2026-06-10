---
name: pyrefly-type-coverage
description: Migrate a file to use stricter Pyrefly type checking with annotations required for all functions, classes, and attributes.
---

# Pyrefly Type Coverage Skill

## Prerequisites
- The file must live in a project with a `pyrefly.toml`.
- `pyrefly`, `lintrunner`, and the project's test runner must be on PATH. **If any
  are missing, stop and ask whether a conda environment needs activating** — don't
  install or substitute (per repo CLAUDE.md).

### Step 1: Remove file-level type-check suppressions

Delete any of these from the top of the file (pyrefly honors `# mypy: ignore-errors`
for mypy compat, so that one must go too):

```python
# pyre-ignore-all-errors
# pyre-ignore-all-errors[16,21,53,56]
# @lint-ignore-every PYRELINT
# mypy: ignore-errors
```

### Step 2: Add a sub-config entry to `pyrefly.toml`

```toml
[[sub-config]]
matches = "path/to/directory/**"
[sub-config.errors]
implicit-import = false
implicit-any = true
bad-param-name-override = false
unannotated-return = true
unannotated-parameter = true
```

**IMPORTANT**: Setting any error key in `[sub-config.errors]` overrides only that key
relative to the parent — but enabling `unannotated-return` / `unannotated-parameter` /
`implicit-any` will resurface errors that were previously hidden file-wide. If you see
unrelated errors (e.g., `bad-param-name-override`) flooding the output, mirror the
parent config's setting for that key in the sub-config to silence them.

### Step 3: Run pyrefly

```bash
pyrefly check <FILENAME>
```

**Goal:** resolve all `unannotated-return`, `unannotated-parameter`, and `implicit-any`
errors by adding annotations — see Step 4's ladder. These three target categories are
always resolvable; **never** suppress them with `# pyrefly: ignore`. The single
exception is `@compatibility(is_backward_compatible=True)` (Step 4).

Other categories (`bad-argument-type`, `missing-attribute`, …) are real type bugs.
Handle them by where pyrefly reports them:

- **Reported in another file** (path != target): leave it. Don't widen scope. If
  the error is now blocking the target, suppress at the report site with
  `# pyrefly: ignore[<category>]  # TODO`.
- **Reported in the target file but the message names a symbol defined elsewhere**
  (e.g., `bad-return` because an imported function's annotation is wrong):
  suppress locally with the same TODO comment. Don't invent a `cast()` that
  papers over the upstream gap.
- **Reported in the target file, originates locally**: fix it.

Use `# pyrefly: ignore[...]` only as a last resort, and only on non-target categories.

### Step 4: Add annotations

Examine call sites when the right type isn't obvious from the function body.

#### Annotation conventions

- Use PEP 604 / PEP 585 syntax (`int | None`, `list[str]`) — assume Python >= 3.10.
- Prefer `collections.abc` over `typing` for ABCs (`Callable`, `Sequence`, `Generator`, ...).
- For generic helpers, import from `typing` when available on the project's minimum
  Python version, and from `typing_extensions` only when you need a newer feature
  (e.g., `Self` and `override` if supporting < 3.11/3.12, or PEP 696 `default=` for
  `TypeVar` / `ParamSpec`). Don't blanket-import from `typing_extensions`.
- Always parameterize `Callable` — `Callable[..., Any]` when the signature is
  genuinely unknown, never bare `Callable`. (See ParamSpec below for the
  signature-preserving wrapper case.)
- Class attributes assigned in `__init__` should get a class-level annotation so pyrefly can see them.
- Break import cycles with `if TYPE_CHECKING:` — annotation-only imports go inside the
  guard, and use `from __future__ import annotations` (or string forward refs) so
  runtime imports stay lazy:
  ```python
  from __future__ import annotations
  from typing import TYPE_CHECKING
  if TYPE_CHECKING:
      from torch.fx import GraphModule
  def transform(gm: GraphModule) -> GraphModule: ...
  ```
- **Never suppress the three target categories.** `unannotated-return`,
  `unannotated-parameter`, and `implicit-any` are always resolvable by adding
  an annotation; `# pyrefly: ignore[<one of those>]` is not an acceptable
  outcome. The single exception is the Backward compatibility carve-out below.
- **Widen, don't bail.** When the right type is hard to infer, walk down this
  ladder rather than reaching for an ignore:
  1. Most specific concrete type observable from call sites and return paths.
  2. A union (`X | Y`), `Sequence[X]`-style abstract type, or a bound `TypeVar`
     for genuinely generic functions (identity-passthrough, container helpers).
  3. `object` — strictest fallback that still type-checks. Forces callers to
     narrow before use, e.g., `def serialize(value: object) -> str:`. Visually
     similar to `Any` but stricter — pyrefly rejects `value.foo()` without an
     `isinstance`.
  4. `Any` — last rung. Always preferred over a `# pyrefly: ignore` on a target
     category, but only after rungs 1–3 fail. Be able to articulate why each
     earlier rung doesn't fit (e.g., "union exceeds 8 types", "no observable
     common bound", "callers genuinely never narrow").
- Read at least three call sites before deciding a parameter must be `Any` —
  don't pattern-match "looks dynamic" on the first try.
- Narrow-scope `# pyrefly: ignore[...]` (on a non-target category) is reserved
  for cases where pyrefly is *actually wrong* about a specific local error —
  dynamic metaprogramming, third-party stub gaps:
  ```python
  # pyrefly: ignore[attr-defined]
  result = getattr(obj, dynamic_name)()
  ```

#### Backward compatibility (the one exception to never-suppress)

**CRITICAL**: Functions decorated with `@compatibility(is_backward_compatible=True)`
must NOT have their signatures changed. The backward-compat test
(`test_function_back_compat`) compares stringified `inspect.signature` against a golden
file — adding annotations (even `-> None`) changes that string and the test fails.
Use pyrefly ignore comments instead:

```python
@compatibility(is_backward_compatible=True)
def my_function(  # pyrefly: ignore[unannotated-return]
    self,
    arg1,  # can't add type here either
):
    ...
```

The `# pyrefly: ignore` comment must be on the `def` line (where pyrefly reports the error),
not on the closing `)`.

**ParamSpec for signature-preserving wrappers** (decorators, `functools.wraps`-style
helpers). Use `Callable[P, R]` so the wrapped function's signature flows through
to the caller — `Callable[..., Any]` loses it. Skip ParamSpec if the wrapper
genuinely accepts arbitrary callables. Pair with `Concatenate[X, P]` when the
wrapper prepends or appends args.

```python
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

def log_calls(fn: Callable[P, R]) -> Callable[P, R]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return fn(*args, **kwargs)
    return wrapper
```

### Step 5: Iterate

Re-run `pyrefly check`. New annotations often surface `bad-return` errors where the
function actually returns an incompatible type — fix those. Repeat until clean.

### Step 6: Lint

Required before handing off — annotations frequently shift import order and line
length:

```bash
lintrunner -a <files...>
```

Resolve anything `lintrunner` can't auto-fix manually.

### Step 7: Test

**Precedence when something fails**: tests passing > pyrefly clean > annotation
strictness. If a freshly-added annotation breaks a test, narrow it one rung in
the discipline ladder (e.g., concrete → `object`, or remove an `Any` widening
that broke a downstream `isinstance` check) before reverting the file.

1. **Backward-compat check.** Run iff
   `grep -l '@compatibility(is_backward_compatible=True)' <target>` returns the
   file — the decorator is the actual precondition for the golden file. The
   broader "imports `torch.fx`" heuristic catches half of `torch/`.
   ```bash
   python -m pytest test/test_fx.py::TestFXAPIBackwardCompatibility -x -v
   ```

2. **Unit tests for the modified module.** Search both ways before concluding
   no coverage exists:
   ```bash
   # torch/foo/bar.py is usually covered by test/test_foo.py or test/test_bar.py
   ls test/ | grep -i <module-name>
   # or by import
   grep -rl "from torch.foo.bar import\|import torch.foo.bar" test/
   ```

   If both come up empty, tell the user — don't silently skip. Type changes can
   introduce real runtime regressions (`Optional[X]` vs `X`, `Sequence` vs
   `list` when `.append` is called, etc.).

## Notes

- **Forward refs in class bodies** without `from __future__ import annotations`
  still need string quoting:
  ```python
  class MyClass:
      def __new__(cls) -> "MyClass": ...
  ```
- **Committing**: don't commit unless the user explicitly asks (per repo
  CLAUDE.md). Stop and surface the diff for review when the file is clean.
