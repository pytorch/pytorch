---
name: pyrefly-type-coverage
description: Migrate a file to use stricter Pyrefly type checking with annotations required for all functions, classes, and attributes.
---

# Pyrefly Type Coverage Skill

This skill guides you through improving type coverage in Python files using Pyrefly, Meta's type checker. Follow this systematic process to add proper type annotations to files.

## Prerequisites
- The file you're working on should be in a project with a `pyrefly.toml` configuration

## Step-by-Step Process

### Step 1: Remove Ignore Errors Directive

First, locate and remove any `pyre-ignore-all-errors` comments at the top of the file:

```python
# REMOVE lines like these:
# pyre-ignore-all-errors
# pyre-ignore-all-errors[16,21,53,56]
# @lint-ignore-every PYRELINT
```

These directives suppress type checking for the entire file and must be removed to enable proper type coverage.

### Step 2: Add Entry to pyrefly.toml

Add a sub-config entry for stricter type checking. Open `pyrefly.toml` and add an entry following this pattern:

```toml
[[sub-config]]
matches = "path/to/your/file.py"
[sub-config.errors]
implicit-import = false
implicit-any = true
```

For directory-level coverage:
```toml
[[sub-config]]
matches = "path/to/directory/**"
[sub-config.errors]
implicit-import = false
implicit-any = true
```

You can also enable stricter options as needed:
```toml
[[sub-config]]
matches = "path/to/your/file.py"
[sub-config.errors]
implicit-import = false
implicit-any = true
# Uncomment these for stricter checking:
# unannotated-attribute = true
# unannotated-parameter = true
# unannotated-return = true
```

### Step 3: Run Pyrefly to Identify Missing Coverage

Execute the type checker to see all type errors:

```bash
pyrefly check <FILENAME>
```

Example:
```bash
pyrefly check torch/_dynamo/utils.py
```

This will output a list of type errors with line numbers and descriptions. Common error types include:
- Missing return type annotations
- Missing parameter type annotations
- Incompatible types
- Missing attribute definitions
- Implicit `Any` usage

**CRITICAL**: Your goal is to resolve all errors. If you cannot resolve an error, you can use `# pyrefly: ignore[...]` to suppress but you should try to resolve the error first

### Step 4: Add Type Annotations

Work through each error systematically:

1. **Read the function/code carefully** - Understand what the function does
2. **Examine usage patterns** - Look at how the function is called to understand expected types
3. **Add appropriate annotations** - Add type hints based on your analysis

#### Common Annotation Patterns

**Function signatures:**
```python
# Before
def process_data(items, callback):
    ...

# After
from collections.abc import Callable
def process_data(items: list[str], callback: Callable[[str], bool]) -> None:
    ...
```

**Class attributes:**
```python
# Before
class MyClass:
    def __init__(self):
        self.value = None
        self.items = []

# After
class MyClass:
    value: int | None
    items: list[str]

    def __init__(self) -> None:
        self.value = None
        self.items = []
```

**Complex types:**
**CRITICAL**: use syntax for Python >3.10 and prefer collections.abc as opposed to
typing for better code standards.

**Critical**: For more advanced/generic types such as `TypeAlias`, `TypeVar`, `Generic`, `Protocol`, etc. use `typing_extensions`

```python

# Optional values
def get_value(key: str) -> int | None: ...

# Union types
def process(value: str | int) -> str: ...

# Dict and List
def transform(data: dict[str, list[int]]) -> list[str]: ...

# Callable
from collections.abc import Callable
def apply(func: Callable[[int, int], int], a: int, b: int) -> int: ...

# TypeVar for generics
from typing_extensions import TypeVar
T = TypeVar('T')
def first(items: list[T]) -> T: ...
```

**Using `# pyre-ignore` for specific lines:**

If a specific line is difficult to type correctly (e.g., dynamic metaprogramming), you can ignore just that line:

```python
# pyrefly: ignore[attr-defined]
result = getattr(obj, dynamic_name)()
```

**CRITICAL**: Avoid using `# pyre-ignore` unless it is necessary.
When possible, we can implement stubs, or refactor code to make it more type-safe.

### Step 5: Iterate and Verify

After adding annotations:

1. **Re-run pyrefly check** to verify errors are resolved:
   ```bash
   pyrefly check <FILENAME>
   ```

2. **Fix any new errors** that may appear from the annotations you added

3. **Repeat until clean** - Continue until pyrefly reports no errors


### Step 6: Commit Changes
To keep type coverage PRs manageable, you should commit your change once finished with a file.

## Tips for Success

1. **Start with function signatures** - Return types and parameter types are usually the highest priority

2. **Use `from __future__ import annotations`** - Add this at the top of the file for forward references:
   ```python
   from __future__ import annotations
   ```

3. **Leverage type inference** - Pyrefly can infer many types; focus on function boundaries

4. **Check existing type stubs** - For external libraries, check if type stubs exist

5. **Use `typing_extensions` for newer features** - For compatibility:
   ```python
   from typing_extensions import TypeAlias, Self, ParamSpec
   ```

6. **Document complex types with TypeAlias**:
   ```python
   from typing import Dict, List, TypeAlias

   ConfigType: TypeAlias = Dict[str, List[int]]

   def process_config(config: ConfigType) -> None: ...
   ```

## Example Workflow

```bash
# 1. Open the file and remove pyre-ignore-all-errors
# 2. Add entry to pyrefly.toml

# 3. Check initial errors
pyrefly check torch/my_module.py

# 4. Add annotations iteratively

# 5. Re-check after changes
pyrefly check torch/my_module.py

# 6. Repeat until clean
```
