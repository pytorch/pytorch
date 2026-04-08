---
name: document-public-apis
description: Document undocumented public APIs in PyTorch by removing functions from coverage_ignore_functions and coverage_ignore_classes in docs/source/conf.py, running Sphinx coverage, and adding the appropriate autodoc directives to the correct .md or .rst doc files. Use when a user asks to remove functions from conf.py ignore lists.
---

# Document Public APIs

This skill documents undocumented public APIs in PyTorch by removing entries from the coverage ignore lists in `docs/source/conf.py` and adding Sphinx autodoc directives (e.g., `autosummary`, `currentmodule`, `autoclass`, `automodule`) to the corresponding `.md` or `.rst` doc source files in `docs/source/`.

**"Documenting" means adding autodoc directives to doc source files — NEVER modifying Python source code.** Do not add or edit docstrings in `.py` files. Do not read or inspect Python source files. Sphinx will pull whatever docstring exists (or render an empty entry if none exists). Your only job is to add the correct directive to the correct doc file.

## Overview

`docs/source/conf.py` contains two lists that suppress Sphinx coverage warnings for undocumented APIs:

- `coverage_ignore_functions`: undocumented functions
- `coverage_ignore_classes`: undocumented classes

Entries are organized by **module comment groups**. Each group has a module label comment followed by the function/class names that belong to that module:

```python
coverage_ignore_functions = [
    # torch.ao.quantization.fx.convert              <-- module label comment
    "convert",                                       # <-- entries belonging to this module
    "convert_custom_module",
    "convert_standalone_module",
    "convert_weighted_module",
    # torch.ao.quantization.fx.fuse                 <-- next module group
    "fuse",
    # torch.nn.functional
    "assert_int_or_pair",  # looks unintentionally public   <-- entry with inline comment
    "constant",  # deprecated                                <-- entry with inline comment
]
```

There are two kinds of comments:
- **Module label comments** (`# torch.ao.quantization.fx.convert`): these label which module the entries below belong to. They appear on their own line before a group of entries.
- **Inline comments** (`# deprecated`, `# documented as adaptive_max_pool1d`): these appear after a string entry on the same line and explain *why* the entry is in the ignore list.

The module label comment directly tells you:
1. Which module the functions belong to
2. Where to add them in the docs (e.g., `# torch.ao.quantization.fx.convert` → the functions go under `torch.ao.quantization.fx.convert` in the doc file)

## Instructions

Each invocation of this skill processes **one batch** of module groups. Pick one or more complete module groups from the ignore lists, document their functions, and verify.

### Step 1: Select module groups to document

Read `docs/source/conf.py` and select one or more **complete module groups** to document. A module group is a module label comment and all entries beneath it up to the next module label comment. Process entire groups — never split a group across batches.

For example, selecting the `torch.ao.quantization.fx.convert` group means taking all of:

```python
# torch.ao.quantization.fx.convert
"convert",
"convert_custom_module",
"convert_standalone_module",
"convert_weighted_module",
```

Work through the lists top-to-bottom. Choose enough groups to make meaningful progress (aim for 5–15 functions total, but always include complete groups even if that means going slightly over).

**Check inline comments before including an entry.** Some entries have inline comments that indicate they should not be documented:

- `# deprecated` — The function is deprecated. Leave it in the ignore list.
- `# documented as <other_name>` — Already documented under a different name. Leave it.
- `# looks unintentionally public` — Probably not meant to be public API. Leave it.
- `# legacy helper for ...` — Same as deprecated. Leave it.
- `# utility function` - Leave it.

If a module group has a **mix** of regular entries and entries with inline comments, still process the group — but only comment out the regular entries. Leave entries with inline comments untouched in the ignore list.

### Step 2: Present the batch to the user

**Before making any edits**, present the selected module groups and their functions to the user. Show them organized by module:

```
Module: torch.ao.quantization.fx.convert
  - convert
  - convert_custom_module
  - convert_standalone_module
  - convert_weighted_module

Module: torch.ao.quantization.fx.fuse
  - fuse
```

Then use the `AskUserQuestion` tool to let the user confirm, with options like:
- "Proceed with this batch"
- "Skip some entries" (user can specify which to remove)
- "Pick a different batch"

### Step 3: Comment out entries in conf.py

After the user confirms, edit `docs/source/conf.py` and **comment out** (do not delete) the selected entries. Use a `#` prefix on each string entry line:

```python
# torch.ao.quantization.fx.convert
# "convert",
# "convert_custom_module",
# "convert_standalone_module",
# "convert_weighted_module",
```

This preserves the original entries so they can be restored if verification fails.

### Step 4: Run Sphinx coverage

```bash
cd docs && make coverage
```

**Ignore the terminal output of `make coverage`.** It often contains unrelated tracebacks and errors from Sphinx extensions (e.g., `onnx_ir`, `katex`, `sphinxcontrib`) that have nothing to do with coverage. The only thing that matters is whether `docs/build/coverage/python.txt` was generated. Read that file to see the specific undocumented APIs.

The format of `python.txt` lists each undocumented API as:

```
torch.ao.quantization.fx.convert
   * convert
   * convert_custom_module
   * convert_standalone_module
   * convert_weighted_module
```

**Not all commented-out functions will appear in `python.txt`.** Some may already be documented elsewhere. This is fine — only add directives for functions that actually appear in `python.txt`.

If `make coverage` fails due to missing dependencies, first run:

```bash
cd docs && pip install -r requirements.txt
```

### Step 5: Add documentation directives

For each function listed in `python.txt`, use the **module label comment** from `conf.py` to determine where it should be added. The module comment gives you the full module path, which maps to a doc source file and a section within that file.

#### Finding the correct doc file

The module comment maps to a doc source file in `docs/source/`. When unsure, search for other functions from the same module:

```bash
grep -rn "torch.module_name" docs/source/*.md docs/source/*.rst
```

Or list candidate files:

```bash
ls docs/source/*module_name*
```

If no doc file exists for a submodule, check whether a parent module's doc file has a section for it (e.g., `backends.md` has sections for `torch.backends.cuda`, `torch.backends.cudnn`, etc.). If not, add a new section to the parent file following existing patterns.

#### Adding the directives

**Read the target doc file first** and match the exact patterns already used there. Do not invent new patterns or use bare `autofunction` with fully qualified names — always use the proper hierarchical structure with `automodule`, `currentmodule`, and short names. Do not use `. py:module::` since that just suppresses errors and doesn't actually document the function. Look at other files that match the target file's format (e.g., `.md` vs. `.rst`) under `docs/source/` to see examples.

There are two file formats. Match the one used in the target file.

**Pattern A — MyST Markdown files (`.md`):** Used in files like `accelerator.md`, `backends.md`, `cuda.md`.

The hierarchical structure uses `automodule` to register the module, `currentmodule` to set context, then short names:

```markdown
## torch.ao.quantization.fx.convert

```{eval-rst}
.. automodule:: torch.ao.quantization.fx.convert
```

```{eval-rst}
.. currentmodule:: torch.ao.quantization.fx.convert
```

```{eval-rst}
.. autofunction:: convert
```

```{eval-rst}
.. autofunction:: convert_custom_module
```
```

For `autosummary` blocks (used in some files instead of individual directives):

```markdown
```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    existing_function
    your_new_function
`` `
```

For classes:

```markdown
```{eval-rst}
.. autoclass:: YourClass
    :members:
`` `
```

**Pattern B — reStructuredText files (`.rst`):** Used in files like `torch.rst`, `nn.rst`.

Same hierarchical structure without the markdown fences:

```rst
torch.ao.quantization.fx.convert
---------------------------------

.. automodule:: torch.ao.quantization.fx.convert

.. currentmodule:: torch.ao.quantization.fx.convert

.. autosummary::
    :toctree: generated
    :nosignatures:

    convert
    convert_custom_module
    convert_standalone_module
    convert_weighted_module
```

For individual directives:

```rst
.. automodule:: torch.submodule

.. currentmodule:: torch.submodule

.. autofunction:: function_name

.. autoclass:: ClassName
    :members:
```

**Key rules:**
- The module label comment from `conf.py` (e.g., `# torch.ao.quantization.fx.convert`) tells you exactly which `automodule` and `currentmodule` to use.
- Always set `.. automodule::` and `.. currentmodule::` before documenting functions from a module.
- Use **short names** (e.g., `convert`, not `torch.ao.quantization.fx.convert.convert`) after `currentmodule` is set.
- If the module already has an `automodule`/`currentmodule` in the file, don't add another — just add your function under the existing one.
- Match whichever style the file already uses (`autosummary` blocks vs. individual `autofunction` directives).

#### Placing in the right section

Read the target doc file and find the appropriate section. If the module already has a section (e.g., `## torch.backends.cuda` in `backends.md`), add the functions there. If no section exists yet, create one following the existing section patterns in the file. Group all functions from the same module group together.

### Step 6: Verify with coverage

Run coverage again:

```bash
cd docs && make coverage
```

Ignore the terminal output — only read `docs/build/coverage/python.txt`. Verification passes when `python.txt` contains **zero undocumented functions across ALL modules**. It should only have the statistics table with 100% coverage and 0 undocumented for every module. For example:

```
Undocumented Python objects
===========================

Statistics
----------

+---------------------------+----------+--------------+
| Module                    | Coverage | Undocumented |
+===========================+==========+==============+
| torch                     | 100.00%  | 0            |
+---------------------------+----------+--------------+
| torch.accelerator         | 100.00%  | 0            |
+---------------------------+----------+--------------+
```

If any module shows undocumented functions (coverage below 100% or undocumented count > 0), verification has failed.

**If verification succeeds (zero undocumented across all modules):** Go to Step 7.

**If verification fails (any undocumented functions remain):** Read `docs/build/coverage/python.txt` to see which functions are still listed as undocumented. Common issues include:

- Wrong doc file: the function was added to the wrong `.md`/`.rst` file. Move the directive to the correct file.
- Wrong directive type: e.g., used `autofunction` for a class, or `autoclass` for a function. Fix the directive.
- Wrong module path in the directive: e.g., `torch.foo.bar` should be `torch.foo.baz.bar`. Correct the qualified name.
- Function added to an `autosummary` block with the wrong `currentmodule`: make sure the `.. currentmodule::` directive above the block matches.
- Missing `automodule` for a submodule that hasn't been registered yet. Add a `.. automodule:: torch.submodule` directive before documenting functions from that submodule.

Fix the doc directive based on the error, then re-run `make coverage`. Repeat until verification passes.

If a function still fails after multiple attempts, **stop and show the error to the user.** Present the function name and the error, then use the `AskUserQuestion` tool with options like:
- "Uncomment it to restore to ignore list (skip for now)"
- "Try a different approach"
- "Investigate further"

### Step 7: Report progress

**Present a progress summary to the user** showing:

- Which module groups were processed and how many functions were documented
- Which functions were skipped or restored to the ignore list (and why)
- How many entries remain in `coverage_ignore_functions` and `coverage_ignore_classes`

### Step 8: Clean up commented-out entries in conf.py

Now that verification has passed, delete the commented-out string entries from Step 3. These are lines that start with `# "` inside `coverage_ignore_functions` and `coverage_ignore_classes`. Commented-out string entries always contain **quotes** — that's how you distinguish them from module label comments:

```python
# "disable_global_flags",       <-- commented-out string entry (has quotes) → DELETE
# torch.backends                <-- module label comment (no quotes) → KEEP if it has active entries
```

Also delete any module label comments that no longer have active entries beneath them (i.e., all their entries were either commented out and now deleted, or had inline comments and were left in place but the module label is otherwise empty).

## Important notes

- **Follow the steps exactly as written.** Do not add extra investigation steps like importing Python modules to check docstrings, inspecting source code to verify function signatures, or running any commands not specified in the instructions. The `make coverage` step is the only verification needed — let it tell you what's wrong.
- **Never modify Python source files (`.py`).** This skill only edits `docs/source/conf.py` and doc source files (`.md`/`.rst`) in `docs/source/`. Do not add or edit docstrings, do not read Python source to check function signatures, do not inspect implementations.
- Entries are commented out in Step 3, verified in Step 6, and cleaned up in Step 8 after verification passes. Never delete uncommented entries directly.
- **Read inline comments** on entries before deciding to document them. Entries marked `# deprecated`, `# documented as ...`, `# looks unintentionally public`, or `# legacy helper` should stay in the ignore list.
- The `coverage_ignore_functions` list uses bare function names (not fully qualified), so the same name can appear multiple times for different modules. Use the module label comment above each entry to identify which module it belongs to. Be careful during Step 8 cleanup to only delete the correct commented-out lines — commented-out string entries have **quotes** (`# "func_name",`), module label comments do not.
- Always match the existing style of the target doc file — don't mix `.md` style directives into `.rst` files or vice versa.
- **Use the module label comment** (e.g., `# torch.ao.quantization.fx.convert`) as the primary guide for both the `automodule`/`currentmodule` directives and for finding the right section in the doc file.
- Always process complete module groups — never split a group across invocations.
