# PyTorch C++ API Documentation

This directory contains the source for PyTorch's C++ API documentation, built with
Sphinx, Breathe, and Doxygen.

## How it works

The documentation pipeline has three stages:

```
C++ headers ──→ Doxygen ──→ XML ──→ Breathe ──→ Sphinx ──→ HTML
                  ↑                    ↑
              Doxyfile          Markdown files with
           (which headers)    Breathe directives (MyST)
```

1. **Doxygen** reads C++ headers listed in `source/Doxyfile` `INPUT` and produces
   XML in `build/xml/`.
2. **Breathe** is a Sphinx extension that reads Doxygen XML and makes it available
   via directives like `` ```{doxygenclass} ClassName `` (MyST Markdown syntax).
3. **Sphinx** builds the final HTML from `.md` files in `source/`, using Breathe
   directives to pull in C++ API documentation. The `myst_parser` extension
   enables Markdown support.

Only headers listed in the Doxyfile's `INPUT` are processed. Source files must
explicitly reference each symbol — nothing is auto-generated.

## Building

```bash
make html        # Build Doxygen XML + Sphinx HTML
make doxygen     # Build Doxygen XML only
make clean       # Clean build artifacts
```

The output is in `build/html/`.

## Contributing to the C++ docs

### Adding a new API

1. **Ensure the header is in the Doxyfile** — check that `source/Doxyfile` `INPUT`
   includes the header file or its parent directory. If not, add it:

   ```
   INPUT = ... \
           ../../../path/to/your/header.h
   ```

2. **Add a Breathe directive to the appropriate `.md` file** under `source/api/`.
   See [Which directive to use](#which-directive-to-use) below.

3. **Run `make html`** and check the output in `build/html/`.

4. **Run `python check_coverage.py`** to verify your API shows as documented.

### Which directive to use

Use Breathe directives to pull documentation from Doxygen XML. These render the
full C++ signature, doc comments, parameters, and members automatically.

All source files use MyST Markdown syntax (fenced directives with backticks).

**Classes and structs** — use `doxygenclass` or `doxygenstruct`:

````markdown
```{doxygenclass} torch::nn::Linear
:members:
:undoc-members:
```
````

- `:members:` shows all public member functions and variables
- `:undoc-members:` includes members without doc comments
- Omit both flags to show only the class description (useful when `:members:`
  causes rendering issues)

**Free functions** — use `doxygenfunction`:

````markdown
```{doxygenfunction} torch::autograd::grad
```
````

For overloaded functions, Breathe will document all overloads.

**Macros** — use `doxygendefine`:

````markdown
```{doxygendefine} TORCH_LIBRARY
```
````

**Typedefs** — use `doxygentypedef`:

````markdown
```{doxygentypedef} torch::DeviceType
```
````

### When Breathe directives don't work

Some symbols can't be documented with Breathe directives:

- **TORCH_MODULE holder classes** (e.g., `Conv2d`, `Linear`): The `TORCH_MODULE()`
  macro generates these, but Doxygen can't index them. Document the `*Impl` class
  instead (e.g., `Conv2dImpl`) — it contains all the actual methods.

- **Functions with broken `\rst`/`\endrst` blocks**: Some doc comments use Doxygen's
  `\rst` alias to embed RST. When a comment has multiple such blocks, Doxygen
  generates malformed XML and Breathe renders raw text. In these cases, either:
  - Fix the header to use native Doxygen (`@code{.cpp}`/`@endcode`, `@note`,
    `@warning`) instead of `\rst`/`\endrst`
  - Use a hand-written Sphinx C++ domain directive as a fallback

- **Functions with mismatched `\param` names**: If a header's `\param` names don't
  match the actual parameter names, Doxygen may fail to index the function. Use a
  hand-written `cpp:function` directive instead.

**Hand-written Sphinx C++ domain directives** (fallback):

````markdown
```{cpp:function} void torch::autograd::backward(const variable_list& tensors, const variable_list& grad_tensors = {}, std::optional<bool> retain_graph = std::nullopt, bool create_graph = false, const variable_list& inputs = {})

Computes gradients of given tensors with respect to graph leaves.

:param tensors: Tensors of which the derivative will be computed.
:param grad_tensors: The "vector" in the Jacobian-vector product.
```
````

These don't pull from Doxygen — you write the signature and docs manually.

**`{eval-rst}` escape hatch** — if a MyST directive doesn't render correctly,
you can embed raw RST:

````markdown
```{eval-rst}
.. doxygenclass:: X::A
   :members:
   :protected-members:
   :private-members:
```
````

### Source file structure

Each `.md` file under `source/api/` documents one topic area using MyST Markdown.
The typical pattern:

````markdown
# Page Title

Brief description of this API area.

## Section Name

Optional prose explaining usage, with a code example:

```cpp
#include <torch/torch.h>
auto x = torch::randn({2, 3});
```

```{doxygenclass} torch::nn::SomeClass
:members:
:undoc-members:
```

```{doxygenstruct} torch::nn::SomeClassOptions
:members:
:undoc-members:
```
````

**Nesting directives:** When a directive contains other directives or code blocks,
the outer fence must use more backticks than the inner ones:

`````markdown
````{cpp:class} at::Tensor

The primary tensor class.

```{cpp:function} int64_t dim() const

Returns the number of dimensions.
```
````
`````

### Writing doc comments in C++ headers

Doxygen extracts documentation from comments in headers. Use `///` style:

```cpp
/// Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.
///
/// @code{.cpp}
/// auto linear = torch::nn::Linear(torch::nn::LinearOptions(10, 5));
/// auto output = linear->forward(input);
/// @endcode
///
/// @note The weight matrix is transposed compared to the Python API.
class LinearImpl : public Cloneable<LinearImpl> {
```

**Preferred Doxygen commands:**
- `@code{.cpp}` / `@endcode` for code examples
- `@note` for important notes
- `@warning` for warnings
- `@param name` for parameter descriptions
- `@return` for return value descriptions

**Avoid** `\rst` / `\endrst` blocks — they cause rendering issues when a single
doc comment contains multiple blocks. Use native Doxygen commands instead.

## Coverage checking

```bash
make coverage        # Build docs + run coverage check
make coverage-only   # Run coverage check without rebuilding
```

`check_coverage.py` auto-discovers public APIs from Doxygen XML (`build/xml/index.xml`)
and checks which ones have Breathe or Sphinx directives in the source files.

Reports are written to `cpp_coverage.txt` and `cpp_html_issues.txt`.

### How coverage discovery works

The script parses `build/xml/index.xml` to find all classes, structs, functions,
and macros that Doxygen indexed. It then:

1. Filters out internal symbols using `EXCLUDED_PATTERNS` and `EXCLUDED_SYMBOLS`
2. Re-includes any symbols in `INCLUDED_SYMBOLS` (overrides exclusions)
3. Scans `.md` files for Breathe (`doxygenclass`, `doxygenfunction`, etc.) and
   Sphinx C++ domain (`cpp:class`, `cpp:function`, etc.) directives
4. Reports the gap

### Excluding internal APIs from coverage

Not everything in Doxygen XML is public API. The script has three exclusion mechanisms:

**`EXCLUDED_PATTERNS`** — regex patterns for broad categories:

```python
EXCLUDED_PATTERNS = [
    r".*::detail::.*",     # Internal namespaces
    r"torch::jit::.*",     # Deprecated
    r".*::_\w+",           # Underscore-prefixed internals
]
```

**`EXCLUDED_SYMBOLS`** — exact match for specific symbols:

```python
EXCLUDED_SYMBOLS = {
    "torch::autograd::deleteNode",
    "at::native::dataSize",
}
```

**Doxyfile `EXCLUDE`** — prevents Doxygen from indexing files entirely:

```
EXCLUDE = ../../../torch/csrc/api/include/torch/detail
```

### Including "internal" APIs that are actually public

Some APIs live in internal-looking namespaces but are widely used by external
developers. To track these for documentation coverage, add them to
`INCLUDED_SYMBOLS` in `check_coverage.py`:

```python
INCLUDED_SYMBOLS: set[str] = {
    "c10::IValue",  # Used for custom op registration
}
```

`INCLUDED_SYMBOLS` takes priority over all exclusions.

### HTML formatting checks

The script also checks built HTML for:
- Unresolved Breathe directives ("Cannot find class/struct/function")
- Raw directive text in output (build failures)
- Sphinx "problematic" nodes (broken references)
- Near-empty API pages

### coverxygen (optional)

```bash
python check_coverage.py --coverxygen
```

This runs [coverxygen](https://github.com/psycofdj/coverxygen) on the Doxygen XML
to measure what percentage of C++ symbols have doc comments in the source code.
This is complementary to file coverage — it tells you which headers need more
doc comments, not which symbols need Sphinx directives.

## Known issues

- **`\rst`/`\endrst` rendering** — Some C++ headers use Doxygen's `\rst` alias to
  embed RST in doc comments. When a single doc comment has multiple `\rst`/`\endrst`
  blocks, Doxygen generates malformed XML and Breathe renders them as raw text.
  Avoid using `:members:` on classes with this issue. Use hand-written examples
  in the source files instead, or convert the headers to native Doxygen commands.

- **TORCH_MODULE holder classes** — Doxygen can't index classes generated by the
  `TORCH_MODULE()` macro. Use the `*Impl` class name in directives instead.

- **"Subclassed by" plain text** — Without per-class pages (exhale), Breathe
  renders "Subclassed by" lists as plain text instead of links. A `conf.py` hook
  (`remove_subclassed_by`) strips these automatically.
