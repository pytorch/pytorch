# PyTorch C++ API Documentation

This directory contains the source for PyTorch's C++ API documentation, built with
Sphinx, Breathe, and Doxygen.

## Building

```bash
make html        # Build Doxygen XML + Sphinx HTML
make doxygen     # Build Doxygen XML only
make clean       # Clean build artifacts
```

The output is in `build/html/`.

## Coverage

```bash
make coverage        # Build docs + run coverage check
make coverage-only   # Run coverage check without rebuilding
```

This runs `check_coverage.py`, which checks:

1. **Allowlist coverage** — verifies that key public APIs are documented in the RST files
2. **HTML formatting** — checks for broken Breathe directives and Sphinx errors
3. **coverxygen** (optional, `--coverxygen` flag) — measures doc-comment coverage
   in the C++ source, producing an lcov report

Reports are written to `cpp_coverage.txt` and `cpp_html_issues.txt`.

### Excluding APIs from coverxygen

coverxygen measures what percentage of public C++ symbols have doc comments in the
source. Not all symbols are public API — here's how to exclude internal code:

**Exclude files or directories** — add `--exclude` regex patterns in `check_coverage.py`:

```python
"--exclude", ".*/build/.*",           # auto-generated code
"--exclude", ".*/detail/.*",          # internal implementation
"--exclude", ".*/nativert/.*",        # not public API
"--exclude", ".*/stable/library\\.h", # internal macros, no public symbols
```

Files in `HIDDEN_NAMESPACE_BEGIN(...)` or `detail` namespaces are typically internal
and should be excluded. If coverxygen reports low coverage on a file that your team
considers fully documented, check whether the "undocumented" symbols are actually
internal (macro helpers, private constructors, anonymous namespace utilities). If so,
add an `--exclude` pattern for the file.

**Exclude from Doxygen entirely** — add to `source/Doxyfile` `EXCLUDE`:

```
EXCLUDE = ../../../torch/csrc/api/include/torch/nn/pimpl-inl.h \
          ../../../torch/csrc/api/include/torch/detail
```

Files listed here won't appear in docs or coverage.

**Mark symbols as internal in source** — use Doxygen annotations:

```cpp
/// @internal
class InternalHelper { ... };  // skipped by Doxygen

/// @cond INTERNAL
void private_function();
/// @endcond
```

The `--scope public` flag in `check_coverage.py` already excludes `protected` and
`private` class members.

**Why does my file show low coverage when everything is documented?**

Files using `HIDDEN_NAMESPACE_BEGIN(...)` may contain a mix of public API symbols
(with `/** */` doc comments) and internal helpers (no doc comments). coverxygen
counts all public symbols, so internal helpers like `DeleterFnPtr`, anonymous
namespace utilities, and macro-expansion plumbing drag down the percentage.

To fix this, annotate internal symbols in the source with `/// @internal`:

```cpp
HIDDEN_NAMESPACE_BEGIN(torch, stable, accelerator)

/// @internal
using DeleterFnPtr = void (*)(void*);

/// @internal
namespace {
inline void delete_device_guard(void* ptr) { ... }
}

/**
 * @brief Device index type for stable ABI.
 */
using DeviceIndex = int32_t;  // this one IS public, so it keeps its doc comment
```

This is preferred over file-level `--exclude` patterns because it lets you keep
public symbols in the coverage report while hiding internal ones. Each team owns
the annotations in their headers.

## Adding new APIs to the docs

1. Make sure the header is listed in `source/Doxyfile` `INPUT`
2. Add a `doxygenclass`, `doxygenstruct`, or `doxygenfunction` directive to the
   appropriate RST file under `source/api/`
3. Add the API to the allowlist in `check_coverage.py` if it's a key public API
4. Run `make coverage` to verify

## Known issues

**`\rst`/`\endrst` rendering** — Some C++ headers use Doxygen's `\rst` alias to
embed RST in doc comments. When a single doc comment has multiple `\rst`/`\endrst`
blocks, Doxygen generates malformed XML and Breathe renders them as raw text. Avoid
using `:members:` on classes with this issue (e.g., `SequentialImpl`, `ModuleListImpl`).
Use hand-written examples in the RST instead.
