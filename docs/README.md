# PyTorch Documentation

Please see the [Writing documentation section of CONTRIBUTING.md](../CONTRIBUTING.md#writing-documentation)
for details on both writing and building the docs.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Build the documentation
make html

# View locally (opens at http://localhost:8000)
make serve PORT=8000
```

The generated HTML files will be in `docs/build/html/`.

## Checking for Broken Links

Linkchecker verifies internal links and `#anchor` fragments in the **built HTML**
documentation. It cannot check source files directly because links and anchors
are only generated when Sphinx builds the docs.

**You must build the docs first** before running linkcheck.

### Check Links in Changed Files (Recommended for PRs)

```bash
# Build docs first
make html

# Check only docs changed in your last commit
make linkcheck-local
```

This runs quickly by only checking files you've modified.

### Check Links in Specific Commits

```bash
# Check docs changed between two commits
../scripts/lint_doc_anchors.sh main HEAD      # vs main branch
../scripts/lint_doc_anchors.sh HEAD~3 HEAD    # last 3 commits
```

### Full Site Check

```bash
# Check ALL internal links (takes longer, used by CI)
../scripts/lint_doc_anchors.sh --all
```

### Prerequisites

Install linkchecker if not already installed:

```bash
pip install linkchecker
```

### What Gets Checked

- Internal links between documentation pages
- `#anchor` fragments (verifies the target ID exists on the page)
- Links to auto-generated API documentation in `/generated/`

External URLs are **not** checked by this linkchecker configuration (they're validated separately
by `scripts/lint_urls.sh`).

### CI Integration

Link checking runs automatically in CI after nightly builds complete successfully.
If broken links are found, a GitHub issue is created with the `module: docs` label.

See `.github/workflows/docs-link-check.yml` for details.
