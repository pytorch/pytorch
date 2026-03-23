"""Minimal Sphinx conf for previewing the landing page only.

Imports the real conf.py but disables extensions that require
a source-built torch (autodoc, autosummary, coverage, linkcode, doctest)
so the card-grid index page can render with a pip-installed torch.
"""

import sys, os

# Make the real source dir importable (for redirects.py etc.)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "source"))

# ---- execute the real conf.py in this namespace ----
_real_conf = os.path.join(os.path.dirname(__file__), "source", "conf.py")
with open(_real_conf) as _f:
    _code = _f.read()

# Patch: skip the coverage_post_process hook (references CoverageBuilder)
_code = _code.replace(
    'from sphinx.ext.coverage import CoverageBuilder',
    '# SKIPPED for preview\nCoverageBuilder = None',
)

exec(compile(_code, _real_conf, "exec"))  # noqa: S102

# ---- now override the heavy extensions ----
_skip = {
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.linkcode",
    "sphinx.ext.napoleon",
}
extensions = [e for e in extensions if e not in _skip]  # noqa: F821

autosummary_generate = False

# Widen exclude_patterns so Sphinx doesn't try to process API rst files
# that rely on autodoc/autosummary generated content.
exclude_patterns = ["generated/*"]  # noqa: F821
