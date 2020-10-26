"""Compat bindings are for nightly only."""

raise NotImplementedError(
    "Compat bindings should not be needed for a stable release. If you "
    "did not monkey patch benchmark utils into an earlier version of "
    "PyTorch, then this is a bug and should be reported to the PyTorch team."
)
