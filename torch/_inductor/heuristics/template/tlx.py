# TorchTLX lives in fbtriton, not in PyTorch. Importing the integration
# registers the TLX template heuristics and installs
# config.inductor_choices_class; it succeeds only when the active Triton ships
# the integration, and fails cleanly on one that does not.
#
# Deferred rather than done at module import: the integration monkey-patches
# TritonTemplate / TritonTemplateKernel and replaces inductor_choices_class,
# none of which belongs at import time in a process that never reaches
# Inductor's choices handler. virtualized._choices_default() calls
# maybe_install() on first use.
#
# The install is unconditional, not gated on config.triton.tlx_mode: the
# choices handler is created once per thread and cached, so gating on the mode
# at that moment would leave TLX permanently uninstalled for a process that
# enables it later via config.patch. Engagement stays gated at runtime inside
# TLXInductorChoices, which falls through to the base choices when mode is None.

import functools


@functools.cache
def maybe_install() -> None:
    try:
        import triton.language.extra.tlx.inductor.registry  # noqa: F401  # type: ignore[import-not-used]
    except ImportError:
        pass
