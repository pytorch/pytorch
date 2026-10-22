# TorchTLX lives in fbtriton, not in PyTorch. Importing the integration
# registers the TLX template heuristics and installs
# config.inductor_choices_class; it succeeds only when the active Triton ships
# the integration, and fails cleanly on one that does not.
#
# Deferred rather than done at module import: the integration monkey-patches
# TritonTemplate / TritonTemplateKernel and replaces inductor_choices_class,
# and none of that belongs in a process that will never engage TLX.
# virtualized._choices_default() calls maybe_install() on first use.

import functools


@functools.cache
def _install() -> None:
    try:
        import triton.language.extra.tlx.inductor.registry  # noqa: F401  # type: ignore[import-not-used]
    except ImportError:
        pass


def maybe_install() -> None:
    from torch._inductor import config

    # Mode is re-read every call rather than cached with _install: tests flip
    # it with config.patch after the first choices handler already exists.
    if config.triton.tlx_mode is not None:
        _install()
