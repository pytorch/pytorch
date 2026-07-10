# TorchTLX lives in the fbtriton, not in
# PyTorch. Importing the integration registers TLX template heuristics and
# installs config.inductor_choices_class; it succeeds only when the active
# Triton is the fbtriton fork
# and fails cleanly on trunk triton. Actual engagement is
# still gated at runtime by TORCHINDUCTOR_TLX_MODE (config.triton.tlx_mode).
try:
    import triton.language.extra.tlx.inductor.registry  # noqa: F401  # type: ignore[import-not-used]
except ImportError:
    pass
