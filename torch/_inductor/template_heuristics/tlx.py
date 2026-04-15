from torch._inductor import config


if config.is_fbcode():
    import torch._inductor.fb.tlx_templates.registry  # noqa: F401  # type: ignore[import-not-used]
