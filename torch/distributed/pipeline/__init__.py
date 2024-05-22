import warnings


with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "`torch.distributed.pipeline` is deprecated. For up-to-date pipeline parallel "
        "implementation, please refer to the PiPPy library under the PyTorch "
        "organization (Pipeline Parallelism for PyTorch): "
        "https://github.com/pytorch/PiPPy",
        DeprecationWarning,
        stacklevel=2,
    )
