ATEN_CORE_HEADER_FILES = [
    # "aten/src/" prefix is added later
    "ATen/core/ATenGeneral.h",
    "ATen/core/blob.h",
    "ATen/core/DimVector.h",
    "ATen/core/grad_mode.h",
    "ATen/core/UndefinedTensorImpl.h",
]

ATEN_CORE_SRC_FILES = [
    "aten/src/ATen/core/grad_mode.cpp",
    "aten/src/ATen/core/VariableFallbackKernel.cpp",
]
