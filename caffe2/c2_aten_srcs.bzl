ATEN_CORE_HEADER_FILES = [
    # "aten/src/" prefix is added later
    "ATen/core/ATenGeneral.h",
    "ATen/core/blob.h",
    "ATen/core/DimVector.h",
    "ATen/core/grad_mode.h",
    "ATen/core/UndefinedTensorImpl.h",
    "ATen/core/SymIntArrayRef.h",
    "ATen/core/SymInt.h",
]

ATEN_CORE_SRC_FILES = [
    "aten/src/ATen/core/VariableFallbackKernel.cpp",
]
