#include <torch/csrc/python_headers.h>

#include <torch/csrc/cuda/THCP.h>

#include <torch/csrc/cuda/override_macros.h>

#include <system_error>
#include <memory>

#define THC_GENERIC_FILE "torch/csrc/generic/serialization.cpp"
#include <THH/THHGenerateAllTypes.h>

#define THC_GENERIC_FILE "torch/csrc/generic/serialization.cpp"
#include <THH/THHGenerateComplexTypes.h>

#define THC_GENERIC_FILE "torch/csrc/generic/serialization.cpp"
#include <THH/THHGenerateBoolType.h>

#define THC_GENERIC_FILE "torch/csrc/generic/serialization.cpp"
#include <THH/THHGenerateBFloat16Type.h>
