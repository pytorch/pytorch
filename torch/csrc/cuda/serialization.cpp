#include <torch/csrc/python_headers.h>

#include <torch/csrc/cuda/THCP.h>

#include <torch/csrc/cuda/override_macros.h>

#include <system_error>
#include <memory>

#define THC_GENERIC_FILE "torch/csrc/generic/serialization.cpp"
#include <THC/THCGenerateAllTypes.h>

#define THC_GENERIC_FILE "torch/csrc/generic/serialization.cpp"
#include <THC/THCGenerateBoolType.h>

#define THC_GENERIC_FILE "torch/csrc/generic/serialization.cpp"
#include <THC/THCGenerateBFloat16Type.h>
