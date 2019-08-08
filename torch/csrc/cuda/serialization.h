#ifndef THCP_SERIALIZATION_INC
#define THCP_SERIALIZATION_INC

#include <torch/csrc/cuda/override_macros.h>

#define THC_GENERIC_FILE "torch/csrc/generic/serialization.h"
#include <THC/THCGenerateAllTypes.h>

#define THC_GENERIC_FILE "torch/csrc/generic/serialization.h"
#include <THC/THCGenerateBoolType.h>

#define THC_GENERIC_FILE "torch/csrc/generic/serialization.h"
#include <THC/THCGenerateBFloat16Type.h>

#endif
