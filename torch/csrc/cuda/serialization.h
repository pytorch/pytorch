#ifndef THCP_SERIALIZATION_INC
#define THCP_SERIALIZATION_INC

#include <torch/csrc/cuda/override_macros.h>

#define THC_GENERIC_FILE "torch/csrc/generic/serialization.h"
#include <THH/THHGenerateAllTypes.h>

#define THC_GENERIC_FILE "torch/csrc/generic/serialization.h"
#include <THH/THHGenerateComplexTypes.h>

#define THC_GENERIC_FILE "torch/csrc/generic/serialization.h"
#include <THH/THHGenerateBoolType.h>

#define THC_GENERIC_FILE "torch/csrc/generic/serialization.h"
#include <THH/THHGenerateBFloat16Type.h>

#endif
