#define __STDC_FORMAT_MACROS

#include <torch/csrc/python_headers.h>
#include <structmember.h>

// See Note [TH abstraction violation]
//    - Used to get at allocator from storage
#include <TH/THTensor.hpp>
#include <THH/THHTensor.hpp>
#include <torch/csrc/cuda/THCP.h>

#include <torch/csrc/cuda/override_macros.h>
#include <torch/csrc/copy_utils.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/CudaIPCTypes.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>

#define THC_GENERIC_FILE "torch/csrc/generic/Storage.cpp"
#include <THH/THHGenerateAllTypes.h>

#define THC_GENERIC_FILE "torch/csrc/generic/Storage.cpp"
#include <THH/THHGenerateBoolType.h>

#define THC_GENERIC_FILE "torch/csrc/generic/Storage.cpp"
#include <THH/THHGenerateBFloat16Type.h>
