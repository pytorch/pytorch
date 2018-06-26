#define __STDC_FORMAT_MACROS

#include "torch/csrc/python_headers.h"
#include <structmember.h>

#include <stdbool.h>
// See Note [TH abstraction violation]
//    - Used to get at allocator from storage
#include <TH/THTensor.hpp>
#include <THC/THCTensor.hpp>
#include "THCP.h"

#include "override_macros.h"
#include "torch/csrc/allocators.h"
#include "torch/csrc/copy_utils.h"
#include "DynamicTypes.h"

#define THC_GENERIC_FILE "torch/csrc/generic/Storage.cpp"
#include <THC/THCGenerateAllTypes.h>

#ifndef USE_CUDA
#error "Compiling torch/csrc/cuda/Storage.cpp without USE_CUDA"
#endif

// NB: When !USE_CUDA, the implementation of this lives
// in torch/csrc/Storage.cpp.
// If you ever divest libtorch of USE_CUDA, you'll have to virtualize
// the CUDA call.
template<>
void THPPointer<THStorage>::free() {
  if (ptr) {
    if (ptr->backend == at::kCPU) {
      THStorage_free(ptr);
    } else {
      AT_ASSERT(ptr->backend == at::kCUDA);
      THCStorage_free(LIBRARY_STATE ptr);
    }
  }
}
