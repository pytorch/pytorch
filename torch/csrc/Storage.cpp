#define __STDC_FORMAT_MACROS

#include "torch/csrc/python_headers.h"
#ifdef _MSC_VER
#include <Windows.h>
#endif
#include <structmember.h>

#define THP_HOST_HALF

#include <stdbool.h>
#include <TH/TH.h>
// See Note [TH abstraction violation]
//  - Used to get at the allocator associated with a storage
#include <TH/THStorage.hpp>
#include <libshm.h>
#include "THP.h"
#include "allocators.h"
#include "copy_utils.h"
#include "DynamicTypes.h"

#include "generic/Storage.cpp"
#include <TH/THGenerateAllTypes.h>

#include "generic/Storage.cpp"
#include <TH/THGenerateHalfType.h>

#ifndef USE_CUDA
// NB: When USE_CUDA, the *full* implementation of
// this lives in torch/csrc/cuda/Storage.cpp.
template<>
void THPPointer<THStorage>::free() {
  if (ptr) {
    AT_ASSERT(ptr->backend == at::kCPU);
    THStorage_free(ptr);
  }
}
#endif
