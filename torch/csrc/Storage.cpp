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
#include <TH/THStorageFunctions.hpp>
#include <libshm.h>
#include "THP.h"
#include "copy_utils.h"
#include "DynamicTypes.h"

#ifdef USE_CUDA
#include <THC/THCStorage.hpp>
#endif

#include "generic/Storage.cpp"
#include <TH/THGenerateAllTypes.h>

#include "generic/Storage.cpp"
#include <TH/THGenerateHalfType.h>

// NB: If you ever divest libtorch of USE_CUDA, you'll have to virtualize
// the CUDA call.
template<>
void THPPointer<THStorage>::free() {
  if (ptr) {
    if (ptr->data_ptr().device().is_cpu()) {
      THStorage_free(ptr);
    } else {
      AT_ASSERT(ptr->data_ptr().device().is_cuda());
#ifdef USE_CUDA
      THStorage_free(ptr);
#else
      AT_ERROR("Cannot free THCStorage when not built with CUDA");
#endif
    }
  }
}
