#include <torch/csrc/python_headers.h>
#ifdef _MSC_VER
#include <c10/util/win32-headers.h>
#endif
#include <structmember.h>

#include <libshm.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/copy_utils.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/CudaIPCTypes.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <c10/core/CPUAllocator.h>

#include <fmt/format.h>

// NOLINTNEXTLINE(bugprone-suspicious-include)
#include <torch/csrc/generic/Storage.cpp>
#include <torch/csrc/THGenerateByteType.h>

#include <c10/util/intrusive_ptr.h>

template<>
void THPPointer<c10::StorageImpl>::free() {
  if (ptr) {
    c10::raw::intrusive_ptr::decref(ptr);
  }
}
