#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <ATen/PythonMode.h>

namespace at {

#if !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
TORCH_LIBRARY_IMPL(aten, PythonMode, m) {
  ${python_mode_registrations};
}
#endif

} // namespace at
