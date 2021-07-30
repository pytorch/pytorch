#pragma once

#include <c10/macros/Macros.h>
#include <torch/library.h>

namespace at {
namespace impl {

struct TORCH_API PythonMode {
  // static void set_enabled(bool enabled);
  // NB: not thread safe
  static void set_torch_dispatch(const Tensor& tensor);
  // NB: not thread safe
  static void reset_torch_dispatch();
};

} // namespace impl
} // namespace at
