#pragma once

#include <c10/util/Exception.h>
#include <sycl/sycl.hpp>

namespace c10::xpu {

static sycl::async_handler asyncHandler = [](sycl::exception_list eL) {
  for (auto& e : eL) {
    try {
      std::rethrow_exception(e);
    } catch (sycl::exception& e) {
      TORCH_WARN(
          "SYCL Exception: ",
          e.what(),
          "file = ",
          __FILE__,
          "line = ",
          __LINE__);
      throw;
    }
  }
};

} // namespace c10::xpu
