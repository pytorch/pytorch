#pragma once

#include <c10/util/Exception.h>
#include <sycl/sycl.hpp>

namespace c10::xpu {

static inline sycl::async_handler asyncHandler =
    [](const sycl::exception_list& el) {
      if (el.size() == 0) {
        return;
      }
      for (const auto& e : el) {
        try {
          std::rethrow_exception(e);
        } catch (sycl::exception& e) {
          TORCH_WARN("SYCL Exception: ", e.what());
        }
      }
      throw;
    };

} // namespace c10::xpu
