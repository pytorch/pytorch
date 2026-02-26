#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/Parallel.h>
#include <torch/csrc/distributed/c10d/NanCheck.hpp>
#include <torch/library.h>

namespace c10d {

namespace {

void check_for_nan_cpu(const at::Tensor& tensor) {
  if (!tensor.is_floating_point()) {
    return;
  }
  if (tensor.numel() == 0) {
    return;
  }

  AT_DISPATCH_FLOATING_TYPES_AND4(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Float8_e4m3fn,
      at::ScalarType::Float8_e5m2,
      tensor.scalar_type(),
      "check_for_nan_cpu",
      [&] {
        auto* data = tensor.data_ptr<scalar_t>();
        auto numel = tensor.numel();
        std::atomic<bool> found{false};
        at::parallel_for(0, numel, 1024, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; i++) {
            if (at::_isnan(data[i])) {
              found.store(true, std::memory_order_relaxed);
              return;
            }
          }
        });
        TORCH_CHECK(!found.load(), "NaN found in input tensor.");
      });
}

TORCH_LIBRARY_IMPL(c10d, CPU, m) {
  m.impl("check_for_nan", check_for_nan_cpu);
}

} // namespace

void checkForNan(const at::Tensor& tensor) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("c10d::check_for_nan", "")
                       .typed<void(const at::Tensor&)>();
  op.call(tensor);
}

} // namespace c10d
