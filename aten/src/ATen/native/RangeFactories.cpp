#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/Dispatch.h>

namespace at { namespace native {


Tensor& linspace_cpu_out(Tensor& result, Scalar start, Scalar end, int64_t steps) {
  AT_CHECK(steps >= 0, "number of steps must be non-negative");

  if (result.numel() != steps) {
    result.resize_({steps});
  }
  Tensor r = result.is_contiguous() ? result : result.contiguous();

  if (steps == 0) {
    // skip
  } else if (steps == 1) {
    r.fill_(start);
  } else {
    AT_DISPATCH_FLOATING_TYPES(r.type(), "linspace", [&]() {
      scalar_t scalar_start = start.to<scalar_t>();
      scalar_t scalar_end = end.to<scalar_t>();
      scalar_t *data_ptr = r.data<scalar_t>();
      scalar_t step = (scalar_end - scalar_start) / static_cast<scalar_t>(steps - 1);
      at::parallel_for(0, steps, internal::GRAIN_SIZE, [&](int64_t p_begin, int64_t p_end) {
        scalar_t is = static_cast<scalar_t>(p_begin);
        for (int64_t i = p_begin; i < p_end; ++i, ++is) {
          data_ptr[i] = scalar_start + step*is;
        }
      });
    });
  }

  if (!result.is_contiguous()) {
    result.copy_(r);
  }
  return result;
}

}} // namespace at::native
