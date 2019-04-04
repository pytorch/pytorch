#include <ATen/native/cpu/CopyKernel.h>

#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/native/Copy.h>

namespace at {
namespace native {
namespace {

// TODO: this number was copied from TH, test to see if it's the right number
constexpr int64_t COPY_GRAIN_SIZE = 20000;

static void copy_kernel_impl(Tensor& dst, const Tensor& src) {
  AT_DISPATCH_ALL_TYPES_AND2(
    at::ScalarType::Half, at::ScalarType::Bool, dst.scalar_type(), "copy_kernel_impl", [&]() {
      scalar_t* self_ptr = dst.data<scalar_t>();
      scalar_t* src_ptr = src.data<scalar_t>();

      auto sample = [&](int64_t begin, int64_t end) {
        int64_t len = end - begin;
        scalar_t* self_seg = self_ptr + begin;
        scalar_t* src_seg = src_ptr + begin;
        at::vec256::convert<scalar_t, scalar_t>(src_seg, self_seg, len);
    };

    parallel_for(0, dst.numel(), COPY_GRAIN_SIZE, sample);
  });
}

} // anonymous namespace

REGISTER_DISPATCH(copy_kernel, &copy_kernel_impl);

} // namespace native
} // namespace at
