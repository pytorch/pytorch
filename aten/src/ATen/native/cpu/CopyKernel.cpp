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
  char* self_ptr = (char*)dst.data_ptr();
  char* src_ptr = (char*)src.data_ptr();

  auto sample = [=](int64_t begin, int64_t end) {
    int64_t len = end - begin;
    char* self_seg = self_ptr + begin;
    char* src_seg = src_ptr + begin;
    memcpy(self_seg, src_seg, len);
  };

  parallel_for(0, dst.nbytes(), COPY_GRAIN_SIZE, sample);
}

} // anonymous namespace

REGISTER_DISPATCH(copy_kernel, &copy_kernel_impl);

} // namespace native
} // namespace at
