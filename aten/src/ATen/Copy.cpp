#include "Copy.h"

#include "ATen/ATen.h"
#include "ATen/CPUApplyUtils.h"
#include "ATen/NativeFunctions.h"

namespace {

template <typename dst_T, typename src_T>
void copy_cpu(at::Tensor& dst, const at::Tensor& src) {
  at::CPU_tensor_apply2<dst_T, src_T>(
      dst, src, [](dst_T& dst_val, const src_T& src_val) {
        dst_val = static_cast<dst_T>(
            static_cast<at::inter_copy_type_t<dst_T>>(src_val));
      });
}

template <typename dst_T>
void copy_cpu(at::Tensor& dst, const at::Tensor& src) {
  AT_DISPATCH_ALL_TYPES_AND_HALF(
      src.type(), "copy_cpu", [&]() { copy_cpu<dst_T, scalar_t>(dst, src); });
}

} // namespace

namespace at {

Tensor copy_cpu(Tensor& dst, const Tensor& src) {
  AT_DISPATCH_ALL_TYPES_AND_HALF(
      dst.type(), "copy_cpu", [&]() { ::copy_cpu<scalar_t>(dst, src); });
  return dst;
}

} // namespace at
