#include "Copy.h"

#include "ATen/ATen.h"
#include "ATen/CPUApplyUtils.h"
#include "ATen/Dispatch.h"
#include "ATen/NativeFunctions.h"

namespace {

template <typename self_T, typename src_T>
void _copy__cpu(at::Tensor& self, const at::Tensor& src) {
  at::CPU_tensor_apply2<self_T, src_T>(
      self, src, [](self_T& self_val, const src_T& src_val) {
        self_val = static_cast<self_T>(
            static_cast<at::native::inter_copy_type_t<self_T>>(src_val));
      });
}

template <typename self_T>
void _copy__cpu(at::Tensor& self, const at::Tensor& src) {
  AT_DISPATCH_ALL_TYPES_AND_HALF(
      src.type(), "_copy__cpu", [&]() { _copy__cpu<self_T, scalar_t>(self, src); });
}

} // namespace

namespace at {
namespace native {

Tensor& _copy__cpu(Tensor& self, const Tensor& src) {
  AT_DISPATCH_ALL_TYPES_AND_HALF(
      self.type(), "_copy__cpu", [&]() { ::_copy__cpu<scalar_t>(self, src); });
  return self;
}

} // namespace native
} // namespace at
