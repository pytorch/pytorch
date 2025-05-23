// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <ATen/native/mtia/MTIAOps.h>
#include <c10/core/DeviceType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorMeta.h>

namespace at::native {

DEFINE_DISPATCH(mm_out_mtia_stub);

TORCH_IMPL_FUNC(mm_out_mtia) (
  const at::Tensor & self, const at::Tensor & mat2, const at::Tensor & out
) {
  mm_out_mtia_stub(c10::DeviceType::MTIA, self, mat2, out);
}

REGISTER_CPU_DISPATCH_NO_OP(
  mm_out, 
  const at::Tensor&, 
  const at::Tensor& self,
  const at::Tensor& mat2,
  const at::Tensor& out);

} // namespace at::native
