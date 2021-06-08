#include "caffe2/contrib/aten/aten_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace internal {
at::Tensor index_with_uint8_handling(
    const at::Tensor& self,
    const torch::List<c10::optional<at::Tensor>>& indices) {
  // Support BC only for the simplest case of mask indexing
  if (indices.size() == 1) {
    c10::optional<at::Tensor> first = indices[0];
    if (first.has_value()
        && first->scalar_type() == at::kByte) {
      TORCH_WARN(
          "Indexing with uint8 mask tensor in ATenOp is now deprecated,"
          " please use a bool mask instead.");
      return at::index(self, {first->to(at::kBool)});
    }
  }
  return at::index(self, indices);
}
} // namespace internal

REGISTER_CPU_OPERATOR(ATen, ATenOp<CPUContext>);
template <>
at::Backend ATenOp<CPUContext>::backend() const {
  return at::Backend::CPU;
}

OPERATOR_SCHEMA(ATen);

namespace math {

template <>
void Set<at::Half, CPUContext>(
    const std::int64_t /* N */,
    const at::Half h,
    at::Half* v,
    CPUContext* c) {
  Set(0, h.x, (uint16_t*)v, c);
}

template <>
void Set<at::BFloat16, CPUContext>(
    const std::int64_t /* N */,
    const at::BFloat16 b,
    at::BFloat16* v,
    CPUContext* c) {
  Set(0, b.x, (uint16_t*)v, c);
}


} // namespace math

} // namespace caffe2
