#include <c10/core/dispatch/KernelRegistration.h>
#include "caffe2/operators/elementwise_ops_utils.h"
#include "caffe2/operators/experimental/c10/schemas/add.h"
#include "caffe2/utils/math.h"

using caffe2::BaseContext;
using caffe2::Tensor;

namespace caffe2 {
namespace {

template <class DataType>
void add_op_cpu_impl(
    const C10Tensor& A_,
    const C10Tensor& B_,
    const C10Tensor& C_,
    bool legacy_broadcast,
    int axis) {
  Tensor A(A_);
  Tensor B(B_);
  Tensor C(C_);
  CPUContext context;
  const DataType* A_data = A.template data<DataType>();
  const DataType* B_data = B.template data<DataType>();
  std::vector<int> A_dims;
  std::vector<int> B_dims;

  if (legacy_broadcast) {
    CAFFE_ENFORCE(
        !B.is_same(C),
        "In-place is allowed only with the first tensor when "
        "legacy-broadcasting");
    C.ResizeLike(A);
    if (B.numel() == 1) {
      A_dims = {static_cast<int>(A.numel())};
      B_dims = {1};
    } else {
      size_t pre, n, post;
      std::tie(pre, n, post) =
          caffe2::elementwise_ops_utils::ComputeLegacyBroadcastSizes(
              A, B, axis);
      A_dims = {
          static_cast<int>(pre), static_cast<int>(n), static_cast<int>(post)};
      B_dims = {static_cast<int>(n), 1};
    }
  } else {
    std::copy(A.sizes().cbegin(), A.sizes().cend(), std::back_inserter(A_dims));
    std::copy(B.sizes().cbegin(), B.sizes().cend(), std::back_inserter(B_dims));
    const std::vector<int> C_dims =
        caffe2::elementwise_ops_utils::ComputeBinaryBroadcastForwardDims(
            A_dims, B_dims);
    if (A.is_same(C)) {
      CAFFE_ENFORCE_EQ(C_dims, A_dims);
    } else if (B.is_same(C)) {
      CAFFE_ENFORCE_EQ(C_dims, B_dims);
    } else {
      C.Resize(C_dims);
    }
  }
  auto* C_data = C.template mutable_data<DataType>();

  caffe2::math::Add(
      A_dims.size(),
      A_dims.data(),
      B_dims.size(),
      B_dims.data(),
      A.data<DataType>(),
      B.data<DataType>(),
      C.mutable_data<DataType>(),
      static_cast<CPUContext*>(&context));
}
} // namespace
} // namespace caffe2

namespace c10 {
C10_REGISTER_KERNEL(caffe2::ops::Add)
    .kernel(&caffe2::add_op_cpu_impl<float>)
    .dispatchKey(c10::DispatchKey<2>{
        c10::details::TensorParameterDispatchKey{DeviceTypeId::CPU,
                                                 LayoutId(0),
                                                 caffe2::TypeMeta::Id<float>()},
        c10::details::TensorParameterDispatchKey{
            DeviceTypeId::CPU,
            LayoutId(0),
            caffe2::TypeMeta::Id<float>()}});
} // namespace c10
