#include <ATen/core/op_registration/op_registration.h>
#include "caffe2/core/operator_c10wrapper.h"
#include "caffe2/core/tensor.h"
#include "caffe2/operators/elementwise_ops_utils.h"
#include "caffe2/utils/math.h"

using caffe2::BaseContext;
using caffe2::Tensor;

namespace caffe2 {
namespace {

template <class DataType>
void mul_op_cpu_impl(
    const at::Tensor& A_,
    const at::Tensor& B_,
    const at::Tensor& C_,
    bool legacy_broadcast,
    int64_t axis) {
  Tensor A{C10Tensor(A_)};
  Tensor B{C10Tensor(B_)};
  Tensor C{C10Tensor(C_)};
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

  caffe2::math::Mul(
      A_dims.size(),
      A_dims.data(),
      B_dims.size(),
      B_dims.data(),
      A.data<DataType>(),
      B.data<DataType>(),
      C.mutable_data<DataType>(),
      static_cast<CPUContext*>(&context));
}

static auto registry = c10::RegisterOperators().op(
    FunctionSchema(
        "_c10_experimental::Mul",
        "",
        (std::vector<c10::Argument>{
            c10::Argument("input1"),
            c10::Argument("input2"),
            c10::Argument("output"),
            c10::Argument("legacy_broadcast", BoolType::get()),
            c10::Argument("axis", IntType::get())}),
        (std::vector<c10::Argument>{})),
    c10::kernel<decltype(mul_op_cpu_impl<float>), &mul_op_cpu_impl<float>>(),
    c10::dispatchKey(CPUTensorId()));

} // namespace

REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_CPU(
    "_c10_experimental::Mul",
    C10Mul_DontUseThisOpYet)

} // namespace caffe2
