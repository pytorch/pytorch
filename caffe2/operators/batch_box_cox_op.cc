#include "caffe2/operators/batch_box_cox_op.h"

#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/perfkernels/batch_box_cox.h"

namespace caffe2 {

namespace {
template <typename T>
void BoxCoxNaive(
    int64_t N,
    int64_t D,
    const T* data_ptr,
    const T* lambda1_ptr,
    const T* lambda2_ptr,
    T* output_ptr) {
  constexpr T k_eps = static_cast<T>(1e-6);
  for (int64_t i = 0; i < N; i++) {
    for (int64_t j = 0; j < D; j++, data_ptr++, output_ptr++) {
      T lambda1_v = lambda1_ptr[j];
      T lambda2_v = lambda2_ptr[j];
      T tmp = std::max(*data_ptr + lambda2_v, k_eps);
      if (lambda1_v == 0) {
        *output_ptr = std::log(tmp);
      } else {
        *output_ptr = (std::pow(tmp, lambda1_v) - 1) / lambda1_v;
      }
    }
  }
}
}

template <>
template <typename T>
bool BatchBoxCoxOp<CPUContext>::DoRunWithType() {
  auto& data = Input(DATA);
  auto& lambda1 = Input(LAMBDA1);
  auto& lambda2 = Input(LAMBDA2);
  CAFFE_ENFORCE_GE(data.dim(), 1);
  auto N = data.size(0);
  auto D = data.size_from_dim(1);

  auto* output = Output(0, Input(DATA).sizes(), at::dtype<T>());
  auto* output_ptr = output->template mutable_data<T>();

  if (data.numel() <= 0) {
    return true;
  }

  CAFFE_ENFORCE_EQ(lambda1.numel(), D);
  CAFFE_ENFORCE_EQ(lambda2.numel(), D);

  const auto* data_ptr = data.template data<T>();
  const auto* lambda1_ptr = lambda1.template data<T>();
  const auto* lambda2_ptr = lambda2.template data<T>();

#ifdef CAFFE2_USE_MKL
  if (min_block_size_ < 1) {
    BoxCoxNaive(N, D, data_ptr, lambda1_ptr, lambda2_ptr, output_ptr);
    return true;
  }
  caffe2::compute_batch_box_cox(
    N, D, min_block_size_, data_ptr, lambda1_ptr, lambda2_ptr, output_ptr);
#else
  BoxCoxNaive(N, D, data_ptr, lambda1_ptr, lambda2_ptr, output_ptr);
#endif
  return true;
}


namespace {

REGISTER_CPU_OPERATOR(BatchBoxCox, BatchBoxCoxOp<CPUContext>);
OPERATOR_SCHEMA(BatchBoxCox)
    .NumInputs(3)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInput(0)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
Input `data` is a N * D matrix. Apply box-cox transform for each column.
`lambda1` and `lambda2` is of size D that defines the hyper-parameters for
the transform of each column `x` of the input `data`:

    ln(x + lambda2), if lambda1 == 0
    ((x + lambda2)^lambda1 - 1)/lambda1, if lambda1 != 0

)DOC")
    .Input(0, "data", "input float or double N * D matrix")
    .Input(1, "lambda1", "tensor of size D with the same type as data")
    .Input(2, "lambda2", "tensor of size D with the same type as data")
    .Output(0, "output", "output matrix that applied box-cox transform");

GRADIENT_NOT_IMPLEMENTED_YET(BatchBoxCox);
} // namespace
} // namespace caffe2

C10_EXPORT_CAFFE2_OP_TO_C10_CPU(
    BatchBoxCox,
    "_caffe2::BatchBoxCox(Tensor data, Tensor lambda1, Tensor lambda2, int min_block_size = 256) -> Tensor results",
    caffe2::BatchBoxCoxOp<caffe2::CPUContext>);
