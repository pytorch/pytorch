#include "caffe2/operators/sparse_lp_regularizer_op.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {

template <>
bool SparseLpRegularizerOp<float, CPUContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
      this, Input(INDICES));
}

template <>
template <typename SIndex>
bool SparseLpRegularizerOp<float, CPUContext>::DoRunWithType() {
  const auto* indices = Input(INDICES).template data<SIndex>();
  auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<float>();

  // n: number of sparse embeddings to be normalized
  auto n = Input(INDICES).numel();
  if (n == 0) {
    return true;
  }

  // embedding length, e.g. 32, 64, 128
  auto block_size = Input(PARAM).size_from_dim(1);

  if (p_ == 2.0) { // L2 regularization
#ifdef LOG_FIRST_N
    LOG_FIRST_N(INFO, 3)
        << "Applying sparse L2 regularization with reg_lambda = "
        << reg_lambda_;
    LOG_FIRST_N(INFO, 3) << "L2 regularization input "
                         << paramOut[indices[0] * block_size];
#endif // LOG_FIRST_N
    for (int i = 0; i < n; ++i) {
      auto idx = indices[i];
      auto offsetIdx = idx * block_size;
      // Should probably be rewritten using Eigen.
      for (int j = 0; j < block_size; j++) {
        paramOut[offsetIdx + j] = paramOut[offsetIdx + j] * (1 - reg_lambda_);
      }
    }
#ifdef LOG_FIRST_N
    LOG_FIRST_N(INFO, 3) << "L2 regularization output "
                         << paramOut[indices[0] * block_size];
#endif // LOG_FIRST_N
  } else if (p_ == 1.0) { // L1 regularization
#ifdef LOG_FIRST_N
    LOG_FIRST_N(INFO, 3)
        << "Applying sparse L1 regularization with reg_lambda = "
        << reg_lambda_;
    LOG_FIRST_N(INFO, 3) << "L1 regularization input "
                         << paramOut[indices[0] * block_size];
#endif // LOG_FIRST_N
    for (int i = 0; i < n; ++i) {
      auto idx = indices[i];
      auto offsetIdx = idx * block_size;

      for (int j = 0; j < block_size; j++) {
        // I assume this can be sped up significantly.
        if (paramOut[offsetIdx + j] < -reg_lambda_) {
          paramOut[offsetIdx + j] += reg_lambda_;
        } else if (paramOut[offsetIdx + j] > reg_lambda_) {
          paramOut[offsetIdx + j] -= reg_lambda_;
        } else {
          paramOut[offsetIdx + j] = 0.0;
        }
      }
    }
#ifdef LOG_FIRST_N
    LOG_FIRST_N(INFO, 3) << "L1 regularization output "
                         << paramOut[indices[0] * block_size];
#endif // LOG_FIRST_N
  } else { // Currently only handling L1 and L2 regularization.
    return false;
  }
  return true;
}

REGISTER_CPU_OPERATOR(
    SparseLpRegularizer,
    SparseLpRegularizerOp<float, CPUContext>);
OPERATOR_SCHEMA(SparseLpRegularizer)
    .NumInputs(2, 3)
    .NumOutputs(1)
    .Input(0, "param", "Parameters to be regularized")
    .Input(1, "indices", "Sparse indices")
    .Input(
        2,
        "grad",
        "Gradient computed (optional - not used, this argument is for backwards compatibility)")
    .Output(0, "output_param", "Regularized parameters")
    .EnforceOneToOneInplace()
    .Arg("p", "Value of p in the Lp regularization to use. The default is 2.0.")
    .Arg(
        "reg_lambda",
        "Value of lambda (multiplier for the regularization term). The default is 1e-5.")
    .SetDoc(R"DOC(
Given a sparse matrix, apply Lp regularization.  Currently only L1 and L2 are implemented.
)DOC");

SHOULD_NOT_DO_GRADIENT(SparseLpNorm);

} // namespace caffe2
