#include "caffe2/operators/elementwise_mul_op.h"
#include "caffe2/quantization/server/elementwise_dnnlowp_op.h"
#include "caffe2/quantization/server/op_wrapper.h"
#include "caffe2/quantization/server/sigmoid.h"

namespace caffe2 {

using namespace std;
using namespace dnnlowp;

using MulFp32Op =
    BinaryElementwiseOp<NumericTypes, CPUContext, MulFunctor<CPUContext>>;

template <typename T>
class MulDNNLowPOp : public BinaryElementwiseDNNLowPOp<T, MulFp32Op> {
 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);
  USE_DNNLOWP_OPERATOR_BASE_FUNCTIONS(T, MulFp32Op);
  using BinaryElementwiseDNNLowPOp<T, MulFp32Op>::axis_;
  using BinaryElementwiseDNNLowPOp<T, MulFp32Op>::enable_broadcast_;
  using BinaryElementwiseDNNLowPOp<T, MulFp32Op>::requantization_params_;

  MulDNNLowPOp(const OperatorDef& operator_def, Workspace* ws)
      : BinaryElementwiseDNNLowPOp<T, MulFp32Op>(operator_def, ws) {}

  bool RunOnDevice() override {
    if (!GetQuantizationParameters_()) {
      return false;
    }

    const auto& A = InputTensorCPU_(0);
    const auto& B = InputTensorCPU_(1);
    auto* C = OutputTensorCPU_(0);
    CAFFE_ENFORCE(
        &B != C || !enable_broadcast_,
        "In-place is allowed only with the first tensor when broadcasting");
    C->ResizeLike(A);

    // Quantize inputs if needed
    vector<T> A_temp, B_temp;
    const T* A_quantized =
        QuantizeInputIfNeeded<T>(this, 0, in_qparams_[0], A_temp);
    const T* B_quantized =
        QuantizeInputIfNeeded<T>(this, 1, in_qparams_[1], B_temp);

    T* C_quantized = GetQuantizedOutputData_();

    if (!enable_broadcast_) {
      CAFFE_ENFORCE_EQ(
          A.sizes(),
          B.sizes(),
          "Dimension mismatch - did you forget to set broadcast=1?");
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (int i = 0; i < C->size(); ++i) {
        int32_t raw = (A_quantized[i] - in_qparams_[0].zero_point) *
            (B_quantized[i] - in_qparams_[1].zero_point);
        C_quantized[i] = fbgemm::Requantize<T>(raw, requantization_params_);
      }
    } else if (B.size() == 1) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (int i = 0; i < C->size(); ++i) {
        int32_t raw = (A_quantized[i] - in_qparams_[0].zero_point) *
            (B_quantized[0] - in_qparams_[1].zero_point);
        C_quantized[i] = fbgemm::Requantize<T>(raw, requantization_params_);
      }
    } else {
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      size_t pre, n, post;
      std::tie(pre, n, post) =
          elementwise_ops_utils::ComputeLegacyBroadcastSizes(A, B, axis_);
#ifdef _OPENMP
#pragma omp parallel for
#endif
      // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
      for (int i = 0; i < pre; ++i) {
        // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
        for (int j = 0; j < n; ++j) {
          // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
          for (int k = 0; k < post; ++k) {
            int32_t raw = (A_quantized[((i * n) + j) * post + k] -
                           in_qparams_[0].zero_point) *
                (B_quantized[j] - in_qparams_[1].zero_point);
            C_quantized[((i * n) + j) * post + k] =
                fbgemm::Requantize<T>(raw, requantization_params_);
          }
        }
      }
    }

    RunOnDeviceEpilogue_();

    return true;
  }

 private:
  bool GetQuantizationParameters_() {
    // Choose quantization for A and B
    in_qparams_[0] =
        GetInputTensorQuantizationParamsOf(this, 0, qfactory_.get());
    in_qparams_[1] =
        GetInputTensorQuantizationParamsOf(this, 1, qfactory_.get());

    GetOutputQuantizationParams_();

    float real_multiplier =
        in_qparams_[0].scale * in_qparams_[1].scale / out_qparams_.scale;
    requantization_params_ = qfactory_->ChooseRequantizationMultiplier(
        real_multiplier, out_qparams_);

    return true;
  }
}; // class MulDNNLowPOp

REGISTER_CPU_OPERATOR_WITH_ENGINE(Mul, DNNLOWP, MulDNNLowPOp<uint8_t>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(Int8Mul, DNNLOWP, MulDNNLowPOp<uint8_t>);

} // namespace caffe2
