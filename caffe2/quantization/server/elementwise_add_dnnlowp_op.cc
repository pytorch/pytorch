#include "elementwise_dnnlowp_op.h"

#include "caffe2/operators/elementwise_add_op.h"
#include "caffe2/quantization/server/sigmoid.h"

#include "dnnlowp_partition.h"
#include "op_wrapper.h"
#include "utility_dnnlowp_ops.h"

namespace caffe2 {

using namespace std;
using namespace dnnlowp;

using AddFp32Op =
    BinaryElementwiseOp<NumericTypes, CPUContext, AddFunctor<CPUContext>>;

template <typename T>
class AddDNNLowPOp : public BinaryElementwiseDNNLowPOp<T, AddFp32Op> {
 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);
  USE_DNNLOWP_OPERATOR_BASE_FUNCTIONS(T, AddFp32Op);
  using BinaryElementwiseDNNLowPOp<T, AddFp32Op>::axis_;
  using BinaryElementwiseDNNLowPOp<T, AddFp32Op>::enable_broadcast_;
  using BinaryElementwiseDNNLowPOp<T, AddFp32Op>::requantization_params_;

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  AddDNNLowPOp(const OperatorDef& operator_def, Workspace* ws)
      : BinaryElementwiseDNNLowPOp<T, AddFp32Op>(operator_def, ws) {}

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

    T* C_quantized = GetQuantizedOutputData_();

    if (A.template IsType<T>() && B.template IsType<T>() &&
        A.numel() == B.numel() && is_same<T, uint8_t>::value &&
        GetCpuId().avx2() && GetCpuId().fma()) {
      // fast path
      // NOTE: this path does addition in floating point unlike slow path that
      // does everything in fixed-point. So they are numerically different.
#ifdef _OPENMP
#pragma omp parallel
#endif
      {
        constexpr int VLEN = 8;
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        int j_begin, j_end;
        tie(j_begin, j_end) = Get1DPartition(
            A.numel(),
            dnnlowp_get_num_threads(),
            dnnlowp_get_thread_num(),
            VLEN);

        internal::ElementWiseSumAVX2<T, false /*ReluFused*/>(
            A.template data<T>() + j_begin,
            B.template data<T>() + j_begin,
            C_quantized + j_begin,
            j_end - j_begin,
            in_qparams_[0].scale,
            in_qparams_[0].zero_point,
            in_qparams_[1].scale,
            in_qparams_[1].zero_point,
            out_qparams_.scale,
            out_qparams_.zero_point);
      } // omp parallel

      RunOnDeviceEpilogue_();

      return true;
    }

    // Quantize inputs if needed
    vector<int32_t> A_quantized(A.numel()), B_quantized(B.numel());
    for (int i = 0; i < 2; ++i) {
      int32_t* quantized_in = i == 0 ? A_quantized.data() : B_quantized.data();
      if (InputTensorCPU_(i).template IsType<T>()) {
        float real_multiplier =
            in_qparams_[i].scale / intermediate_qparams_.scale;
        RequantizationParams in_requantization_params =
            qfactory_->ChooseRequantizationMultiplier(
                real_multiplier, intermediate_qparams_);

        const T* input_data = InputTensorCPU_(i).template data<T>();
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int j = 0; j < InputTensorCPU_(i).numel(); ++j) {
          quantized_in[j] = fbgemm::Requantize<int32_t>(
              input_data[j] - in_qparams_[i].zero_point,
              in_requantization_params);
        }
      } else {
        assert(A.template IsType<float>());
        const float* input_data = InputTensorCPU_(i).template data<float>();
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int j = 0; j < InputTensorCPU_(i).numel(); ++j) {
          quantized_in[j] = fbgemm::Quantize<uint32_t>(
              input_data[j],
              intermediate_qparams_.zero_point,
              intermediate_qparams_.scale,
              qfactory_->GetEltwiseQuantizePrecision());
        }
      }
    }

    int32_t intermediate_zero_point =
        intermediate_qparams_.zero_point * InputSize();

    if (!enable_broadcast_) {
      CAFFE_ENFORCE_EQ(
          A.sizes(),
          B.sizes(),
          "Dimension mismatch - did you forget to set broadcast=1?");
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (int i = 0; i < C->numel(); ++i) {
        int32_t raw = A_quantized[i] + B_quantized[i] - intermediate_zero_point;
        C_quantized[i] = fbgemm::Requantize<T>(raw, requantization_params_);
      }
    } else if (B.numel() == 1) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (int i = 0; i < C->numel(); ++i) {
        int32_t raw = A_quantized[i] + B_quantized[0] - intermediate_zero_point;
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
            int32_t raw = A_quantized[((i * n) + j) * post + k] +
                B_quantized[j] - intermediate_zero_point;
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
    // Find global min and max of all inputs
    float global_min = numeric_limits<float>::max(),
          global_max = numeric_limits<float>::lowest();

    for (int i = 0; i < InputSize(); ++i) {
      in_qparams_[i] =
          GetInputTensorQuantizationParamsOf(this, i, qfactory_.get());

      global_min = std::min(global_min, in_qparams_[i].Min());
      global_max = std::max(global_max, in_qparams_[i].Max());
    }

    intermediate_qparams_ = qfactory_->ChooseQuantizationParams(
        global_min,
        global_max,
        qfactory_->GetEltwiseQuantizePrecision(),
        qfactory_->GetPreserveActivationSparsity());

    GetOutputQuantizationParams_();

    float real_multiplier = intermediate_qparams_.scale / out_qparams_.scale;
    requantization_params_ = qfactory_->ChooseRequantizationMultiplier(
        real_multiplier, out_qparams_);

    return true;
  }

  dnnlowp::TensorQuantizationParams intermediate_qparams_;
}; // class AddDNNLowPOp

REGISTER_CPU_OPERATOR_WITH_ENGINE(Add, DNNLOWP, AddDNNLowPOp<uint8_t>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(Int8Add, DNNLOWP, AddDNNLowPOp<uint8_t>);

} // namespace caffe2
