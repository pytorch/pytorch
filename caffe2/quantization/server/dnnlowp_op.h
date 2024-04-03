#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

#include "caffe2/core/operator.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/quantization/server/caffe2_dnnlowp_utils.h"
#include "caffe2/quantization/server/dnnlowp.h"
#include "caffe2/quantization/server/fbgemm_pack_blob.h"
#include "caffe2/quantization/server/op_wrapper.h"
#include "caffe2/quantization/server/sigmoid.h"
#include "caffe2/quantization/server/tanh.h"

#ifdef _OPENMP
C10_DECLARE_int(caffe2_omp_num_threads);
#endif

namespace caffe2 {

/**
 * @brief A convenient base class for C2 operators with DNNLOWP engine.
 *        DNNLOWP ops give flexibility on the type of input/output blobs.
 *        For example, some inputs can be the usual fp32 tensor and they will be
 *        quantized before doing actual computation.
 *        Otherwise, the inputs should be pre-quantized Int8TensorCPU.
 *        A few constraints: when the weight is pre-quantized if and only if the
 *        bias is also pre-quantized.
 *
 *        static quantization vs. dynamic quantization
 *        When Y_scale and Y_zero_point (optional with default = 0) arg is set,
 *        and dequantize_output is false, we do static quantization, meaning
 *        we're using the same pre-computed scale and zero_point for the output
 *        activation tensor.
 *        Otherwise, we do dynamic quantization by looking at the min/max of
 *        output activation tensor for each batch.
 *        Y_scale and Y_zero_point arguments are used for static quantization.
 *        scale and zero_point of Int8TensorCPU is used for carrying
 *        quantization information across operators both in static and dynamic
 *        quantization. This means scale and zero_point of Int8TensorCPU is
 *        valid only for the current batch and will be reset in the next batch
 *        when dynamic quantization is used.
 *
 *        C2 operators with DNNLOWP engine have the following arguments:
 *        - dequantize_output (default=false): when true, output is dequantized
 *          as fp32. Useful when we're only quantizing individual operators
 *          rather than doing end-to-end quantization. Conv operators don't
            support dequantize_output option as an exception because doing so
            complicate the implementation significantly and having a separate
            Dequantize operator doesn't add much overhead because Conv ops are
            usually used in deep networks where regions of quantization are
            long chains.
 *        - followed_by (default=null): can be relu, sigmoid, or tanh. When
 *          specified, the current operator is only followed by relu, sigmoid,
 *          or tanh, and this fact can be used for more accurate output
 *          quantization.
 *        - measure_quantization_error (default=false): when true, L2 error
 *          with respect to the baseline C2 operator in fp32 is reported.
 *          WARNING: turning this option will make performance very slow and
 *          this option is intended for debugging accuracy issues.
 *
 *        For the following quantization method related options, please refer
 *        to caffe2/quantization/server/dnnlowp.cc for more details.
 *
 *        - activation_quantization_precision (default=8)
 *        - weight_quantization_precision (default=8)
 *        - requantization_multiplier_precision (default=32)
 *        - eltwise_quantization_precision (default=16)
 *        - force_scale_power_of_two (default=0)
 *        - preserve_activation_sparsity (default=0)
 *        - preserve_weight_sparsity (default=0)
 *        - activation_quantization_kind (default=min_max)
 *        - weight_quantization_kind (default=min_max)
 */
template <typename T, typename FP32_OP>
class DNNLowPOp : public Operator<CPUContext> {
  static_assert(std::is_integral<T>::value, "Integral required.");

 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);
  DNNLowPOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws),
        in_qparams_(InputSize()),
        qfactory_(dnnlowp::GetQuantizationFactoryOf(this)) {
#ifdef _OPENMP
    if (FLAGS_caffe2_omp_num_threads > 0) {
      omp_set_num_threads(FLAGS_caffe2_omp_num_threads);
    }
#endif
    if (this->debug_def().engine() == "DNNLOWP_16" ||
        this->debug_def().engine() == "DNNLOWP_ROWWISE_16") {
      LOG(WARNING)
          << this->debug_def().engine()
          << " is an experimental feature mostly for testing accuracy with "
             "fixed-point precision higher than 8 and performance is very slow";
    }
  }

  virtual ~DNNLowPOp() {
    if (measure_quantization_error_) {
      dnnlowp::ReportQuantizationError(this, quantization_error_stats_);
    }
  }

 protected:
  const TensorCPU& InputTensorCPU_(int idx) {
    if (InputIsType<int8::Int8TensorCPU>(idx)) {
      return this->Input<int8::Int8TensorCPU>(idx).t;
    } else if (InputIsType<Int8FCDNNLowPPackedWeightBlob>(idx)) {
      return this->Input<Int8FCDNNLowPPackedWeightBlob>(idx).original_tensor;
    } else {
      return Input(idx);
    }
  }

  TensorCPU* OutputTensorCPU_(int idx) {
    if (dequantize_output_) {
      return Output(idx);
    } else {
      return &Outputs()[idx]->template GetMutable<int8::Int8TensorCPU>()->t;
    }
  }

  Tensor*
  OutputTensorCPU_(int idx, at::IntArrayRef dims, at::TensorOptions options) {
    if (dequantize_output_) {
      return Output(idx, dims, options.device(CPU));
    } else {
      auto* t = &Outputs()[idx]->template GetMutable<int8::Int8TensorCPU>()->t;
      ReinitializeTensor(t, dims, options.device(CPU));
      return t;
    }
  }

  T* GetQuantizedOutputData_() {
    if (dequantize_output_) {
      out_temp_.resize(Output(0)->numel());
      return out_temp_.data();
    } else {
      return OutputTensorCPU_(0)->template mutable_data<T>();
    }
  }

  void MeasureQuantizationError_() {
    if (!measure_quantization_error_ || !Fp32Op_()) {
      return;
    }

    const float* actual = nullptr;
    std::vector<float> actual_temp;
    if (OutputTensorCPU_(0)->template IsType<float>()) {
      actual = OutputTensorCPU_(0)->template data<float>();
      std::string op_type = this->debug_def().type();
      bool relu_fused = op_type.length() >= 4 &&
          op_type.compare(op_type.length() - 4, 4, "Relu") == 0;
      if (GetSingleArgument<std::string>("followed_by", "") == "Relu" &&
          !relu_fused) {
        // If dequantize_output_ is true and relu is not fused,
        // dnnlowp op won't clip negative values. Do it here.
        actual_temp.resize(OutputTensorCPU_(0)->numel());
        for (int i = 0; i < Output(0)->numel(); ++i) {
          actual_temp[i] = std::max(0.f, actual[i]);
        }
        actual = actual_temp.data();
      }
    } else {
      actual_temp.resize(OutputTensorCPU_(0)->numel());
      fbgemm::Dequantize<T>(
          OutputTensorCPU_(0)->template data<T>(),
          actual_temp.data(),
          OutputTensorCPU_(0)->numel(),
          out_qparams_);
      actual = actual_temp.data();
    }

    float* ref = Fp32Op_()->Get()->Output(0)->template mutable_data<float>();
    if (followed_by_ == "Relu") {
      for (int i = 0; i < OutputTensorCPU_(0)->numel(); ++i) {
        ref[i] = std::max(0.f, ref[i]);
      }
    }

    dnnlowp::MeasureQuantizationError(
        actual, ref, OutputTensorCPU_(0)->numel(), &quantization_error_stats_);
  }

  void RunOnDeviceEpilogue_() {
    if (dequantize_output_) {
      fbgemm::Dequantize<T>(
          out_temp_.data(),
          OutputTensorCPU_(0)->template mutable_data<float>(),
          OutputTensorCPU_(0)->numel(),
          out_qparams_);
    } else {
      dnnlowp::PropagateOutputTensorQuantizationParams(this, 0, out_qparams_);
    }

    MeasureQuantizationError_();
  }

  void ParseDNNLowPOperatorArguments_() {
    // Ideally, this should be done in constructor but any modification of
    // arguments in ParseDNNLowPOperatorArguments will be ignored if we call
    // this from constructor.
    // Make sure all derived classes call this "early enough" so that they
    // use correct parameters.
    if (!arguments_parsed_) {
      dnnlowp::ParseDNNLowPOperatorArguments(
          this,
          &dequantize_output_,
          &measure_quantization_error_,
          &followed_by_);
      arguments_parsed_ = true;
    }
  }

  void GetOutputQuantizationParams_(
      dnnlowp::TensorQuantizationParams* out_qparams_overwrite = nullptr) {
    using namespace dnnlowp;

    ParseDNNLowPOperatorArguments_();

    if (HasStaticQuantization(this)) {
      if (out_qparams_overwrite != nullptr) {
        out_qparams_ = *out_qparams_overwrite;
      } else {
        out_qparams_ = GetStaticQuantizationParamsOf(this, 0);
      }

      if (measure_quantization_error_) {
        // To measure quantization error, run ref fp32 impl.
        // This doesn't really belong here but we need to run the reference fp32
        // implementation before quantized computation of some inplace operators
        // will overwrite their inputs.
        Fp32Op_()->DequantizeInput();
        Fp32Op_()->Get()->RunOnDevice();
      }
    } else {
      // TODO: this is only needed when dequantize_output_ == false but leave
      // as it is now because some code relies on out_qparams_ initialized even
      // though it never actually uses it.
      Fp32Op_()->DequantizeInput();
      Fp32Op_()->Get()->RunOnDevice();
      if (out_qparams_overwrite != nullptr) {
        out_qparams_ = *out_qparams_overwrite;
      } else {
        out_qparams_ = Fp32Op_()->GetOutputQuantizationParams(qfactory_.get());
      }
    }
  }

  OpWrapper<FP32_OP, T>* Fp32Op_() {
    if (!fp32_op_) {
      fp32_op_.reset(new OpWrapper<FP32_OP, T>(this, qfactory_.get()));
    }
    return fp32_op_.get();
  }

  bool dequantize_output_{false}, measure_quantization_error_{false};
  std::string followed_by_;

  std::vector<dnnlowp::TensorQuantizationParams> in_qparams_;
  dnnlowp::TensorQuantizationParams out_qparams_;

  std::unique_ptr<OpWrapper<FP32_OP, T>> fp32_op_;
  std::unique_ptr<dnnlowp::QuantizationFactory> qfactory_;

  std::vector<T> out_temp_;
  // Buffer to store quantized output temporarily
  // when we output dequantized values.

  dnnlowp::QuantizationErrorStats quantization_error_stats_;

  bool arguments_parsed_{false};
};

#define USE_DNNLOWP_OPERATOR_BASE_FUNCTIONS(T, FP32_OP)              \
  /* using override */ using BaseType = DNNLowPOp<T, FP32_OP>;       \
  /* using override */ using BaseType::GetOutputQuantizationParams_; \
  /* using override */ using BaseType::GetQuantizedOutputData_;      \
  /* using override */ using BaseType::Fp32Op_;                      \
  /* using override */ using BaseType::InputTensorCPU_;              \
  /* using override */ using BaseType::MeasureQuantizationError_;    \
  /* using override */ using BaseType::OutputTensorCPU_;             \
  /* using override */ using BaseType::RunOnDeviceEpilogue_;         \
  /* using override */ using BaseType::dequantize_output_;           \
  /* using override */ using BaseType::followed_by_;                 \
  /* using override */ using BaseType::in_qparams_;                  \
  /* using override */ using BaseType::measure_quantization_error_;  \
  /* using override */ using BaseType::out_qparams_;                 \
  /* using override */ using BaseType::qfactory_;

inline int dnnlowp_get_num_threads() {
#ifdef _OPENMP
  return omp_get_num_threads();
#else
  return 1;
#endif
}

inline int dnnlowp_get_max_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

inline int dnnlowp_get_thread_num() {
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

} // namespace caffe2
