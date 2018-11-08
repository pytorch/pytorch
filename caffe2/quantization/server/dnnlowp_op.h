#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

#include "caffe2/core/operator.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/quantization/server/caffe2_dnnlowp_utils.h"
#include "caffe2/quantization/server/dnnlowp.h"
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
 *          rather than doing end-to-end quantization.
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
 *        to deeplearning/quantization/dnnlowp/dnnlowp.cc for more details.
 *
 *        - activation_quantization_precision (default=8)
 *        - weight_quantization_precision (default=8)
 *        - requantization_multiplier_precision (default=32)
 *        - eltwise_quantization_precision (default=16)
 *        - force_scale_power_of_two (default=0)
 *        - preserve_sparsity (default=0)
 *        - activation_quantization_kind (default=min_max)
 *        - weight_quantization_kind (default=min_max)
 */
template <typename T, typename FP32_OP>
class DNNLowPOp : public Operator<CPUContext> {
  static_assert(std::is_integral<T>::value, "Integral required.");

 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);
  DNNLowPOp(const OperatorDef& operator_def, Workspace *ws)
    : Operator<CPUContext>(operator_def, ws),
      in_qparams_(InputSize()),
      qfactory_(dnnlowp::GetQuantizationFactoryOf(this)) {

#ifdef _OPENMP
    if (FLAGS_caffe2_omp_num_threads > 0) {
      omp_set_num_threads(FLAGS_caffe2_omp_num_threads);
    }
#endif
  }

  virtual ~DNNLowPOp() {
    if (measure_quantization_error_) {
      dnnlowp::ReportQuantizationError(this, quantization_error_stats_);
    }
  }

 protected:
  const TensorCPU& InputTensorCPU_(int idx) {
    return InputIsType<int8::Int8TensorCPU>(idx)
        ? OperatorBase::Input<int8::Int8TensorCPU>(idx).t
        : Input(idx);
  }

  TensorCPU* OutputTensorCPU_(int idx) {
    if (dequantize_output_) {
      return Output(idx);
    } else {
      return &Outputs()[idx]->template GetMutable<int8::Int8TensorCPU>()->t;
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

    const float *actual = nullptr;
    vector<float> actual_temp;
    if (OutputTensorCPU_(0)->template IsType<float>()) {
      actual = OutputTensorCPU_(0)->template data<float>();
    } else {
      actual_temp.resize(OutputTensorCPU_(0)->numel());
      Dequantize(
          OutputTensorCPU_(0)->template data<float>(),
          actual_temp.data(),
          OutputTensorCPU_(0)->numel(),
          out_qparams_);
      actual = actual_temp.data();
    }

    float *ref = Fp32Op_()->Get()->Output(0)->template mutable_data<float>();
    if (followed_by_ == "Relu") {
      for (int i = 0; i < Output(0)->numel(); ++i) {
        ref[i] = std::max(0.f, ref[i]);
      }
    }

    dnnlowp::MeasureQuantizationError(
        actual, ref, OutputTensorCPU_(0)->numel(), &quantization_error_stats_);
  }

  void RunOnDeviceEpilogue_() {
    if (dequantize_output_) {
      Dequantize(
          out_temp_.data(),
          OutputTensorCPU_(0)->template mutable_data<float>(),
          OutputTensorCPU_(0)->numel(),
          out_qparams_);
    } else {
      PropagateOutputTensorQuantizationParams(this, 0, out_qparams_);
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

  void GetOutputQuantizationParams_() {
    using namespace dnnlowp;

    ParseDNNLowPOperatorArguments_();

    if (HasStaticQuantization(this)) {
      out_qparams_ = GetStaticQuantizationParamsOf(this, 0);

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
      out_qparams_ = Fp32Op_()->GetOutputQuantizationParams(qfactory_.get());
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
