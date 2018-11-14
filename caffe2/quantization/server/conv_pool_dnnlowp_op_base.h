#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

#include "caffe2/core/tensor_int8.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/quantization/server/caffe2_dnnlowp_utils.h"
#include "caffe2/quantization/server/op_wrapper.h"

#ifdef _OPENMP
C10_DECLARE_int(caffe2_omp_num_threads);
#endif

namespace caffe2 {

// TODO: code duplication with dnnlowp_op.h
template <typename T, typename FP32_OP>
class ConvPoolDNNLowPOpBase : public ConvPoolOpBase<CPUContext> {
  static_assert(std::is_integral<T>::value, "Integral required.");

 public:
  USE_CONV_POOL_BASE_FUNCTIONS(CPUContext);
  ConvPoolDNNLowPOpBase(const OperatorDef& operator_def, Workspace *ws)
    : ConvPoolOpBase<CPUContext>(operator_def, ws),
      in_qparams_(InputSize()),
      qfactory_(dnnlowp::GetQuantizationFactoryOf(this)) {
#ifdef _OPENMP
    if (FLAGS_caffe2_omp_num_threads > 0) {
      omp_set_num_threads(FLAGS_caffe2_omp_num_threads);
    }
#endif
  }

  virtual ~ConvPoolDNNLowPOpBase() {
    if (measure_quantization_error_) {
      dnnlowp::ReportQuantizationError(this, quantization_error_stats_);
      LOG(WARNING) << this->debug_def().output(0) << " with type "
                   << this->debug_def().type() << " has output qparams : "
                   << "scale " << out_qparams_.scale << " offset "
                   << out_qparams_.zero_point << "; ";
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
      out_temp_.resize(Output(0)->size());
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
    }
    else {
      actual_temp.resize(OutputTensorCPU_(0)->numel());
      Dequantize(
          OutputTensorCPU_(0)->template data<T>(),
          actual_temp.data(),
          OutputTensorCPU_(0)->numel(),
          out_qparams_);
      actual = actual_temp.data();
    }

    TensorCPU* float_tensor = Fp32Op_()->Get()->Output(0);
    float *ref = float_tensor->template mutable_data<float>();
    if (followed_by_ == "Relu" || debug_def().type() == "ConvRelu" ||
        debug_def().type() == "Int8ConvRelu") {
      for (int i = 0; i < OutputTensorCPU_(0)->numel(); ++i) {
        ref[i] = std::max(0.f, ref[i]);
      }
    }

    dnnlowp::MeasureQuantizationError(
        actual, ref, OutputTensorCPU_(0)->numel(), &quantization_error_stats_);
  }

  void RunOnDeviceEpilogue_() {
    if (dequantize_output_) {
      Dequantize(
        out_temp_.data(), OutputTensorCPU_(0)->template mutable_data<float>(),
        OutputTensorCPU_(0)->size(), out_qparams_);
    } else {
      PropagateOutputTensorQuantizationParams(this, 0, out_qparams_);
    }

    MeasureQuantizationError_();
  }

  void ParseDNNLowPOperatorArguments_() {
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

  void CreateSharedInt32Buffer_() {
    auto* mutexPtr =
        ws_->CreateBlob("__CAFFE2_DNNLOWP_SHARED_INT32_BUFFER_CPU_MUTEX__")
            ->GetMutable<std::unique_ptr<std::mutex>>();
    mutexPtr->reset(new std::mutex());
    ws_->CreateBlob("__CAFFE2_DNNLOWP_SHARED_INT32_BUFFER_CPU__");
  }

  void RunWithSharedInt32Buffer_(
      std::function<void(vector<int32_t>* Y_int32)> f) {
    auto* mutexBlob =
        ws_->GetBlob("__CAFFE2_DNNLOWP_SHARED_INT32_BUFFER_CPU_MUTEX__");
    CAFFE_ENFORCE(mutexBlob, "Must call CreateSharedInt32Buffer() first");

    auto* mutexPtr = mutexBlob->GetMutable<std::unique_ptr<std::mutex>>();
    std::lock_guard<std::mutex> g(**mutexPtr);

    auto* Y_int32 = ws_->GetBlob("__CAFFE2_DNNLOWP_SHARED_INT32_BUFFER_CPU__")
                        ->template GetMutable<vector<int32_t>>();
    f(Y_int32);
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

#define USE_CONV_POOL_DNNLOWP_OPERATOR_BASE_FUNCTIONS(T, FP32_OP)          \
  /* using override */ using BaseType = ConvPoolDNNLowPOpBase<T, FP32_OP>; \
  /* using override */ using BaseType::GetOutputQuantizationParams_;       \
  /* using override */ using BaseType::GetQuantizedOutputData_;            \
  /* using override */ using BaseType::Fp32Op_;                            \
  /* using override */ using BaseType::InputTensorCPU_;                    \
  /* using override */ using BaseType::MeasureQuantizationError_;          \
  /* using override */ using BaseType::OutputTensorCPU_;                   \
  /* using override */ using BaseType::RunOnDeviceEpilogue_;               \
  /* using override */ using BaseType::dequantize_output_;                 \
  /* using override */ using BaseType::followed_by_;                       \
  /* using override */ using BaseType::in_qparams_;                        \
  /* using override */ using BaseType::measure_quantization_error_;        \
  /* using override */ using BaseType::out_qparams_;                       \
  /* using override */ using BaseType::qfactory_;

} // namespace caffe2
