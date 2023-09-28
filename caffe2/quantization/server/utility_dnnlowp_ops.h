#pragma once

#include "caffe2/operators/utility_ops.h"
#include "caffe2/quantization/server/caffe2_dnnlowp_utils.h"
#include "caffe2/quantization/server/dnnlowp.h"
#include "caffe2/quantization/server/dnnlowp_op.h"

namespace caffe2 {

template <typename T, bool ReluFused = false>
class SumDNNLowPOp final : public DNNLowPOp<T, SumOp<CPUContext>> {
 public:
  SumDNNLowPOp(const OperatorDef& operator_def, Workspace* ws);
  bool RunOnDevice() override;

  USE_OPERATOR_FUNCTIONS(CPUContext);
  USE_DNNLOWP_OPERATOR_BASE_FUNCTIONS(T, SumOp<CPUContext>);

 private:
  bool GetQuantizationParameters_();

  dnnlowp::TensorQuantizationParams intermediate_qparams_;

  dnnlowp::RequantizationParams out_requantization_params_;
}; // class SumDNNLowPOp

template <typename T>
class GatherDNNLowPOp final : public GatherOp<CPUContext> {
  static_assert(std::is_integral<T>::value, "Integral required.");

 public:
  GatherDNNLowPOp(const OperatorDef& operator_def, Workspace* ws);
  ~GatherDNNLowPOp() override;
  bool RunOnDevice() override;

  template <typename Index>
  bool DoRunWithType() {
    // If we endup using it on GPU doing O(N) memcpy is probably not best :)
    // TODO: implement prefetching if it starts mattering (TF does it)
    auto& data = (this->template Input<int8::Int8TensorCPU>(DATA)).t;
    auto& indices = Input(INDICES);
    auto* output = &Outputs()[0]->template GetMutable<int8::Int8TensorCPU>()->t;

    CAFFE_ENFORCE_GE(data.ndim(), 1, "DATA should be at least 1-D");
    auto shape = indices.sizes().vec();
    shape.insert(shape.end(), data.sizes().begin() + 1, data.sizes().end());
    output->Resize(shape);

    int block_size = data.size_from_dim(1);
    auto block_bytesize = data.size_from_dim(1) * data.dtype().itemsize();
    int N = indices.numel();

    auto src_base = static_cast<const char*>(data.raw_data());
    const Index* idxs = indices.template data<Index>();
    auto out = static_cast<char*>(output->raw_mutable_data(data.dtype()));

    for (const auto i : c10::irange(N)) {
      auto idx = idxs[i];
      CAFFE_ENFORCE(
          0 <= idx && idx < data.size(0),
          "INDICES element is out of DATA bounds, id=",
          idx,
          " data_dim=",
          data.size(0));
      auto src = src_base + idx * block_bytesize;
      context_.CopyItemsSameDevice(
          data.dtype(), block_size, src, out + block_bytesize * i);
    }
    return true;
  }

  USE_OPERATOR_FUNCTIONS(CPUContext);

 private:
  OpWrapper<GatherOp<CPUContext>, T>* Fp32Op_() {
    if (!fp32_op_) {
      fp32_op_.reset(
          new OpWrapper<GatherOp<CPUContext>, T>(this, qfactory_.get()));
    }
    return fp32_op_.get();
  }

  std::unique_ptr<OpWrapper<GatherOp<CPUContext>, T>> fp32_op_;
  bool dequantize_output_{false}, measure_quantization_error_{false};

  std::unique_ptr<dnnlowp::QuantizationFactory> qfactory_;

  dnnlowp::QuantizationErrorStats quantization_error_stats_;

  bool arguments_parsed_{false};
}; // class GatherDNNLowPOp

namespace internal {

template <typename T, bool ReluFused>
void ElementWiseSumAVX2(
    const T* input0,
    const T* input1,
    T* output,
    int len,
    float a_scale,
    int32_t a_zero_point,
    float b_scale,
    int32_t b_zero_point,
    float c_scale,
    int32_t c_zero_points);

}

} // namespace caffe2
