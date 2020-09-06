#pragma once

#include <caffe2/core/operator.h>

C10_DECLARE_bool(caffe2_fbgemm_fake_fp16_clamp);

namespace caffe2 {

template <class Context>
class SumFP16FP16AccOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(SumFP16FP16AccOp);

  bool DoRunWithFloat() {
    auto& input0 = Input(0);

    size_t N = input0.numel();
    auto* output = Output(0, input0.sizes(), at::dtype<float>());
    // Dimension checking
    for (int i = 1; i < InputSize(); ++i) {
      if (output->sizes() != Input(i).sizes()) {
        CAFFE_THROW(
            "Check failed: output->sizes() == Input(i).sizes().",
            "Description: Input #",
            i,
            ", input dimension:",
            Input(i).sizes(),
            " should match output dimension: ",
            output->sizes());
      }
    }

    float* output_data = output->template mutable_data<float>();
    memset(output_data, 0, sizeof(float) * input0.numel());

    std::vector<float> t1(N);
    std::vector<float> t2(N);

    for (auto i = 0; i < InputSize(); i++) {
      fbgemm::RoundToFloat16(
          Input(i).template data<float>(),
          t1.data(),
          N,
          FLAGS_caffe2_fbgemm_fake_fp16_clamp);
      fbgemm::RoundToFloat16(
          output_data, t2.data(), N, FLAGS_caffe2_fbgemm_fake_fp16_clamp);

      math::Add(N, t1.data(), t2.data(), output_data, &context_);
    }
    fbgemm::RoundToFloat16(
        output_data, output_data, N, FLAGS_caffe2_fbgemm_fake_fp16_clamp);

    return true;
  }

  bool RunOnDevice() override {
    if (Input(0).template IsType<float>()) {
      return DoRunWithFloat();
    } else {
      CAFFE_THROW(
          "Sum operator only supports 32-bit float, but",
          " input was of type ",
          Input(0).dtype().name());
    }
  }
};

} // namespace caffe2
