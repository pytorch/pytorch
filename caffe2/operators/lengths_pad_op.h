#ifndef CAFFE2_OPERATORS_LENGTHS_PAD_OP_H_
#define CAFFE2_OPERATORS_LENGTHS_PAD_OP_H_

#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class LengthsPadOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  LengthsPadOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        OP_SINGLE_ARG(double, "padding_value", padding_value_, -1),
        OP_SINGLE_ARG(int, "target_length", target_length_, -1) {
    CAFFE_ENFORCE_GE(target_length_, 1, "target_length argument must be >= 1");
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, double, int32_t, int64_t>>::call(
        this, Input(DATA));
  }

  template <typename T>
  bool DoRunWithType() {
    auto& data = Input(DATA);
    auto& lengths = Input(LENGTHS);
    auto* output = Output(0);

    CAFFE_ENFORCE_EQ(lengths.ndim(), 1, "LENGTHS must be 1-D");
    CAFFE_ENFORCE_GE(data.ndim(), 1, "DATA should be at least 1-D");

    // Context::CopyFrom and math::Sum need the same context to avoid race
    // conditions
    CPUContext cpuContext;
    lengths_host_.CopyFrom(lengths, &cpuContext);

    auto lengths_size = lengths_host_.size();
    auto* lengths_data = lengths_host_.data<int32_t>();

    int32_t total_length = 0;
    math::Sum<int32_t, CPUContext>(
        lengths_size, lengths_data, &total_length, &cpuContext);

    CAFFE_ENFORCE_EQ(total_length, data.dim(0));

    auto shape = data.dims();
    shape[0] = lengths_size * target_length_;
    output->Resize(shape);

    auto block_size = data.size_from_dim(1);
    auto src_data = data.template data<T>();
    auto out_data = output->template mutable_data<T>();

    math::Set(
        output->size(), static_cast<T>(padding_value_), out_data, &context_);
    for (TIndex i = 0; i < lengths_size; ++i) {
      auto length = lengths_data[i];
      CAFFE_ENFORCE_GE(length, 0);
      CAFFE_ENFORCE_GE(
          target_length_,
          length,
          "Length at index = ",
          i,
          " is larger than target length");

      context_.template Copy<T, Context, Context>(
          block_size * length, src_data, out_data);

      out_data += block_size * target_length_;
      src_data += block_size * length;
    }
    return true;
  }

  INPUT_TAGS(DATA, LENGTHS);

 private:
  double padding_value_;
  int target_length_;
  TensorCPU lengths_host_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_LENGTHS_PAD_OP_H_
