#ifndef CAFFE2_OPERATORS_INT8_RESIZE_NEAREST_OP_H_
#define CAFFE2_OPERATORS_INT8_RESIZE_NEAREST_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/operators/quantized/int8_utils.h"

namespace caffe2 {

namespace int8 {

class Int8ResizeNearestOp final : public Operator<CPUContext> {
 public:
  template <class... Args>
  explicit Int8ResizeNearestOp(Args&&... args)
      : Operator<CPUContext>(std::forward<Args>(args)...) {
    width_scale_ = this->template GetSingleArgument<float>("width_scale", 1);
    height_scale_ = this->template GetSingleArgument<float>("height_scale", 1);
    output_dims = this->template GetRepeatedArgument<int>("output_size", vector<int>{});
    CAFFE_ENFORCE_GT(width_scale_, 0);
    CAFFE_ENFORCE_GT(height_scale_, 0);
  }

  bool RunOnDevice() override {
    // Assume NHWC layout.
    const auto& X = Inputs()[0]->template Get<Int8TensorCPU>();
    auto* Y = Outputs()[0]->template GetMutable<Int8TensorCPU>();

    CAFFE_ENFORCE_EQ(4, X.t.dim());

    const int N = X.t.dim32(0);
    const int IH = X.t.dim32(1);
    const int IW = X.t.dim32(2);
    const int C = X.t.dim32(3);
    if (!output_dims.empty()) {
      CAFFE_ENFORCE_EQ(2, output_dims.size(), "Int8ResizeNearest expects 2 dim output size");
      height_scale_ = output_dims[0] / IH;
      width_scale_ = output_dims[1] / IW;
    }
    const int OW = IW * width_scale_;
    const int OH = IH * height_scale_;
    ReinitializeTensor(&Y->t, {N, OH, OW, C}, at::dtype<uint8_t>().device(CPU));
    Y->scale = X.scale;
    Y->zero_point = X.zero_point;

    int32_t Y_offset = this->template GetSingleArgument<int>("Y_zero_point", 0);
    auto Y_scale = this->template GetSingleArgument<float>("Y_scale", 1);
    CHECK_EQ(Y_offset, X.zero_point);
    CHECK_EQ(Y_scale, X.scale);

    const uint8_t* Xdata = X.t.data<uint8_t>();
    uint8_t* Ydata = Y->t.mutable_data<uint8_t>();

    for (int n = 0; n < N; ++n) {
      for (int y = 0; y < OH; ++y) {
        const int in_y = std::min((int)(y / height_scale_), (IH - 1));
        for (int x = 0; x < OW; ++x) {
          const int in_x = std::min((int)(x / width_scale_), (IW - 1));
          std::memcpy(
              &Ydata[C * x + C * OW * y + C * OW * OH * n],
              &Xdata[C * in_x + C * IW * in_y + C * IW * IH * n],
              C);
        }
      }
    }
    return true;
  }

 private:
  float width_scale_;
  float height_scale_;
  vector<int> output_dims;
};

} // namespace int8

} // namespace caffe2

#endif // CAFFE2_OPERATORS_INT8_RESIZE_NEAREST_OP_H_
