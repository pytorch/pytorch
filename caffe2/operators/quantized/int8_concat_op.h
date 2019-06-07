#ifndef CAFFE2_OPERATORS_INT8_CONCAT_OP_H_
#define CAFFE2_OPERATORS_INT8_CONCAT_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/operators/quantized/int8_utils.h"

namespace caffe2 {

namespace int8 {

class Int8ConcatOp final : public Operator<CPUContext> {
 public:
  template <class... Args>
  explicit Int8ConcatOp(Args&&... args)
      : Operator<CPUContext>(std::forward<Args>(args)...) {
    // concat supports more than NHWC format
    if (this->template GetSingleArgument<string>("order", "") == "NHWC") {
      // Default to C axis
      axis_ = this->template GetSingleArgument<int>("axis", 3);
      CHECK_GE(axis_, 0);
      CHECK_LT(axis_, 4);
    } else if (
        this->template GetSingleArgument<string>("order", "") == "NCHW") {
      axis_ = this->template GetSingleArgument<int>("axis", 1);
      CHECK_GE(axis_, 0);
      CHECK_LT(axis_, 4);
    } else {
      axis_ = this->template GetSingleArgument<int>("axis", 0);
    }
  }

  bool RunOnDevice() override {
    const auto& X0 = Inputs()[0]->template Get<Int8TensorCPU>();
    auto* Y = Outputs()[0]->template GetMutable<Int8TensorCPU>();
    Y->scale = X0.scale;
    Y->zero_point = X0.zero_point;
    int32_t Y_offset = this->template GetSingleArgument<int>("Y_zero_point", 0);
    auto Y_scale = this->template GetSingleArgument<float>("Y_scale", 1);
    CHECK_EQ(Y_offset, X0.zero_point);
    CHECK_EQ(Y_scale, X0.scale);
    CHECK_GE(X0.zero_point, std::numeric_limits<uint8_t>::min());
    CHECK_LE(X0.zero_point, std::numeric_limits<uint8_t>::max());
    auto Y_dims = X0.t.sizes().vec();
    if (this->template GetSingleArgument<string>("order", "") == "NHWC") {
      CHECK_EQ(Y_dims.size(), 4);
    }
    for (auto i = 1; i < InputSize(); ++i) {
      const auto& Xi = Inputs()[i]->template Get<Int8TensorCPU>();
      CHECK_EQ(Xi.t.dim(), Y_dims.size());
      for (auto j = 0; j < Y_dims.size(); ++j) {
        if (j != axis_) {
          CHECK_EQ(Xi.t.size(j), Y_dims[j]);
        }
      }
      Y_dims[axis_] += Xi.t.size(axis_);
    }
    ReinitializeTensor(&Y->t, Y_dims, at::dtype<uint8_t>().device(CPU));
    int before = X0.t.size_to_dim(axis_);
    int after = X0.t.size_from_dim(axis_ + 1);
    const auto C_total = Y_dims[axis_];
    size_t C_offset = 0;
    for (auto i = 0; i < InputSize(); ++i) {
      const auto& Xi = Inputs()[i]->template Get<Int8TensorCPU>();
      // Copy the NxHxWxC input slice to NxHxWx[C_offset:C_offset + C].
      const auto Ci = Xi.t.size(axis_);
      math::CopyMatrix<CPUContext>(
          Xi.t.itemsize(),
          before,
          Ci * after,
          Xi.t.template data<uint8_t>(),
          Ci * after,
          Y->t.template mutable_data<uint8_t>() + C_offset,
          C_total * after,
          &context_,
          Xi.t.dtype().copy());
      C_offset += Ci * after * Xi.t.itemsize();
    }
    return true;
  }

 private:
  int axis_;
};

} // namespace int8

} // namespace caffe2

#endif // CAFFE2_OPERATORS_INT8_CONCAT_OP_H_
