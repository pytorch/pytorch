#ifndef CAFFE2_OPERATORS_INT8_SLICE_OP_H_
#define CAFFE2_OPERATORS_INT8_SLICE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/operators/quantized/int8_utils.h"
#include "caffe2/operators/slice_op.h"

namespace caffe2 {

namespace int8 {

class Int8SliceOp final : public SliceOp<CPUContext> {
 public:
  template <class... Args>
  explicit Int8SliceOp(Args&&... args) : SliceOp(std::forward<Args>(args)...) {}

  bool RunOnDevice() override {
    if (InputSize() > 1) {
      return DispatchHelper<TensorTypes<int, int64_t>>::call(this, Input(1));
    } else {
      return DoRunWithType<int64_t>();
    }
  }

  template <typename SIndex>
  bool DoRunWithType() {
    if (InputSize() > 1) {
      ReinitializeAndCopyFrom(
          &starts_host_, at::dtype<SIndex>().device(CPU), Input(1));
      ReinitializeAndCopyFrom(
          &ends_host_, at::dtype<SIndex>().device(CPU), Input(2));
    } else {
      if (!statically_inited_) {
        if (HasArgument("dim") && HasArgument("start_idx") &&
            HasArgument("end_idx")) {
          auto dim = this->template GetSingleArgument<int>("dim", 0);
          auto start =
              this->template GetSingleArgument<int64_t>("start_idx", 0);
          auto end = this->template GetSingleArgument<int64_t>("end_idx", -1);
          auto& input_tensor = Inputs()[0]->Get<Int8TensorCPU>();
          auto rank = input_tensor.t.sizes().size();
          starts_.resize(rank, 0);
          ends_.resize(rank, -1);
          starts_[dim] = start;
          ends_[dim] = end;
        } else {
          CAFFE_ENFORCE(HasArgument("starts"));
          CAFFE_ENFORCE(HasArgument("ends"));
        }
        CAFFE_ENFORCE_EQ(starts_.size(), ends_.size());

        ReinitializeTensor(
            &starts_host_,
            {static_cast<int64_t>(starts_.size())},
            at::dtype<SIndex>().device(CPU));
        ReinitializeTensor(
            &ends_host_,
            {static_cast<int64_t>(ends_.size())},
            at::dtype<SIndex>().device(CPU));

        memcpy(
            starts_host_.template mutable_data<SIndex>(),
            starts_.data(),
            sizeof(SIndex) * starts_.size());
        memcpy(
            ends_host_.template mutable_data<SIndex>(),
            ends_.data(),
            sizeof(SIndex) * ends_.size());
        statically_inited_ = true;
      }
    }

    auto& X = Inputs()[0]->Get<Int8TensorCPU>();
    auto* Y = Outputs()[0]->GetMutable<Int8TensorCPU>();
    int32_t Y_offset = this->template GetSingleArgument<int>("Y_zero_point", 0);
    auto Y_scale = this->template GetSingleArgument<float>("Y_scale", 1);
    CHECK_EQ(Y_offset, X.zero_point);
    CHECK_EQ(Y_scale, X.scale);
    Y->scale = Y_scale;
    Y->zero_point = Y_offset;

    return SliceImpl<SIndex, CPUContext>(
        &Y->t, X.t, starts_host_, ends_host_, &context_);
  }
};

} // namespace int8

} // namespace caffe2

#endif // CAFFE2_OPERATORS_INT8_SLICE_OP_H_
