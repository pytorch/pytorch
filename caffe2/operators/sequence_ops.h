#ifndef CAFFE2_OPERATORS_SEQUENCE_OPS_H_
#define CAFFE2_OPERATORS_SEQUENCE_OPS_H_

#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class GatherPaddingOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit GatherPaddingOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        startPaddingWidth_(
            this->template GetSingleArgument<int>("padding_width", 1)),
        endPaddingWidth_(
            this->template GetSingleArgument<int>("end_padding_width", -1)) {
    CAFFE_ENFORCE_GE(startPaddingWidth_, 0);
    if (endPaddingWidth_ < 0) {
      endPaddingWidth_ = startPaddingWidth_;
    }
  }

  bool RunOnDevice() override {
    if (startPaddingWidth_ == 0 && endPaddingWidth_ == 0) {
      Output(0)->Resize(std::vector<int64_t>(0));
      auto output_0_data = Output(0)->template mutable_data<int64_t>();
      // TODO(zhengxq): as suggested by salex@, change this to a loop.
      math::Set<int64_t, Context>(
          Output(0)->numel(), 0, output_0_data, &context_);
      if (OutputSize() == 2) {
        Output(1)->Resize(std::vector<int64_t>(0));
        auto output_1_data = Output(1)->template mutable_data<int64_t>();
        math::Set<int64_t, Context>(
            Output(1)->numel(), 0, output_1_data, &context_);
      }
      return true;
    }
    return DispatchHelper<TensorTypes<float, double, int, int64_t, bool>>::call(
        this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& in = Input(0);
    CAFFE_ENFORCE_GE(in.dim(), 1);
    const int32_t outer_size = in.sizes()[0];
    const auto block_size = in.size_from_dim(1);
    const auto pad_width = startPaddingWidth_ + endPaddingWidth_;

    // if no lengths is provided, assume it is a single full-span entry
    const int32_t* lengths_ptr = &outer_size;
    int64_t lengths_size = 1;
    if (InputSize() > 1) {
      const auto& lengths = Input(1);
      lengths_ptr = lengths.template data<int32_t>();
      lengths_size = lengths.numel();
    }
    std::vector<int64_t> padShape(in.sizes().begin() + 1, in.sizes().end());
    // output will contain accumulator over paddings
    Output(0)->Resize(padShape);
    T* padding_start_ptr = Output(0)->template mutable_data<T>();
    math::Set<T, Context>(block_size, 0.0, padding_start_ptr, &context_);

    // if no end_padding is provided, assume it's the same as start_padding
    T* padding_end_ptr = padding_start_ptr;
    if (OutputSize() == 2) {
      Output(1)->Resize(padShape);
      padding_end_ptr = Output(1)->template mutable_data<T>();
      math::Set<T, Context>(block_size, 0.0, padding_end_ptr, &context_);
    }
    GatherPadding<T>(
        outer_size,
        lengths_size,
        block_size,
        pad_width,
        in.template data<T>(),
        lengths_ptr,
        padding_start_ptr,
        padding_end_ptr);
    return true;
  }

 private:
  template <typename T>
  void GatherPadding(
      const int outer_size,
      const int lengths_size,
      const int block_size,
      const int pad_width,
      const T* in_ptr,
      const int* lengths_ptr,
      T* padding_start_ptr,
      T* padding_end_ptr);

  int startPaddingWidth_;
  int endPaddingWidth_;
  // Scratch space required by the CUDA version
  Tensor lengths_prefix_sum_buffer_{Context::GetDeviceType()};
  Tensor lengths_prefix_sum_{Context::GetDeviceType()};
};

template <class Context>
class RemovePaddingOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit RemovePaddingOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        startPaddingWidth_(
            this->template GetSingleArgument<int>("padding_width", 1)),
        endPaddingWidth_(
            this->template GetSingleArgument<int>("end_padding_width", -1)) {
    CAFFE_ENFORCE_GE(startPaddingWidth_, 0);
    if (endPaddingWidth_ < 0) {
      endPaddingWidth_ = startPaddingWidth_;
    }
  }

  bool RunOnDevice() override {
    if (startPaddingWidth_ == 0 && endPaddingWidth_ == 0) {
      Output(0)->CopyFrom(Input(0), true /*async*/);
      if (OutputSize() == 2) {
        Output(1)->CopyFrom(Input(1), true /*async*/);
      }
      return true;
    }
    return DispatchHelper<TensorTypes<float, double, int, int64_t, bool>>::call(
        this, Input(0));
  }

  template <typename T>
  bool DoRunWithType();

 private:
  int startPaddingWidth_;
  int endPaddingWidth_;

  // Scratch space required by the CUDA version
  Tensor lengths_prefix_sum_buffer_{Context::GetDeviceType()};
  Tensor lengths_prefix_sum_{Context::GetDeviceType()};
};

template <class Context>
class AddPaddingOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit AddPaddingOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        startPaddingWidth_(
            this->template GetSingleArgument<int>("padding_width", 1)),
        endPaddingWidth_(
            this->template GetSingleArgument<int>("end_padding_width", -1)) {
    CAFFE_ENFORCE_GE(startPaddingWidth_, 0);
    if (endPaddingWidth_ < 0) {
      endPaddingWidth_ = startPaddingWidth_;
    }
  }

  bool RunOnDevice() override {
    if (startPaddingWidth_ == 0 && endPaddingWidth_ == 0) {
      Output(0)->CopyFrom(Input(0), true /*async*/);
      if (OutputSize() == 2) {
        Output(1)->CopyFrom(Input(1), true /*async*/);
      }
      return true;
    }
    return DispatchHelper<TensorTypes<float, double, int, int64_t, bool>>::call(
        this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& in = Input(0);
    CAFFE_ENFORCE_GE(in.dim(), 1);
    const int32_t outer_size = in.sizes()[0];
    const auto block_size = in.size_from_dim(1);

    // if no lengths is provided, assume it is a single full-span entry
    const int32_t* lengths_ptr = nullptr;
    int32_t lengths_size = 1;
    if (InputSize() > 1) {
      const auto& lengths = Input(1);
      lengths_ptr = lengths.template data<int32_t>();
      lengths_size = lengths.numel();
    }

    // fetch paddings
    // input_size == 2 : pad with zeros
    // input_size == 3 : start and end paddings are the same
    // input_size == 4 : different start and end paddings
    const T* padding_start_ptr = nullptr;
    const T* padding_end_ptr = nullptr;
    if (InputSize() >= 3) {
      auto& padding_start = Input(2);
      CAFFE_ENFORCE_EQ(block_size, padding_start.numel());
      padding_start_ptr = padding_start.template data<T>();
    }
    if (InputSize() == 4) {
      auto& padding_end = Input(3);
      CAFFE_ENFORCE_EQ(block_size, padding_end.numel());
      padding_end_ptr = padding_end.template data<T>();
    } else {
      padding_end_ptr = padding_start_ptr;
    }

    auto out_dims = in.sizes().vec();
    out_dims[0] += (startPaddingWidth_ + endPaddingWidth_) * lengths_size;
    auto* out = Output(0, std::move(out_dims), at::dtype<T>());

    const auto* in_ptr = in.template data<T>();
    auto* out_ptr = out->template mutable_data<T>();

    return MakePadding<T>(
        in_ptr,
        out_ptr,
        lengths_ptr,
        lengths_size,
        outer_size,
        padding_start_ptr,
        padding_end_ptr,
        block_size);
  }

 private:
  template <typename T>
  bool MakePadding(
      const T* in_ptr,
      T* out_ptr,
      const int32_t* lengths_ptr,
      int32_t lengths_size,
      int32_t outer_size,
      const T* padding_start_ptr,
      const T* padding_end_ptr,
      int64_t block_size);

  int startPaddingWidth_;
  int endPaddingWidth_;

  // Scratch space required by the CUDA version
  Tensor lengths_prefix_sum_buffer_{Context::GetDeviceType()};
  Tensor lengths_prefix_sum_{Context::GetDeviceType()};
};

template <class Context>
class PadEmptySamplesOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit PadEmptySamplesOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}

  bool RunOnDevice() override;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_SEQUENCE_OPS_H_
