#ifndef CAFFE2_OPERATORS_SPACE_BATCH_OP_H_
#define CAFFE2_OPERATORS_SPACE_BATCH_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#include "c10/util/irange.h"

namespace caffe2 {

template <typename Context>
void spaceToBatch(
    const Tensor& input,
    int pad_t,
    int pad_l,
    int block_size,
    Tensor* output,
    Context* /*context*/) {
  CAFFE_ENFORCE(input.dim() == 4);
  CAFFE_ENFORCE(output->dim() == 4);

  const int output_batch = output->dim32(0);
  const int output_depth = output->dim32(1);
  const int output_height = output->dim32(2);
  const int output_width = output->dim32(3);

  const int input_batch = input.dim32(0);
  const int input_depth = input.dim32(1);
  const int input_height = input.dim32(2);
  const int input_width = input.dim32(3);

  for (const auto out_b : c10::irange(output_batch)) {
    const int in_b = out_b % input_batch;
    const int offset_w = (out_b / input_batch) % block_size;
    const int offset_h = (out_b / input_batch) / block_size;
    for (const auto d : c10::irange(input_depth)) {
      for (const auto out_h : c10::irange(output_height)) {
        const int in_h = out_h * block_size + offset_h - pad_t;
        for (const auto out_w : c10::irange(output_width)) {
          const int in_w = out_w * block_size + offset_w - pad_l;
          const auto output_offset =
              ((out_b * output_depth + d) * output_height + out_h) *
                  output_width +
              out_w;
          const auto input_offset =
              ((in_b * input_depth + d) * input_height + in_h) * input_width +
              in_w;
          if (in_h >= 0 && in_w >= 0 && in_h < input_height &&
              in_w < input_width) {
            output->template mutable_data<float>()[output_offset] =
                input.template data<float>()[input_offset];
          } else {
            output->template mutable_data<float>()[output_offset] = 0.0;
          }
        }
      }
    }
  }
}

template <typename Context>
void batchToSpace(
    const Tensor& input,
    int pad_t,
    int pad_l,
    int block_size,
    Tensor* output,
    Context* /*context*/) {
  CAFFE_ENFORCE(input.dim() == 4);
  CAFFE_ENFORCE(output->dim() == 4);

  const int output_batch = output->dim32(0);
  const int output_depth = output->dim32(1);
  const int output_height = output->dim32(2);
  const int output_width = output->dim32(3);

  const int input_batch = input.dim32(0);
  const int input_depth = input.dim32(1);
  const int input_height = input.dim32(2);
  const int input_width = input.dim32(3);

  CAFFE_ENFORCE(input_depth == output_depth);
  for (const auto in_b : c10::irange(input_batch)) {
    const int out_b = in_b % output_batch;
    const int offset_w = (in_b / output_batch) % block_size;
    const int offset_h = (in_b / output_batch) / block_size;
    for (const auto d : c10::irange(input_depth)) {
      for (const auto in_h : c10::irange(input_height)) {
        const int out_h = in_h * block_size + offset_h - pad_t;
        for (const auto in_w : c10::irange(input_width)) {
          const int out_w = in_w * block_size + offset_w - pad_l;
          if (out_h >= 0 && out_w >= 0 && out_h < output_height &&
              out_w < output_width) {
            const auto output_offset =
                ((out_b * output_depth + d) * output_height + out_h) *
                    output_width +
                out_w;
            const auto input_offset =
                ((in_b * input_depth + d) * input_height + in_h) * input_width +
                in_w;
            output->template mutable_data<float>()[output_offset] =
                input.template data<float>()[input_offset];
          }
        }
      }
    }
  }
}

template <typename Context>
class SpaceBatchOpBase : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit SpaceBatchOpBase(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        pad_(this->template GetSingleArgument<int>("pad", 0)),
        pad_t_(this->template GetSingleArgument<int>("pad_t", pad_)),
        pad_l_(this->template GetSingleArgument<int>("pad", pad_)),
        pad_b_(this->template GetSingleArgument<int>("pad", pad_)),
        pad_r_(this->template GetSingleArgument<int>("pad", pad_)),
        block_size_(this->template GetSingleArgument<int>("block_size", 2)),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<string>("order", "NCHW"))) {
    CAFFE_ENFORCE(order_ == StorageOrder::NCHW);
  }

 protected:
  int pad_;
  int pad_t_;
  int pad_l_;
  int pad_b_;
  int pad_r_;
  int block_size_;
  StorageOrder order_;
};

template <typename Context>
class SpaceToBatchOp final : public SpaceBatchOpBase<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  using SpaceBatchOpBase<Context>::SpaceBatchOpBase;

  bool RunOnDevice() override {
    const auto& input = Input(0);
    auto* output = Output(0);
    const int batch = input.dim32(0);
    const int depth = input.dim32(1);
    const int height = this->pad_b_ + this->pad_t_ + input.dim32(2);
    const int width = this->pad_l_ + this->pad_r_ + input.dim32(3);
    CAFFE_ENFORCE(
        height % this->block_size_ == 0,
        "Height: ",
        height,
        ", block size: ",
        this->block_size_);
    CAFFE_ENFORCE(width % this->block_size_ == 0);

    const int output_batch = batch * this->block_size_ * this->block_size_;
    const int output_height = height / this->block_size_;
    const int output_width = width / this->block_size_;
    Output(0)->Resize(output_batch, depth, output_height, output_width);

    spaceToBatch<Context>(
        input,
        this->pad_t_,
        this->pad_l_,
        this->block_size_,
        output,
        &context_);

    return true;
  }
};

template <typename Context>
class BatchToSpaceOp final : public SpaceBatchOpBase<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  using SpaceBatchOpBase<Context>::SpaceBatchOpBase;

  bool RunOnDevice() override {
    const auto& input = Input(0);
    auto* output = Output(0);
    const int batch = input.dim32(0);
    const int depth = input.dim32(1);
    const int height = input.dim32(2);
    const int width = input.dim32(3);

    const int output_batch = batch / this->block_size_ / this->block_size_;
    const int output_height =
        height * this->block_size_ - this->pad_b_ - this->pad_t_;
    const int output_width =
        width * this->block_size_ - this->pad_l_ - this->pad_r_;
    Output(0)->Resize(output_batch, depth, output_height, output_width);
    batchToSpace<Context>(
        input,
        this->pad_t_,
        this->pad_l_,
        this->block_size_,
        output,
        &context_);
    return true;
  }
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_SPACE_BATCH_OP_H_
