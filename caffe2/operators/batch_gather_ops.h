#ifndef CAFFE2_OPERATORS_BATCH_GATHER_OPS_H_
#define CAFFE2_OPERATORS_BATCH_GATHER_OPS_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class BatchGatherOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(BatchGatherOp)

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, this->template Input<Tensor>(INDICES, CPU));
  }

  template <typename TInd>
  bool DoRunWithType() {
    auto& data = Input(DATA);
    auto& indices = Input(INDICES);
    auto* output = Output(0);

    CAFFE_ENFORCE_GE(data.dim(), 2, "DATA should be at least 2-D");

    vector<int64_t> shape;
    shape.push_back(data.size(0));
    shape.insert(shape.end(), indices.sizes().begin(), indices.sizes().end());
    shape.insert(shape.end(), data.sizes().begin() + 2, data.sizes().end());
    output->Resize(shape);

    auto block_size = data.size_from_dim(2);
    auto block_bytesize = block_size * data.dtype().itemsize();
    auto N = indices.numel();
    auto data_batch_size = data.size_from_dim(1);
    auto gathered_batch_size = N * data.size_from_dim(2);
    auto data_batch_bytesize = data_batch_size * data.dtype().itemsize();
    auto gathered_batch_bytesize =
        gathered_batch_size * data.dtype().itemsize();
    const TInd* idxs = indices.template data<TInd>();
    auto src_base = static_cast<const char*>(data.raw_data());
    auto out = static_cast<char*>(output->raw_mutable_data(data.dtype()));

    for (auto i = 0; i < N; ++i) {
      auto idx = idxs[i];
      CAFFE_ENFORCE(
          0 <= idx && idx < data.size(1),
          "INDICES element is out of DATA bounds, id=",
          idx,
          " data_dim=",
          data.size(1));
    }

    if (data.template IsType<float>() && block_size == 1) {
      auto src = data.template data<float>();
      auto dst = output->template mutable_data<float>();

      for (auto batch = 0; batch < data.size(0); ++batch) {
        auto src_batch_base = src + batch * data_batch_size;
        auto out_batch_base = dst + batch * gathered_batch_size;

        for (auto i = 0; i < N; ++i) {
          auto idx = idxs[i];
          out_batch_base[i] = src_batch_base[idx];
        }
      }
    } else {
      for (auto batch = 0; batch < data.size(0); ++batch) {
        auto src_batch_base = src_base + batch * data_batch_bytesize;
        auto out_batch_base = out + batch * gathered_batch_bytesize;

        for (auto i = 0; i < N; ++i) {
          auto idx = idxs[i];
          auto src = src_batch_base + idx * block_bytesize;
          auto dst = out_batch_base + i * block_bytesize;
          context_.CopyItemsSameDevice(data.dtype(), block_size, src, dst);
        }
      }
    }
    return true;
  }
  INPUT_TAGS(DATA, INDICES);
};

template <class Context>
class BatchGatherGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(BatchGatherGradientOp);

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, this->template Input<Tensor>(INDICES, CPU));
  }

  template <typename TInd>
  bool DoRunWithType() {
    return DispatchHelper<
        TensorTypes2<float, GenericTensorImplementation>,
        TInd>::call(this, Input(DATA));
  }

  template <typename TInd, typename TData>
  bool DoRunWithType2() {
    auto& data = Input(DATA);
    auto& indices = Input(INDICES);
    auto& grad = Input(GRAD);
    auto* output = Output(0);

    CAFFE_ENFORCE_GE(data.dim(), 2, "DATA should be at least 2-D");
    CAFFE_ENFORCE_EQ(
        data.size(0), grad.size(0), "batch sizes should be the same");

    output->ResizeLike(data);
    TData* out_data = output->template mutable_data<TData>();
    if (data.numel() <= 0) {
      return true;
    }

    memset(out_data, 0, output->nbytes());

    const TData* grad_data = grad.template data<TData>();

    auto block_size = data.size_from_dim(2);
    auto N = indices.numel();
    auto data_batch_size = data.size_from_dim(1);
    auto gathered_batch_size = N * data.size_from_dim(2);
    const TInd* idxs = indices.template data<TInd>();

    for (auto i = 0; i < N; ++i) {
      auto idx = idxs[i];
      CAFFE_ENFORCE(
          0 <= idx && idx < data.size(1),
          "INDICES element is out of DATA bounds, id=",
          idx,
          " data_dim=",
          data.size(1));
    }

    for (auto batch = 0; batch < grad.size(0); ++batch) {
      auto src_batch_base = grad_data + batch * gathered_batch_size;
      auto out_batch_base = out_data + batch * data_batch_size;

      for (auto i = 0; i < N; ++i) {
        auto idx = idxs[i];
        if (block_size == 1) {
          out_batch_base[idx * block_size] += src_batch_base[i * block_size];
        } else {
          math::Add(
              block_size,
              out_batch_base + idx * block_size,
              src_batch_base + i * block_size,
              out_batch_base + idx * block_size,
              &context_);
        }
      }
    }
    return true;
  }

  template <typename TInd>
  bool DoRunWithOtherType2() {
    CAFFE_THROW(
        "BatchGatherGradient is not implemented on tensor of type ",
        Input(DATA).dtype().name(),
        "Consider adding it a type in the list DispatchHelper or implementing "
        "a generic version (which won't work for duplicated indices though)");
  }

  INPUT_TAGS(DATA, INDICES, GRAD);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_BATCH_GATHER_OPS_H_
