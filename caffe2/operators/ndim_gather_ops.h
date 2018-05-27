#ifndef CAFFE2_OPERATORS_BATCH_GATHER_OPS_H_
#define CAFFE2_OPERATORS_BATCH_GATHER_OPS_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class NdimGatherOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  NdimGatherOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        OP_SINGLE_ARG(int, "axis", axis_, 0) {}

  virtual ~NdimGatherOp() noexcept {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, OperatorBase::Input<TensorCPU>(INDICES));
  }

  template <typename TInd>
  bool DoRunWithType() {
    auto& data = Input(DATA);
    auto& indices = Input(INDICES);
    auto* output = Output(0);

    CAFFE_ENFORCE_GE(data.ndim(), 1, "DATA should be at least 1-D");
    CAFFE_ENFORCE_GE(axis_, 0, "Axis should be non-negative");
    CAFFE_ENFORCE_LT(axis_, data.ndim(), "Axis out of range");

    vector<TIndex> shape;
    if (axis_ > 0) {
      shape.insert(
          shape.end(), data.dims().begin(), data.dims().begin() + axis_);
    }
    shape.insert(shape.end(), indices.dims().begin(), indices.dims().end());
    if (axis_ < data.ndim() - 1) {
      shape.insert(
          shape.end(), data.dims().begin() + axis_ + 1, data.dims().end());
    }
    output->Resize(shape);

    auto outer_size = data.size_to_dim(axis_);
    auto block_size = data.size_from_dim(axis_ + 1);
    auto block_bytesize = block_size * data.meta().itemsize();
    auto N = indices.size();
    auto data_batch_bytesize =
        data.size_from_dim(axis_) * data.meta().itemsize();
    auto gathered_batch_bytesize = N * block_size * data.meta().itemsize();
    const TInd* idxs = indices.template data<TInd>();
    auto src_base = static_cast<const char*>(data.raw_data());
    auto out = static_cast<char*>(output->raw_mutable_data(data.meta()));

    for (auto batch = 0; batch < outer_size; ++batch) {
      for (auto i = 0; i < N; ++i) {
        auto idx = idxs[i];
        CAFFE_ENFORCE(
            0 <= idx && idx < data.dim(axis_),
            "INDICES element is out of DATA bounds, id=",
            idx,
            " data_dim=",
            data.dim(axis_));
        auto src =
            src_base + idx * block_bytesize + batch * data_batch_bytesize;
        auto dst = out + i * block_bytesize + batch * gathered_batch_bytesize;
        context_.template CopyItems<Context, Context>(
            data.meta(), block_size, src, dst);
      }
    }
    return true;
  }

  INPUT_TAGS(DATA, INDICES);

protected:
  int axis_;
};

template <class Context>
class NdimGatherGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  NdimGatherGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        OP_SINGLE_ARG(int, "axis", axis_, 0) {}

  virtual ~NdimGatherGradientOp() noexcept {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, OperatorBase::Input<TensorCPU>(INDICES));
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

    CAFFE_ENFORCE_GE(data.ndim(), 1, "DATA should be at least 1-D");
    CAFFE_ENFORCE_GE(axis_, 0, "Axis should be non-negative");
    CAFFE_ENFORCE_LT(axis_, data.ndim(), "Axis out of range");
    for (int i = 0; i < axis_; i++) {
      CAFFE_ENFORCE_EQ(
          data.dim(i),
          grad.dim(i),
          "The ",
          i,
          "-th dimension should be the same");
    }

    output->ResizeLike(data);
    TData* out_data = output->template mutable_data<TData>();
    if (data.size() <= 0) {
      return true;
    }

    memset(out_data, 0, output->nbytes());

    const TData* grad_data = grad.template data<TData>();

    auto outer_size = data.size_to_dim(axis_);
    auto block_size = data.size_from_dim(axis_ + 1);
    auto N = indices.size();
    auto data_batch_size = data.size_from_dim(axis_);
    auto gathered_batch_size = N * block_size;
    const TInd* idxs = indices.template data<TInd>();

    for (auto batch = 0; batch < outer_size; ++batch) {
      for (auto i = 0; i < N; ++i) {
        auto idx = idxs[i];
        CAFFE_ENFORCE(
            0 <= idx && idx < data.dim(axis_),
            "INDICES element is out of DATA bounds, id=",
            idx,
            " data_dim=",
            data.dim(axis_));
        math::Add(
            block_size,
            out_data + idx * block_size + batch * data_batch_size,
            grad_data + i * block_size + batch * gathered_batch_size,
            out_data + idx * block_size + batch * data_batch_size,
            &context_);
      }
    }
    return true;
  }

  template <typename TInd>
  bool DoRunWithOtherType2() {
    CAFFE_THROW(
        "NdimGatherGradientOp is not implemented on tensor of type ",
        Input(DATA).meta().name(),
        "Consider adding it a type in the list DispatchHelper or implementing "
        "a generic version (which won't work for duplicated indices though)");
  }

  INPUT_TAGS(DATA, INDICES, GRAD);

protected:
  int axis_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_BATCH_GATHER_OPS_H_
