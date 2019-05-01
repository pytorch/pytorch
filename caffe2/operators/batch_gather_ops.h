#ifndef CAFFE2_OPERATORS_BATCH_GATHER_OPS_H_
#define CAFFE2_OPERATORS_BATCH_GATHER_OPS_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
// Reuse helper logic from GatherOp since BatchGather is the same with axis=1.
#include "caffe2/operators/gather_op.h"

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
    // BatchGather is a special-case of Gather with Axis = 1.
    return gather_helper::gather_impl<TInd, Context>(
        this, DATA, INDICES, 0, 1, false);
  }
  INPUT_TAGS(DATA, INDICES);
};

template <class Context>
class BatchGatherGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  // Constructor to recieve axis in case it was passed for GatherOp gradient,
  // use default of 1 for batch gather otherwise.
  template <class... Args>
  explicit BatchGatherGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        OP_SINGLE_ARG(int, "axis", axis_, 1) {}
  virtual ~BatchGatherGradientOp() noexcept {}

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

    // ONNX allows negative axis to index from the back, valid range: [-r, r].
    int axis = axis_;
    if (axis < 0) {
      axis = data.dim() + axis;
    }

    CAFFE_ENFORCE_GE(data.dim(), 2, "DATA should be at least 2-D");
    // Outer dimensions of input data and gradient should be the same
    // because they are preserved for gathers with axis > 0.
    for (int acheck = 0; acheck < axis; acheck++) {
      CAFFE_ENFORCE_EQ(
          data.size(acheck),
          grad.size(acheck),
          "batch gather outer dimensions should match");
    }

    auto* output = Output(0, data.sizes(), at::dtype<TData>());
    TData* out_data = output->template mutable_data<TData>();
    if (data.numel() <= 0) {
      return true;
    }
    memset(out_data, 0, output->nbytes());

    const TData* grad_data = grad.template data<TData>();
    const TInd* idxs = indices.template data<TInd>();

    auto outer_dims_product = data.size_to_dim(axis);
    auto batch_size = data.size_from_dim(axis);
    auto block_size = data.size_from_dim(axis + 1);
    auto N = indices.numel();
    auto gathered_grad_batch_size = N * block_size;

    // Check indexing bounds.
    auto src_indexing_axis_dim = data.dim(axis);
    gather_helper::check_indexarray_range<TInd>(
      idxs,
      N,
      src_indexing_axis_dim,
      false);

    for (auto batch = 0; batch < outer_dims_product; ++batch) {
      auto grad_batch_base = grad_data + batch * gathered_grad_batch_size;
      auto out_batch_base = out_data + batch * batch_size;

      for (auto i = 0; i < N; ++i) {
        auto idx = idxs[i];
        if (idx < 0) {
          idx = idx + src_indexing_axis_dim;
        }
        if (block_size == 1) {
          out_batch_base[idx] += grad_batch_base[i];
        } else {
          math::Add(
              block_size,
              out_batch_base + idx * block_size,
              grad_batch_base + i * block_size,
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
        Input(DATA).meta().name(),
        "consider adding it as a type in the DispatchHelper list or "
        "implementing a generic version (which won't work for "
        "duplicated indices though)");
  }

  INPUT_TAGS(DATA, INDICES, GRAD);
protected:
  int axis_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_BATCH_GATHER_OPS_H_
