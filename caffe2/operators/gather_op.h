#ifndef GATHER_OP_H_
#define GATHER_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class GatherOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  GatherOp(const OperatorDef& operator_def, Workspace* ws)
          : Operator<Context>(operator_def, ws),
            OP_SINGLE_ARG(int, "axis", axis_, 0) {}

  virtual ~GatherOp() noexcept {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, this->template Input<Tensor>(INDICES, CPU));
  }

  template <typename TInd>
  bool DoRunWithType() {
    // If we endup using it on GPU doing O(N) memcpy is probably not best :)
    // TODO: implement prefetching if it starts mattering (TF does it)
    auto& data = Input(DATA);
    auto& indices = Input(INDICES);
    auto axis = axis_;
    auto* output = Output(0);

    CAFFE_ENFORCE_GE(
        data.ndim(), axis + 1, "DATA should be at least [axis+1]-D");
    CAFFE_ENFORCE_GE(axis, 0, "Axis should be non-negative");
    CAFFE_ENFORCE_LT(axis, data.ndim(), "Axis out of range");

    vector<TIndex> shape;
    shape.insert(shape.end(), data.dims().begin(), data.dims().begin() + axis);
    shape.insert(shape.end(), indices.dims().begin(), indices.dims().end());
    shape.insert(
        shape.end(), data.dims().begin() + axis + 1, data.dims().end());
    output->Resize(shape);

    auto outer_size = data.size_to_dim(axis);
    auto block_size = data.size_from_dim(axis + 1);
    auto block_bytesize = data.size_from_dim(axis + 1) * data.meta().itemsize();
    auto N = indices.size();

    auto data_batch_bytesize =
        data.size_from_dim(axis) * data.meta().itemsize();
    auto gathered_batch_bytesize = N * block_size * data.meta().itemsize();
    const TInd* idxs = indices.template data<TInd>();
    auto src_base = static_cast<const char*>(data.raw_data());
    auto out = static_cast<char*>(output->raw_mutable_data(data.meta()));
    auto data_axis_dim = data.dim(axis);
    for (auto batch = 0; batch < outer_size; ++batch) {
      for (auto i = 0; i < N; ++i) {
        auto idx = idxs[i];
        if (idx < 0) {
          idx = idx + data_axis_dim;
        }
        CAFFE_ENFORCE(
            0 <= idx && idx < data_axis_dim,
            "INDICES element is out of DATA bounds, id=",
            idx,
            " data_axis_dim=",
            data_axis_dim);
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
} // namespace caffe2
#endif // GATHER_OP_H_
