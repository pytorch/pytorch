#ifndef GATHER_OP_H_
#define GATHER_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class GatherOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(GatherOp);

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, this->template Input<Tensor>(INDICES, CPU));
  }

  template <typename Index>
  bool DoRunWithType() {
    // If we endup using it on GPU doing O(N) memcpy is probably not best :)
    // TODO: implement prefetching if it starts mattering (TF does it)
    auto& data = Input(DATA);
    auto& indices = Input(INDICES);
    auto* output = Output(0);

    CAFFE_ENFORCE_GE(data.dim(), 1, "DATA should be at least 1-D");
    auto shape = data.sizes().vec();
    if (data.size(0) > 0) {
      shape = indices.sizes().vec();
      shape.insert(shape.end(), data.sizes().begin() + 1, data.sizes().end());
    }
    output->Resize(shape);

    int block_size = data.size_from_dim(1);
    auto block_bytesize = data.size_from_dim(1) * data.dtype().itemsize();
    int N = indices.numel();

    auto src_base = static_cast<const char*>(data.raw_data());
    const Index* idxs = indices.template data<Index>();
    auto out = static_cast<char*>(output->raw_mutable_data(data.dtype()));

    if (output->numel() == 0) {
      return true;
    }
    for (int i = 0; i < N; ++i) {
      auto idx = idxs[i];
      if (idx < 0) {
        idx = idx + data.size(0);
      }
      CAFFE_ENFORCE(
          0 <= idx && idx < data.size(0),
          "INDICES element is out of DATA bounds, id=",
          idx,
          " data_dim=",
          data.size(0));
      auto src = src_base + idx * block_bytesize;
      context_.template CopyItems<Context, Context>(
          data.dtype(), block_size, src, out + block_bytesize * i);
    }
    return true;
  }

  INPUT_TAGS(DATA, INDICES);
};
} // namespace caffe2
#endif // GATHER_OP_H_
