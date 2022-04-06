#ifndef CAFFE2_OPERATORS_REMOVE_DATA_BLOCKS_OP_H_
#define CAFFE2_OPERATORS_REMOVE_DATA_BLOCKS_OP_H_

#include <algorithm>
#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "c10/util/irange.h"

namespace caffe2 {

template <class Context>
class RemoveDataBlocksOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(RemoveDataBlocksOp);
  USE_DISPATCH_HELPER;

  bool RunOnDevice() override {
    if (Input(INDICES).sizes()[0] == 0) {
      Output(0)->CopyFrom(Input(0));
      return true;
    } else {
      return DispatchHelper<TensorTypes<int, long>>::call(this, Input(INDICES));
    }
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& data = Input(DATA);
    const auto& indices = Input(INDICES);
    CAFFE_ENFORCE(data.dim() > 0, "DATA should be at leat 1-D.");
    CAFFE_ENFORCE(indices.dim() == 1, "INDICES should be 1-D.");

    const auto outer_size = data.sizes()[0];
    const auto block_size = data.size_from_dim(1);
    const auto block_size_bytes = block_size * data.dtype().itemsize();
    auto indices_size = indices.sizes()[0];
    const char* data_ptr = (char*)data.raw_data();
    const auto* ind_ptr = indices.template data<T>();

    std::vector<T> ind_vec;
    for (const auto i : c10::irange(indices_size)) {
      ind_vec.push_back(ind_ptr[i]);
    }
    std::sort(ind_vec.begin(), ind_vec.end());
    CAFFE_ENFORCE(ind_vec[0] >= 0, "The min index should be larger than zero.");
    CAFFE_ENFORCE(
        ind_vec[indices_size - 1] < outer_size,
        "The max index should be smaller than the data outer size.");
    // removes duplicate indices
    ind_vec.erase(std::unique(ind_vec.begin(), ind_vec.end()), ind_vec.end());
    indices_size = ind_vec.size();

    auto* output = Output(0);
    auto shape = data.sizes().vec();
    shape[0] -= indices_size;
    output->Resize(shape);
    char* out_ptr = (char*)output->raw_mutable_data(data.dtype());

    ind_vec.insert(ind_vec.begin(), -1);
    int64_t ind_vec_size = ind_vec.size();
    for (const auto i : c10::irange(ind_vec_size)) {
      int64_t interval_start = ind_vec[i] + 1;
      int64_t interval_end =
          (i == ind_vec_size - 1) ? outer_size : ind_vec[i + 1];
      auto num_items = interval_end - interval_start;
      context_.CopyItemsSameDevice(
          data.dtype(),
          num_items * block_size,
          data_ptr + block_size_bytes * interval_start,
          out_ptr);
      out_ptr += block_size_bytes * num_items;
    }

    return true;
  }

 private:
  INPUT_TAGS(DATA, INDICES);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_REMOVE_DATA_BLOCKS_OP_H_
