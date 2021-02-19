#ifndef CAFFE2_OPERATORS_FIND_DUPLICATE_ELEMENTS_OP_H
#define CAFFE2_OPERATORS_FIND_DUPLICATE_ELEMENTS_OP_H

#include <unordered_map>
#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

template <class Context>
class FindDuplicateElementsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(FindDuplicateElementsOp);
  USE_DISPATCH_HELPER;

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, double, int, long, std::string>>::
        call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& data = Input(0);
    CAFFE_ENFORCE(data.dim() == 1, "data should be 1-D.");

    const auto* data_ptr = data.template data<T>();
    std::unordered_map<T, int64_t> dict;
    std::vector<int64_t> dupIndices;
    // i is the index of unique elements, j is the index of all elements
    for (int64_t i = 0, j = 0; j < data.sizes()[0]; ++i, ++j) {
      bool retVal = dict.insert({data_ptr[j], i}).second;
      if (!retVal) {
        --i;
        dupIndices.push_back(j);
      }
    }

    const auto dupSize = dupIndices.size();

    auto* output =
        Output(0, {static_cast<int64_t>(dupSize)}, at::dtype<int64_t>());
    auto* out_ptr = output->template mutable_data<int64_t>();
    for (size_t i = 0; i < dupSize; ++i) {
      out_ptr[i] = dupIndices[i];
    }

    return true;
  }
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_FIND_DUPLICATE_ELEMENTS_OP_H
