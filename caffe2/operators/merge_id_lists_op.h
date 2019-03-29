#ifndef CAFFE2_OPERATORS_MERGE_ID_LISTS_OP_H_
#define CAFFE2_OPERATORS_MERGE_ID_LISTS_OP_H_

#include <set>
#include <vector>
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class MergeIdListsOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(MergeIdListsOp);

  template <typename T>
  bool DoRunWithType() {
    auto& first_lengths = Input(0);
    CAFFE_ENFORCE_EQ(first_lengths.dim(), 1, "LENGTHS should be 1-D");
    const auto batch_size = first_lengths.numel();

    auto* out_lengths = Output(0, first_lengths.sizes(), at::dtype<int32_t>());

    auto* out_lengths_data = out_lengths->template mutable_data<int32_t>();

    /**
     * Loop to figure out how much space to reserve for output
     * and perform checks.
     */
    auto M = 0;
    for (size_t i = 0; i < InputSize(); i += 2) {
      auto& lengths = Input(i);
      CAFFE_ENFORCE_EQ(lengths.dim(), 1, "LENGTHS should be 1-D");
      CAFFE_ENFORCE_EQ(lengths.numel(), batch_size, "LENGTHS should be equal");
      auto& values = Input(i + 1);
      CAFFE_ENFORCE_EQ(values.dim(), 1, "VALUES should be 1-D");
      M += values.numel();
    }

    auto* out_values = Output(1, {M}, at::dtype<T>());

    T* out_values_data = out_values->template mutable_data<T>();
    auto pos = 0;

    // TODO(badri): Use unordered_set if performance is an issue
    std::set<T> deduped;
    std::vector<int> offsets(InputSize(), 0);
    for (auto sample = 0; sample < batch_size; sample++) {
      for (size_t i = 0; i < InputSize(); i += 2) {
        auto& lengths = Input(i);
        const auto* lengths_data = lengths.template data<int32_t>();

        auto& values = Input(i + 1);
        const T* values_data = values.template data<T>();
        const auto length = lengths_data[sample];

        for (auto j = offsets[i]; j < offsets[i] + length; j++) {
          deduped.insert(values_data[j]);
        }
        offsets[i] += length;
      }
      for (auto val : deduped) {
        out_values_data[pos++] = val;
      }
      out_lengths_data[sample] = deduped.size();
      deduped.clear();
    }
    out_values->Resize(pos);
    return true;
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(1));
  }
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_MERGE_ID_LISTS_OP_H_
