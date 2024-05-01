#ifndef CAFFE2_OPERATORS_DENSE_VECTOR_TO_ID_LIST_OP_H_
#define CAFFE2_OPERATORS_DENSE_VECTOR_TO_ID_LIST_OP_H_

#include <set>
#include <vector>
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "c10/util/irange.h"

namespace caffe2 {

template <class Context>
class DenseVectorToIdListOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(DenseVectorToIdListOp)

  template <typename T, typename M>
  bool DoRunWithType() {
    auto& input = Input(0);
    const auto* input_data = input.template data<T>();

    CAFFE_ENFORCE_EQ(input.dim(), 2, "Sample should be 2-D");
    const auto batch_size = input.size(0);
    const auto col_num = input.size(1);

    auto* out_lengths = Output(0, {batch_size}, at::dtype<int32_t>());

    auto* out_lengths_data = out_lengths->template mutable_data<int32_t>();

    auto* out_values = Output(1, {batch_size * col_num}, at::dtype<M>());

    auto* out_values_data = out_values->template mutable_data<M>();

    auto v_pos = 0;
    auto l_pos = 0;
    for (const auto i : c10::irange(batch_size)) {
      auto length = 0;
      for (const auto j : c10::irange(col_num)) {
        if ((int)(input_data[i * col_num + j] + 0.5) != 0) {
          out_values_data[v_pos++] = j;
          length++;
        }
      }
      out_lengths_data[l_pos++] = length;
    }
    out_values->Resize(v_pos);
    out_lengths->Resize(l_pos);
    return true;
  }

  bool RunOnDevice() override {
    if (Input(0).template IsType<float>()) {
      return DoRunWithType<float, int>();
    } else {
      CAFFE_THROW(
          "DenseVectorToIdList operator only supports 32-bit float, but",
          " input was of type ",
          Input(0).dtype().name());
    }
  }
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_DENSE_VECTOR_TO_ID_LIST_OP_H_
