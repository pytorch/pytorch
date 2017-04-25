#ifndef CAFFE2_OPERATORS_ELEMENTWISE_LOGICAL_OPS_H_
#define CAFFE2_OPERATORS_ELEMENTWISE_LOGICAL_OPS_H_

#include "caffe2/core/common_omp.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/elementwise_op.h"

#include <unordered_set>

namespace caffe2 {

template <class Context>
class WhereOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(WhereOp);
  USE_OPERATOR_FUNCTIONS(Context);
  USE_DISPATCH_HELPER;

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, double, int, long, std::string>>::
        call(this, Input(1));
  }

  template <typename T>
  bool DoRunWithType() {
    auto& select = Input(0);
    auto& left = Input(1);
    auto& right = Input(2);
    auto* output = Output(0);
    CAFFE_ENFORCE_EQ(select.dims(), left.dims());
    CAFFE_ENFORCE_EQ(select.dims(), right.dims());
    output->ResizeLike(left);

    const bool* select_data = select.template data<bool>();
    const T* left_data = left.template data<T>();
    const T* right_data = right.template data<T>();
    T* output_data = output->template mutable_data<T>();
    for (int i = 0; i < select.size(); ++i) {
      output_data[i] = select_data[i] ? left_data[i] : right_data[i];
    }
    return true;
  }
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_ELEMENTWISE_LOGICAL_OPS_H_
