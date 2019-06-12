#ifndef IS_EMPTY_OP_H_
#define IS_EMPTY_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class IsEmptyOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(IsEmptyOp);

  bool RunOnDevice() override {
    auto& input = Input(0);

    auto* output = Output(0, std::vector<int64_t>{}, at::dtype<bool>());
    *output->template mutable_data<bool>() = (input.numel() == 0);
    return true;
  }
};

} // namespace caffe2

#endif // IS_EMPTY_OP_H_
