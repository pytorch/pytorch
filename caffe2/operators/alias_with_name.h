#ifndef ALIAS_WITH_NAME_OP_H_
#define ALIAS_WITH_NAME_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/export_caffe2_op_to_c10.h"
#include "caffe2/core/operator.h"

C10_DECLARE_EXPORT_CAFFE2_OP_TO_C10(AliasWithName)

namespace caffe2 {

template <class Context>
class AliasWithNameOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit AliasWithNameOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        name_(this->template GetSingleArgument<std::string>(
            "name",
            "invalid_name")),
        is_backward_(
            this->template GetSingleArgument<bool>("is_backward", false)) {
    CAFFE_ENFORCE(
        OperatorBase::HasArgument("name"), "You have to specify argument name");
  }

  bool RunOnDevice() override {
    auto& input = Input(0);
    CAFFE_ENFORCE_GE(input.numel(), 0, "Tensor is not initialized");

    // This doesn't work anymore as this is "newstyle" operator
    // OutputTensorAlias(0, input);

    OperatorBase::SetOutputTensor(0, input.Alias());
    return true;
  }

 protected:
  std::string name_;
  bool is_backward_;
};

} // namespace caffe2

#endif // ALIAS_WITH_NAME_OP_H_
