#ifndef CAFFE2_OPERATORS_IF_OP_H_
#define CAFFE2_OPERATORS_IF_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class IfOp final : public Operator<Context> {
 public:
  explicit IfOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {
    CAFFE_ENFORCE(
        this->template HasSingleArgumentOfType<NetDef>("then_net"),
        "then_net must be specified in If operator");
    auto then_net_def =
        this->template GetSingleArgument<NetDef>("then_net", NetDef());
    then_net_ = CreateNet(then_net_def, ws);
    CAFFE_ENFORCE(then_net_, "Failed to initialize then subnet");

    if (this->template HasSingleArgumentOfType<NetDef>("else_net")) {
      auto else_net_def =
          this->template GetSingleArgument<NetDef>("else_net", NetDef());
      else_net_ = CreateNet(else_net_def, ws);
      CAFFE_ENFORCE(else_net_, "Failed to initialize else subnet");
    }
  }

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    CAFFE_ENFORCE(
        this->InputIsTensorType(0, Context::GetDeviceType()),
        "Invalid condition in If operator: tensor expected");

    const auto& condition = Input(0);
    CAFFE_ENFORCE_EQ(
        condition.numel(),
        1,
        "Invalid condition tensor in If operator: single value expected");

    auto conditionValue = *condition.template data<bool>();
    if (conditionValue) {
      return then_net_->Run();
    } else if (else_net_) {
      return else_net_->Run();
    }

    return true;
  }

 private:
  std::unique_ptr<NetBase> then_net_;
  std::unique_ptr<NetBase> else_net_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_IF_OP_H_
