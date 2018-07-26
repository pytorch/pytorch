#ifndef CAFFE2_OPERATORS_WHILE_OP_H_
#define CAFFE2_OPERATORS_WHILE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class WhileOp final : public Operator<Context> {
 public:
  WhileOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {
    CAFFE_ENFORCE(
        this->template HasSingleArgumentOfType<NetDef>("loop_net"),
        "loop_net must be specified in While operator");
    loop_net_def_ =
        this->template GetSingleArgument<NetDef>("loop_net", NetDef());
    loop_net_ = CreateNet(loop_net_def_, ws);
    CAFFE_ENFORCE(loop_net_, "Failed to initialize loop subnet");

    cond_net_ = nullptr;
    bool has_cond_net =
        this->template HasSingleArgumentOfType<NetDef>("cond_net");
    if (has_cond_net) {
      cond_net_def_ =
          this->template GetSingleArgument<NetDef>("cond_net", NetDef());
      cond_net_ = CreateNet(cond_net_def_, ws);
      CAFFE_ENFORCE(cond_net_, "Failed to initialize condition subnet");
    }
  }

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    CAFFE_ENFORCE(
        this->template InputIsType<Tensor<Context>>(0),
        "Invalid condition in While operator: tensor expected");

    const auto& condition = Input(0);
    CAFFE_ENFORCE_EQ(
        condition.size(),
        1,
        "Invalid condition tensor in While operator: single value expected");

    while (true) {
      if (cond_net_ && !cond_net_->Run()) {
        return false;
      }
      if (!*condition.template data<bool>()) {
        return true;
      }
      if (!loop_net_->Run()) {
        return false;
      }
    }

    return true;
  }

 private:
  NetDef loop_net_def_;
  std::unique_ptr<NetBase> loop_net_;

  NetDef cond_net_def_;
  std::unique_ptr<NetBase> cond_net_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_WHILE_OP_H_
