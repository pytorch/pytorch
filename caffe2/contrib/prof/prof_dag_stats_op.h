#ifndef CAFFE2_OPERATORS_FULLY_CONNECTED_OP_H_
#define CAFFE2_OPERATORS_FULLY_CONNECTED_OP_H_

#include "caffe2/contrib/prof/prof_dag_net.h"
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

// This operator outputs the ProfDAGNet stats
template <typename T, class Context, class Engine = DefaultEngine>
class GetProfDagStatsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  GetProfDagStatsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        net_name_(
            OperatorBase::GetSingleArgument<std::string>("net_name", "")) {
    ws_ = ws;
  }
  ~GetProfDagStatsOp() {}

  bool RunOnDevice() override {
    // Read operator statistics for net_name_
    CAFFE_ENFORCE(!net_name_.empty(), "You need to provide net_name");
    auto* net = ws_->GetNet(net_name_);

    auto prof_dag_net = dynamic_cast_if_rtti<ProfDAGNet*>(net);
    CAFFE_ENFORCE(prof_dag_net);
    auto stats = prof_dag_net->GetOperatorStats();

    // Write protobuf message to the output blob
    std::string serialized_data;
    CAFFE_ENFORCE(stats.SerializeToString(&serialized_data));
    Output(0)->Resize(1);
    Output(0)->template mutable_data<std::string>()[0] = serialized_data;

    return true;
  }

 protected:
  std::string net_name_;
  Workspace* ws_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_FULLY_CONNECTED_OP_H_
