#ifndef CAFFE2_OPERATORS_RECURRENT_BLOB_FETCHER_OP_H_
#define CAFFE2_OPERATORS_RECURRENT_BLOB_FETCHER_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/operators/rnn/recurrent_network_op.h"
#include "google/protobuf/text_format.h"

#include <string>

namespace caffe2 {

template <class Context>
class RecurrentNetworkBlobFetcherOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  RecurrentNetworkBlobFetcherOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {
    prefix_ = OperatorBase::GetSingleArgument<std::string>("prefix", "rnn");
    ws_ = ws;
  }

  bool RunOnDevice() override {
    const detail::ScratchWorkspaces& scratch =
        OperatorBase::Input<detail::ScratchWorkspaces>(0);
    const std::vector<std::shared_ptr<Workspace>>& stepWorkspaces =
        scratch.stepWorkspaces;

    std::vector<std::string> blob_names_vector = {};

    for (TIndex i = 0; i < stepWorkspaces.size(); i++) {
      Workspace* currentStepWorkspace = stepWorkspaces[i].get();
      std::vector<std::string> blob_names = currentStepWorkspace->LocalBlobs();

      for (auto& blob_name : blob_names) {
        const Blob* currentBlob = currentStepWorkspace->GetBlob(blob_name);
        const auto& currentTensor = currentBlob->Get<Tensor<Context>>();

        std::string newBlobName =
            prefix_ + std::string("_") + blob_name + caffe2::to_string(i);
        blob_names_vector.push_back(newBlobName);

        ws_->CreateBlob(newBlobName)
            ->template GetMutable<TensorCPU>()
            ->ResizeLike(currentTensor);

        auto* newTensor =
            ws_->GetBlob(newBlobName)->template GetMutable<Tensor<Context>>();
        newTensor->template CopyFrom<Context>(currentTensor);
      }
    }

    auto* output = Output(0);
    output->Resize(blob_names_vector.size());
    std::copy(
        blob_names_vector.begin(),
        blob_names_vector.end(),
        output->template mutable_data<std::string>());

    return true;
  }

 private:
  std::string prefix_;
  Workspace* ws_;
};
} // namespace caffe2

#endif // CAFFE2_OPERATORS_RECURRENT_BLOB_FETCHER_OP_H_
