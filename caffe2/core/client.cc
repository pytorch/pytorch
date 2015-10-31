#include "caffe2/core/client.h"
#include "caffe2/core/net.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/workspace.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {

Client::Client(const string& client_def_name) : workspace_(new Workspace()) {
  SimpleClientDef client_def;
  CAFFE_CHECK(ReadProtoFromFile(client_def_name, &client_def));
  workspace_->RunNetOnce(client_def.init_net());
  client_def.mutable_main_net()->set_name("main");
  CAFFE_CHECK(workspace_->CreateNet(client_def.main_net()));
  input_blob_ = workspace_->GetBlob(client_def.input());
  output_blob_ = workspace_->GetBlob(client_def.output());
  CAFFE_CHECK(input_blob_ != nullptr);
  CAFFE_CHECK(output_blob_ != nullptr);
}

Client::~Client() {
  delete workspace_;
}

bool Client::Run(const vector<float>& input, vector<float>* output) {
  TensorCPU* input_tensor =
      input_blob_->GetMutable<TensorCPU>();
  CAFFE_CHECK_EQ(input_tensor->size(), input.size());
  memcpy(input_tensor->mutable_data<float>(), input.data(),
         input.size() * sizeof(float));
  workspace_->RunNet("main");
  const TensorCPU& output_tensor =
      output_blob_->Get<TensorCPU>();
  output->resize(output_tensor.size());
  memcpy(output->data(), output_tensor.data<float>(),
         output->size() * sizeof(float));
  return true;
}

}  // namespace caffe2

