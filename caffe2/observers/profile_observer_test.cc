#include "caffe2/core/common.h"
#include "caffe2/core/net.h"
#include "caffe2/core/observer.h"
#include "caffe2/core/operator.h"
#include "profile_observer.h"

#include <gtest/gtest.h>
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

namespace {

OperatorDef* add_op(
    const vector<string>& input,
    const vector<string>& output,
    const string& type,
    NetDef* net) {
  CHECK(net);
  auto& op = *net->add_op();
  op.set_type(type);
  for (const auto& in : input) {
    op.add_input(in);
  }
  for (const auto& out : output) {
    op.add_output(out);
  }

  return net->mutable_op(net->op_size() - 1);
}

void fill_tensor(
    const vector<int64_t>& shape,
    const vector<float>& data,
    TensorCPU* tensor) {
  tensor->Resize(shape);
  CAFFE_ENFORCE_EQ(data.size(), tensor->size());
  auto ptr = tensor->mutable_data<float>();
  for (int i = 0; i < tensor->size(); ++i) {
    ptr[i] = data[i];
  }
}

void add_blob(
    const string& name,
    const vector<int64_t>& shape,
    const vector<float>& data,
    Workspace* ws) {
  auto* blob = ws->CreateBlob(name);
  fill_tensor(shape, data, BlobGetMutableTensor(blob, CPU));
}

} // namespace

TEST(ProfileObserverTest, TestFC) {
  Workspace ws;
  auto create_net_def = [&ws](int M, int N, int K) {
    auto net_def = std::make_shared<NetDef>();
    net_def->set_name("test");
    add_op({"X", "W", "b"}, {"Y"}, "FC", net_def.get());
    add_blob("W", {N, K}, vector<float>(N * K), &ws);
    add_blob("b", {N}, vector<float>(N), &ws);
    add_blob("X", {M, K}, vector<float>(M * K), &ws);
    return net_def;
  };

  int M = 2, N = 3, K = 4;
  NetBase* net = ws.CreateNet(create_net_def(M, N, K), true /*overwrite*/);
  auto net_ob = caffe2::make_unique<ProfileObserver>(net);
  const auto* ob = net_ob.get();
  auto* ref = net->AttachObserver(std::move(net_ob));
  net->Run();
  CAFFE_ENFORCE(ob);
  auto cost_per_op_type = ob->getAggregatedOpTypeCost();
  CAFFE_ENFORCE(cost_per_op_type["FC"].flops == M * N * (2 * K + 1));
  CAFFE_ENFORCE(
      cost_per_op_type["FC"].bytes_read == (K * (M + N) + N) * sizeof(float));
  CAFFE_ENFORCE(cost_per_op_type["FC"].bytes_written == M * N * sizeof(float));
  net->DetachObserver(ref);
}
} // namespace caffe2
