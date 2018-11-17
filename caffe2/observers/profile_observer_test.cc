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

template <typename T = float>
void fill_tensor(
    const vector<int64_t>& shape,
    const vector<T>& data,
    TensorCPU* tensor) {
  tensor->Resize(shape);
  CAFFE_ENFORCE_EQ(data.size(), tensor->size());
  auto ptr = tensor->mutable_data<T>();
  for (int i = 0; i < tensor->size(); ++i) {
    ptr[i] = data[i];
  }
}

template <typename T = float>
void add_blob(
    const string& name,
    const vector<int64_t>& shape,
    const vector<T>& data,
    Workspace* ws) {
  auto* blob = ws->CreateBlob(name);
  fill_tensor<T>(shape, data, BlobGetMutableTensor(blob, CPU));
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

TEST(CostObserverTest, TestLog) {
  Workspace ws;
  auto create_net_def = [&ws](int M, int N) {
    auto net_def = std::make_shared<NetDef>();
    net_def->set_name("test");
    add_op({"X"}, {"Y"}, "Log", net_def.get());
    add_blob("X", {M, N}, vector<float>(M * N), &ws);
    return net_def;
  };

  int M = 2, N = 3;
  NetBase* net = ws.CreateNet(create_net_def(M, N), true /*overwrite*/);
  auto net_ob = caffe2::make_unique<ProfileObserver>(net);
  auto* ob = net_ob.get();
  auto* ref = net->AttachObserver(std::move(net_ob));
  net->Run();
  CAFFE_ENFORCE(ob);
  auto cost_per_op_type = ob->getAggregatedOpTypeCost();
  CAFFE_ENFORCE(cost_per_op_type["Log"].flops == M * N);
  CAFFE_ENFORCE(cost_per_op_type["Log"].bytes_read == M * N * sizeof(float));
  CAFFE_ENFORCE(cost_per_op_type["Log"].bytes_written == M * N * sizeof(float));
  net->DetachObserver(ref);
}

TEST(CostObserverTest, TestLSTMUnit) {
  Workspace ws;
  auto create_net_def = [&ws](int N, int D) {
    auto net_def = std::make_shared<NetDef>();
    net_def->set_name("test");
    add_op(
        {"H_PREV", "C_PREV", "G", "SEQ_LEN", "TIMESTEP"},
        {"H", "C"},
        "LSTMUnit",
        net_def.get());
    add_blob("H_PREV", {1, N, D}, vector<float>(N * D), &ws);
    add_blob("C_PREV", {1, N, D}, vector<float>(N * D), &ws);
    add_blob("G", {1, N, 4 * D}, vector<float>(N * 4 * D), &ws);
    add_blob<int>("SEQ_LEN", {N}, vector<int>(N, 2), &ws);
    add_blob<int>("TIMESTEP", {1}, vector<int>(1, 1), &ws);
    return net_def;
  };

  int N = 2, D = 3;
  NetBase* net = ws.CreateNet(create_net_def(N, D), true /*overwrite*/);
  auto net_ob = caffe2::make_unique<ProfileObserver>(net);
  auto* ob = net_ob.get();
  auto* ref = net->AttachObserver(std::move(net_ob));
  net->Run();
  CAFFE_ENFORCE(ob);
  auto cost_per_op_type = ob->getAggregatedOpTypeCost();
  CAFFE_ENFORCE(
      cost_per_op_type["LSTMUnit"].flops == 5 * D * N + (15 * D + 6) * N);
  CAFFE_ENFORCE(
      cost_per_op_type["LSTMUnit"].bytes_read == 5 * D * N * sizeof(float));
  CAFFE_ENFORCE(
      cost_per_op_type["LSTMUnit"].bytes_written == 2 * D * N * sizeof(float));
  net->DetachObserver(ref);
}

TEST(CostObserverTest, TestSigmoid) {
  Workspace ws;
  auto create_net_def = [&ws](int M, int N) {
    auto net_def = std::make_shared<NetDef>();
    net_def->set_name("test");
    add_op({"X"}, {"Y"}, "Sigmoid", net_def.get());
    add_blob("X", {M, N}, vector<float>(M * N), &ws);
    return net_def;
  };

  int M = 2, N = 3;
  NetBase* net = ws.CreateNet(create_net_def(M, N), true /*overwrite*/);
  auto net_ob = caffe2::make_unique<ProfileObserver>(net);
  auto* ob = net_ob.get();
  auto* ref = net->AttachObserver(std::move(net_ob));
  net->Run();
  CAFFE_ENFORCE(ob);
  auto cost_per_op_type = ob->getAggregatedOpTypeCost();
  CAFFE_ENFORCE(cost_per_op_type["Sigmoid"].flops == M * N);
  CAFFE_ENFORCE(
      cost_per_op_type["Sigmoid"].bytes_read == M * N * sizeof(float));
  CAFFE_ENFORCE(
      cost_per_op_type["Sigmoid"].bytes_written == M * N * sizeof(float));
  net->DetachObserver(ref);
}

TEST(CostObserverTest, TestPow) {
  Workspace ws;
  auto create_net_def = [&ws](int M, int N) {
    auto net_def = std::make_shared<NetDef>();
    net_def->set_name("test");
    add_op({"X", "E"}, {"Y"}, "Pow", net_def.get());
    add_blob("X", {M, N}, vector<float>(M * N), &ws);
    add_blob("E", {M, N}, vector<float>(M * N), &ws);
    return net_def;
  };

  int M = 2, N = 3;
  NetBase* net = ws.CreateNet(create_net_def(M, N), true /*overwrite*/);
  auto net_ob = caffe2::make_unique<ProfileObserver>(net);
  auto* ob = net_ob.get();
  auto* ref = net->AttachObserver(std::move(net_ob));
  net->Run();
  CAFFE_ENFORCE(ob);
  auto cost_per_op_type = ob->getAggregatedOpTypeCost();
  CAFFE_ENFORCE(cost_per_op_type["Pow"].flops == M * N);
  CAFFE_ENFORCE(cost_per_op_type["Pow"].bytes_read == M * N * sizeof(float));
  CAFFE_ENFORCE(cost_per_op_type["Pow"].bytes_written == M * N * sizeof(float));
  net->DetachObserver(ref);
}

TEST(CostObserverTest, TestTanh) {
  Workspace ws;
  auto create_net_def = [&ws](int M, int N) {
    auto net_def = std::make_shared<NetDef>();
    net_def->set_name("test");
    add_op({"X"}, {"Y"}, "Tanh", net_def.get());
    add_blob("X", {M, N}, vector<float>(M * N), &ws);
    return net_def;
  };

  int M = 2, N = 3;
  NetBase* net = ws.CreateNet(create_net_def(M, N), true /*overwrite*/);
  auto net_ob = caffe2::make_unique<ProfileObserver>(net);
  auto* ob = net_ob.get();
  auto* ref = net->AttachObserver(std::move(net_ob));
  net->Run();
  CAFFE_ENFORCE(ob);
  auto cost_per_op_type = ob->getAggregatedOpTypeCost();
  CAFFE_ENFORCE(cost_per_op_type["Tanh"].flops == M * N);
  CAFFE_ENFORCE(cost_per_op_type["Tanh"].bytes_read == M * N * sizeof(float));
  CAFFE_ENFORCE(
      cost_per_op_type["Tanh"].bytes_written == M * N * sizeof(float));
  net->DetachObserver(ref);
}

TEST(CostObserverTest, TestSoftmax) {
  Workspace ws;
  auto create_net_def = [&ws](int M, int N) {
    auto net_def = std::make_shared<NetDef>();
    net_def->set_name("test");
    add_op({"X"}, {"Y"}, "Softmax", net_def.get());
    add_blob("X", {M, N}, vector<float>(M * N), &ws);
    return net_def;
  };

  int M = 2, N = 3;
  NetBase* net = ws.CreateNet(create_net_def(M, N), true /*overwrite*/);
  auto net_ob = caffe2::make_unique<ProfileObserver>(net);
  auto* ob = net_ob.get();
  auto* ref = net->AttachObserver(std::move(net_ob));
  net->Run();
  CAFFE_ENFORCE(ob);
  auto cost_per_op_type = ob->getAggregatedOpTypeCost();
  CAFFE_ENFORCE(cost_per_op_type["Softmax"].flops == M * N);
  CAFFE_ENFORCE(
      cost_per_op_type["Softmax"].bytes_read == M * N * sizeof(float));
  CAFFE_ENFORCE(
      cost_per_op_type["Softmax"].bytes_written == M * N * sizeof(float));
  net->DetachObserver(ref);
}
} // namespace caffe2
