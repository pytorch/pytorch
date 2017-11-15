/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "caffe2/core/net_test_util.h"

namespace caffe2 {

unique_ptr<NetBase> CreateNetTestHelper(
    Workspace* ws,
    const vector<string>& input,
    const vector<string>& output) {
  NetDef net_def;
  {
    auto& op = *(net_def.add_op());
    op.set_type("NetTestDummy");
    op.add_input("in");
    op.add_output("hidden");
  }
  {
    auto& op = *(net_def.add_op());
    op.set_type("NetTestDummy");
    op.add_input("hidden");
    op.add_output("out");
  }

  for (const auto& name : input) {
    net_def.add_external_input(name);
  }
  for (const auto& name : output) {
    net_def.add_external_output(name);
  }
  return CreateNet(net_def, ws);
}

void testExecution(std::unique_ptr<NetBase>& net) {
  int num_cpu_ops = 0;
  for (const auto& op : net->debug_def().op()) {
    if (op.device_option().device_type() == CPU) {
      num_cpu_ops++;
    }
  }
  // Run 100 times
  for (int i = 0; i < 100; i++) {
    NetTestDummyOp<CPUContext>::counter.exchange(0);
    net.get()->Run();
    ASSERT_EQ(num_cpu_ops, NetTestDummyOp<CPUContext>::counter.load());
  }
}

void checkChainingAndRun(
    const char* spec,
    const dag_utils::ExecutionChains& expected) {
  Workspace ws;
  ws.CreateBlob("in");
  NetDef net_def;
  CAFFE_ENFORCE(google::protobuf::TextFormat::ParseFromString(spec, &net_def));
  {
    net_def.set_num_workers(4);
    auto old = FLAGS_caffe2_disable_chaining;
    auto g = MakeGuard([&]() { FLAGS_caffe2_disable_chaining = old; });
    FLAGS_caffe2_disable_chaining = false;

    std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
    auto* dag = dynamic_cast_if_rtti<DAGNetBase*>(net.get());
    CHECK_NOTNULL(dag);
    const auto& chains = dag->TEST_execution_chains();
    EXPECT_TRUE(chains == expected);
    testExecution(net);
  }
}

void checkNumChainsAndRun(const char* spec, const int expected_num_chains) {
  Workspace ws;

  NetDef net_def;
  CAFFE_ENFORCE(google::protobuf::TextFormat::ParseFromString(spec, &net_def));
  net_def.set_num_workers(4);

  // Create all external inputs
  for (auto inp : net_def.external_input()) {
    ws.CreateBlob(inp);
  }

  {
    auto old = FLAGS_caffe2_disable_chaining;
    auto g = MakeGuard([&]() { FLAGS_caffe2_disable_chaining = old; });
    FLAGS_caffe2_disable_chaining = false;

    std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
    auto* dag = dynamic_cast_if_rtti<DAGNetBase*>(net.get());
    CHECK_NOTNULL(dag);
    const auto& chains = dag->TEST_execution_chains();
    EXPECT_EQ(expected_num_chains, chains.size());
    testExecution(net);
  }
}

} // namespace caffe2
