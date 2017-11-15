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

REGISTER_CPU_OPERATOR(NetTestDummy, NetTestDummyOp<CPUContext>);
REGISTER_CPU_OPERATOR(NetTestDummy2, NetTestDummyOp<CPUContext>);

OPERATOR_SCHEMA(NetTestDummy)
    .NumInputs(0, INT_MAX)
    .NumOutputs(0, INT_MAX)
    .AllowInplace({{0, 0}, {1, 1}});
OPERATOR_SCHEMA(NetTestDummy2)
    .NumInputs(0, INT_MAX)
    .NumOutputs(0, INT_MAX)
    .AllowInplace({{1, 0}});

TEST(NetTest, ConstructionNoDeclaredInputOutput) {
  Workspace ws;
  ws.CreateBlob("in");
  unique_ptr<NetBase> net(
      CreateNetTestHelper(&ws, vector<string>(), vector<string>()));
  EXPECT_TRUE(net.get() != nullptr);
}

TEST(NetTest, ConstructionDeclaredInput) {
  Workspace ws;
  ws.CreateBlob("in");
  unique_ptr<NetBase> net(
      CreateNetTestHelper(&ws, vector<string>{"in"}, vector<string>()));
  EXPECT_TRUE(net.get() != nullptr);
}

TEST(NetTest, ConstructionDeclaredOutput) {
  Workspace ws;
  ws.CreateBlob("in");
  unique_ptr<NetBase> net(
      CreateNetTestHelper(&ws, vector<string>(), vector<string>{"out"}));
  EXPECT_TRUE(net.get() != nullptr);
}

TEST(NetTest, DeclaredInputInsufficient) {
  Workspace ws;
  ws.CreateBlob("in");
  ASSERT_THROW(
      CreateNetTestHelper(&ws, vector<string>{"unuseful_in"}, vector<string>()),
      EnforceNotMet);
}

TEST(NetDeathTest, DeclaredOutputNotMet) {
  Workspace ws;
  ws.CreateBlob("in");
  ASSERT_THROW(
      CreateNetTestHelper(
          &ws, vector<string>(), vector<string>{"unproduced_out"}),
      EnforceNotMet);
}

TEST(NetTest, ChainingForLinearModel) {
  const auto spec = R"DOC(
        name: "example"
        type: "dag"
        external_input: "in"
        op {
          input: "in"
          output: "hidden"
          type: "NetTestDummy"
        }
        op {
          input: "hidden"
          output: "out"
          type: "NetTestDummy"
        }
)DOC";
  checkChainingAndRun(spec, {{0, {0, 1}}});
}

TEST(NetTest, ChainingForFork) {
  const auto spec = R"DOC(
        name: "example"
        type: "dag"
        external_input: "in"
        op {
          input: "in"
          output: "hidden"
          type: "NetTestDummy"
        }
        op {
          input: "hidden"
          output: "out1"
          type: "NetTestDummy"
        }
        op {
          input: "hidden"
          output: "out2"
          type: "NetTestDummy"
        }
)DOC";
  checkChainingAndRun(spec, {{0, {0}}, {1, {1}}, {2, {2}}});
}

// TEST(NetTest, ChainingForJoinWithAncestor) {
//   const auto spec = R"DOC(
//         name: "example"
//         type: "dag"
//         external_input: "in"
//         op {
//           input: "in"
//           output: "hidden"
//           type: "NetTestDummy"
//         }
//         op {
//           input: "hidden"
//           output: "out1"
//           type: "NetTestDummy"
//         }
//         op {
//           input: "hidden"
//           output: "out2"
//           type: "NetTestDummy"
//         }
//         op {
//           input: "hidden"
//           input: "out2"
//           type: "NetTestDummy"
//         }
// )DOC";
//   checkChainingAndRun(spec, {{0, {0}}, {1, {1}}, {2, {2, 3}}});
// }

TEST(NetTest, ChainingForForkJoin) {
  const auto spec = R"DOC(
        name: "example"
        type: "dag"
        external_input: "in"
        op {
          input: "in"
          output: "hidden1"
          type: "NetTestDummy"
        }
        op {
          input: "in"
          output: "hidden2"
          type: "NetTestDummy"
        }
        op {
          input: "hidden1"
          input: "hidden2"
          output: "out"
          type: "NetTestDummy"
        }
        op {
          input: "out"
          output: "out2"
          type: "NetTestDummy"
        }
)DOC";
  checkChainingAndRun(spec, {{0, {0}}, {1, {1}}, {2, {2, 3}}});
}

TEST(NetTest, ChainingForwardBackward) {
  const auto spec = R"DOC(
  name: "gpu_0"
  type: "dag"
  op {
    input: "in"
    input: "fc_0_w"
    input: "fc_0_b"
    output: "fc_0"
    name: "0"
    type: "NetTestDummy"
  }
  op {
    input: "fc_0"
    output: "fc_0"
    name: "1"
    type: "NetTestDummy"
  }
  op {
    input: "fc_0"
    input: "fc_1_w"
    input: "fc_1_b"
    output: "fc_1"
    name: "2"
    type: "NetTestDummy"
  }
  op {
    input: "fc_1"
    output: "fc_1"
    name: "3"
    type: "NetTestDummy"
  }
  op {
    input: "fc_1"
    input: "fc_2_w"
    input: "fc_2_b"
    output: "fc_2"
    name: "4"
    type: "NetTestDummy"
  }
  op {
    input: "fc_2"
    output: "fc_2"
    name: "5"
    type: "NetTestDummy"
  }
  op {
    input: "fc_2"
    input: "fc_3_w"
    input: "fc_3_b"
    output: "fc_3"
    name: "6"
    type: "NetTestDummy"
  }
  op {
    input: "fc_3"
    output: "fc_3"
    name: "7"
    type: "NetTestDummy"
  }
  op {
    input: "fc_3"
    input: "fc_4_w"
    input: "fc_4_b"
    output: "fc_4"
    name: "8"
    type: "NetTestDummy"
  }
  op {
    input: "fc_4"
    output: "fc_4"
    name: "9"
    type: "NetTestDummy"
  }
  op {
    input: "fc_4"
    input: "in2"
    output: "LabelCrossEntropy"
    name: "10"
    type: "NetTestDummy"
  }
  op {
    input: "LabelCrossEntropy"
    output: "AveragedLoss"
    name: "11"
    type: "NetTestDummy"
  }
  op {
    input: "AveragedLoss"
    output: "AveragedLoss_autogen_grad"
    name: "12"
    type: "NetTestDummy"
  }
  op {
    input: "LabelCrossEntropy"
    input: "AveragedLoss_autogen_grad"
    output: "LabelCrossEntropy_grad"
    name: "13"
    type: "NetTestDummy"
  }
  op {
    input: "fc_4"
    input: "label"
    input: "LabelCrossEntropy_grad"
    output: "fc_4_grad"
    name: "14"
    type: "NetTestDummy2"
  }
  op {
    input: "fc_4"
    input: "fc_4_grad"
    output: "fc_4_grad"
    name: "15"
    type: "NetTestDummy2"
  }
  op {
    input: "fc_3"
    input: "fc_4_w"
    input: "fc_4_grad"
    output: "fc_4_w_grad"
    output: "fc_4_b_grad"
    output: "fc_3_grad"
    name: "16"
    type: "NetTestDummy"
  }
  op {
    input: "fc_3"
    input: "fc_3_grad"
    output: "fc_3_grad"
    name: "17"
    type: "NetTestDummy2"
  }
  op {
    input: "fc_2"
    input: "fc_3_w"
    input: "fc_3_grad"
    output: "fc_3_w_grad"
    output: "fc_3_b_grad"
    output: "fc_2_grad"
    name: "18"
    type: "NetTestDummy"
  }
  op {
    input: "fc_2"
    input: "fc_2_grad"
    output: "fc_2_grad"
    name: "19"
    type: "NetTestDummy2"
  }
  op {
    input: "fc_1"
    input: "fc_2_w"
    input: "fc_2_grad"
    output: "fc_2_w_grad"
    output: "fc_2_b_grad"
    output: "fc_1_grad"
    name: "20"
    type: "NetTestDummy"
  }
  op {
    input: "fc_1"
    input: "fc_1_grad"
    output: "fc_1_grad"
    name: "21"
    type: "NetTestDummy2"
  }
  op {
    input: "fc_0"
    input: "fc_1_w"
    input: "fc_1_grad"
    output: "fc_1_w_grad"
    output: "fc_1_b_grad"
    output: "fc_0_grad"
    name: "22"
    type: "NetTestDummy"
  }
  op {
    input: "fc_0"
    input: "fc_0_grad"
    output: "fc_0_grad"
    name: "23"
    type: "NetTestDummy2"
  }
  op {
    input: "in"
    input: "fc_0_w"
    input: "fc_0_grad"
    output: "fc_0_w_grad"
    output: "fc_0_b_grad"
    output: "data_grad"
    name: "24"
    type: "NetTestDummy"
  }
  external_input: "in"
  external_input: "in2"
  external_input: "LR"
  external_input: "fc_0_w"
  external_input: "fc_0_b"
  external_input: "fc_1_w"
  external_input: "fc_1_b"
  external_input: "fc_2_w"
  external_input: "fc_2_b"
  external_input: "fc_3_w"
  external_input: "fc_3_b"
  external_input: "fc_4_w"
  external_input: "fc_4_b"
  external_input: "label"
  )DOC";
  checkNumChainsAndRun(spec, 1);
}

TEST(NetTest, ChainingForHogwildModel) {
  const auto spec = R"DOC(
        name: "example"
        type: "dag"
        external_input: "in"
        op {
          input: "in"
          output: "hidden1"
          type: "NetTestDummy"
        }
        op {
          input: "hidden1"
          output: "mid1"
          type: "NetTestDummy"
        }
        op {
          input: "mid1"
          output: "out1"
          type: "NetTestDummy"
        }
        op {
          input: "in"
          output: "hidden2"
          type: "NetTestDummy"
        }
        op {
          input: "hidden2"
          output: "mid2"
          type: "NetTestDummy"
        }
        op {
          input: "mid2"
          output: "out2"
          type: "NetTestDummy"
        }
)DOC";
  checkNumChainsAndRun(spec, 2);
}

TEST(NetTest, FailingOperator) {
  const auto spec = R"DOC(
        name: "example"
        type: "dag"
        external_input: "in"
        op {
          input: "in"
          output: "hidden"
          type: "NetTestDummy"
        }
        op {
          input: "hidden"
          output: "out"
          type: "NetTestDummy"
          arg {
            name: "fail"
            i: 1
          }
        }
)DOC";

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
    for (int i = 0; i < 10; i++) {
      NetTestDummyOp<CPUContext>::counter.exchange(0);
      ASSERT_EQ(false, net.get()->Run());
      ASSERT_EQ(1, NetTestDummyOp<CPUContext>::counter.load());
    }
  }
}

} // namespace caffe2
