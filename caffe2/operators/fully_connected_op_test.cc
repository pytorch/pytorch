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

#include <iostream>

#include "caffe2/operators/fully_connected_op.h"
#include "caffe2/core/flags.h"
#include <gtest/gtest.h>

CAFFE2_DECLARE_string(caffe_test_root);

namespace caffe2 {

static void AddConstInput(const vector<TIndex>& shape, const float value,
                          const string& name, Workspace* ws) {
  DeviceOption option;
  CPUContext context(option);
  Blob* blob = ws->CreateBlob(name);
  auto* tensor = blob->GetMutable<TensorCPU>();
  tensor->Resize(shape);
  math::Set<float, CPUContext>(tensor->size(), value,
                               tensor->mutable_data<float>(),
                               &context);
  return;
}

TEST(FullyConnectedTest, FCTest) {
  Workspace ws;
  OperatorDef def;
  def.set_name("test");
  def.set_type("FC");
  def.add_input("X");
  def.add_input("W");
  def.add_input("B");
  def.add_output("Y");
  AddConstInput(vector<TIndex>{5, 10}, 1., "X", &ws);
  AddConstInput(vector<TIndex>{6, 10}, 1., "W", &ws);
  AddConstInput(vector<TIndex>{6}, 0.1, "B", &ws);
  unique_ptr<OperatorBase> op(CreateOperator(def, &ws));
  ASSERT_NE(nullptr, op.get());
  ASSERT_TRUE(op->Run());
  Blob* Yblob = ws.GetBlob("Y");
  ASSERT_NE(nullptr, Yblob);
  auto& Y = Yblob->Get<TensorCPU>();
  ASSERT_EQ(5 * 6, Y.size());
  for (int i = 0; i < Y.size(); ++i) {
    EXPECT_FLOAT_EQ(10.1f, Y.data<float>()[i]);
  }
}

TEST(FullyConnectedTest, FCTransposedTest) {
  Workspace ws;
  OperatorDef def;
  def.set_name("test");
  def.set_type("FCTransposed");
  def.add_input("X");
  def.add_input("W");
  def.add_input("B");
  def.add_output("Y");
  AddConstInput(vector<TIndex>{5, 10}, 1., "X", &ws);
  AddConstInput(vector<TIndex>{10, 6}, 1., "W", &ws);
  AddConstInput(vector<TIndex>{6}, 0.1, "B", &ws);
  unique_ptr<OperatorBase> op(CreateOperator(def, &ws));
  ASSERT_NE(nullptr, op.get());
  ASSERT_TRUE(op->Run());
  Blob* Yblob = ws.GetBlob("Y");
  ASSERT_NE(nullptr, Yblob);
  auto& Y = Yblob->Get<TensorCPU>();
  ASSERT_EQ(5 * 6, Y.size());
  for (int i = 0; i < Y.size(); ++i) {
    EXPECT_FLOAT_EQ(10.1f, Y.data<float>()[i]);
  }
}

}  // namespace caffe2
