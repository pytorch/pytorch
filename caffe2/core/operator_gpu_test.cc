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

#include <string>

#include <gtest/gtest.h>
#include "caffe2/core/common_gpu.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

class JustTest : public OperatorBase {
 public:
  using OperatorBase::OperatorBase;
  bool Run(int /* unused */ /*stream_id*/) override {
    return true;
  }
  virtual std::string type() {
    return "BASE";
  }
};

class JustTestCUDA : public JustTest {
 public:
  using JustTest::JustTest;
  bool Run(int /* unused */ /*stream_id*/) override {
    return true;
  }
  std::string type() override {
    return "CUDA";
  }
};

class JustTestCUDNN : public JustTest {
 public:
  using JustTest::JustTest;
  bool Run(int /* unused */ /*stream_id*/) override {
    return true;
  }
  std::string type() override {
    return "CUDNN";
  }
};

OPERATOR_SCHEMA(JustTest).NumInputs(0, 1).NumOutputs(0, 1);
REGISTER_CUDA_OPERATOR(JustTest, JustTestCUDA);
REGISTER_CUDNN_OPERATOR(JustTest, JustTestCUDNN);

TEST(EnginePrefTest, GPUDeviceDefaultPreferredEngines) {
  if (!HasCudaGPU())
    return;
  OperatorDef op_def;
  Workspace ws;
  op_def.mutable_device_option()->set_device_type(CUDA);
  op_def.set_type("JustTest");

  {
    const auto op = CreateOperator(op_def, &ws);
    EXPECT_NE(nullptr, op.get());
    // CUDNN should be taken as it's in the default global preferred engines
    // list
    EXPECT_EQ(static_cast<JustTest*>(op.get())->type(), "CUDNN");
  }
}

} // namespace caffe2
