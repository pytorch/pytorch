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

#include "caffe2/core/context.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/flags.h"
#include "caffe2/operators/reshape_op.h"
#include <gtest/gtest.h>

CAFFE2_DECLARE_string(caffe_test_root);

namespace caffe2 {

static void AddConstInput(
    const vector<TIndex>& shape,
    const float value,
    const string& name,
    Workspace* ws) {
  DeviceOption option;
  option.set_device_type(CUDA);
  CUDAContext context(option);
  Blob* blob = ws->CreateBlob(name);
  auto* tensor = blob->GetMutable<Tensor<CUDAContext>>();
  tensor->Resize(shape);
  math::Set<float, CUDAContext>(
      tensor->size(), value, tensor->mutable_data<float>(), &context);
  return;
}

TEST(ReshapeOpGPUTest, testReshapeWithScalar) {
  if (!HasCudaGPU())
    return;
  Workspace ws;
  OperatorDef def;
  def.set_name("test_reshape");
  def.set_type("Reshape");
  def.add_input("X");
  def.add_output("XNew");
  def.add_output("OldShape");
  def.add_arg()->CopyFrom(MakeArgument("shape", vector<int64_t>{1}));
  def.mutable_device_option()->set_device_type(CUDA);
  AddConstInput(vector<TIndex>(), 3.14, "X", &ws);
  // execute the op
  unique_ptr<OperatorBase> op(CreateOperator(def, &ws));
  EXPECT_TRUE(op->Run());
  Blob* XNew = ws.GetBlob("XNew");
  const Tensor<CUDAContext>& XNewTensor = XNew->Get<Tensor<CUDAContext>>();
  EXPECT_EQ(1, XNewTensor.ndim());
  EXPECT_EQ(1, XNewTensor.size());
}

} // namespace caffe2
