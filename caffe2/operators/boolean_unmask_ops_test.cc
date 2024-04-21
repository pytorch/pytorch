#include <iostream>

#include <gtest/gtest.h>
#include "caffe2/core/context.h"
#include "caffe2/core/flags.h"
#include "caffe2/core/operator.h"

C10_DECLARE_string(caffe_test_root);

namespace caffe2 {

template <class DataT>
static void AddScalarInput(
    const DataT& value,
    const string& name,
    Workspace* ws,
    bool isEmpty = false) {
  Blob* blob = ws->CreateBlob(name);
  auto* tensor = BlobGetMutableTensor(blob, CPU);
  if (!isEmpty) {
    tensor->Resize(vector<int64_t>{1});
    *(tensor->template mutable_data<DataT>()) = value;
  } else {
    tensor->Resize(vector<int64_t>{0});
    tensor->template mutable_data<DataT>();
  }
  return;
}

// Test case for BooleanUnmask operator
//  mask1:   [ false ]
//  values1: [ ]
//  mask2:   [ true ]
//  values2: [ 1.0 ]
//
//  Expected Output: [ 1.0 ]
TEST(BooleanUnmaskTest, Test) {
  Workspace ws;
  OperatorDef def;

  def.set_name("test");
  def.set_type("BooleanUnmask");

  def.add_input("mask1");
  def.add_input("values1");
  def.add_input("mask2");
  def.add_input("values2");

  def.add_output("unmasked_data");

  AddScalarInput(false, "mask1", &ws);
  AddScalarInput(float(), "values1", &ws, true);
  AddScalarInput(true, "mask2", &ws);
  AddScalarInput(1.0f, "values2", &ws);

  unique_ptr<OperatorBase> op(CreateOperator(def, &ws));
  EXPECT_NE(nullptr, op.get());

  EXPECT_TRUE(op->Run());

  Blob* unmasked_data_blob = ws.GetBlob("unmasked_data");
  EXPECT_NE(nullptr, unmasked_data_blob);

  auto& unmasked_data = unmasked_data_blob->Get<TensorCPU>();
  EXPECT_EQ(unmasked_data.numel(), 1);

  TORCH_CHECK_EQ(unmasked_data.data<float>()[0], 1.0f);
}

} // namespace caffe2
