#include <iostream>

#include "caffe2/operators/tensor_protos_db_input.h"
#include "caffe2/core/flags.h"
#include "gtest/gtest.h"

CAFFE2_DECLARE_string(caffe_test_root);

const char* kTestDBPath = "/caffe2/data/mnist/mnist-train-nhwc-minidb";

namespace caffe2 {

const int kNumItems = 51200;
const int kLabelsToCheck = 12;
const int kLabels[] = {5, 0, 4, 1, 9, 2, 1, 3, 1, 4, 3, 5};

static void TestMNISTLoad(const int batch_size) {
  Workspace ws;
  OperatorDef def;
  def.set_name("test");
  def.set_type("TensorProtosDBInput");
  def.add_output("data");
  def.add_output("label");
  auto* batch_arg = def.add_arg();
  batch_arg->set_name("batch_size");
  batch_arg->set_i(batch_size);
  auto* db_arg = def.add_arg();
  db_arg->set_name("db");
  db_arg->set_s(FLAGS_caffe_test_root + string(kTestDBPath));
  auto* db_type_arg = def.add_arg();
  db_type_arg->set_name("db_type");
  db_type_arg->set_s("minidb");
  unique_ptr<OperatorBase> op(CreateOperator(def, &ws));
  EXPECT_NE(nullptr, op.get());
  for (int iter = 0; iter < kNumItems / batch_size; ++iter) {
    EXPECT_TRUE(op->Run());
    // Inspect the result
    auto* data_blob = ws.GetBlob("data");
    EXPECT_TRUE((data_blob->IsType<TensorCPU>()));
    auto* label_blob = ws.GetBlob("label");
    EXPECT_TRUE((label_blob->IsType<TensorCPU>()));
    auto& data_tensor = data_blob->Get<TensorCPU>();
    auto& label_tensor = label_blob->Get<TensorCPU>();
    EXPECT_EQ(data_tensor.ndim(), 4);
    EXPECT_EQ(data_tensor.dim32(0), batch_size);
    EXPECT_EQ(data_tensor.dim32(1), 28);
    EXPECT_EQ(data_tensor.dim32(2), 28);
    EXPECT_EQ(data_tensor.dim32(3), 1);
    EXPECT_EQ(label_tensor.ndim(), 1);
    EXPECT_EQ(label_tensor.dim32(0), batch_size);
    /*
    // Visualization just for inspection purpose.
    int idx = 0;
    for (int b = 0; b < batch_size; ++b) {
      for (int row = 0; row < 28; ++row) {
        for (int col = 0; col < 28; ++col) {
          std::cout << (data_tensor.data()[idx++] > 128) << " ";
        }
        std::cout << std::endl;
      }
      std::cout << std::endl << std::endl;
    }
    std::cout << "label: " << label_tensor.data()[0] << std::endl;
    */
    for (int i = 0; i < batch_size; ++i) {
      if (iter * batch_size + i < kLabelsToCheck) {
        EXPECT_EQ(label_tensor.data<int>()[i], kLabels[iter * batch_size + i]);
      }
    }
  }
}

TEST(TensorProtosDBInputTest, TestLoadBatchOne) {
  TestMNISTLoad(1);
}

TEST(TensorProtosDBInputTest, TestLoadBatch64) {
  TestMNISTLoad(64);
}

}  // namespace caffe2
