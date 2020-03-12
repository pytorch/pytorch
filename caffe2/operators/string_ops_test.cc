#include <gtest/gtest.h>
#include <vector>

#include "caffe2/operators/string_ops.h"

namespace caffe2 {

class StringJoinOpTest : public testing::Test {
 public:
  bool runOp(const Tensor& input) {
    auto* blob = ws_.CreateBlob("X");
    BlobSetTensor(blob, input.Alias());

    OperatorDef def;
    def.set_name("test");
    def.set_type("StringJoin");
    def.add_input("X");
    def.add_output("Y");

    auto op = CreateOperator(def, &ws_);
    return op->Run();
  }

  const std::string* checkAndGetOutput(int outputSize) {
    const auto* output = ws_.GetBlob("Y");
    EXPECT_NE(output, nullptr);
    EXPECT_TRUE(BlobIsTensorType(*output, CPU));
    const auto& outputTensor = output->Get<TensorCPU>();
    EXPECT_EQ(outputTensor.dim(), 1);
    EXPECT_EQ(outputTensor.size(0), outputSize);
    EXPECT_EQ(outputTensor.numel(), outputSize);
    return outputTensor.data<std::string>();
  }

 protected:
  Workspace ws_;
};

TEST_F(StringJoinOpTest, testString1DJoin) {
  std::vector<std::string> input = {"a", "xx", "c"};

  auto blob = std::make_unique<Blob>();
  auto* tensor = BlobGetMutableTensor(blob.get(), CPU);
  tensor->Resize(input.size());
  auto* data = tensor->template mutable_data<std::string>();
  for (int i = 0; i < input.size(); ++i) {
    *data++ = input[i];
  }

  EXPECT_TRUE(runOp(*tensor));

  const auto* outputData = checkAndGetOutput(input.size());
  EXPECT_EQ(outputData[0], "a,");
  EXPECT_EQ(outputData[1], "xx,");
  EXPECT_EQ(outputData[2], "c,");
}

TEST_F(StringJoinOpTest, testString2DJoin) {
  std::vector<std::vector<std::string>> input = {{"aa", "bb", "cc"},
                                                 {"dd", "ee", "ff"}};

  auto blob = std::make_unique<Blob>();
  auto* tensor = BlobGetMutableTensor(blob.get(), CPU);
  tensor->Resize(input.size(), input[0].size());
  auto* data = tensor->template mutable_data<std::string>();
  for (int i = 0; i < input.size(); ++i) {
    for (int j = 0; j < input[0].size(); ++j) {
      *data++ = input[i][j];
    }
  }

  EXPECT_TRUE(runOp(*tensor));

  const auto* outputData = checkAndGetOutput(input.size());
  EXPECT_EQ(outputData[0], "aa,bb,cc,");
  EXPECT_EQ(outputData[1], "dd,ee,ff,");
}

TEST_F(StringJoinOpTest, testFloat1DJoin) {
  std::vector<float> input = {3.90f, 5.234f, 8.12f};

  auto blob = std::make_unique<Blob>();
  auto* tensor = BlobGetMutableTensor(blob.get(), CPU);
  tensor->Resize(input.size());
  auto* data = tensor->template mutable_data<float>();
  for (int i = 0; i < input.size(); ++i) {
    *data++ = input[i];
  }

  EXPECT_TRUE(runOp(*tensor));

  const auto* outputData = checkAndGetOutput(input.size());
  EXPECT_EQ(outputData[0], "3.9,");
  EXPECT_EQ(outputData[1], "5.234,");
  EXPECT_EQ(outputData[2], "8.12,");
}

TEST_F(StringJoinOpTest, testFloat2DJoin) {
  std::vector<std::vector<float>> input = {{1.23f, 2.45f, 3.56f},
                                           {4.67f, 5.90f, 6.32f}};

  auto blob = std::make_unique<Blob>();
  auto* tensor = BlobGetMutableTensor(blob.get(), CPU);
  tensor->Resize(input.size(), input[0].size());
  auto* data = tensor->template mutable_data<float>();
  for (int i = 0; i < input.size(); ++i) {
    for (int j = 0; j < input[0].size(); ++j) {
      *data++ = input[i][j];
    }
  }

  EXPECT_TRUE(runOp(*tensor));

  const auto* outputData = checkAndGetOutput(input.size());
  EXPECT_EQ(outputData[0], "1.23,2.45,3.56,");
  EXPECT_EQ(outputData[1], "4.67,5.9,6.32,");
}

TEST_F(StringJoinOpTest, testLong2DJoin) {
  std::vector<std::vector<int64_t>> input = {{100, 200}, {1000, 2000}};

  auto blob = std::make_unique<Blob>();
  auto* tensor = BlobGetMutableTensor(blob.get(), CPU);
  tensor->Resize(input.size(), input[0].size());
  auto* data = tensor->template mutable_data<int64_t>();
  for (int i = 0; i < input.size(); ++i) {
    for (int j = 0; j < input[0].size(); ++j) {
      *data++ = input[i][j];
    }
  }

  EXPECT_TRUE(runOp(*tensor));

  const auto* outputData = checkAndGetOutput(input.size());
  EXPECT_EQ(outputData[0], "100,200,");
  EXPECT_EQ(outputData[1], "1000,2000,");
}
}
