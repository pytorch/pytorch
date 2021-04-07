#include <map>

#include "caffe2/core/flags.h"
#include "caffe2/core/logging.h"
#include "caffe2/operators/half_float_ops.h"
#include "caffe2/utils/conversions.h"

#include <c10/util/irange.h>

#include <gtest/gtest.h>
C10_DECLARE_string(caffe_test_root);

namespace caffe2 {

TEST(Float16, SimpleTest) {
  Workspace ws;
  vector<float> data = {0.1f, 0.23f, 1.6f, 8.2f, -13.9f};

  // loading input data
  Blob* dataBlob = ws.CreateBlob("data");
  auto tensor = BlobGetMutableTensor(dataBlob, CPU);
  tensor->Resize(data.size());
  for (auto i = 0; i < tensor->numel(); ++i) {
    tensor->mutable_data<float>()[i] = data[i];
  }

  // encoding fp32 -> fp16
  OperatorDef def;
  def.set_name("test");
  def.set_type("FloatToHalf");
  def.add_input("data");
  def.add_output("data16");
  unique_ptr<OperatorBase> op(CreateOperator(def, &ws));
  EXPECT_NE(nullptr, op.get());
  EXPECT_TRUE(op->Run());

  // run some sanity checks
  Blob* outputBlob = ws.GetBlob("data16");
  EXPECT_NE(nullptr, outputBlob);
  EXPECT_TRUE(outputBlob->IsType<Tensor>());
  const TensorCPU& outputTensor = outputBlob->Get<Tensor>();
  EXPECT_EQ(outputTensor.numel(), 5);
  EXPECT_NO_THROW(outputTensor.data<at::Half>());

  // decode fp16 -> fp32
  OperatorDef def2;
  def2.set_name("test");
  def2.set_type("HalfToFloat");
  def2.add_input("data16");
  def2.add_output("result");
  unique_ptr<OperatorBase> op2(CreateOperator(def2, &ws));
  EXPECT_NE(nullptr, op2.get());
  EXPECT_TRUE(op2->Run());

  // validate result
  Blob* resultBlob = ws.GetBlob("result");
  EXPECT_NE(nullptr, resultBlob);
  EXPECT_TRUE(resultBlob->IsType<Tensor>());
  const TensorCPU& resultTensor = resultBlob->Get<Tensor>();
  EXPECT_EQ(resultTensor.numel(), 5);

  for (const auto i : c10::irange(data.size())) {
    EXPECT_NEAR(resultTensor.data<float>()[i], data[i], 0.01);
  }
}

TEST(Float16, UniformDistributionTest) {
  Workspace ws;

  OperatorDef def;
  def.set_name("test");
  def.set_type("Float16UniformFill");
  int64_t size = 5000000L;
  std::vector<int64_t> shape = {size, 32};
  long tot_size = shape[0];
  for (const auto i : c10::irange(1, shape.size())) {
    tot_size *= shape[i];
  }
  caffe2::AddArgument<std::vector<int64_t>>("shape", shape, &def);
  caffe2::AddArgument<float>("min", -20.0, &def);
  caffe2::AddArgument<float>("max", 20.0, &def);
  def.add_output("result");

  unique_ptr<OperatorBase> op(CreateOperator(def, &ws));
  EXPECT_NE(nullptr, op.get());
  EXPECT_TRUE(op->Run());

  Blob* resultBlob = ws.GetBlob("result");
  const TensorCPU& resultTensor = resultBlob->Get<Tensor>();
  EXPECT_EQ(resultTensor.numel(), tot_size);
  double mean = 0.0, var = 0.0;
  const at::Half* data = resultTensor.data<at::Half>();
  for (auto i = 0; i < resultTensor.numel(); i++) {
    float x = caffe2::convert::Get<float, at::Half>(data[i]);
    mean += x;
    var += x * x;
  }
  mean /= tot_size;
  var /= tot_size;
  LOG(INFO) << "m " << mean << " " << var;

  // The uniform distribution of [-20,20] should have a mean of 0
  // and a variance of 40^2/12
  EXPECT_TRUE(fabs(mean) < 0.1);
  EXPECT_TRUE(fabs(var - 133.33) / 133.33 < 0.1);
}

} // namespace caffe2
