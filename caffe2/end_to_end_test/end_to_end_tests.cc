#include "caffe2/core/context.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/proto_utils.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

DECLARE_string(caffe_test_root);

namespace caffe2 {

const char kToyRegressionTestPlanPath[] = "/data/toy/toy_regression.pbtxt";
const char kMNISTLinearClassificationPath[] =
    "/data/mnist/linear_classifier_plan.pbtxt";
const char kMNISTTwoLayerReluClassificationPath[] =
    "/data/mnist/mnist_relu_network.pbtxt";
const char kMNISTLeNetClassificationPath[] =
    "/data/mnist/mnist_lenet.pbtxt";
const char kMNISTLeNetClassificationGPUPath[] =
    "/data/mnist/mnist_lenet_gpu.pbtxt";
const char kMNISTLeNetNHWCClassificationPath[] =
    "/data/mnist/mnist_lenet_nhwc.pbtxt";
const char kMNISTLeNetNHWCClassificationGPUPath[] =
    "/data/mnist/mnist_lenet_nhwc_gpu.pbtxt";
const char kMNISTLeNetGroupConvClassificationPath[] =
    "/data/mnist/mnist_lenet_group_convolution.pbtxt";
const char kMNISTLeNetGroupConvNHWCClassificationPath[] =
    "/data/mnist/mnist_lenet_group_convolution_nhwc.pbtxt";


template <typename dtype, class DeviceContext>
void ExpectTensorEquivalence(const Workspace& ws, const string& name_a,
                             const string& name_b,
                             const float relative_error) {
  const Blob* a = ws.GetBlob(name_a);
  EXPECT_TRUE(a != nullptr);
  EXPECT_TRUE((a->IsType<Tensor<dtype, DeviceContext> >()));
  int size = a->Get<Tensor<dtype, DeviceContext> >().size();
  const dtype* a_data = a->Get<Tensor<dtype, DeviceContext> >().data();
  const Blob* b = ws.GetBlob(name_b);
  EXPECT_TRUE(b != nullptr);
  EXPECT_TRUE((b->IsType<Tensor<dtype, DeviceContext> >()));
  EXPECT_EQ(size, (b->Get<Tensor<dtype, DeviceContext> >().size()));
  const dtype* b_data = b->Get<Tensor<dtype, DeviceContext> >().data();
  for (int i = 0; i < size; ++i) {
    EXPECT_NEAR(a_data[i], b_data[i], relative_error);
  }
}

TEST(ToyRegressionTest, TestRunPlan) {
  PlanDef plan_def;
  CHECK(ReadProtoFromFile(
      FLAGS_caffe_test_root + kToyRegressionTestPlanPath, &plan_def));
  Workspace workspace;
  workspace.RunPlan(plan_def);
  ExpectTensorEquivalence<float, CPUContext>(workspace, "W", "W_gt", 0.005);
}

TEST(MNISTLinearClassificationTest, TestRunPlan) {
  PlanDef plan_def;
  CHECK(ReadProtoFromFile(
      FLAGS_caffe_test_root + kMNISTLinearClassificationPath, &plan_def));
  Workspace workspace;
  workspace.RunPlan(plan_def);
  const Blob* accuracy = workspace.GetBlob("accuracy");
  EXPECT_TRUE(accuracy != nullptr);
  EXPECT_TRUE((accuracy->IsType<Tensor<float, CPUContext> >()));
  auto& accuracy_tensor = accuracy->Get<Tensor<float, CPUContext> >();
  EXPECT_EQ(accuracy_tensor.size(), 1);
  // Accuracy should be above 85%.
  EXPECT_GT(accuracy_tensor.data()[0], 0.85);
}

TEST(MNISTTwoLayerReluClassificationTest, TestRunPlan) {
  PlanDef plan_def;
  CHECK(ReadProtoFromFile(
      FLAGS_caffe_test_root + kMNISTTwoLayerReluClassificationPath, &plan_def));
  Workspace workspace;
  workspace.RunPlan(plan_def);
  const Blob* accuracy = workspace.GetBlob("accuracy");
  EXPECT_TRUE(accuracy != nullptr);
  EXPECT_TRUE((accuracy->IsType<Tensor<float, CPUContext> >()));
  auto& accuracy_tensor = accuracy->Get<Tensor<float, CPUContext> >();
  EXPECT_EQ(accuracy_tensor.size(), 1);
  // Accuracy should be above 90%.
  EXPECT_GT(accuracy_tensor.data()[0], 0.90);
}

TEST(MNISTLeNetClassificationTest, LARGE_TestRunPlan) {
  PlanDef plan_def;
  CHECK(ReadProtoFromFile(
      FLAGS_caffe_test_root + kMNISTLeNetClassificationPath, &plan_def));
  Workspace workspace;
  workspace.RunPlan(plan_def);
  const Blob* accuracy = workspace.GetBlob("accuracy");
  EXPECT_TRUE(accuracy != nullptr);
  EXPECT_TRUE((accuracy->IsType<Tensor<float, CPUContext> >()));
  auto& accuracy_tensor = accuracy->Get<Tensor<float, CPUContext> >();
  EXPECT_EQ(accuracy_tensor.size(), 1);
  // Accuracy should be above 90%.
  EXPECT_GT(accuracy_tensor.data()[0], 0.90);
}

TEST(MNISTLeNetClassificationTestGPU, LARGE_TestRunPlan) {
  PlanDef plan_def;
  CHECK(ReadProtoFromFile(
      FLAGS_caffe_test_root + kMNISTLeNetClassificationGPUPath, &plan_def));
  Workspace workspace;
  workspace.RunPlan(plan_def);
  const Blob* accuracy = workspace.GetBlob("accuracy");
  EXPECT_TRUE(accuracy != nullptr);
  EXPECT_TRUE((accuracy->IsType<Tensor<float, CUDAContext> >()));
  CPUContext context;
  Tensor<float, CPUContext> accuracy_tensor(
      accuracy->Get<Tensor<float, CUDAContext> >(), &context);
  EXPECT_EQ(accuracy_tensor.size(), 1);
  // Accuracy should be above 90%.
  EXPECT_GT(accuracy_tensor.data()[0], 0.90);
}


TEST(MNISTLeNetNHWCClassificationTest, LARGE_TestRunPlan) {
  PlanDef plan_def;
  CHECK(ReadProtoFromFile(
      FLAGS_caffe_test_root + kMNISTLeNetNHWCClassificationPath, &plan_def));
  Workspace workspace;
  workspace.RunPlan(plan_def);
  const Blob* accuracy = workspace.GetBlob("accuracy");
  EXPECT_TRUE(accuracy != nullptr);
  EXPECT_TRUE((accuracy->IsType<Tensor<float, CPUContext> >()));
  auto& accuracy_tensor = accuracy->Get<Tensor<float, CPUContext> >();
  EXPECT_EQ(accuracy_tensor.size(), 1);
  // Accuracy should be above 90%.
  EXPECT_GT(accuracy_tensor.data()[0], 0.90);
}

TEST(MNISTLeNetNHWCClassificationGPUTest, LARGE_TestRunPlan) {
  PlanDef plan_def;
  CHECK(ReadProtoFromFile(
      FLAGS_caffe_test_root + kMNISTLeNetNHWCClassificationGPUPath, &plan_def));
  Workspace workspace;
  workspace.RunPlan(plan_def);
  const Blob* accuracy = workspace.GetBlob("accuracy");
  EXPECT_TRUE(accuracy != nullptr);
  EXPECT_TRUE((accuracy->IsType<Tensor<float, CUDAContext> >()));
  CPUContext context;
  Tensor<float, CPUContext> accuracy_tensor(
      accuracy->Get<Tensor<float, CUDAContext> >(), &context);
  EXPECT_EQ(accuracy_tensor.size(), 1);
  // Accuracy should be above 90%.
  EXPECT_GT(accuracy_tensor.data()[0], 0.90);
}



TEST(MNISTLeNetGroupConvolutionClassificationTest, LARGE_TestRunPlan) {
  PlanDef plan_def;
  CHECK(ReadProtoFromFile(
      FLAGS_caffe_test_root + kMNISTLeNetGroupConvClassificationPath,
      &plan_def));
  Workspace workspace;
  workspace.RunPlan(plan_def);
  const Blob* accuracy = workspace.GetBlob("accuracy");
  EXPECT_TRUE(accuracy != nullptr);
  EXPECT_TRUE((accuracy->IsType<Tensor<float, CPUContext> >()));
  auto& accuracy_tensor = accuracy->Get<Tensor<float, CPUContext> >();
  EXPECT_EQ(accuracy_tensor.size(), 1);
  // Accuracy should be above 90%.
  EXPECT_GT(accuracy_tensor.data()[0], 0.90);
}

TEST(MNISTLeNetGroupConvolutionNHWCClassificationTest, LARGE_TestRunPlan) {
  PlanDef plan_def;
  CHECK(ReadProtoFromFile(
      FLAGS_caffe_test_root + kMNISTLeNetGroupConvNHWCClassificationPath,
      &plan_def));
  Workspace workspace;
  workspace.RunPlan(plan_def);
  const Blob* accuracy = workspace.GetBlob("accuracy");
  EXPECT_TRUE(accuracy != nullptr);
  EXPECT_TRUE((accuracy->IsType<Tensor<float, CPUContext> >()));
  auto& accuracy_tensor = accuracy->Get<Tensor<float, CPUContext> >();
  EXPECT_EQ(accuracy_tensor.size(), 1);
  // Accuracy should be above 90%.
  EXPECT_GT(accuracy_tensor.data()[0], 0.90);
}

}  // namespace caffe2
