#include <iostream>

#include "caffe2/core/context.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/core/flags.h"
#include "caffe2/core/logging.h"
#include "gtest/gtest.h"

CAFFE2_DECLARE_string(caffe_test_root);

namespace caffe2 {

const char kToyRegressionTestPlanPath[] = "/data/toy/toy_regression.pbtxt";
const char kMNISTLinearClassificationPath[] =
    "/data/mnist/linear_classifier_plan.pbtxt";
const char kMNISTTwoLayerReluClassificationPath[] =
    "/data/mnist/mnist_relu_network.pbtxt";

const vector<string> kMNISTLeNetVariants {
  "/data/mnist/mnist_lenet_nchw.pbtxt",
  "/data/mnist/mnist_lenet_nhwc.pbtxt",
  "/data/mnist/mnist_lenet_group_convolution_nchw.pbtxt",
  "/data/mnist/mnist_lenet_group_convolution_nhwc.pbtxt",
};

const vector<string> kMNISTLeNetVariantsGPU {
  "/data/mnist/mnist_lenet_nchw_gpu.pbtxt",
  "/data/mnist/mnist_lenet_nhwc_gpu.pbtxt",
  "/data/mnist/mnist_lenet_nchw_cudnn.pbtxt",
};

template <typename T, class Context>
void ExpectTensorEquivalence(const Workspace& ws, const string& name_a,
                             const string& name_b,
                             const float relative_error) {
  const Blob* a = ws.GetBlob(name_a);
  EXPECT_TRUE(a != nullptr);
  EXPECT_TRUE((a->IsType<Tensor<Context> >()));
  int size = a->Get<Tensor<Context> >().size();
  const T* a_data = a->Get<Tensor<Context> >().template data<T>();
  const Blob* b = ws.GetBlob(name_b);
  EXPECT_TRUE(b != nullptr);
  EXPECT_TRUE((b->IsType<Tensor<Context> >()));
  EXPECT_EQ(size, (b->Get<Tensor<Context> >().size()));
  const T* b_data = b->Get<Tensor<Context> >().template data<T>();
  for (int i = 0; i < size; ++i) {
    EXPECT_NEAR(a_data[i], b_data[i], relative_error);
  }
}

TEST(ToyRegressionTest, TestRunPlan) {
  PlanDef plan_def;
  CAFFE_CHECK(ReadProtoFromFile(
      FLAGS_caffe_test_root + kToyRegressionTestPlanPath, &plan_def));
  Workspace workspace;
  workspace.RunPlan(plan_def);
  ExpectTensorEquivalence<float, CPUContext>(workspace, "W", "W_gt", 0.01);
}

TEST(MNISTLinearClassificationTest, TestRunPlan) {
  PlanDef plan_def;
  CAFFE_CHECK(ReadProtoFromFile(
      FLAGS_caffe_test_root + kMNISTLinearClassificationPath, &plan_def));
  Workspace workspace;
  workspace.RunPlan(plan_def);
  const Blob* accuracy = workspace.GetBlob("accuracy");
  EXPECT_TRUE(accuracy != nullptr);
  EXPECT_TRUE((accuracy->IsType<TensorCPU>()));
  auto& accuracy_tensor = accuracy->Get<TensorCPU>();
  EXPECT_EQ(accuracy_tensor.size(), 1);
  // Accuracy should be above 85%.
  EXPECT_GT(accuracy_tensor.data<float>()[0], 0.85);
}

TEST(MNISTTwoLayerReluClassificationTest, TestRunPlan) {
  PlanDef plan_def;
  CAFFE_CHECK(ReadProtoFromFile(
      FLAGS_caffe_test_root + kMNISTTwoLayerReluClassificationPath, &plan_def));
  Workspace workspace;
  workspace.RunPlan(plan_def);
  const Blob* accuracy = workspace.GetBlob("accuracy");
  EXPECT_TRUE(accuracy != nullptr);
  EXPECT_TRUE((accuracy->IsType<TensorCPU>()));
  auto& accuracy_tensor = accuracy->Get<TensorCPU>();
  EXPECT_EQ(accuracy_tensor.size(), 1);
  // Accuracy should be above 90%.
  EXPECT_GT(accuracy_tensor.data<float>()[0], 0.90);
}

TEST(MNISTLeNetClassificationTest, LARGE_TestRunPlan) {
  for (const string& path : kMNISTLeNetVariants) {
    CAFFE_LOG_ERROR << "Testing " << path;
    PlanDef plan_def;
    CAFFE_CHECK(ReadProtoFromFile(
        FLAGS_caffe_test_root + path, &plan_def));
    Workspace workspace;
    workspace.RunPlan(plan_def);
    const Blob* accuracy = workspace.GetBlob("accuracy");
    EXPECT_TRUE(accuracy != nullptr);
    EXPECT_TRUE((accuracy->IsType<TensorCPU>()));
    auto& accuracy_tensor = accuracy->Get<TensorCPU>();
    EXPECT_EQ(accuracy_tensor.size(), 1);
    // Accuracy should be above 90%.
    EXPECT_GT(accuracy_tensor.data<float>()[0], 0.90);
  }
}

TEST(MNISTLeNetClassificationTestGPU, LARGE_TestRunPlan) {
  for (const string& path : kMNISTLeNetVariantsGPU) {
    CAFFE_LOG_ERROR << "Testing " << path;
    PlanDef plan_def;
    CAFFE_CHECK(ReadProtoFromFile(
        FLAGS_caffe_test_root + path, &plan_def));
    Workspace workspace;
    workspace.RunPlan(plan_def);
    const Blob* accuracy = workspace.GetBlob("accuracy");
    EXPECT_TRUE(accuracy != nullptr);
    EXPECT_TRUE((accuracy->IsType<TensorCUDA>()));
    CPUContext context;
    TensorCPU accuracy_tensor(
        accuracy->Get<TensorCUDA>(), &context);
    EXPECT_EQ(accuracy_tensor.size(), 1);
    // Accuracy should be above 90%.
    EXPECT_GT(accuracy_tensor.data<float>()[0], 0.90);
  }
}

}  // namespace caffe2
