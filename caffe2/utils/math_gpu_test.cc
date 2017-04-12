#include <iostream>

#include <gtest/gtest.h>
#include "caffe2/core/context.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/flags.h"
#include "caffe2/operators/utility_ops.h"
#include "caffe2/utils/math.h"

CAFFE2_DECLARE_string(caffe_test_root);

namespace caffe2 {

TEST(MathUtilGPUTest, testAddStripedBatch) {
  if (!HasCudaGPU())
    return;
  Workspace ws;
  DeviceOption option;
  option.set_device_type(CUDA);
  CUDAContext context(option);
  Blob* blobx = ws.CreateBlob("X");
  Blob* bloby = ws.CreateBlob("Y");
  Blob* bloby_host = ws.CreateBlob("Y_host");

  vector<int> shapex{33 * 9, 25};
  vector<int> shapey{33, 25};

  auto* tensorx = blobx->GetMutable<Tensor<CUDAContext>>();
  tensorx->Resize(shapex);
  int stripe = 33 * 25;
  vector<float> tot(33, 0.0);
  for (int j = 0; j < 9; j++) {
    // Have different values for each line
    for (int k = 0; k < 33; k++) {
      math::Set<float, CUDAContext>(
          33,
          1.0 + j + k,
          tensorx->mutable_data<float>() + j * stripe + k * 25,
          &context);
      tot[k] += 1.0 + j + k;
    }
  }

  auto* tensory = bloby->GetMutable<Tensor<CUDAContext>>();
  tensory->Resize(shapey);
  math::Set<float, CUDAContext>(
      stripe, 0.0, tensory->mutable_data<float>(), &context);

  math::AddStripedBatch<float, CUDAContext>(
      stripe,
      tensorx->template data<float>(),
      tensory->mutable_data<float>(),
      stripe,
      9,
      &context);
  context.FinishDeviceComputation();

  // Copy result to CPU so we can inspect it
  auto* tensory_host = bloby_host->GetMutable<Tensor<CPUContext>>();
  tensory_host->CopyFrom<CUDAContext, CUDAContext>(*tensory, &context);
  context.FinishDeviceComputation();

  for (int k = 0; k < 33; k++) {
    for (int i = 0; i < 25; i++) {
      EXPECT_EQ(tensory_host->data<float>()[k * 25 + i], tot[k]);
    }
  }
}

#define TEST_GEMV_WITH_TYPE(field_name)                                      \
  TEST(MathUtilGPUTest, testGemv_##field_name) {                             \
    if (!HasCudaGPU())                                                       \
      return;                                                                \
    Workspace ws;                                                            \
    DeviceOption option;                                                     \
    option.set_device_type(CUDA);                                            \
    CUDAContext context(option);                                             \
    Blob* blobx = ws.CreateBlob("X");                                        \
    Blob* bloby = ws.CreateBlob("Y");                                        \
    Blob* blobz = ws.CreateBlob("Z");                                        \
    Blob* bloby_host = ws.CreateBlob("Y_host");                              \
                                                                             \
    vector<int> shapex{64, 128};                                             \
    vector<int> shapey{64};                                                  \
    vector<int> shapez{128};                                                 \
                                                                             \
    auto* tensorx = blobx->GetMutable<Tensor<CUDAContext>>();                \
    tensorx->Resize(shapex);                                                 \
    math::Set<field_name, CUDAContext>(                                      \
        64 * 128,                                                            \
        (field_name)1.0,                                                     \
        tensorx->mutable_data<field_name>(),                                 \
        &context);                                                           \
                                                                             \
    auto* tensory = bloby->GetMutable<Tensor<CUDAContext>>();                \
    tensory->Resize(shapey);                                                 \
    math::Set<field_name, CUDAContext>(                                      \
        64, (field_name)1.0, tensory->mutable_data<field_name>(), &context); \
                                                                             \
    auto* tensorz = blobz->GetMutable<Tensor<CUDAContext>>();                \
    tensorz->Resize(shapez);                                                 \
                                                                             \
    math::Gemv<field_name, CUDAContext>(                                     \
        CblasTrans,                                                          \
        64,                                                                  \
        128,                                                                 \
        1.0,                                                                 \
        tensorx->template data<field_name>(),                                \
        tensory->mutable_data<field_name>(),                                 \
        0.0,                                                                 \
        tensorz->template mutable_data<field_name>(),                        \
        &context);                                                           \
    context.FinishDeviceComputation();                                       \
                                                                             \
    auto* tensory_host = bloby_host->GetMutable<Tensor<CPUContext>>();       \
    tensory_host->CopyFrom<CUDAContext, CUDAContext>(*tensorz, &context);    \
    context.FinishDeviceComputation();                                       \
                                                                             \
    for (int i = 0; i < 128; i++) {                                          \
      EXPECT_EQ(tensory_host->data<field_name>()[i], 64.0);                  \
    }                                                                        \
  }

TEST_GEMV_WITH_TYPE(float);
TEST_GEMV_WITH_TYPE(double);

} // namespace caffe2
