#include <array>
#include <iostream>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "caffe2/core/context.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/flags.h"
#include "caffe2/operators/utility_ops.h"
#include "caffe2/utils/math.h"

C10_DECLARE_string(caffe_test_root);

namespace caffe2 {

void executeGpuBinaryOpTest(
    int shapex0,
    int shapex1,
    int shapey,
    std::function<float(int)> input0,
    std::function<float(int)> input1,
    std::function<void(
        int N0,
        int N1,
        const float* src0,
        const float* src1,
        float* dst,
        CUDAContext* context)> operation,
    std::function<float(int)> correct_output) {
  if (!HasCudaGPU())
    return;
  Workspace ws;
  DeviceOption option;
  option.set_device_type(PROTO_CUDA);
  CUDAContext context(option);

  Blob* blobx0 = ws.CreateBlob("X0");
  Blob* blobx1 = ws.CreateBlob("X1");
  Blob* bloby = ws.CreateBlob("Y");
  Blob* bloby_host = ws.CreateBlob("Y_host");

  auto* tensorx0 = BlobGetMutableTensor(blobx0, CUDA);
  auto* tensorx1 = BlobGetMutableTensor(blobx1, CUDA);
  auto* tensory = BlobGetMutableTensor(bloby, CUDA);

  vector<int> shapex0_vector{shapex0};
  vector<int> shapex1_vector{shapex1};
  vector<int> shapey_vector{shapey};

  tensorx0->Resize(shapex0_vector);
  tensorx1->Resize(shapex1_vector);
  tensory->Resize(shapey_vector);

  for (int i = 0; i < shapex0; i++) {
    math::Set<float, CUDAContext>(
        1, input0(i), tensorx0->mutable_data<float>() + i, &context);
  }
  for (int i = 0; i < shapex1; i++) {
    math::Set<float, CUDAContext>(
        1, input1(i), tensorx1->mutable_data<float>() + i, &context);
  }
  operation(
      shapex0,
      shapex1,
      tensorx0->template data<float>(),
      tensorx1->template data<float>(),
      tensory->mutable_data<float>(),
      &context);
  context.FinishDeviceComputation();

  // Copy result to CPU so we can inspect it
  auto* tensory_host = BlobGetMutableTensor(bloby_host, CPU);
  tensory_host->CopyFrom(*tensory);

  for (int i = 0; i < shapey; ++i) {
    EXPECT_EQ(tensory_host->data<float>()[i], correct_output(i));
  }
}

TEST(MathUtilGPUTest, testAddStripedBatch) {
  if (!HasCudaGPU())
    return;
  Workspace ws;
  DeviceOption option;
  option.set_device_type(PROTO_CUDA);
  CUDAContext context(option);
  Blob* blobx = ws.CreateBlob("X");
  Blob* bloby = ws.CreateBlob("Y");
  Blob* bloby_host = ws.CreateBlob("Y_host");

  vector<int> shapex{33 * 9, 25};
  vector<int> shapey{33, 25};

  auto* tensorx = BlobGetMutableTensor(blobx, CUDA);
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

  auto* tensory = BlobGetMutableTensor(bloby, CUDA);
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
  auto* tensory_host = BlobGetMutableTensor(bloby_host, CPU);
  tensory_host->CopyFrom(*tensory);

  for (int k = 0; k < 33; k++) {
    for (int i = 0; i < 25; i++) {
      EXPECT_EQ(tensory_host->data<float>()[k * 25 + i], tot[k]);
    }
  }
}

TEST(MathUtilGPUTest, testReduceMin) {
  executeGpuBinaryOpTest(
      6,
      1,
      1,
      [](int /*i*/) { return 11.0f; },
      [](int /*i*/) { return 0.0f; },
      [](int N0,
         int /*N1*/,
         const float* src0,
         const float* /*src1*/,
         float* dst,
         CUDAContext* context) {
        Tensor aux(CUDA);
        math::ReduceMin<float, CUDAContext>(N0, src0, dst, &aux, context);
      },
      [](int /*i*/) { return 11.0f; });
  executeGpuBinaryOpTest(
      6,
      1,
      1,
      [](int i) { return i == 3 ? 11.0f : 17.0f; },
      [](int /*i*/) { return 0.0f; },
      [](int N0,
         int /*N1*/,
         const float* src0,
         const float* /*src1*/,
         float* dst,
         CUDAContext* context) {
        Tensor aux(CUDA);
        math::ReduceMin<float, CUDAContext>(N0, src0, dst, &aux, context);
      },
      [](int /*i*/) { return 11.0f; });
}

TEST(MathUtilGPUTest, testReduceMax) {
  executeGpuBinaryOpTest(
      6,
      1,
      1,
      [](int /*i*/) { return 11.0f; },
      [](int /*i*/) { return 0.0f; },
      [](int N0,
         int /*N1*/,
         const float* src0,
         const float* /*src1*/,
         float* dst,
         CUDAContext* context) {
        Tensor aux(CUDA);
        math::ReduceMax<float, CUDAContext>(N0, src0, dst, &aux, context);
      },
      [](int /*i*/) { return 11.0f; });
  executeGpuBinaryOpTest(
      6,
      1,
      1,
      [](int i) { return i == 3 ? 17.0f : 11.0f; },
      [](int /*i*/) { return 0.0f; },
      [](int N0,
         int /*N1*/,
         const float* src0,
         const float* /*src1*/,
         float* dst,
         CUDAContext* context) {
        Tensor aux(CUDA);
        math::ReduceMax<float, CUDAContext>(N0, src0, dst, &aux, context);
      },
      [](int /*i*/) { return 17.0f; });
}

TEST(MathUtilGPUTest, testCopyVector) {
  executeGpuBinaryOpTest(
      6,
      1,
      6,
      [](int i) { return 5.0f - i; },
      [](int /*i*/) { return 0.0f; },
      [](int N0,
         int /*N1*/,
         const float* src0,
         const float* /*src1*/,
         float* dst,
         CUDAContext* context) {
        math::CopyVector<float, CUDAContext>(N0, src0, dst, context);
      },
      [](int i) { return 5.0f - i; });
}

namespace {

constexpr float kEps = 1e-5;

class GemmBatchedGPUTest
    : public testing::TestWithParam<testing::tuple<bool, bool>> {
 protected:
  void SetUp() override {
    if (!HasCudaGPU()) {
      return;
    }
    option_.set_device_type(PROTO_CUDA);
    cuda_context_ = make_unique<CUDAContext>(option_);
    Blob* X_blob = ws_.CreateBlob("X");
    Blob* W_blob = ws_.CreateBlob("W");
    Blob* Y_blob = ws_.CreateBlob("Y");
    X_ = BlobGetMutableTensor(X_blob, CUDA);
    W_ = BlobGetMutableTensor(W_blob, CUDA);
    Y_ = BlobGetMutableTensor(Y_blob, CUDA);
    X_->Resize(std::vector<int64_t>{3, 5, 10});
    W_->Resize(std::vector<int64_t>{3, 6, 10});
    Y_->Resize(std::vector<int64_t>{3, 5, 6});
    math::Set<float, CUDAContext>(
        X_->numel(), 1.0f, X_->mutable_data<float>(), cuda_context_.get());
    math::Set<float, CUDAContext>(
        W_->numel(), 1.0f, W_->mutable_data<float>(), cuda_context_.get());
    trans_X_ = std::get<0>(GetParam());
    trans_W_ = std::get<1>(GetParam());
  }

  void RunGemmBatched(const float alpha, const float beta) {
    const float* X_data = X_->template data<float>();
    const float* W_data = W_->template data<float>();
    float* Y_data = Y_->template mutable_data<float>();
    const int X_stride = 5 * 10;
    const int W_stride = 6 * 10;
    const int Y_stride = 5 * 6;
    std::array<const float*, 3> X_array = {
        X_data, X_data + X_stride, X_data + 2 * X_stride};
    std::array<const float*, 3> W_array = {
        W_data, W_data + W_stride, W_data + 2 * W_stride};
    std::array<float*, 3> Y_array = {
        Y_data, Y_data + Y_stride, Y_data + 2 * Y_stride};
    math::GemmBatched<float, CUDAContext>(
        trans_X_ ? CblasTrans : CblasNoTrans,
        trans_W_ ? CblasTrans : CblasNoTrans,
        3,
        5,
        6,
        10,
        alpha,
        X_array.data(),
        W_array.data(),
        beta,
        Y_array.data(),
        cuda_context_.get());
  }

  void RunGemmStridedBatched(const float alpha, const float beta) {
    const float* X_data = X_->template data<float>();
    const float* W_data = W_->template data<float>();
    float* Y_data = Y_->template mutable_data<float>();
    const int X_stride = 5 * 10;
    const int W_stride = 6 * 10;
    const int Y_stride = 5 * 6;
    math::GemmStridedBatched<float, CUDAContext>(
        trans_X_ ? CblasTrans : CblasNoTrans,
        trans_W_ ? CblasTrans : CblasNoTrans,
        3,
        5,
        6,
        10,
        alpha,
        X_data,
        X_stride,
        W_data,
        W_stride,
        beta,
        Y_data,
        Y_stride,
        cuda_context_.get());
  }

  void VerifyOutput(const float value) const {
    Tensor Y_cpu(*Y_, CPU);
    for (int i = 0; i < Y_cpu.numel(); ++i) {
      EXPECT_FLOAT_EQ(value, Y_cpu.template data<float>()[i]);
    }
  }

  Workspace ws_;
  DeviceOption option_;
  std::unique_ptr<CUDAContext> cuda_context_;
  Tensor* X_ = nullptr;
  Tensor* W_ = nullptr;
  Tensor* Y_ = nullptr;
  bool trans_X_;
  bool trans_W_;
};

TEST_P(GemmBatchedGPUTest, GemmBatchedGPUFloatTest) {
  if (!HasCudaGPU()) {
    return;
  }
  RunGemmBatched(1.0f, 0.0f);
  VerifyOutput(10.0f);
  RunGemmBatched(1.0f, 0.5f);
  VerifyOutput(15.0f);
  RunGemmBatched(0.5f, 1.0f);
  VerifyOutput(20.0f);
}

TEST_P(GemmBatchedGPUTest, GemmStridedBatchedGPUFloatTest) {
  if (!HasCudaGPU()) {
    return;
  }
  RunGemmStridedBatched(1.0f, 0.0f);
  VerifyOutput(10.0f);
  RunGemmStridedBatched(1.0f, 0.5f);
  VerifyOutput(15.0f);
  RunGemmStridedBatched(0.5f, 1.0f);
  VerifyOutput(20.0f);
}

INSTANTIATE_TEST_CASE_P(
    GemmBatchedGPUTrans,
    GemmBatchedGPUTest,
    testing::Combine(testing::Bool(), testing::Bool()));

class ReduceTensorGPUTest : public testing::Test {
 protected:
  void SetUp() override {
    if (!HasCudaGPU()) {
      return;
    }
    option_.set_device_type(PROTO_CUDA);
    cuda_context_ = make_unique<CUDAContext>(option_);
    Blob* blob_x = ws_.CreateBlob("X");
    Blob* blob_y = ws_.CreateBlob("Y");
    X_ = BlobGetMutableTensor(blob_x, CUDA);
    Y_ = BlobGetMutableTensor(blob_y, CUDA);
  }

  void SetUpData(
      const std::vector<int>& X_dims,
      const std::vector<int>& axes,
      const std::vector<float>& X_data) {
    std::vector<int> Y_dims = X_dims;
    for (const int axis : axes) {
      Y_dims[axis] = 1;
    }
    X_->Resize(X_dims);
    Y_->Resize(Y_dims);
    ASSERT_EQ(X_data.size(), X_->numel());
    cuda_context_->CopyFromCPU<float>(
        X_data.size(), X_data.data(), X_->mutable_data<float>());
  }

  void VerifyResult(const std::vector<float>& expected_output) {
    Blob* blob_y_host = ws_.CreateBlob("Y_host");
    auto* Y_host = BlobGetMutableTensor(blob_y_host, CPU);
    Y_host->CopyFrom(*Y_);
    ASSERT_EQ(expected_output.size(), Y_host->numel());
    for (std::size_t i = 0; i < expected_output.size(); ++i) {
      EXPECT_FLOAT_EQ(expected_output[i], Y_host->data<float>()[i]);
    }
  }

  template <class ReduceFunc>
  void RunRedcueTensorTest(
      const ReduceFunc& reduce_func,
      const std::vector<int>& X_dims,
      const std::vector<int>& axes,
      const std::vector<float>& X_data,
      const std::vector<float>& Y_data) {
    SetUpData(X_dims, axes, X_data);
    reduce_func(
        X_dims.size(),
        X_dims.data(),
        axes.size(),
        axes.data(),
        1.0f,
        X_->data<float>(),
        Y_->mutable_data<float>(),
        cuda_context_.get());
    VerifyResult(Y_data);
  }

  Workspace ws_;
  DeviceOption option_;
  std::unique_ptr<CUDAContext> cuda_context_;
  Tensor* X_ = nullptr;
  Tensor* Y_ = nullptr;
};

TEST_F(ReduceTensorGPUTest, ReduceMinGPUTest) {
  if (!HasCudaGPU()) {
    return;
  }
  const auto& reduce_min = [](const int num_dims,
                              const int* dims,
                              const int num_axes,
                              const int* axes,
                              const float alpha,
                              const float* X,
                              float* Y,
                              CUDAContext* context) {
    return math::ReduceMin<float, CUDAContext>(
        num_dims, dims, num_axes, axes, alpha, X, Y, context);
  };
  // Test for 1D tensor.
  RunRedcueTensorTest(reduce_min, {3}, {0}, {1.0f, 2.0f, 3.0f}, {1.0f});

  // Test for 2D Tensor.
  RunRedcueTensorTest(
      reduce_min,
      {2, 3},
      {1},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      {1.0f, 4.0f});
  RunRedcueTensorTest(
      reduce_min,
      {2, 3},
      {0},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      {1.0f, 2.0f, 3.0f});
  RunRedcueTensorTest(
      reduce_min, {2, 3}, {0, 1}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {1.0f});

  // Test for 3D tensor.
  RunRedcueTensorTest(
      reduce_min,
      {2, 2, 2},
      {1, 2},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {1.0f, 5.0f});
  RunRedcueTensorTest(
      reduce_min,
      {2, 2, 2},
      {0, 1},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {1.0f, 2.0f});
  RunRedcueTensorTest(
      reduce_min,
      {2, 2, 2},
      {0, 2},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {1.0f, 3.0f});
}

TEST_F(ReduceTensorGPUTest, ReduceMaxGPUTest) {
  if (!HasCudaGPU()) {
    return;
  }
  const auto& reduce_max = [](const int num_dims,
                              const int* dims,
                              const int num_axes,
                              const int* axes,
                              const float alpha,
                              const float* X,
                              float* Y,
                              CUDAContext* context) {
    return math::ReduceMax<float, CUDAContext>(
        num_dims, dims, num_axes, axes, alpha, X, Y, context);
  };
  // Test for 1D tensor.
  RunRedcueTensorTest(reduce_max, {3}, {0}, {1.0f, 2.0f, 3.0f}, {3.0f});

  // Test for 2D Tensor.
  RunRedcueTensorTest(
      reduce_max,
      {2, 3},
      {1},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      {3.0f, 6.0f});
  RunRedcueTensorTest(
      reduce_max,
      {2, 3},
      {0},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      {4.0f, 5.0f, 6.0f});
  RunRedcueTensorTest(
      reduce_max, {2, 3}, {0, 1}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {6.0f});

  // Test for 3D tensor.
  RunRedcueTensorTest(
      reduce_max,
      {2, 2, 2},
      {1, 2},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {4.0f, 8.0f});
  RunRedcueTensorTest(
      reduce_max,
      {2, 2, 2},
      {0, 1},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {7.0f, 8.0f});
  RunRedcueTensorTest(
      reduce_max,
      {2, 2, 2},
      {0, 2},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {6.0f, 8.0f});
}

TEST_F(ReduceTensorGPUTest, ReduceSumGPUTest) {
  if (!HasCudaGPU()) {
    return;
  }
  // Test for 1D tensor.
  RunRedcueTensorTest(
      math::ReduceSum<float, CUDAContext>,
      {3},
      {0},
      {1.0f, 2.0f, 3.0f},
      {6.0f});

  // Test for 2D Tensor.
  RunRedcueTensorTest(
      math::ReduceSum<float, CUDAContext>,
      {2, 3},
      {1},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      {6.0f, 15.0f});
  RunRedcueTensorTest(
      math::ReduceSum<float, CUDAContext>,
      {2, 3},
      {0},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      {5.0f, 7.0f, 9.0f});
  RunRedcueTensorTest(
      math::ReduceSum<float, CUDAContext>,
      {2, 3},
      {0, 1},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      {21.0f});

  // Test for 3D tensor.
  RunRedcueTensorTest(
      math::ReduceSum<float, CUDAContext>,
      {2, 2, 2},
      {1, 2},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {10.0f, 26.0f});
  RunRedcueTensorTest(
      math::ReduceSum<float, CUDAContext>,
      {2, 2, 2},
      {0, 1},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {16.0f, 20.0f});
  RunRedcueTensorTest(
      math::ReduceSum<float, CUDAContext>,
      {2, 2, 2},
      {0, 2},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {14.0f, 22.0f});
}

TEST_F(ReduceTensorGPUTest, ReduceMeanGPUTest) {
  if (!HasCudaGPU()) {
    return;
  }
  // Test for 1D tensor.
  RunRedcueTensorTest(
      math::ReduceMean<float, CUDAContext>,
      {3},
      {0},
      {1.0f, 2.0f, 3.0f},
      {2.0f});

  // Test for 2D Tensor.
  RunRedcueTensorTest(
      math::ReduceMean<float, CUDAContext>,
      {2, 3},
      {1},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      {2.0f, 5.0f});
  RunRedcueTensorTest(
      math::ReduceMean<float, CUDAContext>,
      {2, 3},
      {0},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      {2.5f, 3.5f, 4.5f});
  RunRedcueTensorTest(
      math::ReduceMean<float, CUDAContext>,
      {2, 3},
      {0, 1},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      {3.5f});

  // Test for 3D tensor.
  RunRedcueTensorTest(
      math::ReduceMean<float, CUDAContext>,
      {2, 2, 2},
      {1, 2},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {2.5f, 6.5f});
  RunRedcueTensorTest(
      math::ReduceMean<float, CUDAContext>,
      {2, 2, 2},
      {0, 1},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {4.0f, 5.0f});
  RunRedcueTensorTest(
      math::ReduceMean<float, CUDAContext>,
      {2, 2, 2},
      {0, 2},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {3.5f, 5.5f});
}

class BroadcastGPUTest : public testing::Test {
 protected:
  void SetUp() override {
    if (!HasCudaGPU()) {
      return;
    }
    option_.set_device_type(PROTO_CUDA);
    cuda_context_ = make_unique<CUDAContext>(option_);
    Blob* blob_x = ws_.CreateBlob("X");
    Blob* blob_y = ws_.CreateBlob("Y");
    X_ = BlobGetMutableTensor(blob_x, CUDA);
    Y_ = BlobGetMutableTensor(blob_y, CUDA);
  }

  void SetUpData(
      const std::vector<int>& X_dims,
      const std::vector<int>& Y_dims,
      const std::vector<float>& X_data) {
    X_->Resize(X_dims);
    Y_->Resize(Y_dims);
    ASSERT_EQ(X_data.size(), X_->numel());
    cuda_context_->CopyFromCPU<float>(
        X_data.size(), X_data.data(), X_->mutable_data<float>());
  }

  void VerifyResult(const std::vector<float>& expected_output) {
    Blob* blob_y_host = ws_.CreateBlob("Y_host");
    auto* Y_host = BlobGetMutableTensor(blob_y_host, CPU);
    Y_host->CopyFrom(*Y_);
    ASSERT_EQ(expected_output.size(), Y_host->numel());
    for (std::size_t i = 0; i < expected_output.size(); ++i) {
      EXPECT_FLOAT_EQ(expected_output[i], Y_host->data<float>()[i]);
    }
  }

  void RunBroadcastTest(
      const std::vector<int>& X_dims,
      const std::vector<int>& Y_dims,
      const std::vector<float>& X_data,
      const std::vector<float>& Y_data) {
    SetUpData(X_dims, Y_dims, X_data);
    math::Broadcast<float, CUDAContext>(
        X_dims.size(),
        X_dims.data(),
        Y_dims.size(),
        Y_dims.data(),
        1.0f,
        X_->data<float>(),
        Y_->mutable_data<float>(),
        cuda_context_.get());
    VerifyResult(Y_data);
  }

  Workspace ws_;
  DeviceOption option_;
  std::unique_ptr<CUDAContext> cuda_context_;
  Tensor* X_ = nullptr;
  Tensor* Y_ = nullptr;
};

TEST_F(BroadcastGPUTest, BroadcastGPUFloatTest) {
  if (!HasCudaGPU()) {
    return;
  }
  RunBroadcastTest({2}, {2}, {1.0f, 2.0f}, {1.0f, 2.0f});
  RunBroadcastTest({1}, {2}, {1.0f}, {1.0f, 1.0f});
  RunBroadcastTest({1}, {2, 2}, {1.0f}, {1.0f, 1.0f, 1.0f, 1.0f});
  RunBroadcastTest({2, 1}, {2, 2}, {1.0f, 2.0f}, {1.0f, 1.0f, 2.0f, 2.0f});
  RunBroadcastTest(
      {2, 1},
      {2, 2, 2},
      {1.0f, 2.0f},
      {1.0f, 1.0f, 2.0f, 2.0f, 1.0f, 1.0f, 2.0f, 2.0f});
}

class TransposeGPUTest : public testing::Test {
 protected:
  void SetUp() override {
    if (!HasCudaGPU()) {
      return;
    }
    option_.set_device_type(PROTO_CUDA);
    cuda_context_ = make_unique<CUDAContext>(option_);
    Blob* blob_x = ws_.CreateBlob("X");
    Blob* blob_y = ws_.CreateBlob("Y");
    X_ = BlobGetMutableTensor(blob_x, CUDA);
    Y_ = BlobGetMutableTensor(blob_y, CUDA);
  }

  void SetUpData(
      const std::vector<int>& X_dims,
      const std::vector<int>& axes,
      const std::vector<float>& X_data) {
    const int ndim = X_dims.size();
    std::vector<int> Y_dims(ndim);
    for (int i = 0; i < ndim; ++i) {
      Y_dims[i] = X_dims[axes[i]];
    }
    X_->Resize(X_dims);
    Y_->Resize(Y_dims);
    ASSERT_EQ(X_data.size(), X_->numel());
    cuda_context_->CopyFromCPU<float>(
        X_data.size(), X_data.data(), X_->mutable_data<float>());
  }

  void VerifyResult(const std::vector<float>& expected_output) {
    Blob* blob_y_host = ws_.CreateBlob("Y_host");
    auto* Y_host = BlobGetMutableTensor(blob_y_host, CPU);
    Y_host->CopyFrom(*Y_);
    ASSERT_EQ(expected_output.size(), Y_host->numel());
    for (std::size_t i = 0; i < expected_output.size(); ++i) {
      EXPECT_FLOAT_EQ(expected_output[i], Y_host->data<float>()[i]);
    }
  }

  void RunTransposeTest(
      const std::vector<int>& X_dims,
      const std::vector<int>& axes,
      const std::vector<float>& X_data,
      const std::vector<float>& Y_data) {
    SetUpData(X_dims, axes, X_data);
    math::Transpose<float, CUDAContext>(
        X_dims.size(),
        X_dims.data(),
        axes.data(),
        X_->data<float>(),
        Y_->mutable_data<float>(),
        cuda_context_.get());
    cuda_context_->FinishDeviceComputation();
    VerifyResult(Y_data);
  }

  Workspace ws_;
  DeviceOption option_;
  std::unique_ptr<CUDAContext> cuda_context_;
  Tensor* X_ = nullptr;
  Tensor* Y_ = nullptr;
};

TEST_F(TransposeGPUTest, TransposeGPUFloatTest) {
  if (!HasCudaGPU()) {
    return;
  }
  // Test for 1D transpose.
  RunTransposeTest({3}, {0}, {1.0f, 2.0f, 3.0f}, {1.0f, 2.0f, 3.0f});

  // Test for 2D transpose.
  RunTransposeTest(
      {2, 3},
      {1, 0},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f});

  // Test for 3D transpose.
  RunTransposeTest(
      {2, 2, 2},
      {1, 2, 0},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {1.0f, 5.0f, 2.0f, 6.0f, 3.0f, 7.0f, 4.0f, 8.0f});
  RunTransposeTest(
      {2, 2, 2},
      {1, 0, 2},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {1.0f, 2.0f, 5.0f, 6.0f, 3.0f, 4.0f, 7.0f, 8.0f});
}

} // namespace

} // namespace caffe2
