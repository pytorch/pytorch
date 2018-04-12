#include <iostream>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "caffe2/core/context.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/flags.h"
#include "caffe2/operators/utility_ops.h"
#include "caffe2/utils/math.h"

CAFFE2_DECLARE_string(caffe_test_root);

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
  option.set_device_type(CUDA);
  CUDAContext context(option);

  Blob* blobx0 = ws.CreateBlob("X0");
  Blob* blobx1 = ws.CreateBlob("X1");
  Blob* bloby = ws.CreateBlob("Y");
  Blob* bloby_host = ws.CreateBlob("Y_host");

  auto* tensorx0 = blobx0->GetMutable<Tensor<CUDAContext>>();
  auto* tensorx1 = blobx1->GetMutable<Tensor<CUDAContext>>();
  auto* tensory = bloby->GetMutable<Tensor<CUDAContext>>();

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
  auto* tensory_host = bloby_host->GetMutable<Tensor<CPUContext>>();
  tensory_host->CopyFrom<CUDAContext, CUDAContext>(*tensory, &context);
  context.FinishDeviceComputation();

  for (int i = 0; i < shapey; ++i) {
    EXPECT_EQ(tensory_host->data<float>()[i], correct_output(i));
  }
}

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
        Tensor<CUDAContext> aux;
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
        Tensor<CUDAContext> aux;
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
        Tensor<CUDAContext> aux;
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
        Tensor<CUDAContext> aux;
        math::ReduceMax<float, CUDAContext>(N0, src0, dst, &aux, context);
      },
      [](int /*i*/) { return 17.0f; });
}

TEST(MathUtilGPUTest, testElemwiseMax) {
  executeGpuBinaryOpTest(
      13,
      13,
      13,
      [](int i) { return 2.0f - i; },
      [](int i) { return i - 6.0f; },
      [](int N0,
         int /*N1*/,
         const float* src0,
         const float* src1,
         float* dst,
         CUDAContext* context) {
        math::ElemwiseMax<float, CUDAContext>(N0, src0, src1, dst, context);
      },
      [](int i) { return std::max(2.0f - i, i - 6.0f); });
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

class GemmBatchedGPUTest
    : public testing::TestWithParam<testing::tuple<bool, bool>> {
 protected:
  void SetUp() override {
    if (!HasCudaGPU()) {
      return;
    }
    option_.set_device_type(CUDA);
    cuda_context_ = make_unique<CUDAContext>(option_);
    Blob* X_blob = ws_.CreateBlob("X");
    Blob* W_blob = ws_.CreateBlob("W");
    Blob* Y_blob = ws_.CreateBlob("Y");
    X_ = X_blob->GetMutable<Tensor<CUDAContext>>();
    W_ = W_blob->GetMutable<Tensor<CUDAContext>>();
    Y_ = Y_blob->GetMutable<Tensor<CUDAContext>>();
    X_->Resize(std::vector<TIndex>{3, 5, 10});
    W_->Resize(std::vector<TIndex>{3, 6, 10});
    Y_->Resize(std::vector<TIndex>{3, 5, 6});
    math::Set<float, CUDAContext>(
        X_->size(), 1.0f, X_->mutable_data<float>(), cuda_context_.get());
    math::Set<float, CUDAContext>(
        W_->size(), 1.0f, W_->mutable_data<float>(), cuda_context_.get());
    trans_X_ = std::get<0>(GetParam());
    trans_W_ = std::get<1>(GetParam());
  }

  void RunGemmBatched(const float alpha, const float beta) {
    math::GemmBatched(
        trans_X_ ? CblasTrans : CblasNoTrans,
        trans_W_ ? CblasTrans : CblasNoTrans,
        3,
        5,
        6,
        10,
        alpha,
        X_->template data<float>(),
        W_->template data<float>(),
        beta,
        Y_->template mutable_data<float>(),
        cuda_context_.get());
  }

  void VerifyOutput(const float value) const {
    TensorCPU Y_cpu(*Y_);
    for (int i = 0; i < Y_cpu.size(); ++i) {
      EXPECT_FLOAT_EQ(value, Y_cpu.template data<float>()[i]);
    }
  }

  Workspace ws_;
  DeviceOption option_;
  std::unique_ptr<CUDAContext> cuda_context_;
  Tensor<CUDAContext>* X_ = nullptr;
  Tensor<CUDAContext>* W_ = nullptr;
  Tensor<CUDAContext>* Y_ = nullptr;
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
    option_.set_device_type(CUDA);
    cuda_context_ = make_unique<CUDAContext>(option_);
    Blob* blob_x = ws_.CreateBlob("X");
    Blob* blob_y = ws_.CreateBlob("Y");
    Blob* blob_dims = ws_.CreateBlob("dims");
    Blob* blob_axes = ws_.CreateBlob("axes");
    Blob* blob_scratch = ws_.CreateBlob("scratch");
    X_ = blob_x->GetMutable<Tensor<CUDAContext>>();
    Y_ = blob_y->GetMutable<Tensor<CUDAContext>>();
    dims_device_ = blob_dims->GetMutable<Tensor<CUDAContext>>();
    axes_device_ = blob_axes->GetMutable<Tensor<CUDAContext>>();
    scratch_ptr_ = blob_scratch->GetMutable<Tensor<CUDAContext>>();
  }

  void SetUpData(
      const std::vector<int>& x_dims,
      const std::vector<int>& y_dims,
      const std::vector<int>& axes,
      const std::vector<float>& x_data) {
    dims_device_->Resize(x_dims.size());
    cuda_context_->Copy<int, CPUContext, CUDAContext>(
        x_dims.size(), x_dims.data(), dims_device_->mutable_data<int>());
    axes_device_->Resize(axes.size());
    cuda_context_->Copy<int, CPUContext, CUDAContext>(
        axes.size(), axes.data(), axes_device_->mutable_data<int>());
    X_->Resize(x_dims);
    Y_->Resize(y_dims);
    cuda_context_->Copy<float, CPUContext, CUDAContext>(
        x_data.size(), x_data.data(), X_->mutable_data<float>());
  }

  void VerifyResult(const std::vector<float>& expected_output) {
    Blob* blob_y_host = ws_.CreateBlob("Y_host");
    auto* Y_host = blob_y_host->GetMutable<TensorCPU>();
    Y_host->CopyFrom<CUDAContext, CUDAContext>(*Y_, cuda_context_.get());
    cuda_context_->FinishDeviceComputation();
    ASSERT_EQ(expected_output.size(), Y_host->size());
    for (std::size_t i = 0; i < expected_output.size(); ++i) {
      EXPECT_FLOAT_EQ(expected_output[i], Y_host->data<float>()[i]);
    }
  }

  template <class ReduceFunc>
  void RunRedcueTensorTest(
      const ReduceFunc& reduce_func,
      const std::vector<int>& x_dims,
      const std::vector<int>& y_dims,
      const std::vector<int>& axes,
      const std::vector<float>& x_data,
      const std::vector<float>& y_data) {
    SetUpData(x_dims, y_dims, axes, x_data);
    reduce_func(
        X_->size(),
        Y_->size(),
        dims_device_->size(),
        dims_device_->data<int>(),
        axes_device_->size(),
        axes_device_->data<int>(),
        X_->data<float>(),
        Y_->mutable_data<float>(),
        cuda_context_.get(),
        scratch_ptr_);
    VerifyResult(y_data);
  }

  Workspace ws_;
  DeviceOption option_;
  std::unique_ptr<CUDAContext> cuda_context_;
  Tensor<CUDAContext>* X_ = nullptr;
  Tensor<CUDAContext>* Y_ = nullptr;
  Tensor<CUDAContext>* dims_device_ = nullptr;
  Tensor<CUDAContext>* axes_device_ = nullptr;
  Tensor<CUDAContext>* scratch_ptr_ = nullptr;
};

TEST_F(ReduceTensorGPUTest, ReduceSumGPUTest) {
  if (!HasCudaGPU()) {
    return;
  }
  // Test for 1D tensor.
  RunRedcueTensorTest(
      math::ReduceSum<float, CUDAContext>,
      {3},
      {1},
      {0},
      {1.0f, 2.0f, 3.0f},
      {6.0f});

  // Test for 2D Tensor.
  RunRedcueTensorTest(
      math::ReduceSum<float, CUDAContext>,
      {2, 3},
      {2, 1},
      {1},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      {6.0f, 15.0f});
  RunRedcueTensorTest(
      math::ReduceSum<float, CUDAContext>,
      {2, 3},
      {1, 3},
      {0},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      {5.0f, 7.0f, 9.0f});
  RunRedcueTensorTest(
      math::ReduceSum<float, CUDAContext>,
      {2, 3},
      {1, 1},
      {0, 1},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      {21.0f});

  // Test for 3D tensor.
  RunRedcueTensorTest(
      math::ReduceSum<float, CUDAContext>,
      {2, 2, 2},
      {2, 1, 1},
      {1, 2},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {10.0f, 26.0f});
  RunRedcueTensorTest(
      math::ReduceSum<float, CUDAContext>,
      {2, 2, 2},
      {1, 1, 2},
      {0, 1},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {16.0f, 20.0f});
  RunRedcueTensorTest(
      math::ReduceSum<float, CUDAContext>,
      {2, 2, 2},
      {1, 2, 1},
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
      {1},
      {0},
      {1.0f, 2.0f, 3.0f},
      {2.0f});

  // Test for 2D Tensor.
  RunRedcueTensorTest(
      math::ReduceMean<float, CUDAContext>,
      {2, 3},
      {2, 1},
      {1},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      {2.0f, 5.0f});
  RunRedcueTensorTest(
      math::ReduceMean<float, CUDAContext>,
      {2, 3},
      {1, 3},
      {0},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      {2.5f, 3.5f, 4.5f});
  RunRedcueTensorTest(
      math::ReduceMean<float, CUDAContext>,
      {2, 3},
      {1, 1},
      {0, 1},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      {3.5f});

  // Test for 3D tensor.
  RunRedcueTensorTest(
      math::ReduceMean<float, CUDAContext>,
      {2, 2, 2},
      {2, 1, 1},
      {1, 2},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {2.5f, 6.5f});
  RunRedcueTensorTest(
      math::ReduceMean<float, CUDAContext>,
      {2, 2, 2},
      {1, 1, 2},
      {0, 1},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {4.0f, 5.0f});
  RunRedcueTensorTest(
      math::ReduceMean<float, CUDAContext>,
      {2, 2, 2},
      {1, 2, 1},
      {0, 2},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {3.5f, 5.5f});
}

class TransposeGPUTest : public testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    if (!HasCudaGPU()) {
      return;
    }
    option_.set_device_type(CUDA);
    cuda_context_ = make_unique<CUDAContext>(option_);
    Blob* blob_x = ws_.CreateBlob("X");
    Blob* blob_y = ws_.CreateBlob("Y");
    Blob* blob_x_dims = ws_.CreateBlob("x_dims");
    Blob* blob_y_dims = ws_.CreateBlob("y_dims");
    Blob* blob_axes = ws_.CreateBlob("axes");
    X_ = blob_x->GetMutable<Tensor<CUDAContext>>();
    Y_ = blob_y->GetMutable<Tensor<CUDAContext>>();
    x_dims_device_ = blob_x_dims->GetMutable<Tensor<CUDAContext>>();
    y_dims_device_ = blob_y_dims->GetMutable<Tensor<CUDAContext>>();
    axes_device_ = blob_axes->GetMutable<Tensor<CUDAContext>>();
  }

  void SetUpData(
      const std::vector<int>& x_dims,
      const std::vector<int>& axes,
      const std::vector<float>& x_data) {
    const int ndim = x_dims.size();
    std::vector<int> y_dims(ndim);
    for (int i = 0; i < ndim; ++i) {
      y_dims[i] = x_dims[axes[i]];
    }
    x_dims_device_->Resize(x_dims.size());
    cuda_context_->Copy<int, CPUContext, CUDAContext>(
        x_dims.size(), x_dims.data(), x_dims_device_->mutable_data<int>());
    y_dims_device_->Resize(y_dims.size());
    cuda_context_->Copy<int, CPUContext, CUDAContext>(
        y_dims.size(), y_dims.data(), y_dims_device_->mutable_data<int>());
    axes_device_->Resize(axes.size());
    cuda_context_->Copy<int, CPUContext, CUDAContext>(
        axes.size(), axes.data(), axes_device_->mutable_data<int>());
    X_->Resize(x_dims);
    Y_->Resize(y_dims);
    cuda_context_->Copy<float, CPUContext, CUDAContext>(
        x_data.size(), x_data.data(), X_->mutable_data<float>());
  }

  void VerifyResult(const std::vector<float>& expected_output) {
    Blob* blob_y_host = ws_.CreateBlob("Y_host");
    auto* Y_host = blob_y_host->GetMutable<TensorCPU>();
    Y_host->CopyFrom<CUDAContext, CUDAContext>(*Y_, cuda_context_.get());
    cuda_context_->FinishDeviceComputation();
    ASSERT_EQ(expected_output.size(), Y_host->size());
    for (std::size_t i = 0; i < expected_output.size(); ++i) {
      EXPECT_FLOAT_EQ(expected_output[i], Y_host->data<float>()[i]);
    }
  }

  void RunTransposeTest(
      const std::vector<int>& x_dims,
      const std::vector<int>& axes,
      const std::vector<float>& x_data,
      const std::vector<float>& y_data) {
    SetUpData(x_dims, axes, x_data);
    if (GetParam()) {
      math::Transpose<float, CUDAContext>(
          x_data.size(),
          x_dims.size(),
          x_dims_device_->data<int>(),
          y_dims_device_->data<int>(),
          axes_device_->data<int>(),
          X_->data<float>(),
          Y_->mutable_data<float>(),
          cuda_context_.get());
    } else {
      math::Transpose<float, CUDAContext>(
          x_data.size(),
          x_dims.size(),
          x_dims_device_->data<int>(),
          axes_device_->data<int>(),
          X_->data<float>(),
          Y_->mutable_data<float>(),
          cuda_context_.get());
    }
    cuda_context_->FinishDeviceComputation();
    VerifyResult(y_data);
  }

  Workspace ws_;
  DeviceOption option_;
  std::unique_ptr<CUDAContext> cuda_context_;
  Tensor<CUDAContext>* X_ = nullptr;
  Tensor<CUDAContext>* Y_ = nullptr;
  Tensor<CUDAContext>* x_dims_device_ = nullptr;
  Tensor<CUDAContext>* y_dims_device_ = nullptr;
  Tensor<CUDAContext>* axes_device_ = nullptr;
};

TEST_P(TransposeGPUTest, TransposeGPUFloatTest) {
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

INSTANTIATE_TEST_CASE_P(WithYDims, TransposeGPUTest, testing::Bool());

} // namespace

} // namespace caffe2
