#ifdef USE_CUDA

#include <sstream>
#include <stdexcept>
#include "test/cpp/tensorexpr/test_base.h"

#include <cmath>

#include "test/cpp/tensorexpr/padded_buffer.h"
#include "torch/csrc/jit/tensorexpr/buffer.h"
#include "torch/csrc/jit/tensorexpr/cuda_codegen.h"
#include "torch/csrc/jit/tensorexpr/schedule.h"
#include "torch/csrc/jit/tensorexpr/tensor.h"

#include <c10/cuda/CUDACachingAllocator.h>

namespace torch {
namespace jit {
using namespace torch::jit::tensorexpr;
using namespace torch::jit::tensorexpr::schedule;

void testCudaTestVectorAdd01() {
  KernelScope kernel_scope;
  const int num_iter = 3;
  const int block_count = 16;
  const int block_size = 128;
  Buffer a_buf("a", kFloat32, {num_iter, block_count, block_size});
  Buffer b_buf("b", kFloat32, {num_iter, block_count, block_size});
  Tensor* c = Compute(
      "c",
      {
          {num_iter, "n"},
          {block_count, "b_id"},
          {block_size, "t_id"},
      },
      [&](const VarHandle& n, const VarHandle& b_id, const VarHandle& t_id) {
        return a_buf(n, b_id, t_id) + b_buf(n, b_id, t_id);
      });
  Schedule sch({c});
  VarHandle b_id(c->function()->arg(1));
  VarHandle t_id(c->function()->arg(2));
  c->GPUExecConfig({b_id}, {t_id});
  Stmt* stmt = sch.Lower();
  CudaCodeGen cuda_cg(stmt, c, a_buf, b_buf);
  const int N = block_count * block_size * num_iter;
  PaddedBuffer<float> a_v(N);
  PaddedBuffer<float> b_v(N);
  PaddedBuffer<float> c_v(N);
  PaddedBuffer<float> c_ref(N);

  for (int i = 0; i < N; i++) {
    a_v(i) = i;
    b_v(i) = i * 3 + 7;
    c_ref(i) = a_v(i) + b_v(i);
  }

  // TODO: move gpu support into PaddedBuffer
  float* a_dev = nullptr;
  cudaMalloc(&a_dev, N * sizeof(float));
  float* b_dev = nullptr;
  cudaMalloc(&b_dev, N * sizeof(float));
  float* c_dev = nullptr;
  cudaMalloc(&c_dev, N * sizeof(float));
  cudaMemcpy(a_dev, a_v.data(), N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b_dev, b_v.data(), N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(c_dev, c_v.data(), N * sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  cuda_cg(c_dev, a_dev, b_dev);

  cudaDeviceSynchronize();
  cudaMemcpy(c_v.data(), c_dev, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  ExpectAllNear(c_v, c_ref, 1e-5);

  cudaFree(a_dev);
  cudaFree(b_dev);
  cudaFree(c_dev);
}

static void testCudaTestVectorAdd02_impl(int N, int block_size) {
  KernelScope kernel_scope;
  Buffer a_buf("a", kFloat32, {N});
  Buffer b_buf("b", kFloat32, {N});
  Tensor* c = Compute(
      "c",
      {
          {N, "N"},
      },
      [&](const VarHandle& n) { return a_buf(n) + b_buf(n); });
  Schedule sch({c});
  VarHandle n(c->arg(0));
  VarHandle n_outer;
  VarHandle n_inner;
  c->SplitWithMask(n, block_size, true, &n_outer, &n_inner);
  c->GPUExecConfig({n_outer}, {n_inner});
  Stmt* stmt = sch.Lower();
  CudaCodeGen cuda_cg(stmt, c, a_buf, b_buf);
  PaddedBuffer<float> a_v(N);
  PaddedBuffer<float> b_v(N);
  PaddedBuffer<float> c_v(N);
  PaddedBuffer<float> c_ref(N);

  for (int i = 0; i < N; i++) {
    a_v(i) = i;
    b_v(i) = i * 3 + 7;
    c_ref(i) = a_v(i) + b_v(i);
  }

  // TODO: move gpu support into PaddedBuffer
  float* a_dev = nullptr;
  cudaMalloc(&a_dev, N * sizeof(float));
  float* b_dev = nullptr;
  cudaMalloc(&b_dev, N * sizeof(float));
  float* c_dev = nullptr;
  cudaMalloc(&c_dev, N * sizeof(float));
  cudaMemcpy(a_dev, a_v.data(), N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b_dev, b_v.data(), N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(c_dev, c_v.data(), N * sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  cuda_cg(c_dev, a_dev, b_dev);

  cudaDeviceSynchronize();
  cudaMemcpy(c_v.data(), c_dev, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  ExpectAllNear(c_v, c_ref, 1e-5);

  cudaFree(a_dev);
  cudaFree(b_dev);
  cudaFree(c_dev);
}

void testCudaTestVectorAdd02() {
  testCudaTestVectorAdd02_impl(1024, 128);
  testCudaTestVectorAdd02_impl(1030, 128);
}

void testCudaDynamicShape2D() {
  KernelScope kernel_scope;
  auto testWithSize = [](int32_t M, int32_t N) {
    VarHandle m("m", kInt32);
    VarHandle n("n", kInt32);
    Buffer a(VarHandle("a", kHandle), kFloat32, {m, n});
    Buffer b(VarHandle("b", kHandle), kFloat32, {m, n});
    Tensor* c =
        Compute("c", {{m, "m"}, {n, "n"}}, [&](const VarHandle& i, const VarHandle& j) {
          return a(i, j) + b(i, j);
        });
    auto sch = Schedule::make({c});
    Stmt* s = sch.Lower();
    CudaCodeGen cg(s, {a, b, c, m, n});

    std::vector<float> aData(M * N, 1.0f);
    std::vector<float> bData(M * N, 2.0f);
    std::vector<float> cData(M * N, 0.0f);
    float* aDev = nullptr;
    float* bDev = nullptr;
    float* cDev = nullptr;
    cudaMalloc(&aDev, aData.size() * sizeof(aData[0]));
    cudaMalloc(&bDev, bData.size() * sizeof(bData[0]));
    cudaMalloc(&cDev, cData.size() * sizeof(cData[0]));
    cudaMemcpy(
        aDev,
        aData.data(),
        aData.size() * sizeof(aData[0]),
        cudaMemcpyHostToDevice);
    cudaMemcpy(
        bDev,
        bData.data(),
        bData.size() * sizeof(bData[0]),
        cudaMemcpyHostToDevice);
    cudaMemcpy(
        cDev,
        cData.data(),
        cData.size() * sizeof(cData[0]),
        cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    cg.call({aDev, bDev, cDev, M, N});
    cudaDeviceSynchronize();

    cudaMemcpy(
        cData.data(),
        cDev,
        cData.size() * sizeof(cData[0]),
        cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    ExpectAllNear(cData, std::vector<float>(M * N, 3.0f), 1e-7);

    cudaFree(aDev);
    cudaFree(bDev);
    cudaFree(cDev);
  };
  testWithSize(32, 32);
  testWithSize(1, 16);
  testWithSize(27, 13);
}

void testCudaTestRand01() {
  KernelScope kernel_scope;
  const int num_iter = 3;
  const int block_count = 16;
  const int block_size = 128;
  Tensor* c = Compute(
      "c",
      {
          {num_iter, "n"},
          {block_count, "b_id"},
          {block_size, "t_id"},
      },
      [&](const VarHandle& n, const VarHandle& b_id, const VarHandle& t_id) {
        return Intrinsics::make(IntrinsicsOp::kRand, kFloat32);
      });
  Schedule sch({c});
  VarHandle b_id(c->arg(1));
  VarHandle t_id(c->arg(2));
  c->GPUExecConfig({b_id}, {t_id});
  Stmt* stmt = sch.Lower();
  CudaCodeGen cuda_cg(stmt, c);
  const int N = block_count * block_size * num_iter;
  PaddedBuffer<float> c_v(N);

  // TODO: move gpu support into PaddedBuffer
  float* c_dev = nullptr;
  cudaMalloc(&c_dev, N * sizeof(float));
  cudaDeviceSynchronize();

  cuda_cg(c_dev);

  cudaDeviceSynchronize();
  cudaMemcpy(c_v.data(), c_dev, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  float sum1 = 0;
  float sum2 = 0;
  float sum3 = 0;
  for (int i = 0; i < N; i++) {
    float v = c_v.data()[i];
    sum1 += v;
    sum2 += v * v;
    sum3 += v * v * v;
    EXPECT_TRUE(v >= 0 && v < 1) << "invalid value: " << i << ", " << v;
  }
  sum1 /= N;
  sum2 /= N;
  sum3 /= N;
  float sum1_mean = 1.f / 2;
  float sum2_mean = 1.f / 3;
  float sum3_mean = 1.f / 4;

  EXPECT_NEAR(sum1, sum1_mean, 2e-2);
  EXPECT_NEAR(sum2, sum2_mean, 2e-2);
  EXPECT_NEAR(sum3, sum3_mean, 2e-2);
  cudaFree(c_dev);
}

void testCudaDynamicShapeSplit() {
  KernelScope ks;
  constexpr int N = 4096;
  VarHandle n("n", kInt32);
  Buffer a(VarHandle("a", kHandle), kFloat32, {n});
  Tensor* b =
      Compute("b", {{n, "n"}}, [&](const VarHandle& i) { return a(i) * 2.0f; });
  auto sch = Schedule::make({b});
  VarHandle outer;
  VarHandle inner;
  b->SplitWithMask(VarHandle(b->arg(0)), 1024, true, &outer, &inner);
  b->GPUExecConfig({outer}, {inner});
  Stmt* s = sch.Lower();
  CudaCodeGen cg(s, {a, b, n});

  std::vector<float> aData(N, 1.0f);
  std::vector<float> bData(N, 1.0f);
  float* aDev = nullptr;
  float* bDev = nullptr;
  cudaMalloc(&aDev, aData.size() * sizeof(aData[0]));
  cudaMalloc(&bDev, bData.size() * sizeof(bData[0]));
  cudaMemcpy(
      aDev,
      aData.data(),
      aData.size() * sizeof(aData[0]),
      cudaMemcpyHostToDevice);
  cudaMemcpy(
      bDev,
      bData.data(),
      bData.size() * sizeof(aData[0]),
      cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  cg.call({aDev, bDev, N});
  cudaDeviceSynchronize();

  cudaMemcpy(
      bData.data(),
      bDev,
      bData.size() * sizeof(aData[0]),
      cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  ExpectAllNear(bData, std::vector<float>(N, 2.0f), 1e-7);

  cudaFree(aDev);
  cudaFree(bDev);
}

} // namespace jit
} // namespace torch

#endif
