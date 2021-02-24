#ifdef USE_CUDA

#include <cmath>
#include <sstream>
#include <stdexcept>

#include <gtest/gtest.h>

#include "test/cpp/tensorexpr/test_base.h"

#include <torch/csrc/jit/testing/file_check.h>
#include "test/cpp/tensorexpr/padded_buffer.h"
#include "torch/csrc/jit/tensorexpr/cuda_codegen.h"
#include "torch/csrc/jit/tensorexpr/ir_simplifier.h"
#include "torch/csrc/jit/tensorexpr/loopnest.h"
#include "torch/csrc/jit/tensorexpr/tensor.h"

#include <torch/csrc/jit/testing/file_check.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/Half.h>

namespace torch {
namespace jit {
using namespace torch::jit::tensorexpr;
using namespace torch::jit::tensorexpr;

template <typename ctype>
static void testCudaTestVectorAdd01_impl() {
  KernelScope kernel_scope;
  const int num_iter = 3;
  const int block_count = 16;
  const int block_size = 128;
  Dtype dtype = ToDtype<ctype>();
  Placeholder a_buf("a", dtype, {num_iter, block_count, block_size});
  Placeholder b_buf("b", dtype, {num_iter, block_count, block_size});
  Tensor* c = Compute(
      "c",
      {
          {num_iter, "n"},
          {block_count, "b_id"},
          {block_size, "t_id"},
      },
      [&](const VarHandle& n, const VarHandle& b_id, const VarHandle& t_id) {
        return a_buf.load(n, b_id, t_id) + b_buf.load(n, b_id, t_id);
      });
  LoopNest l({c});
  std::vector<For*> loops = l.getLoopStmtsFor(c);
  l.setGPUBlockIndex(loops[1], 0);
  l.setGPUThreadIndex(loops[2], 0);
  l.prepareForCodegen();
  Stmt* stmt = l.root_stmt();
  CudaCodeGen cuda_cg(stmt, c, a_buf, b_buf);
  const int N = block_count * block_size * num_iter;
  PaddedBuffer<ctype> a_v(N);
  PaddedBuffer<ctype> b_v(N);
  PaddedBuffer<ctype> c_v(N);
  PaddedBuffer<ctype> c_ref(N);

  for (int i = 0; i < N; i++) {
    a_v(i) = ctype(i);
    b_v(i) = ctype(i * 3 + 7);
    c_ref(i) = a_v(i) + b_v(i);
  }

  // TODO: move gpu support into PaddedBuffer
  ctype* a_dev = nullptr;
  cudaMalloc(&a_dev, N * sizeof(ctype));
  ctype* b_dev = nullptr;
  cudaMalloc(&b_dev, N * sizeof(ctype));
  ctype* c_dev = nullptr;
  cudaMalloc(&c_dev, N * sizeof(ctype));
  cudaMemcpy(a_dev, a_v.data(), N * sizeof(ctype), cudaMemcpyHostToDevice);
  cudaMemcpy(b_dev, b_v.data(), N * sizeof(ctype), cudaMemcpyHostToDevice);
  cudaMemcpy(c_dev, c_v.data(), N * sizeof(ctype), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  cuda_cg(c_dev, a_dev, b_dev);

  cudaDeviceSynchronize();
  cudaMemcpy(c_v.data(), c_dev, N * sizeof(ctype), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  ExpectAllNear(c_v, c_ref, 1e-5);

  cudaFree(a_dev);
  cudaFree(b_dev);
  cudaFree(c_dev);
}

float sigmoid(float x) {
  return 1.0f / (1.0f + expf(-0.0f - x));
}

TEST(Cuda, Sigmoid_CUDA) {
  KernelScope kernel_scope;
  const int num_iter = 3;
  const int block_count = 16;
  const int block_size = 128;
  Dtype dtype = ToDtype<float>();
  Placeholder a_buf("a", dtype, {num_iter, block_count, block_size});
  Tensor* c = Compute(
      "c",
      {
          {num_iter, "n"},
          {block_count, "b_id"},
          {block_size, "t_id"},
      },
      [&](const VarHandle& n, const VarHandle& b_id, const VarHandle& t_id) {
        return sigmoid(sigmoid(a_buf.load(n, b_id, t_id)));
      });
  LoopNest l({c});
  std::vector<For*> loops = l.getLoopStmtsFor(c);
  l.setGPUBlockIndex(loops[1], 0);
  l.setGPUThreadIndex(loops[2], 0);
  l.prepareForCodegen();
  Stmt* stmt = l.root_stmt();
  CudaCodeGen cuda_cg(stmt, c, a_buf);
  const int N = block_count * block_size * num_iter;
  PaddedBuffer<float> a_v(N);
  PaddedBuffer<float> c_v(N);
  PaddedBuffer<float> c_ref(N);

  for (int i = 0; i < N; i++) {
    a_v(i) = float(i);
    c_ref(i) = sigmoid(sigmoid(a_v(i)));
  }

  // TODO: move gpu support into PaddedBuffer
  float* a_dev = nullptr;
  cudaMalloc(&a_dev, N * sizeof(float));
  float* c_dev = nullptr;
  cudaMalloc(&c_dev, N * sizeof(float));
  cudaMemcpy(a_dev, a_v.data(), N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(c_dev, c_v.data(), N * sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  cuda_cg(c_dev, a_dev);

  cudaDeviceSynchronize();
  cudaMemcpy(c_v.data(), c_dev, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  ExpectAllNear(c_v, c_ref, 1e-5);

  cudaFree(a_dev);
  cudaFree(c_dev);
}

TEST(Cuda, TestVectorAdd01_CUDA) {
  // floating types.
  testCudaTestVectorAdd01_impl<float>();
  testCudaTestVectorAdd01_impl<at::Half>();
  testCudaTestVectorAdd01_impl<double>();

  // integer types.
  testCudaTestVectorAdd01_impl<int8_t>();
  testCudaTestVectorAdd01_impl<uint8_t>();
  testCudaTestVectorAdd01_impl<int16_t>();
  testCudaTestVectorAdd01_impl<int32_t>();
  testCudaTestVectorAdd01_impl<int64_t>();
}

static void testCudaTestVectorAdd02_impl(int N, int block_size) {
  KernelScope kernel_scope;
  Placeholder a_buf("a", kFloat, {N});
  Placeholder b_buf("b", kFloat, {N});
  Tensor* c = Compute(
      "c",
      {
          {N, "N"},
      },
      [&](const VarHandle& n) { return a_buf.load(n) + b_buf.load(n); });
  LoopNest l({c});
  For* n_outer;
  For* n_inner;
  std::vector<For*> loops = l.getLoopStmtsFor(c);
  l.splitWithMask(loops[0], block_size, &n_outer, &n_inner);
  l.setGPUBlockIndex(n_outer, 0);
  l.setGPUThreadIndex(n_inner, 0);
  l.prepareForCodegen();
  Stmt* stmt = l.root_stmt();
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

TEST(Cuda, TestVectorAdd02_CUDA) {
  testCudaTestVectorAdd02_impl(1024, 128);
  testCudaTestVectorAdd02_impl(1030, 128);
}

TEST(Cuda, HalfCast_CUDA) {
  KernelScope ks;
  auto half = ToDtype<at::Half>();
  Placeholder a("a", half, {4});
  Tensor* b = Compute("b", {{4, "n"}}, [&](const VarHandle& i) {
    return Cast::make(kFloat, a.load(i));
  });

  LoopNest l({b});
  l.prepareForCodegen();
  Stmt* s = l.root_stmt();
  CudaCodeGen cg(s, {a, b});

  std::vector<at::Half> aData(4, 2.0f);
  std::vector<float> bData(4, 0.0f);
  at::Half* aDev = nullptr;
  float* bDev = nullptr;
  auto aSize = aData.size() * sizeof(aData[0]);
  auto bSize = bData.size() * sizeof(bData[0]);

  cudaMalloc(&aDev, aSize);
  cudaMalloc(&bDev, bSize);
  cudaMemcpy(aDev, aData.data(), aSize, cudaMemcpyHostToDevice);
  cudaMemcpy(bDev, bData.data(), bSize, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  cg.call({aDev, bDev});
  cudaDeviceSynchronize();

  cudaMemcpy(aData.data(), aDev, aSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(bData.data(), bDev, bSize, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  assertAllEqual(bData, 2.0f);

  cudaFree(aDev);
  cudaFree(bDev);
}

TEST(Cuda, DynamicShape2D_CUDA) {
  KernelScope kernel_scope;
  auto testWithSize = [](int32_t M, int32_t N) {
    VarHandle m("m", kInt);
    VarHandle n("n", kInt);
    Placeholder a(BufHandle("a", {m, n}, kFloat));
    Placeholder b(BufHandle("b", {m, n}, kFloat));
    Tensor* c = Compute(
        "c", {{m, "m"}, {n, "n"}}, [&](const VarHandle& i, const VarHandle& j) {
          return a.load(i, j) + b.load(i, j);
        });
    LoopNest l({c});
    l.prepareForCodegen();
    Stmt* s = l.root_stmt();
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

TEST(Cuda, TestRand01_CUDA) {
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
        return Intrinsics::make(IntrinsicsOp::kRand, kFloat);
      });
  LoopNest l({c});
  std::vector<For*> loops = l.getLoopStmtsFor(c);
  l.setGPUBlockIndex(loops[1], 0);
  l.setGPUThreadIndex(loops[2], 0);
  l.prepareForCodegen();
  Stmt* stmt = l.root_stmt();
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
    ASSERT_TRUE(v >= 0 && v < 1);
  }
  sum1 /= N;
  sum2 /= N;
  sum3 /= N;
  float sum1_mean = 1.f / 2;
  float sum2_mean = 1.f / 3;
  float sum3_mean = 1.f / 4;

  ASSERT_NEAR(sum1, sum1_mean, 2e-2);
  ASSERT_NEAR(sum2, sum2_mean, 2e-2);
  ASSERT_NEAR(sum3, sum3_mean, 2e-2);
  cudaFree(c_dev);
}

TEST(Cuda, DynamicShapeSplit_CUDA) {
  KernelScope ks;
  constexpr int N = 4096;
  VarHandle n("n", kInt);
  Placeholder a(BufHandle("a", {n}, kFloat));
  Tensor* b = Compute(
      "b", {{n, "n"}}, [&](const VarHandle& i) { return a.load(i) * 2.0f; });
  LoopNest l({b});
  For* outer;
  For* inner;
  std::vector<For*> loops = l.getLoopStmtsFor(b);
  l.splitWithMask(loops[0], 1024, &outer, &inner);
  l.setGPUBlockIndex(outer, 0);
  l.setGPUThreadIndex(inner, 0);
  Stmt* s = l.root_stmt();
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

TEST(Cuda, OneBlockOneThreadGlobalReduce1_CUDA) {
  const static int N = 1024;
  KernelScope kernel_scope;
  Placeholder data_buf("data", kFloat, {N});
  Placeholder output_buf("output", kFloat, {1});

  // The test adds the following code for trivial reduction:
  // for (int bidx = 0; bidx < 1; bidx++) { // blockIdx.x
  //   for (int tidx = 0; tidx < 1; tidx++) { // threadIdx.x
  //     output[0] = 0.f;
  //     for (int i1 = 0; i1 < 1024; i1++) {
  //       output[0] = output[0] + data[i1];
  //     }
  //   }
  // }

  Store* init_store = output_buf.store({0}, 0.f);
  VarHandle i1("i1", kInt);
  ExprHandle load_data = Load::make(BufHandle(data_buf.data()), {i1}, 1);
  ExprHandle load_output = Load::make(BufHandle(output_buf.data()), {0}, 1);
  ExprHandle add_value = load_output + load_data;
  Store* store_output = output_buf.store({0}, add_value);
  For* for_output = For::make(i1, 0, N, store_output);
  Stmt* reduce_block = Block::make({init_store, for_output});
  VarHandle thread_idx("tidx", kInt);
  LoopOptions thread_idx_options;
  thread_idx_options.set_gpu_thread_index(0);
  For* thread_idx_loop =
      For::make(thread_idx, 0, 1, reduce_block, thread_idx_options);
  VarHandle block_idx("bidx", kInt);
  LoopOptions block_idx_options;
  block_idx_options.set_gpu_block_index(0);
  For* block_idx_loop =
      For::make(block_idx, 0, 1, thread_idx_loop, block_idx_options);

  CudaCodeGen cuda_cg(block_idx_loop, data_buf, output_buf);
  PaddedBuffer<float> data_v(N);
  PaddedBuffer<float> output_v(1, "output_v");
  PaddedBuffer<float> output_ref(1, "output_ref");

  output_ref(0) = 0;
  for (int i = 0; i < N; i++) {
    data_v(i) = i;
    output_ref(0) += data_v(i);
  }

  float* data_dev = nullptr;
  cudaMalloc(&data_dev, N * sizeof(float));
  cudaMemcpy(
      data_dev, data_v.data(), N * sizeof(float), cudaMemcpyHostToDevice);
  float* output_dev = nullptr;
  cudaMalloc(&output_dev, 1 * sizeof(float));
  cudaDeviceSynchronize();

  cuda_cg(data_dev, output_dev);

  cudaDeviceSynchronize();
  cudaMemcpy(
      output_v.data(), output_dev, 1 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  ExpectAllNear(output_v, output_ref, 1e-5);

  cudaFree(data_dev);
  cudaFree(output_dev);
}

TEST(Cuda, OneBlockMultiThreadGlobalReduce1_CUDA) {
  const static int N = 1024;
  KernelScope kernel_scope;

  // This test does the following reduction:
  // clang-format off
  //   for b in 0..1 // block-idx
  //    for t in 0..1024: // thread-idx
  //      if t < 1:
  //        b[0] = 0
  //    // implied sync_threads
  //    for t in 0..1024: // thread-idx
  //      b[0] = b[0] + a[t] // implied atomic
  // clang-format on

  Placeholder a_buf("a", kFloat, {N});
  Placeholder b_buf("b", kFloat, {1});

  Store* init_store = b_buf.store({0}, 0.f);
  VarHandle t("t", kInt);
  VarHandle b("b", kInt);

  //  for t in 0..1024: // thread-idx
  //    if t < 1:
  //      b[0] = 0
  ExprHandle cond_t_lt_1 =
      CompareSelect::make(t, 1, CompareSelectOperation::kLT);
  Cond* masked_init_b = Cond::make(cond_t_lt_1, init_store, nullptr);
  LoopOptions thread_idx_options;
  thread_idx_options.set_gpu_thread_index(0);
  For* for_init = For::make(t, 0, N, masked_init_b, thread_idx_options);

  //  for t in 0..1024: // thread-idx
  //    b[0] = b[0] + a[t] // implied atomic
  ExprHandle load_a = Load::make(BufHandle(a_buf.data()), {t}, 1);
  ExprHandle load_b = Load::make(BufHandle(b_buf.data()), {0}, 1);
  ExprHandle add_value = load_b + load_a;
  Store* store_b = b_buf.store({0}, add_value);
  For* for_b = For::make(t, 0, N, store_b, thread_idx_options);

  Stmt* reduce_block = Block::make({for_init, for_b});

  VarHandle block_idx("bidx", kInt);
  LoopOptions block_idx_options;
  block_idx_options.set_gpu_block_index(0);
  For* block_idx_loop =
      For::make(block_idx, 0, 1, reduce_block, block_idx_options);

  CudaCodeGen cuda_cg(block_idx_loop, a_buf, b_buf);
  PaddedBuffer<float> a_v(N);
  PaddedBuffer<float> b_v(1, "b_v");
  PaddedBuffer<float> b_ref(1, "b_ref");

  b_ref(0) = 0;
  for (int i = 0; i < N; i++) {
    a_v(i) = i;
    b_ref(0) += a_v(i);
  }

  float* a_dev = nullptr;
  cudaMalloc(&a_dev, N * sizeof(float));
  cudaMemcpy(a_dev, a_v.data(), N * sizeof(float), cudaMemcpyHostToDevice);
  float* b_dev = nullptr;
  cudaMalloc(&b_dev, 1 * sizeof(float));
  cudaDeviceSynchronize();

  cuda_cg(a_dev, b_dev);

  cudaDeviceSynchronize();
  cudaMemcpy(b_v.data(), b_dev, 1 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  ExpectAllNear(b_v, b_ref, 1e-5);

  cudaFree(a_dev);
  cudaFree(b_dev);
}

TEST(Cuda, NoThreadIdxWrite_1_CUDA) {
  KernelScope kernel_scope;

  // This test does the following reduction:
  //
  // for k in 0..1: // block-idx
  //   a[0] = 0
  //   for n in 0..2:
  //     a[0] = a[0] + n
  //   for m in 0..1024: // thread-idx
  //     b[m] = m
  //   a[1] = 1
  //   for l in 0..2:
  //     a[1] = a[1] + n
  //
  //  note that the statements not covered by thread-idx are supposed to be
  //  covered by its own thread-idx

  const static int N = 1024;
  Placeholder a_buf("a", kFloat, {2});
  Placeholder b_buf("b", kFloat, {N});

  VarHandle k("k", kInt);
  VarHandle l("l", kInt);
  VarHandle m("m", kInt);
  VarHandle n("n", kInt);

  //   a[0] = 0
  //   for n in 0..2:
  //     a[0] = a[0] + n
  Store* store_a0_0 = a_buf.store({0}, 0.f);
  ExprHandle load_a0 = Load::make(BufHandle(a_buf.data()), {0}, 1);
  ExprHandle v1 = load_a0 + n;
  Store* store_a0_v1 = a_buf.store({0}, v1);
  For* loop_a_0 = For::make(n, 0, 2, store_a0_v1);

  //   for m in 0..1024: // thread-idx
  //     b[m] = m
  Store* store_bm_m = b_buf.store({m}, m + 0.f);
  LoopOptions thread_idx_options;
  thread_idx_options.set_gpu_thread_index(0);
  For* loop_b_1 = For::make(m, 0, N, store_bm_m, thread_idx_options);

  //   a[1] = 1
  //   for l in 0..2:
  //     a[1] = a[1] + l
  Store* store_a1_1 = a_buf.store({1}, 1.f);
  ExprHandle load_a1 = a_buf.load(1);
  ExprHandle v2 = load_a1 + l;
  Store* store_a1_v2 = a_buf.store({1}, v2);
  For* loop_a_1 = For::make(l, 0, 2, store_a1_v2);

  Stmt* reduce_block =
      Block::make({store_a0_0, loop_a_0, loop_b_1, store_a1_1, loop_a_1});

  VarHandle block_idx("bidx", kInt);
  LoopOptions block_idx_options;
  block_idx_options.set_gpu_block_index(0);
  For* block_idx_loop =
      For::make(block_idx, 0, 1, reduce_block, block_idx_options);

  CudaCodeGen cuda_cg(block_idx_loop, a_buf, b_buf);
  PaddedBuffer<float> a_v(2);
  PaddedBuffer<float> b_v(N, "b_v");
  PaddedBuffer<float> a_ref(2, "a_ref");
  PaddedBuffer<float> b_ref(N, "b_ref");

  a_ref(0) = 0;
  for (int i = 0; i < 2; i++) {
    a_ref(0) += i;
  }
  a_ref(1) = a_ref(0) + 1;
  for (int i = 0; i < N; i++) {
    b_ref(i) = i;
  }

  // TODO: add check of the generated code.
  float* a_dev = nullptr;
  cudaMalloc(&a_dev, 2 * sizeof(float));
  float* b_dev = nullptr;
  cudaMalloc(&b_dev, N * sizeof(float));
  cudaDeviceSynchronize();

  cuda_cg(a_dev, b_dev);

  cudaDeviceSynchronize();
  cudaMemcpy(a_v.data(), a_dev, 2 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(b_v.data(), b_dev, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  ExpectAllNear(a_v, a_ref, 1e-5);
  ExpectAllNear(b_v, b_ref, 1e-5);

  cudaFree(a_dev);
  cudaFree(b_dev);
}

TEST(Cuda, SharedMemReduce_1_CUDA) {
  // FIXME: this test is flaky in CI.
  KernelScope kernel_scope;
  // This test does the following:
  //  for k in 0..1:  // block-idx
  //    alloc(c, 64)
  //    for n in 0..64:  // thread-idx
  //      c(n) = 0
  //    for m in 0..128:
  //      for n in 0..64:  // thread_idx
  //        c(n) = c(n) + a(k, m, n)
  //    b(k) = 0
  //    for n in 0..64:  // thread_idx
  //      b(k) = b(k) + c(n)
  //    free(c)

  const int M = 128;
  const int N = 64;
  const int kTotalSize = M * N;
  LoopOptions thread_idx_opt;
  thread_idx_opt.set_gpu_thread_index(0);
  LoopOptions block_idx_opt;
  block_idx_opt.set_gpu_block_index(0);

  Placeholder a("a", kFloat, {1, M, N});
  Placeholder b("b", kFloat, {1});
  VarHandle k("k", kInt);
  VarHandle m("m", kInt);
  VarHandle n("n", kInt);

  std::vector<Stmt*> block;
  std::vector<const Expr*> dims;
  dims.push_back(ExprHandle(N).node());
  BufHandle c{new Buf("c", dims, kFloat)};
  {
    // alloc(c, 64);
    Allocate* alloc = Allocate::make(c);
    block.push_back(alloc);
  }

  {
    //    for n in 0..64:  // thread-idx
    //      c(n) = 0
    Store* store_cn_0 = Store::make(c, {n}, 0.f, 1);
    For* loop_n1 = For::make(n, 0, N, store_cn_0, thread_idx_opt);
    block.push_back(loop_n1);
  }

  {
    //  for m in 0..128:
    //    for n in 0..64:  // thread_idx
    //      c(n) = c(n) + a(k, m, n)
    ExprHandle load_cn = Load::make(kFloat, c, {n}, 1);
    ExprHandle a_kmn =
        Load::make(BufHandle(a.data()), {k * (M * N) + m * N + n}, 1);
    ExprHandle v_add = load_cn + a_kmn;
    Store* store_cn_v = Store::make(c, {n}, v_add);
    For* loop_n2 = For::make(n, 0, N, store_cn_v, thread_idx_opt);
    For* loop_m1 = For::make(m, 0, M, loop_n2);
    block.push_back(loop_m1);
  }

  {
    //    b(k) = 0
    //    for n in 0..64:  // thread_idx
    //      b(k) = b(k) + c(n)
    Store* store_bk_0 = b.store({k}, 0.f);
    block.push_back(store_bk_0);
    ExprHandle load_bk = b.load(k);
    ExprHandle load_cn = Load::make(kFloat, c, {n}, 1);
    ExprHandle v_add = load_bk + load_cn;
    Store* store_bk = b.store({k}, v_add);
    For* loop_n3 = For::make(n, 0, N, store_bk, thread_idx_opt);
    block.push_back(loop_n3);
  }

  {
    //    free(c)
    Free* free_stmt = Free::make(c);
    block.push_back(free_stmt);
  }

  Block* reduce_body = Block::make(block);
  For* loop_k1 = For::make(k, 0, 1, reduce_body, block_idx_opt);

  // TODO: check the generated code for correctness.
  CudaCodeGen cuda_cg(loop_k1, a, b);

  std::ostringstream oss;
  oss << *cuda_cg.stmt();

  // Check the c write is not masked, but the d write is.
  const std::string& verification_pattern =
      R"IR(
# CHECK: c_1 = 0
# CHECK: for (int m = 0; m < 128
# CHECK:   c_1 = c_1 +
# CHECK: __syncthreads();
# CHECK: if (threadIdx.x<1
# CHECK:   b[blockIdx.x] =
# CHECK: __syncthreads();
# CHECK: atomicAdd(&b[blockIdx.x], c_1)
)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  PaddedBuffer<float> a_v(1, M, N, "a_v");
  PaddedBuffer<float> b_v(1, "b_v");
  PaddedBuffer<float> b_ref(1, "b_ref");

  b_ref(0) = 0;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      int v = i + j;
      a_v(0, i, j) = v;
      b_ref(0) += v;
    }
  }

  float* a_dev = nullptr;
  cudaMalloc(&a_dev, kTotalSize * sizeof(float));
  cudaMemcpy(
      a_dev, a_v.data(), kTotalSize * sizeof(float), cudaMemcpyHostToDevice);
  float* b_dev = nullptr;
  cudaMalloc(&b_dev, 1 * sizeof(float));
  cudaDeviceSynchronize();

  cuda_cg(a_dev, b_dev);

  cudaDeviceSynchronize();
  cudaMemcpy(b_v.data(), b_dev, 1 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  ExpectAllNear(b_v, b_ref, 1e-5);

  cudaFree(a_dev);
  cudaFree(b_dev);
}

TEST(Cuda, LocalMemReduce_1_CUDA) {
  KernelScope kernel_scope;
  // This test does the following:
  //  for k in 0..1:  // block-idx
  //    b(k) = 0
  //    for n in 0..64:  // thread-idx
  //      alloc(c, 1)
  //      c(0) = 0
  //      for m in 0..128:
  //        c(0) = c(0) + a(k, m, n)
  //      b(k) = b(k) + c(0)
  //      free(c)

  const int M = 128;
  const int N = 64;
  const int kTotalSize = M * N;
  LoopOptions thread_idx_opt;
  thread_idx_opt.set_gpu_thread_index(0);
  LoopOptions block_idx_opt;
  block_idx_opt.set_gpu_block_index(0);

  Placeholder a("a", kFloat, {1, M, N});
  Placeholder b("b", kFloat, {1});
  VarHandle k("k", kInt);
  VarHandle m("m", kInt);
  VarHandle n("n", kInt);

  BufHandle c{new Buf("c", {new IntImm(1)}, kFloat)};
  std::vector<Stmt*> block_k;
  {
    //    b(k) = 0
    Store* store_bk_0 = b.store({k}, 0.f);
    block_k.push_back(store_bk_0);
  }
  std::vector<Stmt*> block_n;
  {
    // alloc(c, 1);
    Allocate* alloc = Allocate::make(c);
    block_n.push_back(alloc);
  }
  {
    // c(0) = 0
    Store* store_c0_0 = Store::make(c, {0}, 0.f, 1);
    block_n.push_back(store_c0_0);
  }
  {
    //      for m in 0..128:
    //        c(0) = c(0) + a(k, m, n)
    ExprHandle load_c0 = Load::make(kFloat, c, {0}, 1);
    ExprHandle a_kmn = a.load(k * (M * N) + m * N + n);
    ExprHandle v_add = load_c0 + a_kmn;
    Store* store_c0_v = Store::make(c, {0}, v_add);
    For* loop_m = For::make(m, 0, M, store_c0_v);
    block_n.push_back(loop_m);
  }
  {
    //      b(k) = b(k) + c(0)
    ExprHandle load_bk = b.load(k);
    ExprHandle load_c0 = Load::make(kFloat, c, {0}, 1);
    ExprHandle v_add = load_bk + load_c0;
    Store* store_bk = b.store({k}, v_add);
    block_n.push_back(store_bk);
  }
  {
    //      free(c)
    Free* free_stmt = Free::make(c);
    block_n.push_back(free_stmt);
  }
  {
    Block* block_n_stmt = Block::make(block_n);
    For* for_n = For::make(n, 0, N, block_n_stmt, thread_idx_opt);
    block_k.push_back(for_n);
  }
  Block* block_k_stmt = Block::make(block_k);
  For* loop_k = For::make(k, 0, 1, block_k_stmt, block_idx_opt);

  CudaCodeGen cuda_cg(loop_k, a, b);
  PaddedBuffer<float> a_v(1, M, N, "a_v");
  PaddedBuffer<float> b_v(1, "b_v");
  PaddedBuffer<float> b_ref(1, "b_ref");

  b_ref(0) = 0;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      int v = i + j;
      a_v(0, i, j) = v;
      b_ref(0) += v;
    }
  }

  float* a_dev = nullptr;
  cudaMalloc(&a_dev, kTotalSize * sizeof(float));
  cudaMemcpy(
      a_dev, a_v.data(), kTotalSize * sizeof(float), cudaMemcpyHostToDevice);
  float* b_dev = nullptr;
  cudaMalloc(&b_dev, 1 * sizeof(float));
  cudaDeviceSynchronize();

  cuda_cg(a_dev, b_dev);

  cudaDeviceSynchronize();
  cudaMemcpy(b_v.data(), b_dev, 1 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  ExpectAllNear(b_v, b_ref, 1e-5);

  cudaFree(a_dev);
  cudaFree(b_dev);
}

TEST(Cuda, HalfSupport_CUDA) {
  KernelScope ks;
  auto half = ToDtype<at::Half>();
  Placeholder a("a", half, {4});
  Tensor* b = Compute("b", {{4, "n"}}, [&](const VarHandle& i) {
    return Cast::make(half, ExprHandle(2.0f) * a.load(i));
  });

  Tensor* c = Compute("c", {{4, "n"}}, [&](const VarHandle& i) {
    return Cast::make(kFloat, Cast::make(half, ExprHandle(42)) + b->call(i));
  });

  Tensor* d = Compute("d", {{4, "n"}}, [&](const VarHandle& i) {
    return Cast::make(half, c->call(i));
  });

  LoopNest l({b, c, d});
  l.prepareForCodegen();
  Stmt* s = l.root_stmt();
  CudaCodeGen cg(s, {a, b, c, d});

  std::vector<at::Half> aData(4, 2.0f);
  std::vector<float> cData(4, 0.0f);
  std::vector<at::Half> dData(4, 0.0f);
  at::Half* aDev = nullptr;
  at::Half* bDev = nullptr;
  at::Half* cDev = nullptr;
  at::Half* dDev = nullptr;
  auto aSize = aData.size() * sizeof(aData[0]);
  auto bSize = aData.size() * sizeof(aData[0]);
  auto cSize = cData.size() * sizeof(float);
  auto dSize = dData.size() * sizeof(dData[0]);

  cudaMalloc(&aDev, aSize);
  cudaMalloc(&bDev, bSize);
  cudaMalloc(&cDev, cSize);
  cudaMalloc(&dDev, dSize);
  cudaMemcpy(aDev, aData.data(), aSize, cudaMemcpyHostToDevice);
  cudaMemcpy(cDev, cData.data(), cSize, cudaMemcpyHostToDevice);
  cudaMemcpy(dDev, dData.data(), dSize, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  cg.call({aDev, bDev, cDev, dDev});
  cudaDeviceSynchronize();

  cudaMemcpy(aData.data(), aDev, aSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(cData.data(), cDev, cSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(dData.data(), dDev, dSize, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  assertAllEqual(cData, 46.0f);

  cudaFree(aDev);
  cudaFree(bDev);
  cudaFree(cDev);
  cudaFree(dDev);
}

TEST(Cuda, HalfPropagation_CUDA) {
  KernelScope kernel_scope;
  auto half = ToDtype<at::Half>();
  Placeholder a("a", half, {4});
  Tensor* relu = Compute("relu", {{4, "n"}}, [&](const VarHandle& i) {
    return Max::make(a.load(i), ExprHandle(new HalfImm(0)), true);
  });

  LoopNest l({relu});
  l.prepareForCodegen();
  Stmt* s = l.root_stmt();
  CudaCodeGen cg(s, {a, relu});

  std::ostringstream oss;
  oss << *cg.stmt();

  // Check the types used by the Max are Float.
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (
# CHECK:  float v = float(a[n]);
# CHECK:  relu[n] = half(Max(v, 0.f
# CHECK: })IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  std::vector<at::Half> aData(4, 2.0f);
  std::vector<at::Half> reluData(4, 0.0f);
  at::Half* aDev = nullptr;
  at::Half* reluDev = nullptr;
  auto aSize = aData.size() * sizeof(aData[0]);
  auto reluSize = reluData.size() * sizeof(reluData[0]);

  cudaMalloc(&aDev, aSize);
  cudaMalloc(&reluDev, reluSize);
  cudaMemcpy(aDev, aData.data(), aSize, cudaMemcpyHostToDevice);
  cudaMemcpy(reluDev, reluData.data(), reluSize, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  cg.call({aDev, reluDev});
  cudaMemcpy(reluData.data(), reluDev, reluSize, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  assertAllEqual(aData, reluData);

  cudaFree(aDev);
  cudaFree(reluDev);
}

TEST(Cuda, UnusedHalfArgument_CUDA) {
  KernelScope kernel_scope;
  Placeholder a("a", kFloat, {4});
  auto half = ToDtype<at::Half>();
  Placeholder b("b", half, {4});
  Tensor* relu = Compute("relu", {{4, "n"}}, [&](const VarHandle& i) {
    return Max::make(a.load(i), ExprHandle(new FloatImm(0)), true);
  });

  LoopNest l({relu});
  l.prepareForCodegen();
  Stmt* s = l.root_stmt();
  CudaCodeGen cg(s, {a, b, relu});

  std::ostringstream oss;
  oss << *cg.stmt();

  // Check the types used by the Max are Float.
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (
# CHECK:  float v = a[n];
# CHECK:  relu[n] = Max(v, 0.f
# CHECK: })IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // Sanity Cbeck;
  std::vector<float> aData(4, 2.0f);
  std::vector<at::Half> bData(4, 2.0f);
  std::vector<float> reluData(4, 0.0f);
  at::Half* aDev = nullptr;
  at::Half* bDev = nullptr;
  at::Half* reluDev = nullptr;
  auto aSize = aData.size() * sizeof(aData[0]);
  auto bSize = bData.size() * sizeof(bData[0]);
  auto reluSize = reluData.size() * sizeof(reluData[0]);

  cudaMalloc(&aDev, aSize);
  cudaMalloc(&bDev, bSize);
  cudaMalloc(&reluDev, reluSize);
  cudaMemcpy(aDev, aData.data(), aSize, cudaMemcpyHostToDevice);
  cudaMemcpy(bDev, bData.data(), bSize, cudaMemcpyHostToDevice);
  cudaMemcpy(reluDev, reluData.data(), reluSize, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  cg.call({aDev, bDev, reluDev});
  cudaMemcpy(reluData.data(), reluDev, reluSize, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  assertAllEqual(aData, reluData);

  cudaFree(aDev);
  cudaFree(bDev);
  cudaFree(reluDev);
}

TEST(Cuda, PrioritizeDependents_CUDA) {
  KernelScope kernel_scope;
  Placeholder a("a", kFloat, {10});
  Placeholder b("b", kFloat, {12});
  Placeholder c("c", kFloat, {12});

  LoopOptions block_idx_opt;
  block_idx_opt.set_gpu_block_index(0);

  VarHandle i("i", kInt);
  VarHandle j("j", kInt);

  /*
   * for (int i = 0; i < 12; ++i) {
   *   c[i] = (i < 10 ? a[i] + b[i] : b[i]);
   * }
   */
  ExprHandle load_a = Load::make(BufHandle(a.data()), {i}, 1);
  ExprHandle load_b = Load::make(BufHandle(b.data()), {i}, 1);
  ExprHandle cmp = CompareSelect::make(i, 10, CompareSelectOperation::kLT);
  ExprHandle ite = IfThenElse::make(cmp, Add::make(load_a, load_b), load_b);

  For* loop =
      For::make(i, 0, 12, Block::make({c.store({i}, ite)}), block_idx_opt);

  CudaCodeGen cuda_cg(loop, a, b, c);

  PaddedBuffer<float> a_v(10, "a_v");
  PaddedBuffer<float> b_v(12, "b_v");
  PaddedBuffer<float> c_v(12, "c_v");
  PaddedBuffer<float> c_ref(12, "c_ref");

  for (int i = 0; i < 10; ++i) {
    a_v(i) = i * 100;
    b_v(i) = i;
    c_v(i) = 0;
  }

  for (int i = 10; i < 12; ++i) {
    b_v(i) = i;
    c_v(i) = 0;
  }

  float* a_dev = nullptr;
  float* b_dev = nullptr;
  float* c_dev = nullptr;
  cudaMalloc(&a_dev, 10 * sizeof(float));
  cudaMalloc(&b_dev, 12 * sizeof(float));
  cudaMalloc(&c_dev, 12 * sizeof(float));

  cudaMemcpy(a_dev, a_v.data(), 10 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b_dev, b_v.data(), 12 * sizeof(float), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();

  cuda_cg(a_dev, b_dev, c_dev);

  cudaDeviceSynchronize();
  cudaMemcpy(c_v.data(), c_dev, 12 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  for (int i = 0; i < 12; ++i) {
    if (i < 10) {
      c_ref(i) = i + i * 100;
    } else {
      c_ref(i) = i;
    }
  }

  ExpectAllNear(c_v, c_ref, 1e-5);
}

/// Tests the case where there are two loops which have different extents bound
/// to the same block dimension. We must mask the smaller extent loop body.
TEST(Cuda, MaskBlockDim_CUDA) {
  KernelScope kernel_scope;
  int A_SIZE = 100;
  int B_SIZE = 50;
  Placeholder a_buf("a", kFloat, {A_SIZE});
  Placeholder b_buf("b", kFloat, {B_SIZE});
  Tensor* c = Compute("c", {{A_SIZE, "i"}}, [&](const VarHandle& i) {
    return a_buf.load(i) + 10;
  });
  Tensor* d = Compute("d", {{B_SIZE, "i"}}, [&](const VarHandle& i) {
    return a_buf.load(i) + b_buf.load(i);
  });

  LoopNest l({c, d});
  std::vector<For*> loops = l.getLoopStmtsFor(c);
  l.setGPUBlockIndex(loops[0], 0);
  loops = l.getLoopStmtsFor(d);
  l.setGPUBlockIndex(loops[0], 0);

  l.prepareForCodegen();
  Stmt* stmt = l.root_stmt();
  CudaCodeGen cuda_cg(stmt, c, d, a_buf, b_buf);

  std::ostringstream oss;
  oss << *cuda_cg.stmt();

  // Check the c write is not masked, but the d write is.
  const std::string& verification_pattern =
      R"IR(
# CHECK-NOT: if (blockIdx
# CHECK: c[blockIdx.x] =
# CHECK: if (blockIdx.x<50
# CHECK:   d[blockIdx.x] =)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  auto blockExtents = cuda_cg.gpu_block_extents();
  auto threadExtents = cuda_cg.gpu_thread_extents();
  ASSERT_TRUE(exprEquals(blockExtents[0], new IntImm(A_SIZE)));
  ASSERT_TRUE(exprEquals(threadExtents[0], new IntImm(1)));

  // Sanity check that the kernel works.
  PaddedBuffer<float> a_v(A_SIZE);
  PaddedBuffer<float> b_v(B_SIZE);
  PaddedBuffer<float> c_v(A_SIZE);
  PaddedBuffer<float> d_v(B_SIZE);

  PaddedBuffer<float> c_ref(A_SIZE);
  PaddedBuffer<float> d_ref(B_SIZE);

  for (int i = 0; i < A_SIZE; i++) {
    a_v(i) = (float)i;
    c_ref(i) = (float)(i + 10);
  }

  for (int i = 0; i < B_SIZE; i++) {
    b_v(i) = (float)(B_SIZE - i);
    d_ref(i) = a_v(i) + b_v(i);
  }

  float* a_dev = nullptr;
  cudaMalloc(&a_dev, A_SIZE * sizeof(float));
  float* b_dev = nullptr;
  cudaMalloc(&b_dev, B_SIZE * sizeof(float));
  float* c_dev = nullptr;
  cudaMalloc(&c_dev, A_SIZE * sizeof(float));
  float* d_dev = nullptr;
  cudaMalloc(&d_dev, B_SIZE * sizeof(float));
  cudaMemcpy(a_dev, a_v.data(), A_SIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b_dev, b_v.data(), B_SIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(c_dev, c_v.data(), A_SIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dev, d_v.data(), B_SIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  cuda_cg(c_dev, d_dev, a_dev, b_dev);

  cudaDeviceSynchronize();
  cudaMemcpy(c_v.data(), c_dev, A_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(d_v.data(), d_dev, B_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  ExpectAllNear(c_v, c_ref, 1e-5);
  ExpectAllNear(d_v, d_ref, 1e-5);

  cudaFree(a_dev);
  cudaFree(b_dev);
  cudaFree(c_dev);
  cudaFree(d_dev);
}

/// Tests the case with two loops, which have different extents that are bound
/// to the same thread dimension. This is the same as the above - the smaller
/// rank write should be masked. But this time we also need to syncthreads.
TEST(Cuda, MaskThreadDim_CUDA) {
  KernelScope kernel_scope;
  int A_SIZE = 50;
  int B_SIZE = 100;
  Placeholder a_buf("a", kFloat, {A_SIZE});
  Placeholder b_buf("b", kFloat, {B_SIZE});
  Tensor* c = Compute("c", {{A_SIZE, "i"}}, [&](const VarHandle& i) {
    return a_buf.load(i) + 10;
  });
  Tensor* d = Compute("d", {{B_SIZE, "i"}}, [&](const VarHandle& i) {
    return a_buf.load(i / 2) + b_buf.load(i);
  });

  LoopNest l({c, d});
  std::vector<For*> loops = l.getLoopStmtsFor(c);
  l.setGPUThreadIndex(loops[0], 0);
  loops = l.getLoopStmtsFor(d);
  l.setGPUThreadIndex(loops[0], 0);

  l.prepareForCodegen();
  Stmt* stmt = l.root_stmt();
  CudaCodeGen cuda_cg(stmt, c, d, a_buf, b_buf);

  std::ostringstream oss;
  oss << *cuda_cg.stmt();

  // Check the c write is masked, but the d write is not.
  const std::string& verification_pattern =
      R"IR(
# CHECK: if (threadIdx.x<50
# CHECK:   c[threadIdx.x] =
# CHECK: __syncthreads();
# CHECK-NOT: if (threadIdx.x
# CHECK: d[threadIdx.x] =)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  auto blockExtents = cuda_cg.gpu_block_extents();
  auto threadExtents = cuda_cg.gpu_thread_extents();
  ASSERT_TRUE(exprEquals(blockExtents[0], new IntImm(1)));
  ASSERT_TRUE(exprEquals(threadExtents[0], new IntImm(B_SIZE)));

  PaddedBuffer<float> a_v(A_SIZE);
  PaddedBuffer<float> b_v(B_SIZE);
  PaddedBuffer<float> c_v(A_SIZE);
  PaddedBuffer<float> d_v(B_SIZE);

  PaddedBuffer<float> c_ref(A_SIZE);
  PaddedBuffer<float> d_ref(B_SIZE);

  for (int i = 0; i < A_SIZE; i++) {
    a_v(i) = (float)i;
    c_ref(i) = (float)(i + 10);
  }

  for (int i = 0; i < B_SIZE; i++) {
    b_v(i) = (float)(B_SIZE - i);
    d_ref(i) = a_v(i / 2) + b_v(i);
  }

  float* a_dev = nullptr;
  cudaMalloc(&a_dev, A_SIZE * sizeof(float));
  float* b_dev = nullptr;
  cudaMalloc(&b_dev, B_SIZE * sizeof(float));
  float* c_dev = nullptr;
  cudaMalloc(&c_dev, A_SIZE * sizeof(float));
  float* d_dev = nullptr;
  cudaMalloc(&d_dev, B_SIZE * sizeof(float));
  cudaMemcpy(a_dev, a_v.data(), A_SIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b_dev, b_v.data(), B_SIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(c_dev, c_v.data(), A_SIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dev, d_v.data(), B_SIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  cuda_cg(c_dev, d_dev, a_dev, b_dev);

  cudaDeviceSynchronize();
  cudaMemcpy(c_v.data(), c_dev, A_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(d_v.data(), d_dev, B_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  ExpectAllNear(c_v, c_ref, 1e-5);
  ExpectAllNear(d_v, d_ref, 1e-5);

  cudaFree(a_dev);
  cudaFree(b_dev);
  cudaFree(c_dev);
  cudaFree(d_dev);
}

/// Tests the case where there are two loops, and each is bound to a different
/// block dimension. In this case all writes should be masked since they occur
/// in distinct dimensions.
// Note: this is an extremely dumb pattern which we should never see, but is a
// useful edge case to make sure we've got things covered.
TEST(Cuda, MaskMultiBlockDim_CUDA) {
  KernelScope kernel_scope;
  int A_SIZE = 100;
  int B_SIZE = 50;
  Placeholder a_buf("a", kFloat, {A_SIZE});
  Placeholder b_buf("b", kFloat, {B_SIZE});
  Tensor* c = Compute("c", {{A_SIZE, "i"}}, [&](const VarHandle& i) {
    return a_buf.load(i) + 10;
  });
  Tensor* d = Compute("d", {{B_SIZE, "i"}}, [&](const VarHandle& i) {
    return a_buf.load(i) + b_buf.load(i);
  });

  LoopNest l({c, d});
  std::vector<For*> loops = l.getLoopStmtsFor(c);
  l.setGPUBlockIndex(loops[0], 0);
  loops = l.getLoopStmtsFor(d);
  l.setGPUBlockIndex(loops[0], 1);

  l.prepareForCodegen();
  Stmt* stmt = l.root_stmt();
  CudaCodeGen cuda_cg(stmt, c, d, a_buf, b_buf);

  std::ostringstream oss;
  oss << *cuda_cg.stmt();

  // Write to c should be masked against y, write to d against x.
  const std::string& verification_pattern =
      R"IR(
# CHECK: if (blockIdx.y<1
# CHECK:   c[blockIdx.x] =
# CHECK: if (blockIdx.x<1
# CHECK:   d[blockIdx.y] =)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  auto blockExtents = cuda_cg.gpu_block_extents();
  auto threadExtents = cuda_cg.gpu_thread_extents();
  ASSERT_TRUE(exprEquals(blockExtents[0], new IntImm(A_SIZE)));
  ASSERT_TRUE(exprEquals(blockExtents[1], new IntImm(B_SIZE)));

  PaddedBuffer<float> a_v(A_SIZE);
  PaddedBuffer<float> b_v(B_SIZE);
  PaddedBuffer<float> c_v(A_SIZE);
  PaddedBuffer<float> d_v(B_SIZE);

  PaddedBuffer<float> c_ref(A_SIZE);
  PaddedBuffer<float> d_ref(B_SIZE);

  for (int i = 0; i < A_SIZE; i++) {
    a_v(i) = (float)i;
    c_ref(i) = (float)(i + 10);
  }

  for (int i = 0; i < B_SIZE; i++) {
    b_v(i) = (float)(B_SIZE - i);
    d_ref(i) = a_v(i) + b_v(i);
  }

  float* a_dev = nullptr;
  cudaMalloc(&a_dev, A_SIZE * sizeof(float));
  float* b_dev = nullptr;
  cudaMalloc(&b_dev, B_SIZE * sizeof(float));
  float* c_dev = nullptr;
  cudaMalloc(&c_dev, A_SIZE * sizeof(float));
  float* d_dev = nullptr;
  cudaMalloc(&d_dev, B_SIZE * sizeof(float));
  cudaMemcpy(a_dev, a_v.data(), A_SIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b_dev, b_v.data(), B_SIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(c_dev, c_v.data(), A_SIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dev, d_v.data(), B_SIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  cuda_cg(c_dev, d_dev, a_dev, b_dev);

  cudaDeviceSynchronize();
  cudaMemcpy(c_v.data(), c_dev, A_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(d_v.data(), d_dev, B_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  ExpectAllNear(c_v, c_ref, 1e-5);
  ExpectAllNear(d_v, d_ref, 1e-5);

  cudaFree(a_dev);
  cudaFree(b_dev);
  cudaFree(c_dev);
  cudaFree(d_dev);
}

/// Tests the case where both the blockDim and threadDim are bound to different
/// loops. In this instance both stores should be masked since they are
/// distinct.
// Note: this is an extremely dumb pattern which we should never see, but is a
// useful edge case to make sure we've got things covered.
TEST(Cuda, MaskBlockAndThreadDim_CUDA) {
  KernelScope kernel_scope;
  int A_SIZE = 100;
  int B_SIZE = 50;
  Placeholder a_buf("a", kFloat, {A_SIZE});
  Placeholder b_buf("b", kFloat, {B_SIZE});
  Tensor* c = Compute("c", {{A_SIZE, "i"}}, [&](const VarHandle& i) {
    return a_buf.load(i) + 10;
  });
  Tensor* d = Compute("d", {{B_SIZE, "i"}}, [&](const VarHandle& i) {
    return a_buf.load(i) + b_buf.load(i);
  });

  LoopNest l({c, d});
  std::vector<For*> loops = l.getLoopStmtsFor(c);
  l.setGPUBlockIndex(loops[0], 0);
  loops = l.getLoopStmtsFor(d);
  l.setGPUThreadIndex(loops[0], 0);

  l.prepareForCodegen();
  Stmt* stmt = l.root_stmt();
  CudaCodeGen cuda_cg(stmt, c, d, a_buf, b_buf);

  std::ostringstream oss;
  oss << *cuda_cg.stmt();

  const std::string& verification_pattern =
      R"IR(
# CHECK: if (threadIdx.x<1
# CHECK:   c[blockIdx.x] =
# CHECK: }
# CHECK: if (blockIdx.x<1
# CHECK:   d[threadIdx.x] =)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  auto blockExtents = cuda_cg.gpu_block_extents();
  auto threadExtents = cuda_cg.gpu_thread_extents();
  ASSERT_TRUE(exprEquals(blockExtents[0], new IntImm(A_SIZE)));
  ASSERT_TRUE(exprEquals(threadExtents[0], new IntImm(B_SIZE)));

  PaddedBuffer<float> a_v(A_SIZE);
  PaddedBuffer<float> b_v(B_SIZE);
  PaddedBuffer<float> c_v(A_SIZE);
  PaddedBuffer<float> d_v(B_SIZE);

  PaddedBuffer<float> c_ref(A_SIZE);
  PaddedBuffer<float> d_ref(B_SIZE);

  for (int i = 0; i < A_SIZE; i++) {
    a_v(i) = (float)i;
    c_ref(i) = (float)(i + 10);
  }

  for (int i = 0; i < B_SIZE; i++) {
    b_v(i) = (float)(B_SIZE - i);
    d_ref(i) = a_v(i) + b_v(i);
  }

  float* a_dev = nullptr;
  cudaMalloc(&a_dev, A_SIZE * sizeof(float));
  float* b_dev = nullptr;
  cudaMalloc(&b_dev, B_SIZE * sizeof(float));
  float* c_dev = nullptr;
  cudaMalloc(&c_dev, A_SIZE * sizeof(float));
  float* d_dev = nullptr;
  cudaMalloc(&d_dev, B_SIZE * sizeof(float));
  cudaMemcpy(a_dev, a_v.data(), A_SIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b_dev, b_v.data(), B_SIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(c_dev, c_v.data(), A_SIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dev, d_v.data(), B_SIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  cuda_cg(c_dev, d_dev, a_dev, b_dev);

  cudaDeviceSynchronize();
  cudaMemcpy(c_v.data(), c_dev, A_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(d_v.data(), d_dev, B_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  ExpectAllNear(c_v, c_ref, 1e-5);
  ExpectAllNear(d_v, d_ref, 1e-5);

  cudaFree(a_dev);
  cudaFree(b_dev);
  cudaFree(c_dev);
  cudaFree(d_dev);
}

/// Tests the case where the loopnest has two loops of depth two: each with the
/// outer loop bound to blockDim.x and the inner loop bound to threadDim.x. In
/// this case all writes with a rank smaller than the max should be masked.
TEST(Cuda, MaskMultiDim_CUDA) {
  KernelScope kernel_scope;
  int OUTER_SIZE = 10;
  int A_SIZE = 100;
  int B_SIZE = 50;
  Placeholder a_buf("a", kFloat, {OUTER_SIZE, A_SIZE});
  Placeholder b_buf("b", kFloat, {OUTER_SIZE, B_SIZE});
  Tensor* c = Compute(
      "C",
      {{OUTER_SIZE, "i"}, {A_SIZE, "j"}},
      [&](const VarHandle& i, const VarHandle& j) {
        return ExprHandle(2) * a_buf.load(i, j);
      });
  Tensor* d = Compute(
      "D",
      {{OUTER_SIZE, "i"}, {B_SIZE, "j"}},
      [&](const VarHandle& i, const VarHandle& j) {
        return c->call(i, j * 2) + b_buf.load(i, j);
      });

  LoopNest l({c, d});
  std::vector<For*> loops = l.getLoopStmtsFor(c);
  l.setGPUBlockIndex(loops[0], 0);
  l.setGPUThreadIndex(loops[1], 0);
  loops = l.getLoopStmtsFor(d);
  l.setGPUBlockIndex(loops[0], 0);
  l.setGPUThreadIndex(loops[1], 0);

  l.prepareForCodegen();
  Stmt* stmt = l.root_stmt();
  CudaCodeGen cuda_cg(stmt, c, d, a_buf, b_buf);

  std::ostringstream oss;
  oss << *cuda_cg.stmt();

  // The write to D should be masked, but not the write to C.
  const std::string& verification_pattern =
      R"IR(
# CHECK-NOT: if (
# CHECK: C[100 * blockIdx.x + threadIdx.x] =
# CHECK: __syncthreads();
# CHECK: if (threadIdx.x<50
# CHECK:   D[50 * blockIdx.x + threadIdx.x] =)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  auto blockExtents = cuda_cg.gpu_block_extents();
  auto threadExtents = cuda_cg.gpu_thread_extents();
  ASSERT_TRUE(exprEquals(blockExtents[0], new IntImm(OUTER_SIZE)));
  ASSERT_TRUE(exprEquals(threadExtents[0], new IntImm(A_SIZE)));

  PaddedBuffer<float> a_v(OUTER_SIZE, A_SIZE);
  PaddedBuffer<float> b_v(OUTER_SIZE, B_SIZE);
  PaddedBuffer<float> c_v(OUTER_SIZE, A_SIZE);
  PaddedBuffer<float> d_v(OUTER_SIZE, B_SIZE);

  PaddedBuffer<float> c_ref(OUTER_SIZE, A_SIZE);
  PaddedBuffer<float> d_ref(OUTER_SIZE, B_SIZE);

  for (int o = 0; o < OUTER_SIZE; ++o) {
    for (int i = 0; i < A_SIZE; i++) {
      a_v(o, i) = (float)i;
      c_ref(o, i) = (float)(i * 2);
    }
  }

  for (int o = 0; o < OUTER_SIZE; ++o) {
    for (int i = 0; i < B_SIZE; i++) {
      b_v(o, i) = (float)(B_SIZE - i);
      d_ref(o, i) = c_ref(o, i * 2) + b_v(o, i);
    }
  }

  float* a_dev = nullptr;
  cudaMalloc(&a_dev, OUTER_SIZE * A_SIZE * sizeof(float));
  float* b_dev = nullptr;
  cudaMalloc(&b_dev, OUTER_SIZE * B_SIZE * sizeof(float));
  float* c_dev = nullptr;
  cudaMalloc(&c_dev, OUTER_SIZE * A_SIZE * sizeof(float));
  float* d_dev = nullptr;
  cudaMalloc(&d_dev, OUTER_SIZE * B_SIZE * sizeof(float));
  cudaMemcpy(
      a_dev,
      a_v.data(),
      OUTER_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyHostToDevice);
  cudaMemcpy(
      b_dev,
      b_v.data(),
      OUTER_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyHostToDevice);
  cudaMemcpy(
      c_dev,
      c_v.data(),
      OUTER_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyHostToDevice);
  cudaMemcpy(
      d_dev,
      d_v.data(),
      OUTER_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  cuda_cg(c_dev, d_dev, a_dev, b_dev);

  cudaDeviceSynchronize();
  cudaMemcpy(
      c_v.data(),
      c_dev,
      OUTER_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyDeviceToHost);
  cudaMemcpy(
      d_v.data(),
      d_dev,
      OUTER_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  ExpectAllNear(c_v, c_ref, 1e-5);
  ExpectAllNear(d_v, d_ref, 1e-5);

  cudaFree(a_dev);
  cudaFree(b_dev);
  cudaFree(c_dev);
  cudaFree(d_dev);
}

// Tests the case where loop extents are symbolic and not known at compile time.
// In this case both stores must be masked against the extent of the other loop,
// incase it is larger.
TEST(Cuda, MaskMultiDimSymbolic_CUDA) {
  KernelScope kernel_scope;
  VarHandle OUTER_SIZE("OUTER_SIZE", kInt);
  VarHandle A_SIZE("A_SIZE", kInt);
  VarHandle B_SIZE("B_SIZE", kInt);
  Placeholder a_buf("a", kFloat, {OUTER_SIZE, A_SIZE});
  Placeholder b_buf("b", kFloat, {OUTER_SIZE, B_SIZE});
  Tensor* c = Compute(
      "C",
      {{OUTER_SIZE, "i"}, {A_SIZE, "j"}},
      [&](const VarHandle& i, const VarHandle& j) {
        return ExprHandle(2) * a_buf.load(i, j);
      });
  Tensor* d = Compute(
      "D",
      {{OUTER_SIZE, "i"}, {B_SIZE, "j"}},
      [&](const VarHandle& i, const VarHandle& j) {
        return c->call(i, j * 2) + b_buf.load(i, j);
      });

  LoopNest l({c, d});
  std::vector<For*> loops = l.getLoopStmtsFor(c);
  l.setGPUBlockIndex(loops[0], 0);
  l.setGPUThreadIndex(loops[1], 0);
  loops = l.getLoopStmtsFor(d);
  l.setGPUBlockIndex(loops[0], 0);
  l.setGPUThreadIndex(loops[1], 0);

  l.prepareForCodegen();
  Stmt* stmt = l.root_stmt();
  CudaCodeGen cuda_cg(stmt, c, d, OUTER_SIZE, A_SIZE, B_SIZE, a_buf, b_buf);

  std::ostringstream oss;
  oss << *cuda_cg.stmt();

  // Since we don't know which is bigger (A_SIZE or B_SIZE) we must mask both.
  const std::string& verification_pattern =
      R"IR(
# CHECK: if (threadIdx.x<A_SIZE
# CHECK:   C[threadIdx.x + A_SIZE * blockIdx.x] =
# CHECK: __syncthreads();
# CHECK: if (threadIdx.x<B_SIZE
# CHECK:   D[threadIdx.x + B_SIZE * blockIdx.x] =)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  auto blockExtents = cuda_cg.gpu_block_extents();
  auto threadExtents = cuda_cg.gpu_thread_extents();
  ASSERT_TRUE(exprEquals(blockExtents[0], OUTER_SIZE.node()));
  ASSERT_TRUE(exprEquals(
      threadExtents[0], new Max(A_SIZE.node(), B_SIZE.node(), true)));

  int OUTER_EXTENT = 10;
  int A_EXTENT = 100;
  int B_EXTENT = 50;

  PaddedBuffer<float> a_v(OUTER_EXTENT, A_EXTENT);
  PaddedBuffer<float> b_v(OUTER_EXTENT, B_EXTENT);
  PaddedBuffer<float> c_v(OUTER_EXTENT, A_EXTENT);
  PaddedBuffer<float> d_v(OUTER_EXTENT, B_EXTENT);

  PaddedBuffer<float> c_ref(OUTER_EXTENT, A_EXTENT);
  PaddedBuffer<float> d_ref(OUTER_EXTENT, B_EXTENT);

  for (int o = 0; o < OUTER_EXTENT; ++o) {
    for (int i = 0; i < A_EXTENT; i++) {
      a_v(o, i) = (float)i;
      c_ref(o, i) = (float)(i * 2);
    }
  }

  for (int o = 0; o < OUTER_EXTENT; ++o) {
    for (int i = 0; i < B_EXTENT; i++) {
      b_v(o, i) = (float)(B_EXTENT - i);
      d_ref(o, i) = c_ref(o, i * 2) + b_v(o, i);
    }
  }

  float* a_dev = nullptr;
  cudaMalloc(&a_dev, OUTER_EXTENT * A_EXTENT * sizeof(float));
  float* b_dev = nullptr;
  cudaMalloc(&b_dev, OUTER_EXTENT * B_EXTENT * sizeof(float));
  float* c_dev = nullptr;
  cudaMalloc(&c_dev, OUTER_EXTENT * A_EXTENT * sizeof(float));
  float* d_dev = nullptr;
  cudaMalloc(&d_dev, OUTER_EXTENT * B_EXTENT * sizeof(float));
  cudaMemcpy(
      a_dev,
      a_v.data(),
      OUTER_EXTENT * A_EXTENT * sizeof(float),
      cudaMemcpyHostToDevice);
  cudaMemcpy(
      b_dev,
      b_v.data(),
      OUTER_EXTENT * B_EXTENT * sizeof(float),
      cudaMemcpyHostToDevice);
  cudaMemcpy(
      c_dev,
      c_v.data(),
      OUTER_EXTENT * A_EXTENT * sizeof(float),
      cudaMemcpyHostToDevice);
  cudaMemcpy(
      d_dev,
      d_v.data(),
      OUTER_EXTENT * B_EXTENT * sizeof(float),
      cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  cuda_cg(c_dev, d_dev, OUTER_EXTENT, A_EXTENT, B_EXTENT, a_dev, b_dev);

  cudaDeviceSynchronize();
  cudaMemcpy(
      c_v.data(),
      c_dev,
      OUTER_EXTENT * A_EXTENT * sizeof(float),
      cudaMemcpyDeviceToHost);
  cudaMemcpy(
      d_v.data(),
      d_dev,
      OUTER_EXTENT * B_EXTENT * sizeof(float),
      cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  ExpectAllNear(c_v, c_ref, 1e-5);
  ExpectAllNear(d_v, d_ref, 1e-5);

  cudaFree(a_dev);
  cudaFree(b_dev);
  cudaFree(c_dev);
  cudaFree(d_dev);
}

// Tests the case where two loops are fused at a common parent loop, which is
// bound to the block dimension. Internally the inner loops have different
// extents but are bound to the same thread dimension. The smaller loop should
// be masked.
TEST(Cuda, MaskCompoundInnerLoop_CUDA) {
  KernelScope kernel_scope;
  int OUTER_SIZE = 10;
  int A_SIZE = 100;
  int B_SIZE = 50;
  Placeholder a_buf("a", kFloat, {OUTER_SIZE, A_SIZE});
  Placeholder b_buf("b", kFloat, {OUTER_SIZE, B_SIZE});
  Placeholder c_buf("c", kFloat, {OUTER_SIZE, A_SIZE});
  Placeholder d_buf("d", kFloat, {OUTER_SIZE, B_SIZE});

  // Can't build this using Compute and transforms yet.
  LoopOptions blockBound;
  blockBound.set_gpu_block_index(0);
  LoopOptions threadBound;
  threadBound.set_gpu_thread_index(0);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);

  Stmt* stmt = For::make(
      i,
      0,
      OUTER_SIZE,
      Block::make(
          {For::make(
               j,
               0,
               A_SIZE,
               c_buf.store({i, j}, ExprHandle(2) * a_buf.load(i, j)),
               threadBound),
           For::make(
               k,
               0,
               B_SIZE,
               d_buf.store({i, k}, c_buf.load(i, k * 2) + b_buf.load(i, k)),
               threadBound)}),
      blockBound);

  stmt = FlattenIndexes(stmt);
  stmt = IRSimplifier::simplify(stmt);

  CudaCodeGen cuda_cg(stmt, a_buf, b_buf, c_buf, d_buf);

  std::ostringstream oss;
  oss << *cuda_cg.stmt();

  // The write to D should be masked, but not the write to C.
  const std::string& verification_pattern =
      R"IR(
# CHECK-NOT: if (
# CHECK: c[100 * blockIdx.x + threadIdx.x] =
# CHECK: __syncthreads();
# CHECK: if (threadIdx.x<50
# CHECK:   d[50 * blockIdx.x + threadIdx.x] =)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  auto blockExtents = cuda_cg.gpu_block_extents();
  auto threadExtents = cuda_cg.gpu_thread_extents();
  ASSERT_TRUE(exprEquals(blockExtents[0], new IntImm(OUTER_SIZE)));
  ASSERT_TRUE(exprEquals(threadExtents[0], new IntImm(A_SIZE)));

  PaddedBuffer<float> a_v(OUTER_SIZE, A_SIZE);
  PaddedBuffer<float> b_v(OUTER_SIZE, B_SIZE);
  PaddedBuffer<float> c_v(OUTER_SIZE, A_SIZE);
  PaddedBuffer<float> d_v(OUTER_SIZE, B_SIZE);

  PaddedBuffer<float> c_ref(OUTER_SIZE, A_SIZE);
  PaddedBuffer<float> d_ref(OUTER_SIZE, B_SIZE);

  for (int o = 0; o < OUTER_SIZE; ++o) {
    for (int i = 0; i < A_SIZE; i++) {
      a_v(o, i) = (float)i;
      c_ref(o, i) = (float)(i * 2);
    }
    for (int i = 0; i < B_SIZE; i++) {
      b_v(o, i) = (float)(B_SIZE - i);
      d_ref(o, i) = c_ref(o, i * 2) + b_v(o, i);
    }
  }

  float* a_dev = nullptr;
  cudaMalloc(&a_dev, OUTER_SIZE * A_SIZE * sizeof(float));
  float* b_dev = nullptr;
  cudaMalloc(&b_dev, OUTER_SIZE * B_SIZE * sizeof(float));
  float* c_dev = nullptr;
  cudaMalloc(&c_dev, OUTER_SIZE * A_SIZE * sizeof(float));
  float* d_dev = nullptr;
  cudaMalloc(&d_dev, OUTER_SIZE * B_SIZE * sizeof(float));
  cudaMemcpy(
      a_dev,
      a_v.data(),
      OUTER_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyHostToDevice);
  cudaMemcpy(
      b_dev,
      b_v.data(),
      OUTER_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyHostToDevice);
  cudaMemcpy(
      c_dev,
      c_v.data(),
      OUTER_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyHostToDevice);
  cudaMemcpy(
      d_dev,
      d_v.data(),
      OUTER_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  cuda_cg(a_dev, b_dev, c_dev, d_dev);

  cudaDeviceSynchronize();
  cudaMemcpy(
      c_v.data(),
      c_dev,
      OUTER_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyDeviceToHost);
  cudaMemcpy(
      d_v.data(),
      d_dev,
      OUTER_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  ExpectAllNear(c_v, c_ref, 1e-5);
  ExpectAllNear(d_v, d_ref, 1e-5);

  cudaFree(a_dev);
  cudaFree(b_dev);
  cudaFree(c_dev);
  cudaFree(d_dev);
}

// Tests the case with two loops fused into a common parent, which is not bound
// to any block or thread dimension - however it's two inner loops are bound to
// the first thread dimensions. This should work just like the MaskThreadDim
// test where the bigger loop is unmasked but the smaller is masked.
TEST(Cuda, MaskInnerLoopOneBlock_CUDA) {
  KernelScope kernel_scope;
  int OUTER_SIZE = 10;
  int A_SIZE = 100;
  int B_SIZE = 50;
  Placeholder a_buf("a", kFloat, {OUTER_SIZE, A_SIZE});
  Placeholder b_buf("b", kFloat, {OUTER_SIZE, B_SIZE});
  Placeholder c_buf("c", kFloat, {OUTER_SIZE, A_SIZE});
  Placeholder d_buf("d", kFloat, {OUTER_SIZE, B_SIZE});

  // Can't build this using Compute and transforms yet.
  LoopOptions blockBound;
  blockBound.set_gpu_block_index(0);
  LoopOptions threadBound;
  threadBound.set_gpu_thread_index(0);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);

  Stmt* stmt = For::make(
      i,
      0,
      OUTER_SIZE,
      Block::make(
          {For::make(
               j,
               0,
               A_SIZE,
               c_buf.store({i, j}, ExprHandle(2) * a_buf.load(i, j)),
               threadBound),
           For::make(
               k,
               0,
               B_SIZE,
               d_buf.store({i, k}, c_buf.load(i, k * 2) + b_buf.load(i, k)),
               threadBound)}));

  stmt = FlattenIndexes(stmt);
  stmt = IRSimplifier::simplify(stmt);

  CudaCodeGen cuda_cg(stmt, a_buf, b_buf, c_buf, d_buf);

  std::ostringstream oss;
  oss << *cuda_cg.stmt();

  // The other loop remains the D write is masked.
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i = 0; i < 10
# CHECK-NOT: if (
# CHECK: c[100 * i + threadIdx.x] =
# CHECK: __syncthreads();
# CHECK: if (threadIdx.x<50
# CHECK:   d[50 * i + threadIdx.x] =)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  auto blockExtents = cuda_cg.gpu_block_extents();
  auto threadExtents = cuda_cg.gpu_thread_extents();
  ASSERT_TRUE(exprEquals(blockExtents[0], new IntImm(1)));
  ASSERT_TRUE(exprEquals(threadExtents[0], new IntImm(A_SIZE)));

  PaddedBuffer<float> a_v(OUTER_SIZE, A_SIZE);
  PaddedBuffer<float> b_v(OUTER_SIZE, B_SIZE);
  PaddedBuffer<float> c_v(OUTER_SIZE, A_SIZE);
  PaddedBuffer<float> d_v(OUTER_SIZE, B_SIZE);

  PaddedBuffer<float> c_ref(OUTER_SIZE, A_SIZE);
  PaddedBuffer<float> d_ref(OUTER_SIZE, B_SIZE);

  for (int o = 0; o < OUTER_SIZE; ++o) {
    for (int i = 0; i < A_SIZE; i++) {
      a_v(o, i) = (float)i;
      c_ref(o, i) = (float)(i * 2);
    }
    for (int i = 0; i < B_SIZE; i++) {
      b_v(o, i) = (float)(B_SIZE - i);
      d_ref(o, i) = c_ref(o, i * 2) + b_v(o, i);
    }
  }

  float* a_dev = nullptr;
  cudaMalloc(&a_dev, OUTER_SIZE * A_SIZE * sizeof(float));
  float* b_dev = nullptr;
  cudaMalloc(&b_dev, OUTER_SIZE * B_SIZE * sizeof(float));
  float* c_dev = nullptr;
  cudaMalloc(&c_dev, OUTER_SIZE * A_SIZE * sizeof(float));
  float* d_dev = nullptr;
  cudaMalloc(&d_dev, OUTER_SIZE * B_SIZE * sizeof(float));
  cudaMemcpy(
      a_dev,
      a_v.data(),
      OUTER_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyHostToDevice);
  cudaMemcpy(
      b_dev,
      b_v.data(),
      OUTER_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyHostToDevice);
  cudaMemcpy(
      c_dev,
      c_v.data(),
      OUTER_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyHostToDevice);
  cudaMemcpy(
      d_dev,
      d_v.data(),
      OUTER_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  cuda_cg(a_dev, b_dev, c_dev, d_dev);

  cudaDeviceSynchronize();
  cudaMemcpy(
      c_v.data(),
      c_dev,
      OUTER_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyDeviceToHost);
  cudaMemcpy(
      d_v.data(),
      d_dev,
      OUTER_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  ExpectAllNear(c_v, c_ref, 1e-5);
  ExpectAllNear(d_v, d_ref, 1e-5);

  cudaFree(a_dev);
  cudaFree(b_dev);
  cudaFree(c_dev);
  cudaFree(d_dev);
}

// Tests the case with two loop nests, each of which bound to the same block
// size, but with internal loops bound to different thread rank (ie x and y). In
// this case both bodies must be masked against the other dimension being > 0.
// Note: this is a bit degenerate no one would actually write this for perf.
TEST(Cuda, MaskMultiDimMultiAxis_CUDA) {
  KernelScope kernel_scope;
  int OUTER_SIZE = 10;
  int A_SIZE = 30;
  int B_SIZE = 15;
  Placeholder a_buf("a", kFloat, {OUTER_SIZE, A_SIZE});
  Placeholder b_buf("b", kFloat, {OUTER_SIZE, B_SIZE});
  Tensor* c = Compute(
      "C",
      {{OUTER_SIZE, "i"}, {A_SIZE, "j"}},
      [&](const VarHandle& i, const VarHandle& j) {
        return ExprHandle(2) * a_buf.load(i, j);
      });
  Tensor* d = Compute(
      "D",
      {{OUTER_SIZE, "i"}, {B_SIZE, "j"}},
      [&](const VarHandle& i, const VarHandle& j) {
        return c->call(i, j * 2) + b_buf.load(i, j);
      });

  LoopNest l({c, d});
  std::vector<For*> loops = l.getLoopStmtsFor(c);
  l.setGPUBlockIndex(loops[0], 0);
  l.setGPUThreadIndex(loops[1], 0);
  loops = l.getLoopStmtsFor(d);
  l.setGPUBlockIndex(loops[0], 0);
  l.setGPUThreadIndex(loops[1], 1);

  l.prepareForCodegen();
  Stmt* stmt = l.root_stmt();
  CudaCodeGen cuda_cg(stmt, c, d, a_buf, b_buf);

  std::ostringstream oss;
  oss << *cuda_cg.stmt();

  // Both stores masked agaist the other thread dim < 1.
  const std::string& verification_pattern =
      R"IR(
# CHECK: if (threadIdx.y<1
# CHECK:   C[30 * blockIdx.x + threadIdx.x] =
# CHECK: __syncthreads();
# CHECK: if (threadIdx.x<1
# CHECK:   D[threadIdx.y + 15 * blockIdx.x] =)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  auto blockExtents = cuda_cg.gpu_block_extents();
  auto threadExtents = cuda_cg.gpu_thread_extents();
  ASSERT_TRUE(exprEquals(blockExtents[0], new IntImm(OUTER_SIZE)));
  ASSERT_TRUE(exprEquals(threadExtents[0], new IntImm(A_SIZE)));

  PaddedBuffer<float> a_v(OUTER_SIZE, A_SIZE);
  PaddedBuffer<float> b_v(OUTER_SIZE, B_SIZE);
  PaddedBuffer<float> c_v(OUTER_SIZE, A_SIZE);
  PaddedBuffer<float> d_v(OUTER_SIZE, B_SIZE);

  PaddedBuffer<float> c_ref(OUTER_SIZE, A_SIZE);
  PaddedBuffer<float> d_ref(OUTER_SIZE, B_SIZE);

  for (int o = 0; o < OUTER_SIZE; ++o) {
    for (int i = 0; i < A_SIZE; i++) {
      a_v(o, i) = (float)i;
      c_ref(o, i) = (float)(i * 2);
    }
  }

  for (int o = 0; o < OUTER_SIZE; ++o) {
    for (int i = 0; i < B_SIZE; i++) {
      b_v(o, i) = (float)(B_SIZE - i);
      d_ref(o, i) = c_ref(o, i * 2) + b_v(o, i);
    }
  }

  float* a_dev = nullptr;
  cudaMalloc(&a_dev, OUTER_SIZE * A_SIZE * sizeof(float));
  float* b_dev = nullptr;
  cudaMalloc(&b_dev, OUTER_SIZE * B_SIZE * sizeof(float));
  float* c_dev = nullptr;
  cudaMalloc(&c_dev, OUTER_SIZE * A_SIZE * sizeof(float));
  float* d_dev = nullptr;
  cudaMalloc(&d_dev, OUTER_SIZE * B_SIZE * sizeof(float));
  cudaMemcpy(
      a_dev,
      a_v.data(),
      OUTER_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyHostToDevice);
  cudaMemcpy(
      b_dev,
      b_v.data(),
      OUTER_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyHostToDevice);
  cudaMemcpy(
      c_dev,
      c_v.data(),
      OUTER_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyHostToDevice);
  cudaMemcpy(
      d_dev,
      d_v.data(),
      OUTER_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  cuda_cg(c_dev, d_dev, a_dev, b_dev);

  cudaDeviceSynchronize();
  cudaMemcpy(
      c_v.data(),
      c_dev,
      OUTER_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyDeviceToHost);
  cudaMemcpy(
      d_v.data(),
      d_dev,
      OUTER_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  ExpectAllNear(c_v, c_ref, 1e-5);
  ExpectAllNear(d_v, d_ref, 1e-5);

  cudaFree(a_dev);
  cudaFree(b_dev);
  cudaFree(c_dev);
  cudaFree(d_dev);
}

// Tests the case with two loop nests, each bound to both Block and Thread but
// the second loop is smaller in both cases - the second store must be masked
// for both the block and thread dimension.
TEST(Cuda, MaskMultiDimMultiLevel_CUDA) {
  KernelScope kernel_scope;
  int OUTER_A_SIZE = 10;
  int OUTER_B_SIZE = 5;
  int A_SIZE = 30;
  int B_SIZE = 15;
  Placeholder a_buf("a", kFloat, {OUTER_A_SIZE, A_SIZE});
  Placeholder b_buf("b", kFloat, {OUTER_B_SIZE, B_SIZE});
  Tensor* c = Compute(
      "C",
      {{OUTER_A_SIZE, "i"}, {A_SIZE, "j"}},
      [&](const VarHandle& i, const VarHandle& j) {
        return ExprHandle(2) * a_buf.load(i, j);
      });
  Tensor* d = Compute(
      "D",
      {{OUTER_B_SIZE, "i"}, {B_SIZE, "j"}},
      [&](const VarHandle& i, const VarHandle& j) {
        return c->call(i, j * 2) + b_buf.load(i, j);
      });

  LoopNest l({c, d});
  std::vector<For*> loops = l.getLoopStmtsFor(c);
  l.setGPUBlockIndex(loops[0], 0);
  l.setGPUThreadIndex(loops[1], 0);
  loops = l.getLoopStmtsFor(d);
  l.setGPUBlockIndex(loops[0], 0);
  l.setGPUThreadIndex(loops[1], 0);

  l.prepareForCodegen();
  Stmt* stmt = l.root_stmt();
  CudaCodeGen cuda_cg(stmt, c, d, a_buf, b_buf);

  std::ostringstream oss;
  oss << *cuda_cg.stmt();

  // The write to D should be masked twice, but not the write to C.
  const std::string& verification_pattern =
      R"IR(
# CHECK-NOT: if (
# CHECK: C[30 * blockIdx.x + threadIdx.x] =
# CHECK: __syncthreads();
# CHECK: if (blockIdx.x<5
# CHECK:   if (threadIdx.x<15
# CHECK:     D[threadIdx.x + 15 * blockIdx.x] =)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  auto blockExtents = cuda_cg.gpu_block_extents();
  auto threadExtents = cuda_cg.gpu_thread_extents();
  ASSERT_TRUE(exprEquals(blockExtents[0], new IntImm(OUTER_A_SIZE)));
  ASSERT_TRUE(exprEquals(threadExtents[0], new IntImm(A_SIZE)));

  PaddedBuffer<float> a_v(OUTER_A_SIZE, A_SIZE);
  PaddedBuffer<float> b_v(OUTER_B_SIZE, B_SIZE);
  PaddedBuffer<float> c_v(OUTER_A_SIZE, A_SIZE);
  PaddedBuffer<float> d_v(OUTER_B_SIZE, B_SIZE);

  PaddedBuffer<float> c_ref(OUTER_A_SIZE, A_SIZE);
  PaddedBuffer<float> d_ref(OUTER_B_SIZE, B_SIZE);

  for (int o = 0; o < OUTER_A_SIZE; ++o) {
    for (int i = 0; i < A_SIZE; i++) {
      a_v(o, i) = (float)i;
      c_ref(o, i) = (float)(i * 2);
    }
  }

  for (int o = 0; o < OUTER_B_SIZE; ++o) {
    for (int i = 0; i < B_SIZE; i++) {
      b_v(o, i) = (float)(B_SIZE - i);
      d_ref(o, i) = c_ref(o, i * 2) + b_v(o, i);
    }
  }

  float* a_dev = nullptr;
  cudaMalloc(&a_dev, OUTER_A_SIZE * A_SIZE * sizeof(float));
  float* b_dev = nullptr;
  cudaMalloc(&b_dev, OUTER_B_SIZE * B_SIZE * sizeof(float));
  float* c_dev = nullptr;
  cudaMalloc(&c_dev, OUTER_A_SIZE * A_SIZE * sizeof(float));
  float* d_dev = nullptr;
  cudaMalloc(&d_dev, OUTER_B_SIZE * B_SIZE * sizeof(float));
  cudaMemcpy(
      a_dev,
      a_v.data(),
      OUTER_A_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyHostToDevice);
  cudaMemcpy(
      b_dev,
      b_v.data(),
      OUTER_B_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyHostToDevice);
  cudaMemcpy(
      c_dev,
      c_v.data(),
      OUTER_A_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyHostToDevice);
  cudaMemcpy(
      d_dev,
      d_v.data(),
      OUTER_B_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  cuda_cg(c_dev, d_dev, a_dev, b_dev);

  cudaDeviceSynchronize();
  cudaMemcpy(
      c_v.data(),
      c_dev,
      OUTER_A_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyDeviceToHost);
  cudaMemcpy(
      d_v.data(),
      d_dev,
      OUTER_B_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  ExpectAllNear(c_v, c_ref, 1e-5);
  ExpectAllNear(d_v, d_ref, 1e-5);

  cudaFree(a_dev);
  cudaFree(b_dev);
  cudaFree(c_dev);
  cudaFree(d_dev);
}

} // namespace jit
} // namespace torch

#endif
