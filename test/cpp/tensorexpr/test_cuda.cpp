#ifdef USE_CUDA

#include <sstream>
#include <stdexcept>
#include "test/cpp/tensorexpr/test_base.h"

#include <cmath>

#include "test/cpp/tensorexpr/padded_buffer.h"
#include "torch/csrc/jit/tensorexpr/buffer.h"
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
void testCudaTestVectorAdd01_impl() {
  KernelScope kernel_scope;
  const int num_iter = 3;
  const int block_count = 16;
  const int block_size = 128;
  Dtype dtype = ToDtype<ctype>();
  Buffer a_buf("a", dtype, {num_iter, block_count, block_size});
  Buffer b_buf("b", dtype, {num_iter, block_count, block_size});
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

void testCudaSigmoid() {
  KernelScope kernel_scope;
  const int num_iter = 3;
  const int block_count = 16;
  const int block_size = 128;
  Dtype dtype = ToDtype<float>();
  Buffer a_buf("a", dtype, {num_iter, block_count, block_size});
  Tensor* c = Compute(
      "c",
      {
          {num_iter, "n"},
          {block_count, "b_id"},
          {block_size, "t_id"},
      },
      [&](const VarHandle& n, const VarHandle& b_id, const VarHandle& t_id) {
        return sigmoid(sigmoid(a_buf(n, b_id, t_id)));
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

void testCudaTestVectorAdd01() {
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
  Buffer a_buf("a", kFloat, {N});
  Buffer b_buf("b", kFloat, {N});
  Tensor* c = Compute(
      "c",
      {
          {N, "N"},
      },
      [&](const VarHandle& n) { return a_buf(n) + b_buf(n); });
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

void testCudaTestVectorAdd02() {
  testCudaTestVectorAdd02_impl(1024, 128);
  testCudaTestVectorAdd02_impl(1030, 128);
}

void testCudaHalfCast() {
  KernelScope ks;
  auto half = ToDtype<at::Half>();
  Buffer a("a", half, {4});
  Tensor* b = Compute("b", {{4, "n"}}, [&](const VarHandle& i) {
    return Cast::make(kFloat, a(i));
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

void testCudaDynamicShape2D() {
  KernelScope kernel_scope;
  auto testWithSize = [](int32_t M, int32_t N) {
    VarHandle m("m", kInt);
    VarHandle n("n", kInt);
    Buffer a(BufHandle("a", {m, n}, kFloat));
    Buffer b(BufHandle("b", {m, n}, kFloat));
    Tensor* c = Compute(
        "c", {{m, "m"}, {n, "n"}}, [&](const VarHandle& i, const VarHandle& j) {
          return a(i, j) + b(i, j);
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
    ASSERT_TRUE(v >= 0 && v < 1, "invalid value: ", i, ", ", v);
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

void testCudaDynamicShapeSplit() {
  KernelScope ks;
  constexpr int N = 4096;
  VarHandle n("n", kInt);
  Buffer a(BufHandle("a", {n}, kFloat));
  Tensor* b =
      Compute("b", {{n, "n"}}, [&](const VarHandle& i) { return a(i) * 2.0f; });
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

void testCudaOneBlockOneThreadGlobalReduce1() {
  const static int N = 1024;
  KernelScope kernel_scope;
  Buffer data_buf("data", kFloat, {N});
  Buffer output_buf("output", kFloat, {1});

  // The test adds the following code for trivial reduction:
  // for (int bidx = 0; bidx < 1; bidx++) { // blockIdx.x
  //   for (int tidx = 0; tidx < 1; tidx++) { // threadIdx.x
  //     output[0] = 0.f;
  //     for (int i1 = 0; i1 < 1024; i1++) {
  //       output[0] = output[0] + data[i1];
  //     }
  //   }
  // }

  Store* init_store = Store::make(output_buf, {0}, 0.f, 1);
  VarHandle i1("i1", kInt);
  ExprHandle load_data = Load::make(data_buf, {i1}, 1);
  ExprHandle load_output = Load::make(output_buf, {0}, 1);
  ExprHandle add_value = load_output + load_data;
  Store* store_output = Store::make(output_buf, {0}, add_value, 1);
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

void testCudaOneBlockMultiThreadGlobalReduce1() {
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

  Buffer a_buf("a", kFloat, {N});
  Buffer b_buf("b", kFloat, {1});

  Store* init_store = Store::make(b_buf, {0}, 0.f, 1);
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
  ExprHandle load_a = Load::make(a_buf, {t}, 1);
  ExprHandle load_b = Load::make(b_buf, {0}, 1);
  ExprHandle add_value = load_b + load_a;
  Store* store_b = Store::make(b_buf, {0}, add_value, 1);
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

void testCudaNoThreadIdxWrite_1() {
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
  Buffer a_buf("a", kFloat, {2});
  Buffer b_buf("b", kFloat, {N});

  VarHandle k("k", kInt);
  VarHandle l("l", kInt);
  VarHandle m("m", kInt);
  VarHandle n("n", kInt);

  //   a[0] = 0
  //   for n in 0..2:
  //     a[0] = a[0] + n
  Store* store_a0_0 = Store::make(a_buf, {0}, 0.f, 1);
  ExprHandle load_a0 = Load::make(a_buf, {0}, 1);
  ExprHandle v1 = load_a0 + n;
  Store* store_a0_v1 = Store::make(a_buf, {0}, v1, 1);
  For* loop_a_0 = For::make(n, 0, 2, store_a0_v1);

  //   for m in 0..1024: // thread-idx
  //     b[m] = m
  Store* store_bm_m = Store::make(b_buf, {m}, m + 0.f, 1);
  LoopOptions thread_idx_options;
  thread_idx_options.set_gpu_thread_index(0);
  For* loop_b_1 = For::make(m, 0, N, store_bm_m, thread_idx_options);

  //   a[1] = 1
  //   for l in 0..2:
  //     a[1] = a[1] + l
  Store* store_a1_1 = Store::make(a_buf, {1}, 1.f, 1);
  ExprHandle load_a1 = Load::make(a_buf, {1}, 1);
  ExprHandle v2 = load_a1 + l;
  Store* store_a1_v2 = Store::make(a_buf, {1}, v2, 1);
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

void testCudaSharedMemReduce_1() {
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

  Buffer a("a", kFloat, {1, M, N});
  Buffer b("b", kFloat, {1});
  VarHandle k("k", kInt);
  VarHandle m("m", kInt);
  VarHandle n("n", kInt);

  std::vector<Stmt*> block;
  VarHandle c_var("c", kHandle);
  std::vector<const Expr*> dims;
  dims.push_back(ExprHandle(N).node());
  BufHandle c{new Buf(c_var.node(), dims, kFloat)};
  {
    // alloc(c, 64);
    Allocate* alloc = Allocate::make(c_var, kFloat, {N});
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
    ExprHandle a_kmn = Load::make(a, {k * (M * N) + m * N + n}, 1);
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
    Store* store_bk_0 = Store::make(b, {k}, 0.f, 1);
    block.push_back(store_bk_0);
    ExprHandle load_bk = Load::make(b, {k}, 1);
    ExprHandle load_cn = Load::make(kFloat, c, {n}, 1);
    ExprHandle v_add = load_bk + load_cn;
    Store* store_bk = Store::make(b, {k}, v_add, 1);
    For* loop_n3 = For::make(n, 0, N, store_bk, thread_idx_opt);
    block.push_back(loop_n3);
  }

  {
    //    free(c)
    Free* free_stmt = Free::make(c_var);
    block.push_back(free_stmt);
  }

  Block* reduce_body = Block::make(block);
  For* loop_k1 = For::make(k, 0, 1, reduce_body, block_idx_opt);

  // TODO: check the generated code for correctness.
  CudaCodeGen cuda_cg(loop_k1, a, b);
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

void testCudaLocalMemReduce_1() {
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

  Buffer a("a", kFloat, {1, M, N});
  Buffer b("b", kFloat, {1});
  VarHandle k("k", kInt);
  VarHandle m("m", kInt);
  VarHandle n("n", kInt);

  VarHandle c_var("c", kHandle);
  std::vector<const Expr*> dims;
  dims.push_back(ExprHandle(N).node());
  BufHandle c{new Buf(c_var.node(), dims, kFloat)};
  std::vector<Stmt*> block_k;
  {
    //    b(k) = 0
    Store* store_bk_0 = Store::make(b, {k}, 0.f, 1);
    block_k.push_back(store_bk_0);
  }
  std::vector<Stmt*> block_n;
  {
    // alloc(c, 1);
    Allocate* alloc = Allocate::make(c_var, kFloat, {1});
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
    ExprHandle a_kmn = Load::make(a, {k * (M * N) + m * N + n}, 1);
    ExprHandle v_add = load_c0 + a_kmn;
    Store* store_c0_v = Store::make(c, {0}, v_add);
    For* loop_m = For::make(m, 0, M, store_c0_v);
    block_n.push_back(loop_m);
  }
  {
    //      b(k) = b(k) + c(0)
    ExprHandle load_bk = Load::make(b, {k}, 1);
    ExprHandle load_c0 = Load::make(kFloat, c, {0}, 1);
    ExprHandle v_add = load_bk + load_c0;
    Store* store_bk = Store::make(b, {k}, v_add, 1);
    block_n.push_back(store_bk);
  }
  {
    //      free(c)
    Free* free_stmt = Free::make(c_var);
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

void testCudaHalfSupport() {
  KernelScope ks;
  auto half = ToDtype<at::Half>();
  Buffer a("a", half, {4});
  Tensor* b = Compute("b", {{4, "n"}}, [&](const VarHandle& i) {
    return Cast::make(half, ExprHandle(2.0f) * a(i));
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

} // namespace jit
} // namespace torch

#endif
