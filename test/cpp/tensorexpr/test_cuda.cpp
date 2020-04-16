#ifdef USE_CUDA

#include <sstream>
#include <stdexcept>
#include "test/cpp/tensorexpr/test_base.h"

#include <cmath>

#include "test/cpp/tensorexpr/padded_buffer.h"
#include "torch/csrc/jit/tensorexpr/buffer.h"
#include "torch/csrc/jit/tensorexpr/cuda_codegen.h"
#include "torch/csrc/jit/tensorexpr/loopnest.h"
#include "torch/csrc/jit/tensorexpr/tensor.h"

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

void testCudaDynamicShape2D() {
  KernelScope kernel_scope;
  auto testWithSize = [](int32_t M, int32_t N) {
    VarHandle m("m", kInt);
    VarHandle n("n", kInt);
    Buffer a(BufHandle("a", {m, n}), kFloat);
    Buffer b(BufHandle("b", {m, n}), kFloat);
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
  Buffer a(BufHandle("a", {n}), kFloat);
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

} // namespace jit
} // namespace torch

#endif
