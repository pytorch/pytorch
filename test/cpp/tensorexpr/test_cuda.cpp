#ifdef USE_CUDA

#include <cmath>
#include <sstream>
#include <stdexcept>

#include <gtest/gtest.h>

#include <test/cpp/tensorexpr/test_base.h>

#include <test/cpp/tensorexpr/padded_buffer.h>
#include <torch/csrc/jit/tensorexpr/cuda_codegen.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/csrc/jit/testing/file_check.h>

#include <torch/csrc/jit/testing/file_check.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/Half.h>
#include <c10/util/irange.h>

namespace torch {
namespace jit {
using namespace torch::jit::tensorexpr;
using namespace torch::jit::tensorexpr;

template <typename ctype>
static void testCudaTestVectorAdd01_impl() {
  const int num_iter = 3;
  const int block_count = 16;
  const int block_size = 128;
  Dtype dtype = ToDtype<ctype>();
  BufHandle a_buf("a", {num_iter, block_count, block_size}, dtype);
  BufHandle b_buf("b", {num_iter, block_count, block_size}, dtype);
  Tensor c = Compute(
      "c",
      {
          num_iter,
          block_count,
          block_size,
      },
      [&](const VarHandle& n, const VarHandle& b_id, const VarHandle& t_id) {
        return a_buf.load(n, b_id, t_id) + b_buf.load(n, b_id, t_id);
      });
  LoopNest l({c});
  std::vector<ForPtr> loops = l.getLoopStmtsFor(c);
  loops[1]->set_gpu_block_index(0);
  loops[2]->set_gpu_thread_index(0);
  l.prepareForCodegen();
  StmtPtr stmt = l.root_stmt();
  CudaCodeGen cuda_cg(stmt, c, a_buf, b_buf);
  const int N = block_count * block_size * num_iter;
  PaddedBuffer<ctype> a_v(N);
  PaddedBuffer<ctype> b_v(N);
  PaddedBuffer<ctype> c_v(N);
  PaddedBuffer<ctype> c_ref(N);

  for (const auto i : c10::irange(N)) {
    a_v(i) = ctype(i);
    b_v(i) = ctype(i * 3 + 7);
    c_ref(i) = a_v(i) + b_v(i);
  }

  // TODO: move gpu support into PaddedBuffer
  ctype* a_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&a_dev, N * sizeof(ctype)));
  ctype* b_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&b_dev, N * sizeof(ctype)));
  ctype* c_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&c_dev, N * sizeof(ctype)));
  C10_CUDA_CHECK(
      cudaMemcpy(a_dev, a_v.data(), N * sizeof(ctype), cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(
      cudaMemcpy(b_dev, b_v.data(), N * sizeof(ctype), cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(
      cudaMemcpy(c_dev, c_v.data(), N * sizeof(ctype), cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  cuda_cg(c_dev, a_dev, b_dev);

  C10_CUDA_CHECK(cudaDeviceSynchronize());
  C10_CUDA_CHECK(
      cudaMemcpy(c_v.data(), c_dev, N * sizeof(ctype), cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  ExpectAllNear(c_v, c_ref, 1e-5);

  C10_CUDA_CHECK(cudaFree(a_dev));
  C10_CUDA_CHECK(cudaFree(b_dev));
  C10_CUDA_CHECK(cudaFree(c_dev));
}

float sigmoid(float x) {
  return 1.0f / (1.0f + expf(-0.0f - x));
}

TEST(Cuda, Sigmoid_CUDA) {
  const int num_iter = 3;
  const int block_count = 16;
  const int block_size = 128;
  Dtype dtype = ToDtype<float>();
  BufHandle a_buf("a", {num_iter, block_count, block_size}, dtype);
  Tensor c = Compute(
      "c",
      {
          num_iter,
          block_count,
          block_size,
      },
      [&](const VarHandle& n, const VarHandle& b_id, const VarHandle& t_id) {
        return sigmoid(sigmoid(a_buf.load(n, b_id, t_id)));
      });
  LoopNest l({c});
  std::vector<ForPtr> loops = l.getLoopStmtsFor(c);
  loops[1]->set_gpu_block_index(0);
  loops[2]->set_gpu_thread_index(0);
  l.prepareForCodegen();
  StmtPtr stmt = l.root_stmt();
  CudaCodeGen cuda_cg(stmt, c, a_buf);
  const int N = block_count * block_size * num_iter;
  PaddedBuffer<float> a_v(N);
  PaddedBuffer<float> c_v(N);
  PaddedBuffer<float> c_ref(N);

  for (const auto i : c10::irange(N)) {
    a_v(i) = float(i);
    c_ref(i) = sigmoid(sigmoid(a_v(i)));
  }

  // TODO: move gpu support into PaddedBuffer
  float* a_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&a_dev, N * sizeof(float)));
  float* c_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&c_dev, N * sizeof(float)));
  C10_CUDA_CHECK(
      cudaMemcpy(a_dev, a_v.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(
      cudaMemcpy(c_dev, c_v.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  cuda_cg(c_dev, a_dev);

  C10_CUDA_CHECK(cudaDeviceSynchronize());
  C10_CUDA_CHECK(
      cudaMemcpy(c_v.data(), c_dev, N * sizeof(float), cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  ExpectAllNear(c_v, c_ref, 1e-5);

  C10_CUDA_CHECK(cudaFree(a_dev));
  C10_CUDA_CHECK(cudaFree(c_dev));
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

static void testCudaTestVectorAdd02_impl(int64_t N, int64_t block_size) {
  BufHandle a_buf("a", {N}, kFloat);
  BufHandle b_buf("b", {N}, kFloat);
  Tensor c = Compute("c", {N}, [&](const VarHandle& n) {
    return a_buf.load(n) + b_buf.load(n);
  });
  LoopNest l({c});
  ForPtr n_inner;
  std::vector<ForPtr> loops = l.getLoopStmtsFor(c);
  l.splitWithMask(loops[0], block_size, &n_inner);
  loops[0]->set_gpu_block_index(0);
  n_inner->set_gpu_thread_index(0);
  l.prepareForCodegen();
  StmtPtr stmt = l.root_stmt();
  CudaCodeGen cuda_cg(stmt, c, a_buf, b_buf);
  PaddedBuffer<float> a_v(N);
  PaddedBuffer<float> b_v(N);
  PaddedBuffer<float> c_v(N);
  PaddedBuffer<float> c_ref(N);

  for (const auto i : c10::irange(N)) {
    a_v(i) = i;
    b_v(i) = i * 3 + 7;
    c_ref(i) = a_v(i) + b_v(i);
  }

  // TODO: move gpu support into PaddedBuffer
  float* a_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&a_dev, N * sizeof(float)));
  float* b_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&b_dev, N * sizeof(float)));
  float* c_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&c_dev, N * sizeof(float)));
  C10_CUDA_CHECK(
      cudaMemcpy(a_dev, a_v.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(
      cudaMemcpy(b_dev, b_v.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(
      cudaMemcpy(c_dev, c_v.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  cuda_cg(c_dev, a_dev, b_dev);

  C10_CUDA_CHECK(cudaDeviceSynchronize());
  C10_CUDA_CHECK(
      cudaMemcpy(c_v.data(), c_dev, N * sizeof(float), cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  ExpectAllNear(c_v, c_ref, 1e-5);

  C10_CUDA_CHECK(cudaFree(a_dev));
  C10_CUDA_CHECK(cudaFree(b_dev));
  C10_CUDA_CHECK(cudaFree(c_dev));
}

TEST(Cuda, TestVectorAdd02_CUDA) {
  testCudaTestVectorAdd02_impl(1024, 128);
  testCudaTestVectorAdd02_impl(1030, 128);
}

TEST(Cuda, HalfCast_CUDA) {
  auto half = ToDtype<at::Half>();
  BufHandle a("a", {4}, half);
  Tensor b = Compute("b", {4}, [&](const VarHandle& i) {
    return Cast::make(kFloat, a.load(i));
  });

  LoopNest l({b});
  l.prepareForCodegen();
  StmtPtr s = l.root_stmt();
  CudaCodeGen cg(s, {a, b});

  std::vector<at::Half> aData(4, 2.0f);
  std::vector<float> bData(4, 0.0f);
  at::Half* aDev = nullptr;
  float* bDev = nullptr;
  auto aSize = aData.size() * sizeof(aData[0]);
  auto bSize = bData.size() * sizeof(bData[0]);

  C10_CUDA_CHECK(cudaMalloc(&aDev, aSize));
  C10_CUDA_CHECK(cudaMalloc(&bDev, bSize));
  C10_CUDA_CHECK(cudaMemcpy(aDev, aData.data(), aSize, cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(bDev, bData.data(), bSize, cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  cg.call({aDev, bDev});
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  C10_CUDA_CHECK(cudaMemcpy(aData.data(), aDev, aSize, cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaMemcpy(bData.data(), bDev, bSize, cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  assertAllEqual(bData, 2.0f);

  C10_CUDA_CHECK(cudaFree(aDev));
  C10_CUDA_CHECK(cudaFree(bDev));
}

TEST(Cuda, DynamicShape2D_CUDA) {
  auto testWithSize = [](int32_t M, int32_t N) {
    VarHandle m("m", kInt);
    VarHandle n("n", kInt);
    BufHandle a("a", {m, n}, kFloat);
    BufHandle b("b", {m, n}, kFloat);
    Tensor c =
        Compute("c", {m, n}, [&](const VarHandle& i, const VarHandle& j) {
          return a.load(i, j) + b.load(i, j);
        });
    LoopNest l({c});
    l.prepareForCodegen();
    StmtPtr s = l.root_stmt();
    CudaCodeGen cg(s, {a, b, c, m, n});

    std::vector<float> aData(M * N, 1.0f);
    std::vector<float> bData(M * N, 2.0f);
    std::vector<float> cData(M * N, 0.0f);
    float* aDev = nullptr;
    float* bDev = nullptr;
    float* cDev = nullptr;
    C10_CUDA_CHECK(cudaMalloc(&aDev, aData.size() * sizeof(aData[0])));
    C10_CUDA_CHECK(cudaMalloc(&bDev, bData.size() * sizeof(bData[0])));
    C10_CUDA_CHECK(cudaMalloc(&cDev, cData.size() * sizeof(cData[0])));
    C10_CUDA_CHECK(cudaMemcpy(
        aDev,
        aData.data(),
        aData.size() * sizeof(aData[0]),
        cudaMemcpyHostToDevice));
    C10_CUDA_CHECK(cudaMemcpy(
        bDev,
        bData.data(),
        bData.size() * sizeof(bData[0]),
        cudaMemcpyHostToDevice));
    C10_CUDA_CHECK(cudaMemcpy(
        cDev,
        cData.data(),
        cData.size() * sizeof(cData[0]),
        cudaMemcpyHostToDevice));
    C10_CUDA_CHECK(cudaDeviceSynchronize());

    cg.call({aDev, bDev, cDev, M, N});
    C10_CUDA_CHECK(cudaDeviceSynchronize());

    C10_CUDA_CHECK(cudaMemcpy(
        cData.data(),
        cDev,
        cData.size() * sizeof(cData[0]),
        cudaMemcpyDeviceToHost));
    C10_CUDA_CHECK(cudaDeviceSynchronize());

    ExpectAllNear(cData, std::vector<float>(M * N, 3.0f), 1e-7);

    C10_CUDA_CHECK(cudaFree(aDev));
    C10_CUDA_CHECK(cudaFree(bDev));
    C10_CUDA_CHECK(cudaFree(cDev));
  };
  testWithSize(32, 32);
  testWithSize(1, 16);
  testWithSize(27, 13);
}

TEST(Cuda, TestRand01_CUDA) {
  const int num_iter = 3;
  const int block_count = 16;
  const int block_size = 128;
  Tensor c = Compute(
      "c",
      {
          num_iter,
          block_count,
          block_size,
      },
      [&](const VarHandle& n, const VarHandle& b_id, const VarHandle& t_id) {
        return Intrinsics::make(IntrinsicsOp::kRand, kFloat);
      });
  LoopNest l({c});
  std::vector<ForPtr> loops = l.getLoopStmtsFor(c);
  loops[1]->set_gpu_block_index(0);
  loops[2]->set_gpu_thread_index(0);
  l.prepareForCodegen();
  StmtPtr stmt = l.root_stmt();
  CudaCodeGen cuda_cg(stmt, c);
  const int N = block_count * block_size * num_iter;
  PaddedBuffer<float> c_v(N);

  // TODO: move gpu support into PaddedBuffer
  float* c_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&c_dev, N * sizeof(float)));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  cuda_cg(c_dev);

  C10_CUDA_CHECK(cudaDeviceSynchronize());
  C10_CUDA_CHECK(
      cudaMemcpy(c_v.data(), c_dev, N * sizeof(float), cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  float sum1 = 0;
  float sum2 = 0;
  float sum3 = 0;
  for (const auto i : c10::irange(N)) {
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
  C10_CUDA_CHECK(cudaFree(c_dev));
}

TEST(Cuda, DynamicShapeSplit_CUDA) {
  constexpr int64_t N = 4096;
  VarHandle n("n", kLong);
  BufHandle a("a", {n}, kFloat);
  Tensor b =
      Compute("b", {n}, [&](const VarHandle& i) { return a.load(i) * 2.0f; });
  LoopNest l({b});
  ForPtr inner;
  std::vector<ForPtr> loops = l.getLoopStmtsFor(b);
  l.splitWithMask(loops[0], 1024, &inner);
  loops[0]->set_gpu_block_index(0);
  inner->set_gpu_thread_index(0);
  StmtPtr s = l.root_stmt();
  CudaCodeGen cg(s, {a, b, n});

  std::vector<float> aData(N, 1.0f);
  std::vector<float> bData(N, 1.0f);
  float* aDev = nullptr;
  float* bDev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&aDev, aData.size() * sizeof(aData[0])));
  C10_CUDA_CHECK(cudaMalloc(&bDev, bData.size() * sizeof(bData[0])));
  C10_CUDA_CHECK(cudaMemcpy(
      aDev,
      aData.data(),
      aData.size() * sizeof(aData[0]),
      cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      bDev,
      bData.data(),
      bData.size() * sizeof(aData[0]),
      cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  cg.call({aDev, bDev, N});
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  C10_CUDA_CHECK(cudaMemcpy(
      bData.data(),
      bDev,
      bData.size() * sizeof(aData[0]),
      cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  ExpectAllNear(bData, std::vector<float>(N, 2.0f), 1e-7);

  C10_CUDA_CHECK(cudaFree(aDev));
  C10_CUDA_CHECK(cudaFree(bDev));
}

TEST(Cuda, OneBlockOneThreadGlobalReduce1_CUDA) {
  const static int N = 1024;
  BufHandle data_buf("data", {N}, kFloat);
  BufHandle output_buf("output", {1}, kFloat);

  // The test adds the following code for trivial reduction:
  // for (const auto bidx : c10::irange(1)) { // blockIdx.x
  //   for (const auto tidx : c10::irange(1)) { // threadIdx.x
  //     output[0] = 0.f;
  //     for (const auto i1 : c10::irange(1024)) {
  //       output[0] = output[0] + data[i1];
  //     }
  //   }
  // }

  StorePtr init_store = output_buf.store({0}, 0.f);
  VarHandle i1("i1", kInt);
  ExprHandle load_data = Load::make(data_buf, {i1});
  ExprHandle load_output = Load::make(output_buf, {0});
  ExprHandle add_value = load_output + load_data;
  StorePtr store_output = output_buf.store({0}, add_value);
  ForPtr for_output = For::make(i1, 0, N, store_output);
  StmtPtr reduce_block = Block::make({init_store, for_output});
  VarHandle thread_idx("tidx", kInt);
  LoopOptions thread_idx_options;
  thread_idx_options.set_gpu_thread_index(0);
  ForPtr thread_idx_loop =
      For::make(thread_idx, 0, 1, reduce_block, thread_idx_options);
  VarHandle block_idx("bidx", kInt);
  LoopOptions block_idx_options;
  block_idx_options.set_gpu_block_index(0);
  ForPtr block_idx_loop =
      For::make(block_idx, 0, 1, thread_idx_loop, block_idx_options);

  CudaCodeGen cuda_cg(block_idx_loop, data_buf, output_buf);
  PaddedBuffer<float> data_v(N);
  PaddedBuffer<float> output_v(1, "output_v");
  PaddedBuffer<float> output_ref(1, "output_ref");

  output_ref(0) = 0;
  for (const auto i : c10::irange(N)) {
    data_v(i) = i;
    output_ref(0) += data_v(i);
  }

  float* data_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&data_dev, N * sizeof(float)));
  C10_CUDA_CHECK(cudaMemcpy(
      data_dev, data_v.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  float* output_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&output_dev, 1 * sizeof(float)));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  cuda_cg(data_dev, output_dev);

  C10_CUDA_CHECK(cudaDeviceSynchronize());
  C10_CUDA_CHECK(cudaMemcpy(
      output_v.data(), output_dev, 1 * sizeof(float), cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  ExpectAllNear(output_v, output_ref, 1e-5);

  C10_CUDA_CHECK(cudaFree(data_dev));
  C10_CUDA_CHECK(cudaFree(output_dev));
}

TEST(Cuda, OneBlockMultiThreadGlobalReduce1_CUDA) {
  const static int N = 1024;

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

  BufHandle a_buf("a", {N}, kFloat);
  BufHandle b_buf("b", {1}, kFloat);

  StorePtr init_store = b_buf.store({0}, 0.f);
  VarHandle t("t", kInt);
  VarHandle b("b", kInt);

  //  for t in 0..1024: // thread-idx
  //    if t < 1:
  //      b[0] = 0
  ExprHandle cond_t_lt_1 =
      CompareSelect::make(t, 1, CompareSelectOperation::kLT);
  CondPtr masked_init_b = Cond::make(cond_t_lt_1, init_store, nullptr);
  LoopOptions thread_idx_options;
  thread_idx_options.set_gpu_thread_index(0);
  ForPtr for_init = For::make(t, 0, N, masked_init_b, thread_idx_options);

  //  for t in 0..1024: // thread-idx
  //    b[0] = b[0] + a[t] // implied atomic
  ExprHandle load_a = Load::make(a_buf, {t});
  ExprHandle load_b = Load::make(b_buf, {0});
  ExprHandle add_value = load_b + load_a;
  StorePtr store_b = b_buf.store({0}, add_value);
  ForPtr for_b = For::make(t, 0, N, store_b, thread_idx_options);

  StmtPtr reduce_block = Block::make({for_init, for_b});

  VarHandle block_idx("bidx", kInt);
  LoopOptions block_idx_options;
  block_idx_options.set_gpu_block_index(0);
  ForPtr block_idx_loop =
      For::make(block_idx, 0, 1, reduce_block, block_idx_options);

  CudaCodeGen cuda_cg(block_idx_loop, a_buf, b_buf);
  PaddedBuffer<float> a_v(N);
  PaddedBuffer<float> b_v(1, "b_v");
  PaddedBuffer<float> b_ref(1, "b_ref");

  b_ref(0) = 0;
  for (const auto i : c10::irange(N)) {
    a_v(i) = i;
    b_ref(0) += a_v(i);
  }

  float* a_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&a_dev, N * sizeof(float)));
  C10_CUDA_CHECK(
      cudaMemcpy(a_dev, a_v.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  float* b_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&b_dev, 1 * sizeof(float)));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  cuda_cg(a_dev, b_dev);

  C10_CUDA_CHECK(cudaDeviceSynchronize());
  C10_CUDA_CHECK(
      cudaMemcpy(b_v.data(), b_dev, 1 * sizeof(float), cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  ExpectAllNear(b_v, b_ref, 1e-5);

  C10_CUDA_CHECK(cudaFree(a_dev));
  C10_CUDA_CHECK(cudaFree(b_dev));
}

TEST(Cuda, NoThreadIdxWrite_1_CUDA) {
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
  BufHandle a_buf("a", {2}, kFloat);
  BufHandle b_buf("b", {N}, kFloat);

  VarHandle k("k", kInt);
  VarHandle l("l", kInt);
  VarHandle m("m", kInt);
  VarHandle n("n", kInt);

  //   a[0] = 0
  //   for n in 0..2:
  //     a[0] = a[0] + n
  StorePtr store_a0_0 = a_buf.store({0}, 0.f);
  ExprHandle load_a0 = Load::make(a_buf, {0});
  ExprHandle v1 = load_a0 + n;
  StorePtr store_a0_v1 = a_buf.store({0}, v1);
  ForPtr loop_a_0 = For::make(n, 0, 2, store_a0_v1);

  //   for m in 0..1024: // thread-idx
  //     b[m] = m
  StorePtr store_bm_m = b_buf.store({m}, m + 0.f);
  LoopOptions thread_idx_options;
  thread_idx_options.set_gpu_thread_index(0);
  ForPtr loop_b_1 = For::make(m, 0, N, store_bm_m, thread_idx_options);

  //   a[1] = 1
  //   for l in 0..2:
  //     a[1] = a[1] + l
  StorePtr store_a1_1 = a_buf.store({1}, 1.f);
  ExprHandle load_a1 = a_buf.load(1);
  ExprHandle v2 = load_a1 + l;
  StorePtr store_a1_v2 = a_buf.store({1}, v2);
  ForPtr loop_a_1 = For::make(l, 0, 2, store_a1_v2);

  StmtPtr reduce_block =
      Block::make({store_a0_0, loop_a_0, loop_b_1, store_a1_1, loop_a_1});

  VarHandle block_idx("bidx", kInt);
  LoopOptions block_idx_options;
  block_idx_options.set_gpu_block_index(0);
  ForPtr block_idx_loop =
      For::make(block_idx, 0, 1, reduce_block, block_idx_options);

  CudaCodeGen cuda_cg(block_idx_loop, a_buf, b_buf);
  PaddedBuffer<float> a_v(2);
  PaddedBuffer<float> b_v(N, "b_v");
  PaddedBuffer<float> a_ref(2, "a_ref");
  PaddedBuffer<float> b_ref(N, "b_ref");

  a_ref(0) = 0;
  for (const auto i : c10::irange(2)) {
    a_ref(0) += i;
  }
  a_ref(1) = a_ref(0) + 1;
  for (const auto i : c10::irange(N)) {
    b_ref(i) = i;
  }

  // TODO: add check of the generated code.
  float* a_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&a_dev, 2 * sizeof(float)));
  float* b_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&b_dev, N * sizeof(float)));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  cuda_cg(a_dev, b_dev);

  C10_CUDA_CHECK(cudaDeviceSynchronize());
  C10_CUDA_CHECK(
      cudaMemcpy(a_v.data(), a_dev, 2 * sizeof(float), cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(
      cudaMemcpy(b_v.data(), b_dev, N * sizeof(float), cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  ExpectAllNear(a_v, a_ref, 1e-5);
  ExpectAllNear(b_v, b_ref, 1e-5);

  C10_CUDA_CHECK(cudaFree(a_dev));
  C10_CUDA_CHECK(cudaFree(b_dev));
}

TEST(Cuda, SharedMemReduce_1_CUDA) {
  // FIXME: this test is flaky in CI.
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

  BufHandle a("a", {1, M, N}, kFloat);
  BufHandle b("b", {1}, kFloat);
  VarHandle k("k", kInt);
  VarHandle m("m", kInt);
  VarHandle n("n", kInt);

  std::vector<StmtPtr> block;
  std::vector<ExprPtr> dims;
  dims.push_back(ExprHandle(N).node());
  BufHandle c{alloc<Buf>("c", dims, kFloat)};
  {
    // alloc(c, 64);
    AllocatePtr alloc = Allocate::make(c);
    block.push_back(alloc);
  }

  {
    //    for n in 0..64:  // thread-idx
    //      c(n) = 0
    StorePtr store_cn_0 = Store::make(c, {n}, 0.f);
    ForPtr loop_n1 = For::make(n, 0, N, store_cn_0, thread_idx_opt);
    block.push_back(loop_n1);
  }

  {
    //  for m in 0..128:
    //    for n in 0..64:  // thread_idx
    //      c(n) = c(n) + a(k, m, n)
    ExprHandle load_cn = Load::make(kFloat, c, {n});
    ExprHandle a_kmn = Load::make(a, {k * (M * N) + m * N + n});
    ExprHandle v_add = load_cn + a_kmn;
    StorePtr store_cn_v = Store::make(c, {n}, v_add);
    ForPtr loop_n2 = For::make(n, 0, N, store_cn_v, thread_idx_opt);
    ForPtr loop_m1 = For::make(m, 0, M, loop_n2);
    block.push_back(loop_m1);
  }

  {
    //    b(k) = 0
    //    for n in 0..64:  // thread_idx
    //      b(k) = b(k) + c(n)
    StorePtr store_bk_0 = b.store({k}, 0.f);
    block.push_back(store_bk_0);
    ExprHandle load_bk = b.load(k);
    ExprHandle load_cn = Load::make(kFloat, c, {n});
    ExprHandle v_add = load_bk + load_cn;
    StorePtr store_bk = b.store({k}, v_add);
    ForPtr loop_n3 = For::make(n, 0, N, store_bk, thread_idx_opt);
    block.push_back(loop_n3);
  }

  {
    //    free(c)
    FreePtr free_stmt = Free::make(c);
    block.push_back(free_stmt);
  }

  BlockPtr reduce_body = Block::make(block);
  ForPtr loop_k1 = For::make(k, 0, 1, reduce_body, block_idx_opt);

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
  for (const auto i : c10::irange(M)) {
    for (const auto j : c10::irange(N)) {
      int v = i + j;
      a_v(0, i, j) = v;
      b_ref(0) += v;
    }
  }

  float* a_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&a_dev, kTotalSize * sizeof(float)));
  C10_CUDA_CHECK(cudaMemcpy(
      a_dev, a_v.data(), kTotalSize * sizeof(float), cudaMemcpyHostToDevice));
  float* b_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&b_dev, 1 * sizeof(float)));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  cuda_cg(a_dev, b_dev);

  C10_CUDA_CHECK(cudaDeviceSynchronize());
  C10_CUDA_CHECK(
      cudaMemcpy(b_v.data(), b_dev, 1 * sizeof(float), cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  ExpectAllNear(b_v, b_ref, 1e-5);

  C10_CUDA_CHECK(cudaFree(a_dev));
  C10_CUDA_CHECK(cudaFree(b_dev));
}

TEST(Cuda, LocalMemReduce_1_CUDA) {
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

  BufHandle a("a", {1, M, N}, kFloat);
  BufHandle b("b", {1}, kFloat);
  VarHandle k("k", kInt);
  VarHandle m("m", kInt);
  VarHandle n("n", kInt);

  BufHandle c{
      alloc<Buf>("c", std::vector<ExprPtr>({alloc<IntImm>(1)}), kFloat)};
  std::vector<StmtPtr> block_k;
  {
    //    b(k) = 0
    StorePtr store_bk_0 = b.store({k}, 0.f);
    block_k.push_back(store_bk_0);
  }
  std::vector<StmtPtr> block_n;
  {
    // alloc(c, 1);
    AllocatePtr alloc = Allocate::make(c);
    block_n.push_back(alloc);
  }
  {
    // c(0) = 0
    StorePtr store_c0_0 = Store::make(c, {0}, 0.f);
    block_n.push_back(store_c0_0);
  }
  {
    //      for m in 0..128:
    //        c(0) = c(0) + a(k, m, n)
    ExprHandle load_c0 = Load::make(kFloat, c, {0});
    ExprHandle a_kmn = a.load(k * (M * N) + m * N + n);
    ExprHandle v_add = load_c0 + a_kmn;
    StorePtr store_c0_v = Store::make(c, {0}, v_add);
    ForPtr loop_m = For::make(m, 0, M, store_c0_v);
    block_n.push_back(loop_m);
  }
  {
    //      b(k) = b(k) + c(0)
    ExprHandle load_bk = b.load(k);
    ExprHandle load_c0 = Load::make(kFloat, c, {0});
    ExprHandle v_add = load_bk + load_c0;
    StorePtr store_bk = b.store({k}, v_add);
    block_n.push_back(store_bk);
  }
  {
    //      free(c)
    FreePtr free_stmt = Free::make(c);
    block_n.push_back(free_stmt);
  }
  {
    BlockPtr block_n_stmt = Block::make(block_n);
    ForPtr for_n = For::make(n, 0, N, block_n_stmt, thread_idx_opt);
    block_k.push_back(for_n);
  }
  BlockPtr block_k_stmt = Block::make(block_k);
  ForPtr loop_k = For::make(k, 0, 1, block_k_stmt, block_idx_opt);

  CudaCodeGen cuda_cg(loop_k, a, b);
  PaddedBuffer<float> a_v(1, M, N, "a_v");
  PaddedBuffer<float> b_v(1, "b_v");
  PaddedBuffer<float> b_ref(1, "b_ref");

  b_ref(0) = 0;
  for (const auto i : c10::irange(M)) {
    for (const auto j : c10::irange(N)) {
      int v = i + j;
      a_v(0, i, j) = v;
      b_ref(0) += v;
    }
  }

  float* a_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&a_dev, kTotalSize * sizeof(float)));
  C10_CUDA_CHECK(cudaMemcpy(
      a_dev, a_v.data(), kTotalSize * sizeof(float), cudaMemcpyHostToDevice));
  float* b_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&b_dev, 1 * sizeof(float)));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  cuda_cg(a_dev, b_dev);

  C10_CUDA_CHECK(cudaDeviceSynchronize());
  C10_CUDA_CHECK(
      cudaMemcpy(b_v.data(), b_dev, 1 * sizeof(float), cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  ExpectAllNear(b_v, b_ref, 1e-5);

  C10_CUDA_CHECK(cudaFree(a_dev));
  C10_CUDA_CHECK(cudaFree(b_dev));
}

TEST(Cuda, HalfSupport_CUDA) {
  auto half = ToDtype<at::Half>();
  BufHandle a("a", {4}, half);
  Tensor b = Compute("b", {4}, [&](const VarHandle& i) {
    return Cast::make(half, ExprHandle(2.0f) * a.load(i));
  });

  Tensor c = Compute("c", {4}, [&](const VarHandle& i) {
    return Cast::make(kFloat, Cast::make(half, ExprHandle(42)) + b.load(i));
  });

  Tensor d = Compute("d", {4}, [&](const VarHandle& i) {
    return Cast::make(half, c.load(i));
  });

  LoopNest l({b, c, d});
  l.prepareForCodegen();
  StmtPtr s = l.root_stmt();
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

  C10_CUDA_CHECK(cudaMalloc(&aDev, aSize));
  C10_CUDA_CHECK(cudaMalloc(&bDev, bSize));
  C10_CUDA_CHECK(cudaMalloc(&cDev, cSize));
  C10_CUDA_CHECK(cudaMalloc(&dDev, dSize));
  C10_CUDA_CHECK(cudaMemcpy(aDev, aData.data(), aSize, cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(cDev, cData.data(), cSize, cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(dDev, dData.data(), dSize, cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  cg.call({aDev, bDev, cDev, dDev});
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  C10_CUDA_CHECK(cudaMemcpy(aData.data(), aDev, aSize, cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaMemcpy(cData.data(), cDev, cSize, cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaMemcpy(dData.data(), dDev, dSize, cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  assertAllEqual(cData, 46.0f);

  C10_CUDA_CHECK(cudaFree(aDev));
  C10_CUDA_CHECK(cudaFree(bDev));
  C10_CUDA_CHECK(cudaFree(cDev));
  C10_CUDA_CHECK(cudaFree(dDev));
}

TEST(Cuda, HalfPropagation_CUDA) {
  auto half = ToDtype<at::Half>();
  BufHandle a("a", {4}, half);
  Tensor relu = Compute("relu", {4}, [&](const VarHandle& i) {
    return Max::make(a.load(i), ExprHandle(alloc<HalfImm>(0)), true);
  });

  LoopNest l({relu});
  l.prepareForCodegen();
  StmtPtr s = l.root_stmt();
  CudaCodeGen cg(s, {a, relu});

  std::ostringstream oss;
  oss << *cg.stmt();

  // Check the types used by the Max are Float.
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (
# CHECK:  float v = float(a[i]);
# CHECK:  relu[i] = half(Max(v, 0.f
# CHECK: })IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  std::vector<at::Half> aData(4, 2.0f);
  std::vector<at::Half> reluData(4, 0.0f);
  at::Half* aDev = nullptr;
  at::Half* reluDev = nullptr;
  auto aSize = aData.size() * sizeof(aData[0]);
  auto reluSize = reluData.size() * sizeof(reluData[0]);

  C10_CUDA_CHECK(cudaMalloc(&aDev, aSize));
  C10_CUDA_CHECK(cudaMalloc(&reluDev, reluSize));
  C10_CUDA_CHECK(cudaMemcpy(aDev, aData.data(), aSize, cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(
      cudaMemcpy(reluDev, reluData.data(), reluSize, cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  cg.call({aDev, reluDev});
  C10_CUDA_CHECK(
      cudaMemcpy(reluData.data(), reluDev, reluSize, cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  assertAllEqual(aData, reluData);

  C10_CUDA_CHECK(cudaFree(aDev));
  C10_CUDA_CHECK(cudaFree(reluDev));
}

TEST(Cuda, UnusedHalfArgument_CUDA) {
  BufHandle a("a", {4}, kFloat);
  auto half = ToDtype<at::Half>();
  BufHandle b("b", {4}, half);
  Tensor relu = Compute("relu", {4}, [&](const VarHandle& i) {
    return Max::make(a.load(i), ExprHandle(alloc<FloatImm>(0)), true);
  });

  LoopNest l({relu});
  l.prepareForCodegen();
  StmtPtr s = l.root_stmt();
  CudaCodeGen cg(s, {a, b, relu});

  std::ostringstream oss;
  oss << *cg.stmt();

  // Check the types used by the Max are Float.
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (
# CHECK:  float v = a[i];
# CHECK:  relu[i] = Max(v, 0.f
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

  C10_CUDA_CHECK(cudaMalloc(&aDev, aSize));
  C10_CUDA_CHECK(cudaMalloc(&bDev, bSize));
  C10_CUDA_CHECK(cudaMalloc(&reluDev, reluSize));
  C10_CUDA_CHECK(cudaMemcpy(aDev, aData.data(), aSize, cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(bDev, bData.data(), bSize, cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(
      cudaMemcpy(reluDev, reluData.data(), reluSize, cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  cg.call({aDev, bDev, reluDev});
  C10_CUDA_CHECK(
      cudaMemcpy(reluData.data(), reluDev, reluSize, cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  assertAllEqual(aData, reluData);

  C10_CUDA_CHECK(cudaFree(aDev));
  C10_CUDA_CHECK(cudaFree(bDev));
  C10_CUDA_CHECK(cudaFree(reluDev));
}

TEST(Cuda, PrioritizeDependents_CUDA) {
  BufHandle a("a", {10}, kFloat);
  BufHandle b("b", {12}, kFloat);
  BufHandle c("c", {12}, kFloat);

  LoopOptions block_idx_opt;
  block_idx_opt.set_gpu_block_index(0);

  VarHandle i("i", kInt);
  VarHandle j("j", kInt);

  /*
   * for (const auto i : c10::irange(12)) {
   *   c[i] = (i < 10 ? a[i] + b[i] : b[i]);
   * }
   */
  ExprHandle load_a = a.load({i});
  ExprHandle load_b = b.load({i});
  ExprHandle cmp = CompareSelect::make(i, 10, CompareSelectOperation::kLT);
  ExprHandle ite = IfThenElse::make(cmp, Add::make(load_a, load_b), load_b);

  ForPtr loop =
      For::make(i, 0, 12, Block::make({c.store({i}, ite)}), block_idx_opt);

  CudaCodeGen cuda_cg(loop, a, b, c);

  PaddedBuffer<float> a_v(10, "a_v");
  PaddedBuffer<float> b_v(12, "b_v");
  PaddedBuffer<float> c_v(12, "c_v");
  PaddedBuffer<float> c_ref(12, "c_ref");

  for (const auto i : c10::irange(10)) {
    a_v(i) = i * 100;
    b_v(i) = i;
    c_v(i) = 0;
  }

  for (const auto i : c10::irange(10, 12)) {
    b_v(i) = i;
    c_v(i) = 0;
  }

  float* a_dev = nullptr;
  float* b_dev = nullptr;
  float* c_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&a_dev, 10 * sizeof(float)));
  C10_CUDA_CHECK(cudaMalloc(&b_dev, 12 * sizeof(float)));
  C10_CUDA_CHECK(cudaMalloc(&c_dev, 12 * sizeof(float)));

  C10_CUDA_CHECK(cudaMemcpy(
      a_dev, a_v.data(), 10 * sizeof(float), cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      b_dev, b_v.data(), 12 * sizeof(float), cudaMemcpyHostToDevice));

  C10_CUDA_CHECK(cudaDeviceSynchronize());

  cuda_cg(a_dev, b_dev, c_dev);

  C10_CUDA_CHECK(cudaDeviceSynchronize());
  C10_CUDA_CHECK(cudaMemcpy(
      c_v.data(), c_dev, 12 * sizeof(float), cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  for (const auto i : c10::irange(12)) {
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
  int A_SIZE = 100;
  int B_SIZE = 50;
  BufHandle a_buf("a", {A_SIZE}, kFloat);
  BufHandle b_buf("b", {B_SIZE}, kFloat);
  Tensor c = Compute(
      "c", {A_SIZE}, [&](const VarHandle& i) { return a_buf.load(i) + 10; });
  Tensor d = Compute("d", {B_SIZE}, [&](const VarHandle& i) {
    return a_buf.load(i) + b_buf.load(i);
  });

  LoopNest l({c, d});
  std::vector<ForPtr> loops = l.getLoopStmtsFor(c);
  loops[0]->set_gpu_block_index(0);
  loops = l.getLoopStmtsFor(d);
  loops[0]->set_gpu_block_index(0);

  l.prepareForCodegen();
  StmtPtr stmt = l.root_stmt();
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
  ASSERT_TRUE(exprEquals(blockExtents[0], alloc<IntImm>(A_SIZE)));
  ASSERT_TRUE(exprEquals(threadExtents[0], alloc<IntImm>(1)));

  // Sanity check that the kernel works.
  PaddedBuffer<float> a_v(A_SIZE);
  PaddedBuffer<float> b_v(B_SIZE);
  PaddedBuffer<float> c_v(A_SIZE);
  PaddedBuffer<float> d_v(B_SIZE);

  PaddedBuffer<float> c_ref(A_SIZE);
  PaddedBuffer<float> d_ref(B_SIZE);

  for (const auto i : c10::irange(A_SIZE)) {
    a_v(i) = (float)i;
    c_ref(i) = (float)(i + 10);
  }

  for (const auto i : c10::irange(B_SIZE)) {
    b_v(i) = (float)(B_SIZE - i);
    d_ref(i) = a_v(i) + b_v(i);
  }

  float* a_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&a_dev, A_SIZE * sizeof(float)));
  float* b_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&b_dev, B_SIZE * sizeof(float)));
  float* c_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&c_dev, A_SIZE * sizeof(float)));
  float* d_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&d_dev, B_SIZE * sizeof(float)));
  C10_CUDA_CHECK(cudaMemcpy(
      a_dev, a_v.data(), A_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      b_dev, b_v.data(), B_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      c_dev, c_v.data(), A_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      d_dev, d_v.data(), B_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  cuda_cg(c_dev, d_dev, a_dev, b_dev);

  C10_CUDA_CHECK(cudaDeviceSynchronize());
  C10_CUDA_CHECK(cudaMemcpy(
      c_v.data(), c_dev, A_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaMemcpy(
      d_v.data(), d_dev, B_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  ExpectAllNear(c_v, c_ref, 1e-5);
  ExpectAllNear(d_v, d_ref, 1e-5);

  C10_CUDA_CHECK(cudaFree(a_dev));
  C10_CUDA_CHECK(cudaFree(b_dev));
  C10_CUDA_CHECK(cudaFree(c_dev));
  C10_CUDA_CHECK(cudaFree(d_dev));
}

/// Tests the case with two loops, which have different extents that are bound
/// to the same thread dimension. This is the same as the above - the smaller
/// rank write should be masked. But this time we also need to syncthreads.
TEST(Cuda, MaskThreadDim_CUDA) {
  int A_SIZE = 50;
  int B_SIZE = 100;
  BufHandle a_buf("a", {A_SIZE}, kFloat);
  BufHandle b_buf("b", {B_SIZE}, kFloat);
  Tensor c = Compute(
      "c", {A_SIZE}, [&](const VarHandle& i) { return a_buf.load(i) + 10; });
  Tensor d = Compute("d", {B_SIZE}, [&](const VarHandle& i) {
    return a_buf.load(i / 2) + b_buf.load(i);
  });

  LoopNest l({c, d});
  std::vector<ForPtr> loops = l.getLoopStmtsFor(c);
  loops[0]->set_gpu_thread_index(0);
  loops = l.getLoopStmtsFor(d);
  loops[0]->set_gpu_thread_index(0);

  l.prepareForCodegen();
  StmtPtr stmt = l.root_stmt();
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
  ASSERT_TRUE(exprEquals(blockExtents[0], alloc<IntImm>(1)));
  ASSERT_TRUE(exprEquals(threadExtents[0], alloc<IntImm>(B_SIZE)));

  PaddedBuffer<float> a_v(A_SIZE);
  PaddedBuffer<float> b_v(B_SIZE);
  PaddedBuffer<float> c_v(A_SIZE);
  PaddedBuffer<float> d_v(B_SIZE);

  PaddedBuffer<float> c_ref(A_SIZE);
  PaddedBuffer<float> d_ref(B_SIZE);

  for (const auto i : c10::irange(A_SIZE)) {
    a_v(i) = (float)i;
    c_ref(i) = (float)(i + 10);
  }

  for (const auto i : c10::irange(B_SIZE)) {
    b_v(i) = (float)(B_SIZE - i);
    d_ref(i) = a_v(i / 2) + b_v(i);
  }

  float* a_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&a_dev, A_SIZE * sizeof(float)));
  float* b_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&b_dev, B_SIZE * sizeof(float)));
  float* c_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&c_dev, A_SIZE * sizeof(float)));
  float* d_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&d_dev, B_SIZE * sizeof(float)));
  C10_CUDA_CHECK(cudaMemcpy(
      a_dev, a_v.data(), A_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      b_dev, b_v.data(), B_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      c_dev, c_v.data(), A_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      d_dev, d_v.data(), B_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  cuda_cg(c_dev, d_dev, a_dev, b_dev);

  C10_CUDA_CHECK(cudaDeviceSynchronize());
  C10_CUDA_CHECK(cudaMemcpy(
      c_v.data(), c_dev, A_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaMemcpy(
      d_v.data(), d_dev, B_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  ExpectAllNear(c_v, c_ref, 1e-5);
  ExpectAllNear(d_v, d_ref, 1e-5);

  C10_CUDA_CHECK(cudaFree(a_dev));
  C10_CUDA_CHECK(cudaFree(b_dev));
  C10_CUDA_CHECK(cudaFree(c_dev));
  C10_CUDA_CHECK(cudaFree(d_dev));
}

/// Tests the case where there are two loops, and each is bound to a different
/// block dimension. In this case all writes should be masked since they occur
/// in distinct dimensions.
// Note: this is an extremely dumb pattern which we should never see, but is a
// useful edge case to make sure we've got things covered.
TEST(Cuda, MaskMultiBlockDim_CUDA) {
  int A_SIZE = 100;
  int B_SIZE = 50;
  BufHandle a_buf("a", {A_SIZE}, kFloat);
  BufHandle b_buf("b", {B_SIZE}, kFloat);
  Tensor c = Compute(
      "c", {A_SIZE}, [&](const VarHandle& i) { return a_buf.load(i) + 10; });
  Tensor d = Compute("d", {B_SIZE}, [&](const VarHandle& i) {
    return a_buf.load(i) + b_buf.load(i);
  });

  LoopNest l({c, d});
  std::vector<ForPtr> loops = l.getLoopStmtsFor(c);
  loops[0]->set_gpu_block_index(0);
  loops = l.getLoopStmtsFor(d);
  loops[0]->set_gpu_block_index(1);

  l.prepareForCodegen();
  StmtPtr stmt = l.root_stmt();
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
  ASSERT_TRUE(exprEquals(blockExtents[0], alloc<IntImm>(A_SIZE)));
  ASSERT_TRUE(exprEquals(blockExtents[1], alloc<IntImm>(B_SIZE)));

  PaddedBuffer<float> a_v(A_SIZE);
  PaddedBuffer<float> b_v(B_SIZE);
  PaddedBuffer<float> c_v(A_SIZE);
  PaddedBuffer<float> d_v(B_SIZE);

  PaddedBuffer<float> c_ref(A_SIZE);
  PaddedBuffer<float> d_ref(B_SIZE);

  for (const auto i : c10::irange(A_SIZE)) {
    a_v(i) = (float)i;
    c_ref(i) = (float)(i + 10);
  }

  for (const auto i : c10::irange(B_SIZE)) {
    b_v(i) = (float)(B_SIZE - i);
    d_ref(i) = a_v(i) + b_v(i);
  }

  float* a_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&a_dev, A_SIZE * sizeof(float)));
  float* b_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&b_dev, B_SIZE * sizeof(float)));
  float* c_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&c_dev, A_SIZE * sizeof(float)));
  float* d_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&d_dev, B_SIZE * sizeof(float)));
  C10_CUDA_CHECK(cudaMemcpy(
      a_dev, a_v.data(), A_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      b_dev, b_v.data(), B_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      c_dev, c_v.data(), A_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      d_dev, d_v.data(), B_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  cuda_cg(c_dev, d_dev, a_dev, b_dev);

  C10_CUDA_CHECK(cudaDeviceSynchronize());
  C10_CUDA_CHECK(cudaMemcpy(
      c_v.data(), c_dev, A_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaMemcpy(
      d_v.data(), d_dev, B_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  ExpectAllNear(c_v, c_ref, 1e-5);
  ExpectAllNear(d_v, d_ref, 1e-5);

  C10_CUDA_CHECK(cudaFree(a_dev));
  C10_CUDA_CHECK(cudaFree(b_dev));
  C10_CUDA_CHECK(cudaFree(c_dev));
  C10_CUDA_CHECK(cudaFree(d_dev));
}

/// Tests the case where both the blockDim and threadDim are bound to different
/// loops. In this instance both stores should be masked since they are
/// distinct.
// Note: this is an extremely dumb pattern which we should never see, but is a
// useful edge case to make sure we've got things covered.
TEST(Cuda, MaskBlockAndThreadDim_CUDA) {
  int A_SIZE = 100;
  int B_SIZE = 50;
  BufHandle a_buf("a", {A_SIZE}, kFloat);
  BufHandle b_buf("b", {B_SIZE}, kFloat);
  Tensor c = Compute(
      "c", {A_SIZE}, [&](const VarHandle& i) { return a_buf.load(i) + 10; });
  Tensor d = Compute("d", {B_SIZE}, [&](const VarHandle& i) {
    return a_buf.load(i) + b_buf.load(i);
  });

  LoopNest l({c, d});
  std::vector<ForPtr> loops = l.getLoopStmtsFor(c);
  loops[0]->set_gpu_block_index(0);
  loops = l.getLoopStmtsFor(d);
  loops[0]->set_gpu_thread_index(0);

  l.prepareForCodegen();
  StmtPtr stmt = l.root_stmt();
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
  ASSERT_TRUE(exprEquals(blockExtents[0], alloc<IntImm>(A_SIZE)));
  ASSERT_TRUE(exprEquals(threadExtents[0], alloc<IntImm>(B_SIZE)));

  PaddedBuffer<float> a_v(A_SIZE);
  PaddedBuffer<float> b_v(B_SIZE);
  PaddedBuffer<float> c_v(A_SIZE);
  PaddedBuffer<float> d_v(B_SIZE);

  PaddedBuffer<float> c_ref(A_SIZE);
  PaddedBuffer<float> d_ref(B_SIZE);

  for (const auto i : c10::irange(A_SIZE)) {
    a_v(i) = (float)i;
    c_ref(i) = (float)(i + 10);
  }

  for (const auto i : c10::irange(B_SIZE)) {
    b_v(i) = (float)(B_SIZE - i);
    d_ref(i) = a_v(i) + b_v(i);
  }

  float* a_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&a_dev, A_SIZE * sizeof(float)));
  float* b_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&b_dev, B_SIZE * sizeof(float)));
  float* c_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&c_dev, A_SIZE * sizeof(float)));
  float* d_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&d_dev, B_SIZE * sizeof(float)));
  C10_CUDA_CHECK(cudaMemcpy(
      a_dev, a_v.data(), A_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      b_dev, b_v.data(), B_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      c_dev, c_v.data(), A_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      d_dev, d_v.data(), B_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  cuda_cg(c_dev, d_dev, a_dev, b_dev);

  C10_CUDA_CHECK(cudaDeviceSynchronize());
  C10_CUDA_CHECK(cudaMemcpy(
      c_v.data(), c_dev, A_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaMemcpy(
      d_v.data(), d_dev, B_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  ExpectAllNear(c_v, c_ref, 1e-5);
  ExpectAllNear(d_v, d_ref, 1e-5);

  C10_CUDA_CHECK(cudaFree(a_dev));
  C10_CUDA_CHECK(cudaFree(b_dev));
  C10_CUDA_CHECK(cudaFree(c_dev));
  C10_CUDA_CHECK(cudaFree(d_dev));
}

/// Tests the case where the loopnest has two loops of depth two: each with the
/// outer loop bound to blockDim.x and the inner loop bound to threadDim.x. In
/// this case all writes with a rank smaller than the max should be masked.
TEST(Cuda, MaskMultiDim_CUDA) {
  int OUTER_SIZE = 10;
  int A_SIZE = 100;
  int B_SIZE = 50;
  BufHandle a_buf("a", {OUTER_SIZE, A_SIZE}, kFloat);
  BufHandle b_buf("b", {OUTER_SIZE, B_SIZE}, kFloat);
  Tensor c = Compute(
      "C", {OUTER_SIZE, A_SIZE}, [&](const VarHandle& i, const VarHandle& j) {
        return ExprHandle(2) * a_buf.load(i, j);
      });
  Tensor d = Compute(
      "D", {OUTER_SIZE, B_SIZE}, [&](const VarHandle& i, const VarHandle& j) {
        return c.load(i, j * 2) + b_buf.load(i, j);
      });

  LoopNest l({c, d});
  std::vector<ForPtr> loops = l.getLoopStmtsFor(c);
  loops[0]->set_gpu_block_index(0);
  loops[1]->set_gpu_thread_index(0);
  loops = l.getLoopStmtsFor(d);
  loops[0]->set_gpu_block_index(0);
  loops[1]->set_gpu_thread_index(0);

  l.prepareForCodegen();
  StmtPtr stmt = l.root_stmt();
  CudaCodeGen cuda_cg(stmt, c, d, a_buf, b_buf);

  std::ostringstream oss;
  oss << *cuda_cg.stmt();

  // The write to D should be masked, but not the write to C.
  const std::string& verification_pattern =
      R"IR(
# CHECK-NOT: if (
# CHECK: C[threadIdx.x + 100 * blockIdx.x] =
# CHECK: __syncthreads();
# CHECK: if (threadIdx.x<50
# CHECK:   D[threadIdx.x + 50 * blockIdx.x] =)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  auto blockExtents = cuda_cg.gpu_block_extents();
  auto threadExtents = cuda_cg.gpu_thread_extents();
  ASSERT_TRUE(exprEquals(blockExtents[0], alloc<IntImm>(OUTER_SIZE)));
  ASSERT_TRUE(exprEquals(threadExtents[0], alloc<IntImm>(A_SIZE)));

  PaddedBuffer<float> a_v(OUTER_SIZE, A_SIZE);
  PaddedBuffer<float> b_v(OUTER_SIZE, B_SIZE);
  PaddedBuffer<float> c_v(OUTER_SIZE, A_SIZE);
  PaddedBuffer<float> d_v(OUTER_SIZE, B_SIZE);

  PaddedBuffer<float> c_ref(OUTER_SIZE, A_SIZE);
  PaddedBuffer<float> d_ref(OUTER_SIZE, B_SIZE);

  for (const auto o : c10::irange(OUTER_SIZE)) {
    for (const auto i : c10::irange(A_SIZE)) {
      a_v(o, i) = (float)i;
      c_ref(o, i) = (float)(i * 2);
    }
  }

  for (const auto o : c10::irange(OUTER_SIZE)) {
    for (const auto i : c10::irange(B_SIZE)) {
      b_v(o, i) = (float)(B_SIZE - i);
      d_ref(o, i) = c_ref(o, i * 2) + b_v(o, i);
    }
  }

  float* a_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&a_dev, OUTER_SIZE * A_SIZE * sizeof(float)));
  float* b_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&b_dev, OUTER_SIZE * B_SIZE * sizeof(float)));
  float* c_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&c_dev, OUTER_SIZE * A_SIZE * sizeof(float)));
  float* d_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&d_dev, OUTER_SIZE * B_SIZE * sizeof(float)));
  C10_CUDA_CHECK(cudaMemcpy(
      a_dev,
      a_v.data(),
      OUTER_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      b_dev,
      b_v.data(),
      OUTER_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      c_dev,
      c_v.data(),
      OUTER_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      d_dev,
      d_v.data(),
      OUTER_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  cuda_cg(c_dev, d_dev, a_dev, b_dev);

  C10_CUDA_CHECK(cudaDeviceSynchronize());
  C10_CUDA_CHECK(cudaMemcpy(
      c_v.data(),
      c_dev,
      OUTER_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaMemcpy(
      d_v.data(),
      d_dev,
      OUTER_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  ExpectAllNear(c_v, c_ref, 1e-5);
  ExpectAllNear(d_v, d_ref, 1e-5);

  C10_CUDA_CHECK(cudaFree(a_dev));
  C10_CUDA_CHECK(cudaFree(b_dev));
  C10_CUDA_CHECK(cudaFree(c_dev));
  C10_CUDA_CHECK(cudaFree(d_dev));
}

// Tests the case where loop extents are symbolic and not known at compile time.
// In this case both stores must be masked against the extent of the other loop,
// incase it is larger.
TEST(Cuda, MaskMultiDimSymbolic_CUDA) {
  VarHandle OUTER_SIZE("OUTER_SIZE", kLong);
  VarHandle A_SIZE("A_SIZE", kLong);
  VarHandle B_SIZE("B_SIZE", kLong);
  BufHandle a_buf("a", {OUTER_SIZE, A_SIZE}, kFloat);
  BufHandle b_buf("b", {OUTER_SIZE, B_SIZE}, kFloat);
  Tensor c = Compute(
      "C", {OUTER_SIZE, A_SIZE}, [&](const VarHandle& i, const VarHandle& j) {
        return ExprHandle(2) * a_buf.load(i, j);
      });
  Tensor d = Compute(
      "D", {OUTER_SIZE, B_SIZE}, [&](const VarHandle& i, const VarHandle& j) {
        return c.load(i, j * 2) + b_buf.load(i, j);
      });

  LoopNest l({c, d});
  std::vector<ForPtr> loops = l.getLoopStmtsFor(c);
  loops[0]->set_gpu_block_index(0);
  loops[1]->set_gpu_thread_index(0);
  loops = l.getLoopStmtsFor(d);
  loops[0]->set_gpu_block_index(0);
  loops[1]->set_gpu_thread_index(0);

  l.prepareForCodegen();
  StmtPtr stmt = l.root_stmt();
  CudaCodeGen cuda_cg(stmt, c, d, OUTER_SIZE, A_SIZE, B_SIZE, a_buf, b_buf);

  std::ostringstream oss;
  oss << *cuda_cg.stmt();

  // Since we don't know which is bigger (A_SIZE or B_SIZE) we must mask both.
  const std::string& verification_pattern =
      R"IR(
# CHECK: if (threadIdx.x<A_SIZE
# CHECK:   C[A_SIZE * int64_t(blockIdx.x) + int64_t(threadIdx.x)] =
# CHECK: __syncthreads();
# CHECK: if (threadIdx.x<B_SIZE
# CHECK:   D[B_SIZE * int64_t(blockIdx.x) + int64_t(threadIdx.x)] =)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  auto blockExtents = cuda_cg.gpu_block_extents();
  auto threadExtents = cuda_cg.gpu_thread_extents();
  ASSERT_TRUE(exprEquals(blockExtents[0], OUTER_SIZE.node()));
  ASSERT_TRUE(exprEquals(
      threadExtents[0], alloc<Max>(A_SIZE.node(), B_SIZE.node(), true)));

  int64_t OUTER_EXTENT = 10;
  int64_t A_EXTENT = 100;
  int64_t B_EXTENT = 50;

  PaddedBuffer<float> a_v(OUTER_EXTENT, A_EXTENT);
  PaddedBuffer<float> b_v(OUTER_EXTENT, B_EXTENT);
  PaddedBuffer<float> c_v(OUTER_EXTENT, A_EXTENT);
  PaddedBuffer<float> d_v(OUTER_EXTENT, B_EXTENT);

  PaddedBuffer<float> c_ref(OUTER_EXTENT, A_EXTENT);
  PaddedBuffer<float> d_ref(OUTER_EXTENT, B_EXTENT);

  for (const auto o : c10::irange(OUTER_EXTENT)) {
    for (const auto i : c10::irange(A_EXTENT)) {
      a_v(o, i) = (float)i;
      c_ref(o, i) = (float)(i * 2);
    }
  }

  for (const auto o : c10::irange(OUTER_EXTENT)) {
    for (const auto i : c10::irange(B_EXTENT)) {
      b_v(o, i) = (float)(B_EXTENT - i);
      d_ref(o, i) = c_ref(o, i * 2) + b_v(o, i);
    }
  }

  float* a_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&a_dev, OUTER_EXTENT * A_EXTENT * sizeof(float)));
  float* b_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&b_dev, OUTER_EXTENT * B_EXTENT * sizeof(float)));
  float* c_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&c_dev, OUTER_EXTENT * A_EXTENT * sizeof(float)));
  float* d_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&d_dev, OUTER_EXTENT * B_EXTENT * sizeof(float)));
  C10_CUDA_CHECK(cudaMemcpy(
      a_dev,
      a_v.data(),
      OUTER_EXTENT * A_EXTENT * sizeof(float),
      cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      b_dev,
      b_v.data(),
      OUTER_EXTENT * B_EXTENT * sizeof(float),
      cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      c_dev,
      c_v.data(),
      OUTER_EXTENT * A_EXTENT * sizeof(float),
      cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      d_dev,
      d_v.data(),
      OUTER_EXTENT * B_EXTENT * sizeof(float),
      cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  cuda_cg(c_dev, d_dev, OUTER_EXTENT, A_EXTENT, B_EXTENT, a_dev, b_dev);

  C10_CUDA_CHECK(cudaDeviceSynchronize());
  C10_CUDA_CHECK(cudaMemcpy(
      c_v.data(),
      c_dev,
      OUTER_EXTENT * A_EXTENT * sizeof(float),
      cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaMemcpy(
      d_v.data(),
      d_dev,
      OUTER_EXTENT * B_EXTENT * sizeof(float),
      cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  ExpectAllNear(c_v, c_ref, 1e-5);
  ExpectAllNear(d_v, d_ref, 1e-5);

  C10_CUDA_CHECK(cudaFree(a_dev));
  C10_CUDA_CHECK(cudaFree(b_dev));
  C10_CUDA_CHECK(cudaFree(c_dev));
  C10_CUDA_CHECK(cudaFree(d_dev));
}

// Tests the case where two loops are fused at a common parent loop, which is
// bound to the block dimension. Internally the inner loops have different
// extents but are bound to the same thread dimension. The smaller loop should
// be masked.
TEST(Cuda, MaskCompoundInnerLoop_CUDA) {
  int OUTER_SIZE = 10;
  int A_SIZE = 100;
  int B_SIZE = 50;
  BufHandle a_buf("a", {OUTER_SIZE, A_SIZE}, kFloat);
  BufHandle b_buf("b", {OUTER_SIZE, B_SIZE}, kFloat);
  BufHandle c_buf("c", {OUTER_SIZE, A_SIZE}, kFloat);
  BufHandle d_buf("d", {OUTER_SIZE, B_SIZE}, kFloat);

  // Can't build this using Compute and transforms yet.
  LoopOptions blockBound;
  blockBound.set_gpu_block_index(0);
  LoopOptions threadBound;
  threadBound.set_gpu_thread_index(0);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);

  StmtPtr stmt = For::make(
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
# CHECK: c[threadIdx.x + 100 * blockIdx.x] =
# CHECK: __syncthreads();
# CHECK: if (threadIdx.x<50
# CHECK:   d[threadIdx.x + 50 * blockIdx.x] =)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  auto blockExtents = cuda_cg.gpu_block_extents();
  auto threadExtents = cuda_cg.gpu_thread_extents();
  ASSERT_TRUE(exprEquals(blockExtents[0], alloc<IntImm>(OUTER_SIZE)));
  ASSERT_TRUE(exprEquals(threadExtents[0], alloc<IntImm>(A_SIZE)));

  PaddedBuffer<float> a_v(OUTER_SIZE, A_SIZE);
  PaddedBuffer<float> b_v(OUTER_SIZE, B_SIZE);
  PaddedBuffer<float> c_v(OUTER_SIZE, A_SIZE);
  PaddedBuffer<float> d_v(OUTER_SIZE, B_SIZE);

  PaddedBuffer<float> c_ref(OUTER_SIZE, A_SIZE);
  PaddedBuffer<float> d_ref(OUTER_SIZE, B_SIZE);

  for (const auto o : c10::irange(OUTER_SIZE)) {
    for (const auto i : c10::irange(A_SIZE)) {
      a_v(o, i) = (float)i;
      c_ref(o, i) = (float)(i * 2);
    }
    for (const auto i : c10::irange(B_SIZE)) {
      b_v(o, i) = (float)(B_SIZE - i);
      d_ref(o, i) = c_ref(o, i * 2) + b_v(o, i);
    }
  }

  float* a_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&a_dev, OUTER_SIZE * A_SIZE * sizeof(float)));
  float* b_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&b_dev, OUTER_SIZE * B_SIZE * sizeof(float)));
  float* c_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&c_dev, OUTER_SIZE * A_SIZE * sizeof(float)));
  float* d_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&d_dev, OUTER_SIZE * B_SIZE * sizeof(float)));
  C10_CUDA_CHECK(cudaMemcpy(
      a_dev,
      a_v.data(),
      OUTER_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      b_dev,
      b_v.data(),
      OUTER_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      c_dev,
      c_v.data(),
      OUTER_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      d_dev,
      d_v.data(),
      OUTER_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  cuda_cg(a_dev, b_dev, c_dev, d_dev);

  C10_CUDA_CHECK(cudaDeviceSynchronize());
  C10_CUDA_CHECK(cudaMemcpy(
      c_v.data(),
      c_dev,
      OUTER_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaMemcpy(
      d_v.data(),
      d_dev,
      OUTER_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  ExpectAllNear(c_v, c_ref, 1e-5);
  ExpectAllNear(d_v, d_ref, 1e-5);

  C10_CUDA_CHECK(cudaFree(a_dev));
  C10_CUDA_CHECK(cudaFree(b_dev));
  C10_CUDA_CHECK(cudaFree(c_dev));
  C10_CUDA_CHECK(cudaFree(d_dev));
}

// Tests the case with two loops fused into a common parent, which is not bound
// to any block or thread dimension - however it's two inner loops are bound to
// the first thread dimensions. This should work just like the MaskThreadDim
// test where the bigger loop is unmasked but the smaller is masked.
TEST(Cuda, MaskInnerLoopOneBlock_CUDA) {
  int OUTER_SIZE = 10;
  int A_SIZE = 100;
  int B_SIZE = 50;
  BufHandle a_buf("a", {OUTER_SIZE, A_SIZE}, kFloat);
  BufHandle b_buf("b", {OUTER_SIZE, B_SIZE}, kFloat);
  BufHandle c_buf("c", {OUTER_SIZE, A_SIZE}, kFloat);
  BufHandle d_buf("d", {OUTER_SIZE, B_SIZE}, kFloat);

  // Can't build this using Compute and transforms yet.
  LoopOptions blockBound;
  blockBound.set_gpu_block_index(0);
  LoopOptions threadBound;
  threadBound.set_gpu_thread_index(0);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);

  StmtPtr stmt = For::make(
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
# CHECK: c[threadIdx.x + 100 * i] =
# CHECK: __syncthreads();
# CHECK: if (threadIdx.x<50
# CHECK:   d[threadIdx.x + 50 * i] =)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  auto blockExtents = cuda_cg.gpu_block_extents();
  auto threadExtents = cuda_cg.gpu_thread_extents();
  ASSERT_TRUE(exprEquals(blockExtents[0], alloc<IntImm>(1)));
  ASSERT_TRUE(exprEquals(threadExtents[0], alloc<IntImm>(A_SIZE)));

  PaddedBuffer<float> a_v(OUTER_SIZE, A_SIZE);
  PaddedBuffer<float> b_v(OUTER_SIZE, B_SIZE);
  PaddedBuffer<float> c_v(OUTER_SIZE, A_SIZE);
  PaddedBuffer<float> d_v(OUTER_SIZE, B_SIZE);

  PaddedBuffer<float> c_ref(OUTER_SIZE, A_SIZE);
  PaddedBuffer<float> d_ref(OUTER_SIZE, B_SIZE);

  for (const auto o : c10::irange(OUTER_SIZE)) {
    for (const auto i : c10::irange(A_SIZE)) {
      a_v(o, i) = (float)i;
      c_ref(o, i) = (float)(i * 2);
    }
    for (const auto i : c10::irange(B_SIZE)) {
      b_v(o, i) = (float)(B_SIZE - i);
      d_ref(o, i) = c_ref(o, i * 2) + b_v(o, i);
    }
  }

  float* a_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&a_dev, OUTER_SIZE * A_SIZE * sizeof(float)));
  float* b_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&b_dev, OUTER_SIZE * B_SIZE * sizeof(float)));
  float* c_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&c_dev, OUTER_SIZE * A_SIZE * sizeof(float)));
  float* d_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&d_dev, OUTER_SIZE * B_SIZE * sizeof(float)));
  C10_CUDA_CHECK(cudaMemcpy(
      a_dev,
      a_v.data(),
      OUTER_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      b_dev,
      b_v.data(),
      OUTER_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      c_dev,
      c_v.data(),
      OUTER_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      d_dev,
      d_v.data(),
      OUTER_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  cuda_cg(a_dev, b_dev, c_dev, d_dev);

  C10_CUDA_CHECK(cudaDeviceSynchronize());
  C10_CUDA_CHECK(cudaMemcpy(
      c_v.data(),
      c_dev,
      OUTER_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaMemcpy(
      d_v.data(),
      d_dev,
      OUTER_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  ExpectAllNear(c_v, c_ref, 1e-5);
  ExpectAllNear(d_v, d_ref, 1e-5);

  C10_CUDA_CHECK(cudaFree(a_dev));
  C10_CUDA_CHECK(cudaFree(b_dev));
  C10_CUDA_CHECK(cudaFree(c_dev));
  C10_CUDA_CHECK(cudaFree(d_dev));
}

// Tests the case with two loop nests, each of which bound to the same block
// size, but with internal loops bound to different thread rank (ie x and y). In
// this case both bodies must be masked against the other dimension being > 0.
// Note: this is a bit degenerate no one would actually write this for perf.
TEST(Cuda, MaskMultiDimMultiAxis_CUDA) {
  int OUTER_SIZE = 10;
  int A_SIZE = 30;
  int B_SIZE = 15;
  BufHandle a_buf("a", {OUTER_SIZE, A_SIZE}, kFloat);
  BufHandle b_buf("b", {OUTER_SIZE, B_SIZE}, kFloat);
  Tensor c = Compute(
      "C", {OUTER_SIZE, A_SIZE}, [&](const VarHandle& i, const VarHandle& j) {
        return ExprHandle(2) * a_buf.load(i, j);
      });
  Tensor d = Compute(
      "D", {OUTER_SIZE, B_SIZE}, [&](const VarHandle& i, const VarHandle& j) {
        return c.load(i, j * 2) + b_buf.load(i, j);
      });

  LoopNest l({c, d});
  std::vector<ForPtr> loops = l.getLoopStmtsFor(c);
  loops[0]->set_gpu_block_index(0);
  loops[1]->set_gpu_thread_index(0);
  loops = l.getLoopStmtsFor(d);
  loops[0]->set_gpu_block_index(0);
  loops[1]->set_gpu_thread_index(1);

  l.prepareForCodegen();
  StmtPtr stmt = l.root_stmt();
  CudaCodeGen cuda_cg(stmt, c, d, a_buf, b_buf);

  std::ostringstream oss;
  oss << *cuda_cg.stmt();

  // Both stores masked agaist the other thread dim < 1.
  const std::string& verification_pattern =
      R"IR(
# CHECK: if (threadIdx.y<1
# CHECK:   C[threadIdx.x + 30 * blockIdx.x] =
# CHECK: __syncthreads();
# CHECK: if (threadIdx.x<1
# CHECK:   D[threadIdx.y + 15 * blockIdx.x] =)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  auto blockExtents = cuda_cg.gpu_block_extents();
  auto threadExtents = cuda_cg.gpu_thread_extents();
  ASSERT_TRUE(exprEquals(blockExtents[0], alloc<IntImm>(OUTER_SIZE)));
  ASSERT_TRUE(exprEquals(threadExtents[0], alloc<IntImm>(A_SIZE)));

  PaddedBuffer<float> a_v(OUTER_SIZE, A_SIZE);
  PaddedBuffer<float> b_v(OUTER_SIZE, B_SIZE);
  PaddedBuffer<float> c_v(OUTER_SIZE, A_SIZE);
  PaddedBuffer<float> d_v(OUTER_SIZE, B_SIZE);

  PaddedBuffer<float> c_ref(OUTER_SIZE, A_SIZE);
  PaddedBuffer<float> d_ref(OUTER_SIZE, B_SIZE);

  for (const auto o : c10::irange(OUTER_SIZE)) {
    for (const auto i : c10::irange(A_SIZE)) {
      a_v(o, i) = (float)i;
      c_ref(o, i) = (float)(i * 2);
    }
  }

  for (const auto o : c10::irange(OUTER_SIZE)) {
    for (const auto i : c10::irange(B_SIZE)) {
      b_v(o, i) = (float)(B_SIZE - i);
      d_ref(o, i) = c_ref(o, i * 2) + b_v(o, i);
    }
  }

  float* a_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&a_dev, OUTER_SIZE * A_SIZE * sizeof(float)));
  float* b_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&b_dev, OUTER_SIZE * B_SIZE * sizeof(float)));
  float* c_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&c_dev, OUTER_SIZE * A_SIZE * sizeof(float)));
  float* d_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&d_dev, OUTER_SIZE * B_SIZE * sizeof(float)));
  C10_CUDA_CHECK(cudaMemcpy(
      a_dev,
      a_v.data(),
      OUTER_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      b_dev,
      b_v.data(),
      OUTER_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      c_dev,
      c_v.data(),
      OUTER_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      d_dev,
      d_v.data(),
      OUTER_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  cuda_cg(c_dev, d_dev, a_dev, b_dev);

  C10_CUDA_CHECK(cudaDeviceSynchronize());
  C10_CUDA_CHECK(cudaMemcpy(
      c_v.data(),
      c_dev,
      OUTER_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaMemcpy(
      d_v.data(),
      d_dev,
      OUTER_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  ExpectAllNear(c_v, c_ref, 1e-5);
  ExpectAllNear(d_v, d_ref, 1e-5);

  C10_CUDA_CHECK(cudaFree(a_dev));
  C10_CUDA_CHECK(cudaFree(b_dev));
  C10_CUDA_CHECK(cudaFree(c_dev));
  C10_CUDA_CHECK(cudaFree(d_dev));
}

// Tests the case with two loop nests, each bound to both Block and Thread but
// the second loop is smaller in both cases - the second store must be masked
// for both the block and thread dimension.
TEST(Cuda, MaskMultiDimMultiLevel_CUDA) {
  int OUTER_A_SIZE = 10;
  int OUTER_B_SIZE = 5;
  int A_SIZE = 30;
  int B_SIZE = 15;
  BufHandle a_buf("a", {OUTER_A_SIZE, A_SIZE}, kFloat);
  BufHandle b_buf("b", {OUTER_B_SIZE, B_SIZE}, kFloat);
  Tensor c = Compute(
      "C", {OUTER_A_SIZE, A_SIZE}, [&](const VarHandle& i, const VarHandle& j) {
        return ExprHandle(2) * a_buf.load(i, j);
      });
  Tensor d = Compute(
      "D", {OUTER_B_SIZE, B_SIZE}, [&](const VarHandle& i, const VarHandle& j) {
        return c.load(i, j * 2) + b_buf.load(i, j);
      });

  LoopNest l({c, d});
  std::vector<ForPtr> loops = l.getLoopStmtsFor(c);
  loops[0]->set_gpu_block_index(0);
  loops[1]->set_gpu_thread_index(0);
  loops = l.getLoopStmtsFor(d);
  loops[0]->set_gpu_block_index(0);
  loops[1]->set_gpu_thread_index(0);

  l.prepareForCodegen();
  StmtPtr stmt = l.root_stmt();
  CudaCodeGen cuda_cg(stmt, c, d, a_buf, b_buf);

  std::ostringstream oss;
  oss << *cuda_cg.stmt();

  // The write to D should be masked twice, but not the write to C.
  const std::string& verification_pattern =
      R"IR(
# CHECK-NOT: if (
# CHECK: C[threadIdx.x + 30 * blockIdx.x] =
# CHECK: __syncthreads();
# CHECK: if (blockIdx.x<5
# CHECK:   if (threadIdx.x<15
# CHECK:     D[threadIdx.x + 15 * blockIdx.x] =)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  auto blockExtents = cuda_cg.gpu_block_extents();
  auto threadExtents = cuda_cg.gpu_thread_extents();
  ASSERT_TRUE(exprEquals(blockExtents[0], alloc<IntImm>(OUTER_A_SIZE)));
  ASSERT_TRUE(exprEquals(threadExtents[0], alloc<IntImm>(A_SIZE)));

  PaddedBuffer<float> a_v(OUTER_A_SIZE, A_SIZE);
  PaddedBuffer<float> b_v(OUTER_B_SIZE, B_SIZE);
  PaddedBuffer<float> c_v(OUTER_A_SIZE, A_SIZE);
  PaddedBuffer<float> d_v(OUTER_B_SIZE, B_SIZE);

  PaddedBuffer<float> c_ref(OUTER_A_SIZE, A_SIZE);
  PaddedBuffer<float> d_ref(OUTER_B_SIZE, B_SIZE);

  for (const auto o : c10::irange(OUTER_A_SIZE)) {
    for (const auto i : c10::irange(A_SIZE)) {
      a_v(o, i) = (float)i;
      c_ref(o, i) = (float)(i * 2);
    }
  }

  for (const auto o : c10::irange(OUTER_B_SIZE)) {
    for (const auto i : c10::irange(B_SIZE)) {
      b_v(o, i) = (float)(B_SIZE - i);
      d_ref(o, i) = c_ref(o, i * 2) + b_v(o, i);
    }
  }

  float* a_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&a_dev, OUTER_A_SIZE * A_SIZE * sizeof(float)));
  float* b_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&b_dev, OUTER_B_SIZE * B_SIZE * sizeof(float)));
  float* c_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&c_dev, OUTER_A_SIZE * A_SIZE * sizeof(float)));
  float* d_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&d_dev, OUTER_B_SIZE * B_SIZE * sizeof(float)));
  C10_CUDA_CHECK(cudaMemcpy(
      a_dev,
      a_v.data(),
      OUTER_A_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      b_dev,
      b_v.data(),
      OUTER_B_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      c_dev,
      c_v.data(),
      OUTER_A_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      d_dev,
      d_v.data(),
      OUTER_B_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  cuda_cg(c_dev, d_dev, a_dev, b_dev);

  C10_CUDA_CHECK(cudaDeviceSynchronize());
  C10_CUDA_CHECK(cudaMemcpy(
      c_v.data(),
      c_dev,
      OUTER_A_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaMemcpy(
      d_v.data(),
      d_dev,
      OUTER_B_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  ExpectAllNear(c_v, c_ref, 1e-5);
  ExpectAllNear(d_v, d_ref, 1e-5);

  C10_CUDA_CHECK(cudaFree(a_dev));
  C10_CUDA_CHECK(cudaFree(b_dev));
  C10_CUDA_CHECK(cudaFree(c_dev));
  C10_CUDA_CHECK(cudaFree(d_dev));
}

} // namespace jit
} // namespace torch

#endif
