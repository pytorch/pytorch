#include <sstream>
#include <stdexcept>
#include "test/cpp/tensorexpr/test_base.h"

#include <cmath>

#include "test/cpp/tensorexpr/padded_buffer.h"
#include "torch/csrc/jit/ir/ir.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/csrc/jit/tensorexpr/buffer.h"
#include "torch/csrc/jit/tensorexpr/cuda_codegen.h"
#include "torch/csrc/jit/tensorexpr/kernel.h"
#include "torch/csrc/jit/tensorexpr/loopnest.h"
#include "torch/csrc/jit/tensorexpr/tensor.h"

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/Half.h>

namespace torch {
namespace jit {

using namespace torch::jit;
using namespace torch::jit::tensorexpr;

void testKernel_1() {
  KernelScope kernel_scope;
  const int num_iter = 3;

  const auto graph_string = R"IR(
      graph(%0 : Float(3:3,3:1),
            %1 : Float(3:3,3:1)):
        %2 : Float(3:3,3:1) = aten::mul(%0, %1)
        %3 : Float(3:3,3:1) = aten::mul(%0, %2)
        return (%3))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  //   auto a = at::rand({3, 3}, at::kCUDA);
  //   auto b = at::rand({3, 3}, at::kCUDA);//.transpose(0, 1);
  //   auto o = at::zeros({3, 3}, at::kCUDA);
  auto a = at::rand({3, 3}, at::kCPU);
  auto b = at::rand({3, 3}, at::kCPU); //.transpose(0, 1);
  auto o = at::zeros({3, 3}, at::kCPU);
  TensorExprKernel k(graph);
  std::vector<at::Tensor> inputs = {a, b};
  Stmt* s = k.generateStmtForInputs(fmap<IValue>(inputs));
  std::cerr << "QQQQ:\n" << *s << "\n";
  std::vector<at::Tensor> tensors = {a, b, o};
  std::vector<IValue> stack = fmap<IValue>(tensors);
  k.run(stack);
  //   Stmt* s2 = debugLaunchGraph(graph, {a, b, o});
  //   std::cerr << "ZZZZ:\n" << *s2 << "\n";
  //  auto outputs = debugLaunchGraph(graph, {a, b});
  //   ASSERT_EQ(outputs.size(), 1);
  //   auto o2 = a * b;
  //   float max_diff = (o2 - outputs[0]).abs().max().item<double>();
  // std::cout << "max diff: " << max_diff << "\n";
  //   ASSERT_EQ(max_diff, 0);

  /*
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
  */
}

} // namespace jit
} // namespace torch
