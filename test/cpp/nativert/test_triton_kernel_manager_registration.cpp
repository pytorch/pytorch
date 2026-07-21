#include <gtest/gtest.h>

#include <torch/nativert/executor/triton/TritonKernelManager.h>
#include <torch/nativert/kernels/TritonKernel.h>

using namespace ::testing;
using namespace torch::nativert;

TEST(TritonKernelManagerRegistrationTests, TestRegister) {
  EXPECT_TRUE(TritonKernelManagerRegistry()->Has(at::kCPU));

#ifdef USE_CUDA
#ifdef USE_ROCM
  EXPECT_TRUE(TritonKernelManagerRegistry()->Has(at::kHIP));
  EXPECT_FALSE(TritonKernelManagerRegistry()->Has(at::kCUDA));

#else
  EXPECT_TRUE(TritonKernelManagerRegistry()->Has(at::kCUDA));
  EXPECT_FALSE(TritonKernelManagerRegistry()->Has(at::kHIP));

#endif // USE_ROCM
#else
  EXPECT_FALSE(TritonKernelManagerRegistry()->Has(at::kCUDA));
  EXPECT_FALSE(TritonKernelManagerRegistry()->Has(at::kHIP));
#endif // USE_CUDA
}

TEST(TritonKernelManagerRegistrationTests, ParseTupleGridAttribute) {
  auto graph = Graph::createGraph();
  Node* node = graph->insertNode(
      "torch.ops.higher_order.triton_kernel_wrapper_functional");
  node->addAttribute(Attribute{
      "grid",
      std::vector<c10::IValue>{
          c10::IValue(2), c10::IValue(3), c10::IValue(4)}});

  LaunchParams params;
  params.parseCommonAttributes(node);

  EXPECT_EQ(params.grid_dims.x, 2);
  EXPECT_EQ(params.grid_dims.y, 3);
  EXPECT_EQ(params.grid_dims.z, 4);
}
