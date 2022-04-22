
#include <c10/core/Device.h>
#include <gtest/gtest.h>
#include <test/cpp/lazy/test_lazy_ops_util.h>
#include <torch/csrc/lazy/core/debug_util.h>
#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/core/lazy_graph_executor.h>
#include <torch/csrc/lazy/core/metrics.h>
#include <torch/csrc/lazy/core/permutation_util.h>
#include <torch/csrc/lazy/core/tensor.h>
#include <torch/csrc/lazy/ts_backend/ts_backend_impl.h>
#include <torch/torch.h>
#include <iostream>

namespace torch {
namespace lazy {

// Lazy Tensor is disabled in FBCODE until addressing non-virtual methods (e.g.
// sizes) in TensorImpl
#ifndef FBCODE_CAFFE2

namespace {
// This registers the torchscript backend, without which lazy device won't work
torch::lazy::BackendRegistrar g_registrar(GetTSBackendImpl());

static inline at::DeviceType DefaultDevice() {
  return torch::lazy::getBackend()->EagerFallbackDeviceType();
}

std::vector<bool> getIsSymbolic(at::Tensor& lazy_tensor) {
  auto ltc_tensor = GetLtcTensor(lazy_tensor);
  Value ir_val = ltc_tensor->GetIrValue();
  const Shape& shape = ir_val->shape();
  return shape.is_symbolic().value();
}

class LazyShapeTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {}
  void SetUp() override {
    at::manual_seed(42);
    torch::lazy::LazyGraphExecutor::Get()->SetRngSeed(
        torch::lazy::BackendDevice(), 42);
    FLAGS_ltc_enable_symbolic_shapes = true;
  }
  void TearDown() override {
    FLAGS_ltc_enable_symbolic_shapes = false;
  }
};

class DynamicInputShapeNode : public Node {
 public:
  explicit DynamicInputShapeNode(Shape& shape)
      : Node(OpKind(), /* num_outputs */ 1),
        hash_(0),
        shape_(shape) {}
  ~DynamicInputShapeNode() override = default;

  const std::vector<Output>& operands() const override {
    TORCH_INTERNAL_ASSERT(false, "Can't access operands of test node");
  }

  const Output& operand(size_t i) const override {
    TORCH_INTERNAL_ASSERT(false, "Can't access operand[i] of test node");
  }
  const Shape& shape(size_t i) const override {
    return shape_;
  }
  c10::ArrayRef<Shape> shapes() const override {
    return {shape_};
  }

  hash_t hash() const override { return hash_; }
  hash_t shapeHash() const override { return hash_; }

 private:
  hash_t hash_;
  Shape shape_;
};

} // namespace

Tensor tensorWithSymbolicShape(
    const std::vector<int64_t>& sizes,
    const std::vector<bool>& is_symbolic) {
  Shape shape = Shape(torch::kFloat32, sizes);
  Shape shape_with_symbolic = shape.with_symbolic_dims(is_symbolic);
  auto n = torch::lazy::MakeNode<DynamicInputShapeNode>(shape_with_symbolic);
  auto device = BackendDevice();
  auto lt = torch::lazy::LazyTensor::Create(n, device);
  return torch::lazy::CreateAtenFromLtcTensor(lt);
}

TEST_F(LazyShapeTest, TestMulBasic) {
  // Basic propagation
  torch::Tensor a = tensorWithSymbolicShape({2, 2}, {true, false});
  torch::Tensor b = tensorWithSymbolicShape({2, 2}, {true, false});
  torch::Tensor res = torch::mul(a, b);

  std::vector<bool> expected = {true, false};
  EXPECT_EQ(getIsSymbolic(res), expected);

  // Test when some inputs are symbolic
  a = tensorWithSymbolicShape({2, 2}, {true, true});
  b = tensorWithSymbolicShape({2, 2}, {true, false});
  res = torch::mul(a, b);

  // This is not {true, false}, as the SSA shape propagation
  // is not able to simplify
  // expandedSizes.append(sizeB if sizeA == 1 else sizeA)
  // in broadcast() in shape_functions_1.h
  // due to sizeA being symbolic
  expected = {true, true};
  EXPECT_EQ(getIsSymbolic(res), expected);

  // Test correct handling of broadcasting dim
  a = tensorWithSymbolicShape({2, 2}, {false, true});
  b = tensorWithSymbolicShape({2, 1}, {true, false});
  res = torch::mul(a, b);

  expected = {false, true};
  EXPECT_EQ(getIsSymbolic(res), expected);

  // Test correct handling of scalar values
  a = tensorWithSymbolicShape({2, 2}, {false, true});
  res = torch::mul(a, 3);
  expected = {false, true};
  EXPECT_EQ(getIsSymbolic(res), expected);
};
#endif // FBCODE_CAFFE2

} // namespace lazy
} // namespace torch
