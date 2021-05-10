#include <gtest/gtest.h>

#include <test/cpp/tensorexpr/test_base.h>
#include <torch/csrc/jit/frontend/code_template.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace torch {
namespace jit {

using namespace torch::indexing;
using namespace torch::jit::tensorexpr;

class Kernel : public ::testing::Test {
 public:
  // NOLINTNEXTLINE(modernize-use-override,cppcoreguidelines-explicit-virtual-functions)
  void SetUp() {
    getTEMustUseLLVMOnCPU() = false;
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(Kernel, InliningIntermediates) {
  // here, each mul has only one use, so it should be completely inlined
  {
    const auto graph_string = R"IR(
        graph(%0 : Float(5, 3, strides=[3, 1], device=cpu),
              %1 : Float(5, 3, strides=[3, 1], device=cpu)):
          %2 : Float(5, 3, strides=[3, 1]) = aten::mul(%0, %1)
          %one : int = prim::Constant[value=1]()
          %4 : Float(5, 3, strides=[3, 1]) = aten::mul(%0, %2)
          %5: Float(5, 3, strides=[3, 1]) = aten::add(%4, %1, %one)
          return (%5))IR";
    KernelScope kernel_scope;
    auto graph = std::make_shared<Graph>();
    parseIR(graph_string, &*graph);
    TensorExprKernel k(graph);
    auto stmt = k.getCodeGenStmt();
    std::ostringstream oss;
    oss << *stmt;
    torch::jit::testing::FileCheck().check_not("aten_mul")->run(oss.str());
  }
  {
    const auto graph_template = R"IR(
        graph(%0 : Float(5, 3, strides=[3, 1], device=${device}),
              %1 : Float(5, 3, strides=[3, 1], device=${device})):
          %2 : Float(5, 3, strides=[3, 1]) = aten::mul(%0, %1)
          %one : int = prim::Constant[value=1]()
          %3 : Float(5, 3, strides=[3, 1]) = aten::sub(%0, %2, %one)
          %4 : Float(5, 3, strides=[3, 1]) = aten::add(%3, %0, %one)
          %5 : Float(5, 3, strides=[3, 1]) = aten::div(%3, %0)
          return (%4, %5))IR";
    for (bool use_cuda : {false, true}) {
      if (!torch::cuda::is_available() && use_cuda) {
        continue;
      }

      KernelScope kernel_scope;
      TemplateEnv env;
      env.s("device", use_cuda ? "cuda:0" : "cpu");
      const auto graph_string = format(graph_template, env);
      auto graph = std::make_shared<Graph>();
      parseIR(graph_string, &*graph);
      // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
      auto device = use_cuda ? kCUDA : kCPU;
      TensorExprKernel k(graph);
      auto stmt = k.getCodeGenStmt();
      std::ostringstream oss;
      oss << *stmt;
      // aten_mul only has one use, inlined completely
      torch::jit::testing::FileCheck().check_not("aten_mul")->run(oss.str());

      // aten_sub should be removed by the CUDA backend by metavar rewriting
      // and by the CPU backend by horizontal fusion.
      // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores,cppcoreguidelines-avoid-magic-numbers)
      size_t num_out1_uses = use_cuda ? 0 : 5;
      torch::jit::testing::FileCheck().check_not("aten_sub")->run(oss.str());
    }
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(Kernel, _1) {
  KernelScope kernel_scope;

  const auto graph_string = R"IR(
      graph(%0 : Float(5, 3, strides=[3, 1], device=cpu),
            %1 : Float(5, 3, strides=[3, 1], device=cpu)):
        %2 : Float(5, 3, strides=[3, 1]) = aten::mul(%0, %1)
        %3 : Float(5, 3, strides=[3, 1]) = aten::mul(%0, %2)
        return (%3))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  auto a = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto b = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto o = at::zeros({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto ref = a * (a * b);
  TensorExprKernel k(graph);
  std::vector<at::Tensor> inputs = {a, b};
  Stmt* s = k.getCodeGenStmt();

  std::ostringstream oss;
  oss << *s;

  // Check the IR we produced
  const std::string& verification_pattern =
      R"IR(
# CHECK: for
# CHECK-NEXT: for
# CHECK-NOT: for)IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  o = stack[0].toTensor();
  for (size_t i = 0; i < 5 * 3; i++) {
    CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(Kernel, _2) {
  KernelScope kernel_scope;

  const auto graph_string = R"IR(
      graph(%0 : Float(5, 3, strides=[3, 1], device=cpu),
            %1 : Float(5, 3, strides=[1, 5], device=cpu)):
        %2 : Float(5, 3, strides=[3, 1]) = aten::mul(%0, %1)
        %3 : Float(5, 3, strides=[3, 1]) = aten::mul(%0, %2)
        return (%3))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  auto a = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto b =
      at::rand({3, 5}, TensorOptions(kCPU).dtype(at::kFloat)).transpose(0, 1);
  auto o = at::zeros({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto ref = a * (a * b);
  TensorExprKernel k(graph);
  std::vector<at::Tensor> inputs = {a, b};
  Stmt* s = k.getCodeGenStmt();

  std::ostringstream oss;
  oss << *s;

  // Check the IR we produced
  const std::string& verification_pattern =
      R"IR(
# CHECK: for
# CHECK-NEXT: for
# CHECK-NOT: for)IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  o = stack[0].toTensor();
  for (size_t i = 0; i < 5 * 3; i++) {
    CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(Kernel, _3) {
  KernelScope kernel_scope;

  const auto graph_string = R"IR(
      graph(%0 : Float(5, 3, strides=[3, 1], device=cpu),
            %1 : Float(5, 3, strides=[12, 2], device=cpu)):
        %2 : Float(5, 3, strides=[3, 1]) = aten::mul(%0, %1)
        %3 : Float(5, 3, strides=[3, 1]) = aten::mul(%0, %2)
        return (%3))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  auto a = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto b = at::rand({10, 6}, TensorOptions(kCPU).dtype(at::kFloat))
               .index({Slice(None, None, 2), Slice(None, None, 2)});
  auto o = at::zeros({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto ref = a * (a * b);
  TensorExprKernel k(graph);
  std::vector<at::Tensor> inputs = {a, b};
  Stmt* s = k.getCodeGenStmt();

  std::ostringstream oss;
  oss << *s;

  // Check the IR we produced
  const std::string& verification_pattern =
      R"IR(
# CHECK: for
# CHECK-NEXT: for
# CHECK-NOT: for)IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  o = stack[0].toTensor();
  for (size_t i = 0; i < 5 * 3; i++) {
    CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(Kernel, DISABLED_Shape_Inference) {
  // disabled: doesn't do stride propagation, and isn't being used currently

  // Test TensorExpr shape inference capabilities: it should only require shapes
  // for the inputs
  {
    KernelScope kernel_scope;

    const auto graph_string = R"IR(
      graph(%0 : Float(5, 3, strides=[3, 1], device=cpu),
            %1 : Float(5, 3, strides=[12, 2], device=cpu)):
        %2 : Tensor = aten::mul(%0, %1)
        %3 : Tensor = aten::mul(%0, %2)
        return (%3))IR";
    auto graph = std::make_shared<Graph>();
    parseIR(graph_string, &*graph);

    auto a = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
    auto b = at::rand({10, 6}, TensorOptions(kCPU).dtype(at::kFloat))
                 .index({Slice(None, None, 2), Slice(None, None, 2)});
    auto o = at::zeros({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
    auto ref = a * (a * b);
    TensorExprKernel k(graph);
    std::vector<at::Tensor> inputs = {a, b};
    Stmt* s = k.getCodeGenStmt();

    std::ostringstream oss;
    oss << *s;

    // Check the IR we produced
    const std::string& verification_pattern =
        R"IR(
# CHECK: for
# CHECK-NEXT: for
# CHECK-NOT: for)IR";
    torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

    std::vector<IValue> stack = fmap<IValue>(inputs);
    k.run(stack);
    o = stack[0].toTensor();
    for (size_t i = 0; i < 5 * 3; i++) {
      CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
    }
  }
  {
    KernelScope kernel_scope;

    const auto graph_string = R"IR(
      graph(%0 : Float(8, 8, strides=[8, 1], device=cpu),
            %1 : Float(8, 8, strides=[8, 1], device=cpu)):
        %2 : Tensor = aten::mul(%0, %1)
        %3 : Tensor, %4 : Tensor = prim::ConstantChunk[dim=1,chunks=2](%2)
        %r : Tensor = aten::mul(%3, %4)
        return (%r))IR";
    auto graph = std::make_shared<Graph>();
    parseIR(graph_string, &*graph);

    auto a = at::rand({8, 8}, TensorOptions(kCPU).dtype(at::kFloat));
    auto b = at::rand({8, 8}, TensorOptions(kCPU).dtype(at::kFloat));
    auto o = at::zeros({8, 4}, TensorOptions(kCPU).dtype(at::kFloat));
    auto t = torch::chunk(a * b, 2, 1);
    auto ref = t[0] * t[1];
    TensorExprKernel k(graph);
    std::vector<at::Tensor> inputs = {a, b};
    Stmt* s = k.getCodeGenStmt();

    std::ostringstream oss;
    oss << *s;

    // Check the IR we produced
    const std::string& verification_pattern =
        R"IR(
# CHECK: for)IR";
    torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

    std::vector<IValue> stack = fmap<IValue>(inputs);
    k.run(stack);
    o = stack[0].toTensor();
    CHECK_EQ(o.sizes()[0], 8);
    CHECK_EQ(o.sizes()[1], 4);
    for (size_t i = 0; i < 8 * 4; i++) {
      CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
    }
  }
  {
    // Test that shape inference handles aten::unsqueeze
    KernelScope kernel_scope;

    const auto graph_string = R"IR(
      graph(%a : Float(4, 2, strides=[2, 1], device=cpu),
            %b : Float(4, 3, 2, strides=[6, 2, 1], device=cpu),
            %c : Float(3, 2, 2, strides=[4, 2, 1], device=cpu)):
        %one : int = prim::Constant[value=1]()
        %minus_one : int = prim::Constant[value=-1]()
        %three : int = prim::Constant[value=3]()
        %minus_four : int = prim::Constant[value=-4]()
        %a1 : Tensor = aten::unsqueeze(%a, %one)        # new size: [4,1,2]
        %a2 : Tensor = aten::unsqueeze(%a1, %minus_one) # new size: [4,1,2,1]
        %b1 : Tensor = aten::unsqueeze(%b, %three)      # new size: [4,3,2,1]
        %c1 : Tensor = aten::unsqueeze(%c, %minus_four) # new size: [1,3,2,2]
        %ab : Tensor = aten::mul(%a2, %b1)         # expected size: [4,3,2,1]
        %abc : Tensor = aten::mul(%ab, %c1)        # expected size: [4,3,2,2]
        return (%abc))IR";
    auto graph = std::make_shared<Graph>();
    parseIR(graph_string, &*graph);

    auto a = at::rand({4, 2}, TensorOptions(kCPU).dtype(at::kFloat));
    auto b = at::rand({4, 3, 2}, TensorOptions(kCPU).dtype(at::kFloat));
    auto c = at::rand({3, 2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
    auto o = at::zeros({4, 3, 2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
    auto ref = at::unsqueeze(at::unsqueeze(a, 1), -1) * at::unsqueeze(b, 3) *
        at::unsqueeze(c, -4);

    TensorExprKernel k(graph);
    std::vector<at::Tensor> inputs = {a, b, c};
    Stmt* s = k.getCodeGenStmt();

    std::ostringstream oss;
    oss << *s;

    // Check the IR we produced
    const std::string& verification_pattern =
        R"IR(
# CHECK: for
# CHECK-NEXT: for
# CHECK-NEXT: for
# CHECK-NEXT: for
# CHECK-NEXT: aten_mul)IR";
    torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

    std::vector<IValue> stack = fmap<IValue>(inputs);
    k.run(stack);
    o = stack[0].toTensor();

    // Check sizes
    CHECK_EQ(o.sizes().size(), ref.sizes().size());
    size_t num_el = 1;
    for (size_t idx = 0; idx < ref.sizes().size(); idx++) {
      CHECK_EQ(o.sizes()[idx], ref.sizes()[idx]);
      num_el *= ref.sizes()[idx];
    }

    // Check the contents
    for (size_t i = 0; i < num_el; i++) {
      CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
    }
  }
  {
    // Test that shape inference handles aten::cat
    KernelScope kernel_scope;

    const auto graph_string = R"IR(
      graph(%a : Float(5, 3, 2, strides=[6, 2, 1], device=cpu),
            %b : Float(5, 7, 2, strides=[14, 2, 1], device=cpu),
            %c : Float(5, 9, 2, strides=[18, 2, 1], device=cpu)):
        %dim : int = prim::Constant[value=1]()
        %inputs : Tensor[] = prim::ListConstruct(%a, %b, %c)
        %r : Tensor = aten::cat(%inputs, %dim)               # new size: [5,19,2]
        return (%r))IR";
    auto graph = std::make_shared<Graph>();
    parseIR(graph_string, &*graph);

    auto a = at::rand({5, 3, 2}, TensorOptions(kCPU).dtype(at::kFloat));
    auto b = at::rand({5, 7, 2}, TensorOptions(kCPU).dtype(at::kFloat));
    auto c = at::rand({5, 9, 2}, TensorOptions(kCPU).dtype(at::kFloat));
    auto o = at::zeros({5, 19, 2}, TensorOptions(kCPU).dtype(at::kFloat));
    auto ref = at::cat({a, b, c}, 1);

    TensorExprKernel k(graph);
    std::vector<at::Tensor> inputs = {a, b, c};
    Stmt* s = k.getCodeGenStmt();

    std::ostringstream oss;
    oss << *s;

    // Check the IR we produced
    const std::string& verification_pattern =
        R"IR(
# CHECK: for
# CHECK-NEXT: for
# CHECK-NEXT: for
# CHECK-NEXT: aten_cat)IR";
    torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

    std::vector<IValue> stack = fmap<IValue>(inputs);
    k.run(stack);
    o = stack[0].toTensor();

    // Check sizes
    CHECK_EQ(o.sizes().size(), ref.sizes().size());
    size_t num_el = 1;
    for (size_t idx = 0; idx < ref.sizes().size(); idx++) {
      CHECK_EQ(o.sizes()[idx], ref.sizes()[idx]);
      num_el *= ref.sizes()[idx];
    }

    // Check the contents
    for (size_t i = 0; i < num_el; i++) {
      CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
    }
  }
  {
    // Test that we throw an error when input list for aten::cat is empty
    KernelScope kernel_scope;

    const auto graph_string = R"IR(
      graph():
        %dim : int = prim::Constant[value=1]()
        %inputs : Tensor[] = prim::ListConstruct()
        %r : Tensor = aten::cat(%inputs, %dim)
        return (%r))IR";
    auto graph = std::make_shared<Graph>();
    parseIR(graph_string, &*graph);
    auto compile = [&]() {
      TensorExprKernel k(graph);
      k.getCodeGenStmt();
    };
    ASSERT_THROWS_WITH(compile(), "Empty input list is passed to aten::cat");
  }
  {
    // Test that we throw an error when 'dim' passed to aten::cat is invalid
    KernelScope kernel_scope;

    const auto ir_dim_99 = R"IR(
      graph(%a : Float(5, 3, 2, strides=[6, 2, 1], device=cpu),
            %b : Float(5, 3, 2, strides=[6, 2, 1], device=cpu)):
        %dim : int = prim::Constant[value=99]()
        %inputs : Tensor[] = prim::ListConstruct(%a, %b)
        %r : Float(5, 3, 2, strides=[6, 2, 1], device=cpu) = aten::cat(%inputs, %dim)
        return (%r))IR";
    const auto ir_dim_minus_6 = R"IR(
      graph(%a : Float(5, 3, 2, strides=[6, 2, 1], device=cpu),
            %b : Float(5, 3, 2, strides=[6, 2, 1], device=cpu)):
        %dim : int = prim::Constant[value=-6]()
        %inputs : Tensor[] = prim::ListConstruct(%a, %b)
        %r : Float(5, 3, 2, strides=[6, 2, 1], device=cpu) = aten::cat(%inputs, %dim)
        return (%r))IR";

    auto compile = [](const std::string& graph_string) {
      auto graph = std::make_shared<Graph>();
      parseIR(graph_string, &*graph);
      TensorExprKernel k(graph);
      k.getCodeGenStmt();
    };
    ASSERT_THROWS_WITH(compile(ir_dim_99), "Invalid index");
    ASSERT_THROWS_WITH(compile(ir_dim_minus_6), "Invalid index");
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(Kernel, CatInputTypesPromotion) {
  {
    // Test that we properly promote input types for aten::cat
    KernelScope kernel_scope;

    const auto graph_string = R"IR(
      graph(%a : Float(5, 3, 2, strides=[6, 2, 1], device=cpu),
            %b : Float(5, 7, 2, strides=[14, 2, 1], device=cpu),
            %c : Double(5, 9, 2, strides=[18, 2, 1], device=cpu)):
        %dim : int = prim::Constant[value=1]()
        %inputs : Tensor[] = prim::ListConstruct(%a, %b, %c)
        %r : Double(5, 19, 2, strides=[38, 2, 1]) = aten::cat(%inputs, %dim)
        return (%r))IR";
    auto graph = std::make_shared<Graph>();
    parseIR(graph_string, &*graph);

    auto a = at::rand({5, 3, 2}, TensorOptions(kCPU).dtype(at::kFloat));
    auto b = at::rand({5, 7, 2}, TensorOptions(kCPU).dtype(at::kFloat));
    auto c = at::rand({5, 9, 2}, TensorOptions(kCPU).dtype(at::kDouble));
    auto ref = at::cat({a, b, c}, 1);

    TensorExprKernel k(graph);
    std::vector<at::Tensor> inputs = {a, b, c};
    Stmt* s = k.getCodeGenStmt();

    std::ostringstream oss;
    oss << *s;

    // Check the IR we produced
    const std::string& verification_pattern =
        R"IR(
# CHECK: for
# CHECK-NEXT: for
# CHECK-NEXT: for
# CHECK-NEXT: aten_cat)IR";
    torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

    std::vector<IValue> stack = fmap<IValue>(inputs);
    k.run(stack);
    auto o = stack[0].toTensor();

    // Check sizes
    CHECK_EQ(o.sizes().size(), ref.sizes().size());
    CHECK_EQ(o.dtype(), ref.dtype());
    size_t num_el = 1;
    for (size_t idx = 0; idx < ref.sizes().size(); idx++) {
      CHECK_EQ(o.sizes()[idx], ref.sizes()[idx]);
      num_el *= ref.sizes()[idx];
    }

    // Check the contents
    for (size_t i = 0; i < num_el; i++) {
      CHECK_EQ(((double*)o.data_ptr())[i], ((double*)ref.data_ptr())[i]);
    }
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(Kernel, CatWoConditionals) {
  getCatWoConditionals() = true;
  const auto graph_string = R"IR(
      graph(%a : Float(5, 3, 2, strides=[6, 2, 1], device=cpu),
            %b : Float(5, 7, 2, strides=[14, 2, 1], device=cpu),
            %c : Float(5, 9, 2, strides=[18, 2, 1], device=cpu)):
        %dim : int = prim::Constant[value=1]()
        %inputs : Tensor[] = prim::ListConstruct(%a, %b, %c)
        %r : Float(5, 19, 2, strides=[38, 2, 1]) = aten::cat(%inputs, %dim)
        return (%r))IR";

  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  TensorExprKernel k(graph);
  Stmt* s = k.getCodeGenStmt();
  std::ostringstream oss;
  oss << *s;

  const std::string& verification_pattern =
      R"IR(
# CHECK: for
# CHECK-NEXT: for
# CHECK-NEXT: for
# CHECK-NEXT: aten_cat
# CHECK: for
# CHECK-NEXT: for
# CHECK-NEXT: for
# CHECK-NEXT: aten_cat
# CHECK: for
# CHECK-NEXT: for
# CHECK-NEXT: for
# CHECK-NEXT: aten_cat)IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  auto a = at::rand({5, 3, 2}, TensorOptions(kCPU).dtype(at::kFloat));
  auto b = at::rand({5, 7, 2}, TensorOptions(kCPU).dtype(at::kFloat));
  auto c = at::rand({5, 9, 2}, TensorOptions(kCPU).dtype(at::kFloat));
  auto ref = at::cat({a, b, c}, 1);

  std::vector<at::Tensor> inputs = {a, b, c};
  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  auto o = stack[0].toTensor();

  // Check sizes
  CHECK_EQ(o.sizes().size(), ref.sizes().size());
  CHECK_EQ(o.dtype(), ref.dtype());
  size_t num_el = 1;
  for (size_t idx = 0; idx < ref.sizes().size(); idx++) {
    CHECK_EQ(o.sizes()[idx], ref.sizes()[idx]);
    num_el *= ref.sizes()[idx];
  }

  // Check the contents
  for (size_t i = 0; i < num_el; i++) {
    CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
  }
}

namespace {

std::string dtypeConstant(ScalarType scalar_type) {
  if (scalar_type == ScalarType::Undefined) {
    return "None = prim::Constant()";
  } else {
    TemplateEnv env_dtype;
    env_dtype.d("scalar_type", static_cast<int>(scalar_type));
    return format("int = prim::Constant[value=${scalar_type}]()", env_dtype);
  }
}

at::Tensor iotaTensor(IntArrayRef sizes, const at::TensorOptions& options) {
  int64_t numel = std::accumulate(
      sizes.begin(),
      sizes.end(),
      1,
      // NOLINTNEXTLINE(modernize-use-transparent-functors)
      std::multiplies<int64_t>());
  std::vector<float> values(numel);
  std::iota(values.begin(), values.end(), 0);
  auto a = at::tensor(values, options);
  return a.reshape(sizes);
}

} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(Kernel, DISABLED_SumAllAxes) {
  // [zero-dim tensors]
  // NNC does not yet handle zero-dim tensors. aten::sum with no axis
  // input returns a zero-dim tensors, so these tests must be disabled
  // until we add support for zero-dim tensors.

  // Test lowering of sum on all axes.
  const auto graph_template = R"IR(
      graph(%0 : Float(5, 3, strides=[3, 1], device=cpu)):
        %1 : ${dtype}
        %2 : Tensor = aten::sum(%0, %1)
        return (%2))IR";
  auto a = iotaTensor({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));

  for (auto scalar_type : {ScalarType::Undefined, ScalarType::Double}) {
    KernelScope kernel_scope;
    TemplateEnv env;
    env.s("dtype", dtypeConstant(scalar_type));
    const auto graph_string = format(graph_template, env);

    auto graph = std::make_shared<Graph>();
    parseIR(graph_string, &*graph);

    auto o = at::empty({}, TensorOptions(kCPU));
    c10::optional<c10::ScalarType> dtype;
    if (scalar_type != ScalarType::Undefined) {
      dtype = static_cast<c10::ScalarType>(scalar_type);
    }
    auto ref = a.sum(/*dtype=*/dtype);
    TensorExprKernel k(graph);
    std::vector<at::Tensor> inputs = {a};
    Stmt* s = k.getCodeGenStmt();

    std::ostringstream oss;
    oss << *s;

    // Check the IR we produced
    const std::string& verification_pattern =
        R"IR(
# CHECK: for
# CHECK-NEXT: for)IR";
    torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

    std::vector<IValue> stack = fmap<IValue>(inputs);
    k.run(stack);
    o = stack[0].toTensor();
    ASSERT_EQ(o.sizes(), ref.sizes());
    ASSERT_EQ(o.dtype(), ref.dtype());
    ASSERT_TRUE(at::allclose(o, ref));
  }
}

std::string li_to_str(at::ArrayRef<int64_t> li) {
  std::stringstream out;
  bool first = true;
  for (auto elem : li) {
    if (!first) {
      out << ", ";
    }
    out << elem;
    first = false;
  }
  return out.str();
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(Kernel, SumOneAxis) {
  // Test lowering of sum on one axis.
  const auto graph_template = R"IR(
      graph(%0 : Float(5, 3, strides=[3, 1], device=cpu)):
        %1 : int[] = prim::Constant[value=[${dim}]]()
        %2 : bool = prim::Constant[value=${keepdim}]()
        %3 : ${dtype}
        %4 : ${out_dtype}(${size}, strides=[${strides}], device=cpu) = aten::sum(%0, %1, %2, %3)
        return (%4))IR";
  auto a = iotaTensor({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));

  for (int dim = -a.dim(); dim < a.dim(); ++dim) {
    for (bool keepdim : {false, true}) {
      for (auto scalar_type : {ScalarType::Undefined, ScalarType::Double}) {
        KernelScope kernel_scope;
        TemplateEnv env;
        env.d("dim", dim);
        env.d("keepdim", keepdim);
        env.s("dtype", dtypeConstant(scalar_type));
        c10::optional<c10::ScalarType> dtype;
        if (scalar_type != ScalarType::Undefined) {
          dtype = static_cast<c10::ScalarType>(scalar_type);
        }
        auto ref = a.sum({dim}, /*keepdim=*/keepdim, /*dtype=*/dtype);
        if (scalar_type == ScalarType::Undefined) {
          env.s("out_dtype", "Float");
        } else {
          env.s("out_dtype", "Double");
        }
        env.s("size", li_to_str(ref.sizes()));
        env.s("strides", li_to_str(ref.strides()));
        const auto graph_string = format(graph_template, env);
        auto graph = std::make_shared<Graph>();
        parseIR(graph_string, &*graph);

        auto o = at::empty({}, TensorOptions(kCPU));
        TensorExprKernel k(graph);
        std::vector<at::Tensor> inputs = {a};
        Stmt* s = k.getCodeGenStmt();

        std::ostringstream oss;
        oss << *s;

        // Check the IR we produced
        const std::string& verification_pattern =
            R"IR(
# CHECK: for (int v = 0; v <
# CHECK-NEXT: sum
# CHECK-NEXT: for (int v_1 = 0; v_1 <
# CHECK-NEXT:   sum)IR";
        torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

        std::vector<IValue> stack = fmap<IValue>(inputs);
        k.run(stack);
        o = stack[0].toTensor();
        ASSERT_EQ(o.sizes(), ref.sizes());
        ASSERT_EQ(o.dtype(), ref.dtype());
        ASSERT_TRUE(at::allclose(o, ref));
      }
    }
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(Kernel, SumMultipleAxes) {
  // Test lowering of sum on multiple axes.
  const auto graph_template = R"IR(
      graph(%0 : Float(2, 3, 2, 3, strides=[18, 6, 3, 1], device=cpu)):
        %1 : int = prim::Constant[value=${dim1}]()
        %2 : int = prim::Constant[value=${dim2}]()
        %3 : int[] = prim::ListConstruct(%1, %2)
        %4 : bool = prim::Constant[value=${keepdim}]()
        %5 : ${dtype}
        %6 : Float(${size}, strides=[${strides}]) = aten::sum(%0, %3, %4, %5)
        return (%6))IR";
  auto a = iotaTensor({2, 3, 2, 3}, TensorOptions(kCPU).dtype(at::kFloat));

  // Only iterate over positive values of axes to keep the running time
  // reasonable, since the number of pairs is quadratic.
  for (int dim1 = 0; dim1 < a.dim(); ++dim1) {
    for (int dim2 = dim1 + 1; dim2 < a.dim(); ++dim2) {
      for (bool keepdim : {false, true}) {
        KernelScope kernel_scope;
        TemplateEnv env;
        env.d("dim1", dim1);
        env.d("dim2", dim2);
        env.d("keepdim", keepdim);
        env.s("dtype", dtypeConstant(ScalarType::Undefined));
        auto o = at::empty({}, TensorOptions(kCPU));
        auto ref = a.sum(IntArrayRef{dim1, dim2}, /*keepdim=*/keepdim);

        env.s("size", li_to_str(ref.sizes()));
        env.s("strides", li_to_str(ref.strides()));

        const auto graph_string = format(graph_template, env);

        auto graph = std::make_shared<Graph>();
        parseIR(graph_string, &*graph);

        TensorExprKernel k(graph);
        std::vector<at::Tensor> inputs = {a};
        Stmt* s = k.getCodeGenStmt();

        std::ostringstream oss;
        oss << *s;

        // Check the IR we produced
        const std::string& verification_pattern =
            R"IR(
# CHECK: int v = 0
# CHECK: int v_1 = 0
# CHECK: int v_2 = 0
# CHECK: int v_3 = 0
# CHECK: sum)IR";
        torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

        std::vector<IValue> stack = fmap<IValue>(inputs);
        k.run(stack);
        o = stack[0].toTensor();
        ASSERT_EQ(o.sizes(), ref.sizes());
        ASSERT_EQ(o.dtype(), ref.dtype());
        ASSERT_TRUE(at::allclose(o, ref));
      }
    }
  }
}

// This test and the following ones testing Softmax only tests with dim set
// to one of the valid input dimensions. It does not test with dim=None
// because that is supposed to be deprecated.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(Kernel, Softmax2D) {
  const auto graph_template = R"IR(
      graph(%0 : Float(5, 3, strides=[3, 1], device=cpu)):
        %1 : int = prim::Constant[value=${dim}]()
        %2 : int = prim::Constant[value=7]()
        %3 : Float(${size}, strides=[${strides}]) = aten::${op}(%0, %1, %2)
        return (%3))IR";

  auto a = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));

  const std::string& verification_template =
      R"IR(
        # CHECK: for (int i${other_dim} = 0; i${other_dim} < ${other_dim_size}
        # CHECK: for (int i${softmax_dim} = 0; i${softmax_dim} < ${softmax_dim_size}
        # CHECK-NEXT: aten_softmax_max
        # CHECK: for (int i${other_dim}_1 = 0; i${other_dim}_1 < ${other_dim_size}
        # CHECK: for (int i${softmax_dim}_1 = 0; i${softmax_dim}_1 < ${softmax_dim_size}
        # CHECK-NEXT: aten_softmax_sum
        # CHECK: for (int i0_2 = 0; i0_2 < 5
        # CHECK-NEXT: for (int i1_2 = 0; i1_2 < 3
        # CHECK-NEXT: aten_softmax)IR";

  for (auto log_softmax : {false, true}) {
    for (int softmax_dim = 0; softmax_dim < a.dim(); ++softmax_dim) {
      auto softmax_dim_size = a.sizes()[softmax_dim];
      auto other_dim = (softmax_dim + 1) % a.dim();
      auto ref =
          log_softmax ? a.log_softmax(softmax_dim) : a.softmax(softmax_dim);
      KernelScope kernel_scope;
      TemplateEnv env;
      env.d("dim", softmax_dim);
      env.s("op", log_softmax ? "log_softmax" : "softmax");
      env.s("size", li_to_str(ref.sizes()));
      env.s("strides", li_to_str(ref.strides()));

      const auto graph_string = format(graph_template, env);

      auto graph = std::make_shared<Graph>();
      parseIR(graph_string, &*graph);

      TensorExprKernel k(graph);
      std::vector<at::Tensor> inputs = {a};
      Stmt* s = k.getCodeGenStmt();

      std::ostringstream oss;
      oss << *s;

      TemplateEnv ver_env;
      ver_env.d("other_dim", other_dim);
      ver_env.d("other_dim_size", a.sizes()[other_dim]);
      ver_env.d("softmax_dim", softmax_dim);
      ver_env.d("softmax_dim_size", softmax_dim_size);
      const auto verification_pattern = format(verification_template, ver_env);

      // verication sting temporarily disabled until
      // inlining of exp() is benchmarked and determined
      // torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

      std::vector<IValue> stack = fmap<IValue>(inputs);
      k.run(stack);
      auto output = stack[0].toTensor();
      ASSERT_EQ(output.sizes(), ref.sizes());
      ASSERT_TRUE(at::allclose(output, ref));
    }
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(Kernel, Softmax3D) {
  const auto graph_template = R"IR(
      graph(%0 : Float(3, 4, 5, strides=[20, 5, 1], device=cpu)):
        %1 : int = prim::Constant[value=${dim}]()
        %2 : int = prim::Constant[value=7]()
        %3 : Float(${size}, strides=[${strides}]) = aten::${op}(%0, %1, %2)
        return (%3))IR";

  auto a = at::rand({3, 4, 5}, TensorOptions(kCPU).dtype(at::kFloat));

  const std::string& verification_template =
      R"IR(
        # CHECK: for (int i${dim1} = 0; i${dim1} < ${dim1_size}
        # CHECK-NEXT: for (int i${dim2} = 0; i${dim2} < ${dim2_size}
        # CHECK: for (int i${softmax_dim} = 0; i${softmax_dim} < ${softmax_dim_size}
        # CHECK-NEXT: aten_softmax_max
        # CHECK: for (int i${dim1}_1 = 0; i${dim1}_1 < ${dim1_size}
        # CHECK-NEXT: for (int i${dim2}_1 = 0; i${dim2}_1 < ${dim2_size}
        # CHECK: for (int i${softmax_dim}_1 = 0; i${softmax_dim}_1 < ${softmax_dim_size}
        # CHECK-NEXT: aten_softmax_sum
        # CHECK: for (int i0_2 = 0; i0_2 < 3
        # CHECK-NEXT: for (int i1_2 = 0; i1_2 < 4
        # CHECK-NEXT: for (int i2_2 = 0; i2_2 < 5
        # CHECK-NEXT: aten_softmax)IR";

  for (auto log_softmax : {false, true}) {
    for (int softmax_dim = 0; softmax_dim < a.dim(); ++softmax_dim) {
      auto softmax_dim_size = a.sizes()[softmax_dim];
      std::vector<int> other_dims;
      for (int i = 0; i < a.dim(); ++i) {
        if (i != softmax_dim) {
          other_dims.push_back(i);
        }
      }
      auto ref =
          log_softmax ? a.log_softmax(softmax_dim) : a.softmax(softmax_dim);

      KernelScope kernel_scope;
      TemplateEnv env;
      env.d("dim", softmax_dim);
      env.s("op", log_softmax ? "log_softmax" : "softmax");
      env.s("size", li_to_str(ref.sizes()));
      env.s("strides", li_to_str(ref.strides()));

      const auto graph_string = format(graph_template, env);

      auto graph = std::make_shared<Graph>();
      parseIR(graph_string, &*graph);

      TensorExprKernel k(graph);
      std::vector<at::Tensor> inputs = {a};
      Stmt* s = k.getCodeGenStmt();

      std::ostringstream oss;
      oss << *s;

      TemplateEnv ver_env;
      ver_env.d("dim1", other_dims[0]);
      ver_env.d("dim1_size", a.sizes()[other_dims[0]]);
      ver_env.d("dim2", other_dims[1]);
      ver_env.d("dim2_size", a.sizes()[other_dims[1]]);
      ver_env.d("softmax_dim", softmax_dim);
      ver_env.d("softmax_dim_size", softmax_dim_size);
      const auto verification_pattern = format(verification_template, ver_env);

      // verication sting temporarily disabled until
      // inlining of exp() is benchmarked and determined
      // torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

      std::vector<IValue> stack = fmap<IValue>(inputs);
      k.run(stack);
      auto output = stack[0].toTensor();

      ASSERT_EQ(output.sizes(), ref.sizes());
      ASSERT_TRUE(at::allclose(output, ref));
    }
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(Kernel, Softmax4D) {
  const auto graph_template = R"IR(
      graph(%0 : Float(2, 3, 2, 3, strides=[18, 6, 3, 1], device=cpu)):
        %1 : int = prim::Constant[value=${dim}]()
        %2 : int = prim::Constant[value=7]()
        %3 : Float(${size}, strides=[${strides}]) = aten::${op}(%0, %1, %2)
        return (%3))IR";

  auto a = at::rand({2, 3, 2, 3}, TensorOptions(kCPU).dtype(at::kFloat));

  const std::string& verification_template =
      R"IR(
        # CHECK: for (int i${dim1} = 0; i${dim1} < ${dim1_size}
        # CHECK-NEXT: for (int i${dim2} = 0; i${dim2} < ${dim2_size}
        # CHECK-NEXT: for (int i${dim3} = 0; i${dim3} < ${dim3_size}
        # CHECK: for (int i${softmax_dim} = 0; i${softmax_dim} < ${softmax_dim_size}
        # CHECK-NEXT: aten_softmax_max
        # CHECK: for (int i${dim1}_1 = 0; i${dim1}_1 < ${dim1_size}
        # CHECK-NEXT: for (int i${dim2}_1 = 0; i${dim2}_1 < ${dim2_size}
        # CHECK-NEXT: for (int i${dim3}_1 = 0; i${dim3}_1 < ${dim3_size}
        # CHECK: for (int i${softmax_dim}_1 = 0; i${softmax_dim}_1 < ${softmax_dim_size}
        # CHECK-NEXT: aten_softmax_sum
        # CHECK: for (int i0_2 = 0; i0_2 < 2
        # CHECK-NEXT: for (int i1_2 = 0; i1_2 < 3
        # CHECK-NEXT: for (int i2_2 = 0; i2_2 < 2
        # CHECK-NEXT: for (int i3_2 = 0; i3_2 < 3
        # CHECK-NEXT: aten_softmax)IR";

  for (auto log_softmax : {false, true}) {
    for (int softmax_dim = 0; softmax_dim < a.dim(); ++softmax_dim) {
      auto softmax_dim_size = a.sizes()[softmax_dim];
      std::vector<int> other_dims;
      for (int i = 0; i < a.dim(); ++i) {
        if (i != softmax_dim) {
          other_dims.push_back(i);
        }
      }
      auto ref =
          log_softmax ? a.log_softmax(softmax_dim) : a.softmax(softmax_dim);

      KernelScope kernel_scope;
      TemplateEnv env;
      env.d("dim", softmax_dim);
      env.s("op", log_softmax ? "log_softmax" : "softmax");
      env.s("size", li_to_str(ref.sizes()));
      env.s("strides", li_to_str(ref.strides()));

      const auto graph_string = format(graph_template, env);

      auto graph = std::make_shared<Graph>();
      parseIR(graph_string, &*graph);

      TensorExprKernel k(graph);
      std::vector<at::Tensor> inputs = {a};
      Stmt* s = k.getCodeGenStmt();

      std::ostringstream oss;
      oss << *s;

      TemplateEnv ver_env;
      ver_env.d("dim1", other_dims[0]);
      ver_env.d("dim1_size", a.sizes()[other_dims[0]]);
      ver_env.d("dim2", other_dims[1]);
      ver_env.d("dim2_size", a.sizes()[other_dims[1]]);
      ver_env.d("dim3", other_dims[2]);
      ver_env.d("dim3_size", a.sizes()[other_dims[2]]);
      ver_env.d("softmax_dim", softmax_dim);
      ver_env.d("softmax_dim_size", softmax_dim_size);
      const auto verification_pattern = format(verification_template, ver_env);

      // verication sting temporarily disabled until
      // inlining of exp() is benchmarked and determined
      // torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

      std::vector<IValue> stack = fmap<IValue>(inputs);
      k.run(stack);
      auto output = stack[0].toTensor();
      ASSERT_EQ(output.sizes(), ref.sizes());
      ASSERT_TRUE(at::allclose(output, ref));
    }
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(Kernel, DISABLED_InlineProducerIntoReduction) {
  // see : [zero-dim tensors]
  KernelScope kernel_scope;

  // Inline producer (mul) into reduction (sum).
  const auto graph_string = R"IR(
      graph(%0 : Float(5, 3, strides=[3, 1], device=cpu),
            %1 : Float(5, 3, strides=[3, 1], device=cpu)):
        %2 : Float(5, 3, strides=[3, 1]) = aten::mul(%0, %1)
        %3 : int = prim::Constant[value=7]()
        %4 : Float(5, 3, strides=[3, 1]) = aten::sum(%2, %3)
        return (%4))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  TensorExprKernel k(graph);
  Stmt* s = k.getCodeGenStmt();
  std::ostringstream oss;
  oss << *s;

  // Check the IR we produced.
  // We should have only one loop in the end.
  const std::string& verification_pattern =
      R"IR(
        # CHECK: for (int v = 0; v < 5;
        # CHECK-NEXT: for (int v_1 = 0; v_1 < 3;
        # CHECK-NEXT:   sum
        # CHECK-NOT: for)IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  auto a = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto b = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  std::vector<at::Tensor> inputs = {a, b};
  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  auto o = stack[0].toTensor();
  auto ref = (a * b).sum(at::kDouble);
  ASSERT_TRUE(at::allclose(o, ref));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(Kernel, DISABLED_InlineReductionIntoConsumer) {
  // see : [zero-dim tensors]

  KernelScope kernel_scope;

  // Inline producer (mul %2) into reduction (sum %4) but DO NOT
  // inline the reduction into consumer (mul %4).
  const auto graph_string = R"IR(
      graph(%0 : Float(5, 3, strides=[3, 1], device=cpu),
            %1 : Float(5, 3, strides=[3, 1], device=cpu)):
        %2 : Float(5, 3, strides=[3, 1]) = aten::mul(%0, %1)
        %3 : int = prim::Constant[value=6]()
        %4 : Float(5, 3, strides=[3, 1]) = aten::sum(%2, %3)
        %5 : Float(5, 3, strides=[3, 1]) = aten::mul(%2, %4)
        return (%5))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  TensorExprKernel k(graph);
  Stmt* s = k.getCodeGenStmt();
  std::ostringstream oss;
  oss << *s;

  // Check the IR we produced.
  // We should have two loops in the end.
  const std::string& verification_pattern =
      R"IR(
        # CHECK: for (int v = 0; v < 5;
        # CHECK-NEXT: for (int v_1 = 0; v_1 < 3;
        # CHECK-NEXT:   sum
        # CHECK: for (int v_2 = 0; v_2 < 5;
        # CHECK-NEXT: for (int v_3 = 0; v_3 < 3;
        # CHECK-NEXT:   aten_mul
        # CHECK-NOT: for)IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  auto a = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto b = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  std::vector<at::Tensor> inputs = {a, b};
  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  auto o = stack[0].toTensor();
  auto ref = (a * b).sum(at::kFloat) * (a * b);
  ASSERT_TRUE(at::allclose(o, ref));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(Kernel, SanitizeNames_CUDA) {
  const auto graph_string = R"IR(
      graph(%0 : Float(5, 3, strides=[3, 1], device=cuda:0),
            %1 : Float(5, 3, strides=[3, 1], device=cuda:0)):
        %2 : Float(5, 3, strides=[3, 1]) = aten::mul(%0, %1)
        %4 : Float(5, 3, strides=[3, 1]) = aten::mul(%0, %2)
        return (%4))IR";
  KernelScope kernel_scope;
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);
  graph->inputs().at(0)->setDebugName("aten::add:");
  graph->inputs().at(1)->setDebugName("aten::add_");
  TensorExprKernel k(graph);
  auto a = at::rand({5, 3}, TensorOptions(kCUDA).dtype(at::kFloat));
  auto b = at::rand({5, 3}, TensorOptions(kCUDA).dtype(at::kFloat));
  auto ref = a * (a * b);
  std::vector<at::Tensor> inputs = {a, b};
  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  auto o = stack[0].toTensor();
  ASSERT_TRUE(at::allclose(o, ref));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(Kernel, ConstantTensors) {
  const auto graph_string = R"IR(
        graph(%x : Float(16, 16, strides=[16, 1], device=cpu)):
          %none : NoneType = prim::Constant()
          %size : int = prim::Constant[value=16]()
          %sizes : int[] = prim::ListConstruct(%size, %size)
          %y : Float(16, 16, strides=[16, 1], device=cpu) = aten::ones(%sizes, %none, %none, %none, %none)
          %z : Float(16, 16, strides=[16, 1], device=cpu) = aten::mul(%x, %y)
          return (%z))IR";
  KernelScope kernel_scope;
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);
  // IRParser doesn't support tensor constants, so we insert a call to
  // aten::ones and then const-prop it
  ConstantPropagation(graph);

  TensorExprKernel k(graph);

  auto x = at::rand({16, 16}, TensorOptions(kCPU).dtype(at::kFloat));
  std::vector<at::Tensor> inputs = {x};
  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  auto o = stack[0].toTensor();
  auto y = at::ones({16, 16}, TensorOptions(kCPU).dtype(at::kFloat));
  auto ref = x * y;
  ASSERT_TRUE(at::allclose(o, ref));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(Kernel, ConstantTensorsNonContiguous) {
  const auto graph_string = R"IR(
        graph(%x : Float(16, 16, strides=[16, 1], device=cpu)):
          %none : NoneType = prim::Constant()
          %dtype : int = prim::Constant[value=6]()
          %c0 : int = prim::Constant[value=0]()
          %c256 : int = prim::Constant[value=256]()
          %c16 : int = prim::Constant[value=16]()
          %y_flat : Tensor = aten::arange(%c0, %c256, %dtype, %none, %none, %none)
          %sizes : int[] = prim::ListConstruct(%c16, %c16)
          %y_t : Tensor = aten::view(%y_flat, %sizes)
          %y : Tensor = aten::t(%y_t)
          %z : Float(16, 16, strides=[16, 1], device=cpu) = aten::mul(%x, %y)
          return (%z))IR";
  KernelScope kernel_scope;
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);
  // IRParser doesn't support tensor constants, so we generate several aten
  // calls to produce non-contiguos constant tensor and then const-prop it
  ConstantPropagation(graph);

  TensorExprKernel k(graph);

  auto x = at::rand({16, 16}, TensorOptions(kCPU).dtype(at::kFloat));
  std::vector<at::Tensor> inputs = {x};
  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  auto o = stack[0].toTensor();
  auto y = at::arange(0, 256, TensorOptions(kCPU).dtype(at::kFloat))
               .view({16, 16})
               .t();
  auto ref = x * y;
  ASSERT_TRUE(at::allclose(o, ref));
}

TEST_F(Kernel, RunFast) {
#ifdef TORCH_ENABLE_LLVM
  // TODO: Implement call_raw in IREval and remove the ifdef
  KernelScope kernel_scope;

  const auto graph_string = R"IR(
      graph(%0 : Float(5, 3, strides=[3, 1], device=cpu),
            %1 : Float(5, 3, strides=[1, 5], device=cpu)):
        %2 : Float(5, 3, strides=[3, 1]) = aten::mul(%0, %1)
        %3 : Float(5, 3, strides=[3, 1]) = aten::mul(%0, %2)
        return (%3))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  auto a = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto b =
      at::rand({3, 5}, TensorOptions(kCPU).dtype(at::kFloat)).transpose(0, 1);
  auto o = at::zeros({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto ref = a * (a * b);
  TensorExprKernel k(graph);

  k.runFast({a.data_ptr(), b.data_ptr()}, {o.data_ptr()});
  for (size_t i = 0; i < 5 * 3; i++) {
    CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
  }
#endif
}

TEST_F(Kernel, CodegenInspection) {
#ifdef TORCH_ENABLE_LLVM
  const auto graph_string = R"IR(
        graph(%x : Float(16, 16, strides=[16, 1], device=cpu)):
          %none : NoneType = prim::Constant()
          %dtype : int = prim::Constant[value=6]()
          %c0 : int = prim::Constant[value=0]()
          %c256 : int = prim::Constant[value=256]()
          %c16 : int = prim::Constant[value=16]()
          %y_flat : Tensor = aten::arange(%c0, %c256, %dtype, %none, %none, %none)
          %sizes : int[] = prim::ListConstruct(%c16, %c16)
          %y_t : Tensor = aten::view(%y_flat, %sizes)
          %y : Tensor = aten::t(%y_t)
          %z : Float(16, 16, strides=[16, 1], device=cpu) = aten::mul(%x, %y)
          return (%z))IR";
  KernelScope kernel_scope;
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);
  // IRParser doesn't support tensor constants, so we generate several aten
  // calls to produce non-contiguos constant tensor and then const-prop it
  ConstantPropagation(graph);

  TensorExprKernel k(graph);

  // Check that we could retrieve generated assembly
  auto asm_str = k.getCodeText("asm");
  const std::string& asm_verification_pattern =
      R"ASM(
        # CHECK: .text
        # CHECK: retq)ASM";
  torch::jit::testing::FileCheck().run(asm_verification_pattern, asm_str);

  // Check that we could retrieve info about codegen parameters
  auto constants = k.getConstantDescriptors();
  auto buf_args = k.getBufferArgs();
  // Expected buf args: [input0, output0, constant0]
  ASSERT_EQ(buf_args.size(), 3);
  ASSERT_EQ(constants.size(), 1);
  ASSERT_TRUE(
      !buf_args[0].isVar() && !buf_args[1].isVar() && !buf_args[2].isVar());
#endif
}

} // namespace jit
} // namespace torch
