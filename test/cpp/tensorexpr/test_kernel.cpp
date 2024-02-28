#include <gtest/gtest.h>

#include <ATen/code_template.h>
#include <c10/util/irange.h>
#include <test/cpp/tensorexpr/test_base.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
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
  void SetUp() override {
    getTEMustUseLLVMOnCPU() = false;
  }
};

TEST_F(Kernel, ParallelExternalCallBuf) {
  const auto graph_string = R"IR(
    graph(%0 : Float(1000, 5000, strides=[5000, 1], device=cpu),
          %1 : Float(1000, 5000, strides=[5000, 1], device=cpu),
          %2 : Float(5000, 1000, strides=[5000, 1], device=cpu)):
      %3 : Float(1000, 5000, strides=[5000, 1], device=cpu) = aten::mul(%0, %1)
      %4 : Float(1000, 5000, strides=[5000, 1], device=cpu) = aten::matmul(%3, %2)
      return (%4))IR";
  auto graph = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, &*graph);
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int64_t i = 0ll; i < 5000ll; i++)  /* parallel */{)IR";

#ifdef TORCH_ENABLE_LLVM
  TensorExprKernel k(graph);
  StmtPtr s = k.getCodeGenStmt();
  std::ostringstream oss;
  oss << *s;
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
#endif
}

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

      at::jit::TemplateEnv env;
      env.s("device", use_cuda ? "cuda:0" : "cpu");
      const auto graph_string = format(graph_template, env);
      auto graph = std::make_shared<Graph>();
      parseIR(graph_string, &*graph);
      TensorExprKernel k(graph);
      auto stmt = k.getCodeGenStmt();
      std::ostringstream oss;
      oss << *stmt;
      // aten_mul only has one use, inlined completely
      torch::jit::testing::FileCheck().check_not("aten_mul")->run(oss.str());

      // aten_sub should be removed by the CUDA backend by metavar rewriting
      // and by the CPU backend by horizontal fusion.
      torch::jit::testing::FileCheck().check_not("aten_sub")->run(oss.str());
    }
  }
}

TEST_F(Kernel, PreAllocIntermediateBufs) {
  const auto graph_string = R"IR(
graph(%a.1 : Float(8, 8, strides=[8, 1], requires_grad=0, device=cpu),
      %b.1 : Float(8, 8, strides=[8, 1], requires_grad=0, device=cpu)):
  %2 : int = prim::Constant[value=1]()
  %c.2 : Float(8, 8, strides=[8, 1], requires_grad=0, device=cpu) = aten::matmul(%a.1, %b.1) # test_matmul.py:12:12
  %3 : Float(8, 8, strides=[8, 1], requires_grad=0, device=cpu) = aten::add(%a.1, %c.2, %2) # test_matmul.py:13:15
  return (%3))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  auto a = at::rand({8, 8}, TensorOptions(kCPU).dtype(at::kFloat));
  auto b = at::rand({8, 8}, TensorOptions(kCPU).dtype(at::kFloat));
  auto o = at::zeros({8, 8}, TensorOptions(kCPU).dtype(at::kFloat));
  auto ref = at::matmul(a, b) + a;
  TensorExprKernel k(graph, {}, {}, true);

  std::vector<at::Tensor> inputs = {a, b};
  auto stmt = k.getCodeGenStmt();

  std::ostringstream oss;
  oss << *stmt;

  // Check whether the intermediate buffer has been added to constants
  auto constants = k.getConstantDescriptors();
  ASSERT_EQ(constants.size(), 1);

  // Check the IR we produced
  torch::jit::testing::FileCheck().check_not("Alloc")->run(oss.str());
  torch::jit::testing::FileCheck().check_not("Free")->run(oss.str());

  // Check correctness
  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  o = stack[0].toTensor();
  ASSERT_TRUE(at::allclose(o, ref));
}

TEST_F(Kernel, _1) {
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
  StmtPtr s = k.getCodeGenStmt();

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
    TORCH_CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
  }
}

TEST_F(Kernel, _2) {
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
  StmtPtr s = k.getCodeGenStmt();

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
    TORCH_CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
  }
}

TEST_F(Kernel, _3) {
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
  StmtPtr s = k.getCodeGenStmt();

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
    TORCH_CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
  }
}

TEST_F(Kernel, Huge) {
  const auto graph_string = R"IR(
      graph(%x.1 : Float(4000000000, strides=[1], requires_grad=0, device=cpu)):
        %1 : int = prim::Constant[value=0]()
        %2 : Float(1, 4000000000, strides=[4000000000, 1], requires_grad=0, device=cpu) = aten::unsqueeze(%x.1, %1)
        %3 : Float(1, 4000000000, strides=[4000000000, 1], requires_grad=0, device=cpu) = aten::relu(%2)
        return (%3))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);
  TensorExprKernel k(graph);
  std::ostringstream oss;
  oss << *k.getCodeGenStmt();
  // The 4000000000 iterations loop will be split into 500000000 x 8 and the
  // outer loop will be parallel. If LLVM is not present, it will not be split,
  // and to cover both of these cases we're looking for 00000000ll; in the
  // output.
  const std::string& verification_pattern = R"IR(# CHECK: 00000000ll;)IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

TEST_F(Kernel, ParallelStrided) {
  const auto graph_string = R"IR(
      graph(%0 : Float(5, 3, 40005, strides=[120015, 40005, 1], device=cpu),
            %1 : Float(5, 3, 40005, strides=[960120, 160020, 2], device=cpu)):
        %2 : Float(5, 3, 40005, strides=[120015, 40005, 1]) = aten::mul(%0, %1)
        %3 : Float(5, 3, 40005, strides=[120015, 40005, 1]) = aten::mul(%0, %2)
        return (%3))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  auto a = at::rand({5, 3, 40005}, TensorOptions(kCPU).dtype(at::kFloat));
  auto b = at::rand({10, 6, 80010}, TensorOptions(kCPU).dtype(at::kFloat))
               .index(
                   {Slice(None, None, 2),
                    Slice(None, None, 2),
                    Slice(None, None, 2)});
  auto ref = a * (a * b);
  auto o = at::zeros_like(ref);
  TensorExprKernel k(graph);
  std::vector<at::Tensor> inputs = {a, b};
  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  o = stack[0].toTensor();
  for (size_t i = 0; i < 5 * 3; i++) {
    TORCH_CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
  }
}

TEST_F(Kernel, DISABLED_Shape_Inference) {
  // disabled: doesn't do stride propagation, and isn't being used currently

  // Test TensorExpr shape inference capabilities: it should only require shapes
  // for the inputs
  {
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
    StmtPtr s = k.getCodeGenStmt();

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
      TORCH_CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
    }
  }
  {
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
    StmtPtr s = k.getCodeGenStmt();

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
    TORCH_CHECK_EQ(o.sizes()[0], 8);
    TORCH_CHECK_EQ(o.sizes()[1], 4);
    for (size_t i = 0; i < 8 * 4; i++) {
      TORCH_CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
    }
  }
  {
    // Test that shape inference handles aten::unsqueeze

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
    StmtPtr s = k.getCodeGenStmt();

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
    TORCH_CHECK_EQ(o.sizes().size(), ref.sizes().size());
    size_t num_el = 1;
    for (const auto idx : c10::irange(ref.sizes().size())) {
      TORCH_CHECK_EQ(o.sizes()[idx], ref.sizes()[idx]);
      num_el *= ref.sizes()[idx];
    }

    // Check the contents
    for (const auto i : c10::irange(num_el)) {
      TORCH_CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
    }
  }
  {
    // Test that shape inference handles aten::cat

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
    StmtPtr s = k.getCodeGenStmt();

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
    TORCH_CHECK_EQ(o.sizes().size(), ref.sizes().size());
    size_t num_el = 1;
    for (const auto idx : c10::irange(ref.sizes().size())) {
      TORCH_CHECK_EQ(o.sizes()[idx], ref.sizes()[idx]);
      num_el *= ref.sizes()[idx];
    }

    // Check the contents
    for (const auto i : c10::irange(num_el)) {
      TORCH_CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
    }
  }
  {
    // Test that we throw an error when input list for aten::cat is empty

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

TEST_F(Kernel, CatInputTypesPromotion) {
  {
    // Test that we properly promote input types for aten::cat

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
    StmtPtr s = k.getCodeGenStmt();

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
    TORCH_CHECK_EQ(o.sizes().size(), ref.sizes().size());
    TORCH_CHECK_EQ(o.dtype(), ref.dtype());
    size_t num_el = 1;
    for (const auto idx : c10::irange(ref.sizes().size())) {
      TORCH_CHECK_EQ(o.sizes()[idx], ref.sizes()[idx]);
      num_el *= ref.sizes()[idx];
    }

    // Check the contents
    for (const auto i : c10::irange(num_el)) {
      TORCH_CHECK_EQ(((double*)o.data_ptr())[i], ((double*)ref.data_ptr())[i]);
    }
  }
}

TEST_F(Kernel, ToDType) {
#ifdef TORCH_ENABLE_LLVM
  const auto graph_string = R"IR(
      graph(%x.1 : BFloat16(2, 2, strides=[2, 1], requires_grad=0, device=cpu)):
        %1 : NoneType = prim::Constant()
        %2 : bool = prim::Constant[value=0]()
        %3 : int = prim::Constant[value=6]()
        %4 : int = prim::Constant[value=15]()
        %5 : int = prim::Constant[value=5]()
        %6 : bool = prim::Constant[value=1]()
        %y.3 : BFloat16(2, 2, strides=[2, 1], requires_grad=0, device=cpu) = aten::sigmoid(%x.1)
        %z.3 : BFloat16(2, 2, strides=[2, 1], requires_grad=0, device=cpu) = aten::_autocast_to_reduced_precision(%y.3, %6, %6, %5, %4)
        %h.3 : Float(2, 2, strides=[2, 1], requires_grad=0, device=cpu) = aten::_autocast_to_full_precision(%z.3, %6, %6)
        %i.3 : Float(2, 2, strides=[2, 1], requires_grad=0, device=cpu) = aten::to(%h.3, %3, %2, %2, %1)
        %j.3 : BFloat16(2, 2, strides=[2, 1], requires_grad=0, device=cpu) = aten::to(%i.3, %4, %2, %2, %1)
        %k.3 : Float(2, 2, strides=[2, 1], requires_grad=0, device=cpu) = aten::to(%j.3, %3, %2, %2, %1)
        return (%k.3))IR";

  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);
  TensorExprKernel k(graph);
  StmtPtr s = k.getCodeGenStmt();
  std::ostringstream oss;
  oss << *s;

  const std::string& verification_pattern =
      R"IR(
# CHECK: for
# CHECK-NEXT: for
# CHECK-NEXT: aten_to
# CHECK-NEXT: }
# CHECK-NEXT: })IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  auto a = at::rand({2, 2}, TensorOptions(kCPU).dtype(at::kBFloat16));
  auto ref =
      at::_to_copy(at::sigmoid(a), TensorOptions(kCPU).dtype(at::kFloat));

  std::vector<at::Tensor> inputs = {a};
  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  auto o = stack[0].toTensor();
  ASSERT_EQ(o.sizes(), ref.sizes());
  ASSERT_EQ(o.dtype(), ref.dtype());
  ASSERT_TRUE(at::allclose(o, ref, 4E-3, 4E-3));
#endif
}

TEST_F(Kernel, CatAndInlineWithAConstantDim) {
  const auto graph_string = R"IR(
      graph(%0 : Float(1, 512, strides=[1024, 1], requires_grad=0, device=cpu),
            %1 : Float(1, 512, strides=[1024, 1], requires_grad=0, device=cpu)):
        %2 : bool = prim::Constant[value=0]()
        %3 : int = prim::Constant[value=1]()
        %4 : Tensor[] = prim::ListConstruct(%0, %1)
        %5 : Float(1, 1024, strides=[1024, 1], requires_grad=0, device=cpu) = aten::cat(%4, %3)
        %6 : Tensor[] = prim::ListConstruct(%5)
        %7 : Float(1, 1024, strides=[1024, 1], requires_grad=0, device=cpu) = aten::cat(%6, %3)
        %8 : Float(1, 1024, strides=[1024, 1], requires_grad=0, device=cpu) = aten::_cast_Float(%7, %2)
        return (%8, %7))IR";

  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);
  TensorExprKernel k(graph);

  auto a = at::rand({1, 512}, TensorOptions(kCPU).dtype(at::kFloat));
  auto b = at::rand({1, 512}, TensorOptions(kCPU).dtype(at::kFloat));
  auto ref = at::_cast_Float(at::cat({a, b}, 1), 0);

  std::vector<at::Tensor> inputs = {a, b};
  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  auto o = stack[0].toTensor();
  ASSERT_EQ(o.sizes(), ref.sizes());
  ASSERT_EQ(o.dtype(), ref.dtype());
  ASSERT_TRUE(at::allclose(o, ref));
}

TEST_F(Kernel, CatWithEmptyInputs) {
  bool curr_cat_wo_conditionals = getCatWoConditionals();
  for (auto cat_wo_conditionals : {true, false}) {
    getCatWoConditionals() = cat_wo_conditionals;
    const auto graph_string = R"IR(
        graph(%0 : Float(0, 64, strides=[64, 1], requires_grad=0, device=cpu),
              %1 : Float(10, 64, strides=[64, 1], requires_grad=0, device=cpu)):
          %3 : int = prim::Constant[value=0]()
          %6 : Float(0, 64, strides=[64, 1], requires_grad=0, device=cpu) = aten::tanh(%0)
          %7 : Float(10, 64, strides=[64, 1], requires_grad=0, device=cpu) = aten::tanh(%1)
          %10 : Tensor[] = prim::ListConstruct(%6, %7)
          %11 : Float(10, 64, strides=[64, 1], requires_grad=0, device=cpu) = aten::cat(%10, %3)
          return (%11))IR";

    auto graph = std::make_shared<Graph>();
    parseIR(graph_string, &*graph);
    TensorExprKernel k(graph);

    auto a = at::rand({0, 64}, TensorOptions(kCPU).dtype(at::kFloat));
    auto b = at::rand({10, 64}, TensorOptions(kCPU).dtype(at::kFloat));
    auto ref = at::cat({at::tanh(a), at::tanh(b)}, 0);

    std::vector<at::Tensor> inputs = {a, b};
    std::vector<IValue> stack = fmap<IValue>(inputs);
    k.run(stack);
    auto o = stack[0].toTensor();
    ASSERT_EQ(o.sizes(), ref.sizes());
    ASSERT_EQ(o.dtype(), ref.dtype());
    ASSERT_TRUE(at::allclose(o, ref));
  }
  getCatWoConditionals() = curr_cat_wo_conditionals;
}

TEST_F(Kernel, CatWoConditionals) {
  bool old_cat_wo_conditionals = getCatWoConditionals();
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
  StmtPtr s = k.getCodeGenStmt();
  std::ostringstream oss;
  oss << *s;

  const std::string& verification_pattern =
      R"IR(
# CHECK: for
# CHECK: for
# CHECK: for
# CHECK: aten_cat
# CHECK: for
# CHECK: for
# CHECK: aten_cat
# CHECK: for
# CHECK: for
# CHECK: aten_cat)IR";
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
  TORCH_CHECK_EQ(o.sizes().size(), ref.sizes().size());
  TORCH_CHECK_EQ(o.dtype(), ref.dtype());
  size_t num_el = 1;
  for (const auto idx : c10::irange(ref.sizes().size())) {
    TORCH_CHECK_EQ(o.sizes()[idx], ref.sizes()[idx]);
    num_el *= ref.sizes()[idx];
  }

  // Check the contents
  for (const auto i : c10::irange(num_el)) {
    TORCH_CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
  }
  getCatWoConditionals() = old_cat_wo_conditionals;
}

TEST_F(Kernel, OptimizeConditionals) {
  bool old_cat_wo_conditionals = getCatWoConditionals();
  bool old_opt_conditionals = getOptConditionals();
  getCatWoConditionals() = false;
  getOptConditionals() = true;
  const auto graph_string = R"IR(
      graph(%a : Float(5, 3, strides=[3, 1], device=cpu),
            %b : Float(5, 7, strides=[7, 1], device=cpu),
            %c : Float(5, 9, strides=[9, 1], device=cpu)):
        %dim : int = prim::Constant[value=1]()
        %inputs : Tensor[] = prim::ListConstruct(%a, %b, %c)
        %r : Float(5, 19, strides=[19, 1]) = aten::cat(%inputs, %dim)
        %t : Float(5, 19, strides=[19, 1]) = aten::relu(%r)
        return (%t))IR";

  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  TensorExprKernel k(graph);
  StmtPtr s = k.getCodeGenStmt();
  std::ostringstream oss;
  oss << *s;

  const std::string& verification_pattern =
      R"IR(
# CHECK: for
# CHECK-NEXT: for
# CHECK-NEXT: aten_relu
# CHECK: for
# CHECK-NEXT: aten_relu
# CHECK: for
# CHECK-NEXT: aten_relu
# CHECK-NOT: Allocate
# CHECK-NOT: Free)IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto a = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto b = at::rand({5, 7}, TensorOptions(kCPU).dtype(at::kFloat));
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto c = at::rand({5, 9}, TensorOptions(kCPU).dtype(at::kFloat));
  auto ref = at::relu(at::cat({a, b, c}, 1));

  std::vector<at::Tensor> inputs = {a, b, c};
  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  auto o = stack[0].toTensor();

  // Check sizes
  TORCH_CHECK_EQ(o.sizes().size(), ref.sizes().size());
  TORCH_CHECK_EQ(o.dtype(), ref.dtype());
  size_t num_el = 1;
  for (const auto idx : c10::irange(ref.sizes().size())) {
    TORCH_CHECK_EQ(o.sizes()[idx], ref.sizes()[idx]);
    num_el *= ref.sizes()[idx];
  }

  // Check the contents
  for (const auto i : c10::irange(num_el)) {
    TORCH_CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
  }
  getOptConditionals() = old_opt_conditionals;
  getCatWoConditionals() = old_cat_wo_conditionals;
}

namespace {

std::string dtypeConstant(ScalarType scalar_type) {
  if (scalar_type == ScalarType::Undefined) {
    return "None = prim::Constant()";
  } else {
    at::jit::TemplateEnv env_dtype;
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

TEST_F(Kernel, SumAllAxes) {
  // Test lowering of sum on all axes.
  const auto graph_template = R"IR(
      graph(%0 : Float(5, 3, strides=[3, 1], device=cpu)):
        %1 : ${dtype}
        %2 : ${out_dtype}(requires_grad=0, device=cpu) = aten::sum(%0, %1)
        return (%2))IR";
  auto a = iotaTensor({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));

  for (auto scalar_type : {ScalarType::Undefined, ScalarType::Double}) {
    at::jit::TemplateEnv env;
    env.s("dtype", dtypeConstant(scalar_type));
    if (scalar_type == ScalarType::Undefined) {
      env.s("out_dtype", "Float");
    } else {
      env.s("out_dtype", "Double");
    }
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
    StmtPtr s = k.getCodeGenStmt();

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
        at::jit::TemplateEnv env;
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
        StmtPtr s = k.getCodeGenStmt();

        std::ostringstream oss;
        oss << *s;

        // Check the IR we produced
        const std::string& verification_pattern =
            R"IR(
# CHECK: for (int64_t
# CHECK-NEXT: sum
# CHECK-NEXT: for (int64_t
# CHECK-NEXT:   sum)IR";
        torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

        std::vector<IValue> stack = fmap<IValue>(inputs);
        k.run(stack);
        o = stack[0].toTensor();
        ASSERT_EQ(o.sizes(), ref.sizes());
        ASSERT_EQ(o.dtype(), ref.dtype());
        ASSERT_TRUE(at::allclose(o, ref, 4E-3, 4E-3));
      }
    }
  }
}

TEST_F(Kernel, SumMultipleAxes) {
  // Test lowering of sum on multiple axes.
  const auto graph_template = R"IR(
      graph(%0 : Float(2, 3, 2, 3, strides=[18, 6, 3, 1], requires_grad=0, device=cpu)):
        %1 : int = prim::Constant[value=${dim1}]()
        %2 : int = prim::Constant[value=${dim2}]()
        %3 : int[] = prim::ListConstruct(%1, %2)
        %4 : bool = prim::Constant[value=${keepdim}]()
        %5 : ${dtype}
        %6 : Float(${size}, strides=[${strides}], requires_grad=0, device=cpu) = aten::sum(%0, %3, %4, %5)
        return (%6))IR";
  auto a = iotaTensor({2, 3, 2, 3}, TensorOptions(kCPU).dtype(at::kFloat));

  // Only iterate over positive values of axes to keep the running time
  // reasonable, since the number of pairs is quadratic.
  for (const auto dim1 : c10::irange(a.dim())) {
    for (int dim2 = dim1 + 1; dim2 < a.dim(); ++dim2) {
      for (bool keepdim : {false, true}) {
        at::jit::TemplateEnv env;
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
        StmtPtr s = k.getCodeGenStmt();

        std::ostringstream oss;
        oss << *s;

        // Check the IR we produced
        const std::string& verification_pattern =
            R"IR(
# CHECK: for (int64_t
# CHECK: for (int64_t
# CHECK: for (int64_t
# CHECK: for (int64_t
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
TEST_F(Kernel, Softmax2D) {
  const auto graph_template = R"IR(
      graph(%0 : Float(5, 3, strides=[3, 1], device=cpu)):
        %1 : int = prim::Constant[value=${dim}]()
        %dt_float : int = prim::Constant[value=7]()
        %dt_none : NoneType = prim::Constant()
        %4 : Float(${size}, strides=[${strides}]) = aten::${op}(%0, %1, %${dt})
        return (%4))IR";

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

  for (bool empty_dtype : {false, true}) {
    for (auto log_softmax : {false, true}) {
      for (const auto softmax_dim : c10::irange(a.dim())) {
        auto softmax_dim_size = a.sizes()[softmax_dim];
        auto other_dim = (softmax_dim + 1) % a.dim();
        auto ref =
            log_softmax ? a.log_softmax(softmax_dim) : a.softmax(softmax_dim);
        at::jit::TemplateEnv env;
        env.d("dim", softmax_dim);
        env.s("op", log_softmax ? "log_softmax" : "softmax");
        env.s("size", li_to_str(ref.sizes()));
        env.s("strides", li_to_str(ref.strides()));
        env.s("dt", empty_dtype ? "dt_none" : "dt_float");

        const auto graph_string = format(graph_template, env);

        auto graph = std::make_shared<Graph>();
        parseIR(graph_string, &*graph);

        TensorExprKernel k(graph);
        std::vector<at::Tensor> inputs = {a};
        StmtPtr s = k.getCodeGenStmt();

        std::ostringstream oss;
        oss << *s;

        at::jit::TemplateEnv ver_env;
        ver_env.d("other_dim", other_dim);
        ver_env.d("other_dim_size", a.sizes()[other_dim]);
        ver_env.d("softmax_dim", softmax_dim);
        ver_env.d("softmax_dim_size", softmax_dim_size);
        const auto verification_pattern =
            format(verification_template, ver_env);

        // verification sting temporarily disabled until
        // inlining of exp() is benchmarked and determined
        // torch::jit::testing::FileCheck().run(verification_pattern,
        // oss.str());

        std::vector<IValue> stack = fmap<IValue>(inputs);
        k.run(stack);
        auto output = stack[0].toTensor();
        ASSERT_EQ(output.sizes(), ref.sizes());
        ASSERT_TRUE(at::allclose(output, ref));
      }
    }
  }
}

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
    for (const auto softmax_dim : c10::irange(a.dim())) {
      auto softmax_dim_size = a.sizes()[softmax_dim];
      std::vector<int> other_dims;
      for (const auto i : c10::irange(a.dim())) {
        if (i != softmax_dim) {
          other_dims.push_back(i);
        }
      }
      auto ref =
          log_softmax ? a.log_softmax(softmax_dim) : a.softmax(softmax_dim);

      at::jit::TemplateEnv env;
      env.d("dim", softmax_dim);
      env.s("op", log_softmax ? "log_softmax" : "softmax");
      env.s("size", li_to_str(ref.sizes()));
      env.s("strides", li_to_str(ref.strides()));

      const auto graph_string = format(graph_template, env);

      auto graph = std::make_shared<Graph>();
      parseIR(graph_string, &*graph);

      TensorExprKernel k(graph);
      std::vector<at::Tensor> inputs = {a};
      StmtPtr s = k.getCodeGenStmt();

      std::ostringstream oss;
      oss << *s;

      at::jit::TemplateEnv ver_env;
      ver_env.d("dim1", other_dims[0]);
      ver_env.d("dim1_size", a.sizes()[other_dims[0]]);
      ver_env.d("dim2", other_dims[1]);
      ver_env.d("dim2_size", a.sizes()[other_dims[1]]);
      ver_env.d("softmax_dim", softmax_dim);
      ver_env.d("softmax_dim_size", softmax_dim_size);
      const auto verification_pattern = format(verification_template, ver_env);

      // verification sting temporarily disabled until
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
    for (const auto softmax_dim : c10::irange(a.dim())) {
      auto softmax_dim_size = a.sizes()[softmax_dim];
      std::vector<int> other_dims;
      for (const auto i : c10::irange(a.dim())) {
        if (i != softmax_dim) {
          other_dims.push_back(i);
        }
      }
      auto ref =
          log_softmax ? a.log_softmax(softmax_dim) : a.softmax(softmax_dim);

      at::jit::TemplateEnv env;
      env.d("dim", softmax_dim);
      env.s("op", log_softmax ? "log_softmax" : "softmax");
      env.s("size", li_to_str(ref.sizes()));
      env.s("strides", li_to_str(ref.strides()));

      const auto graph_string = format(graph_template, env);

      auto graph = std::make_shared<Graph>();
      parseIR(graph_string, &*graph);

      TensorExprKernel k(graph);
      std::vector<at::Tensor> inputs = {a};
      StmtPtr s = k.getCodeGenStmt();

      std::ostringstream oss;
      oss << *s;

      at::jit::TemplateEnv ver_env;
      ver_env.d("dim1", other_dims[0]);
      ver_env.d("dim1_size", a.sizes()[other_dims[0]]);
      ver_env.d("dim2", other_dims[1]);
      ver_env.d("dim2_size", a.sizes()[other_dims[1]]);
      ver_env.d("dim3", other_dims[2]);
      ver_env.d("dim3_size", a.sizes()[other_dims[2]]);
      ver_env.d("softmax_dim", softmax_dim);
      ver_env.d("softmax_dim_size", softmax_dim_size);
      const auto verification_pattern = format(verification_template, ver_env);

      // verification sting temporarily disabled until
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

TEST_F(Kernel, SignTest) {
  const auto graph_template = R"IR(
      graph(%0 : ${dtype}(${size}, strides=[1], device=cpu)):
        %2 : ${dtype}(${size}, strides=[1]) = aten::sign(%0)
        return (%2))IR";

  auto run_test = [](const std::string& graph_string, const at::Tensor& input) {
    auto graph = std::make_shared<Graph>();
    parseIR(graph_string, &*graph);

    TensorExprKernel k(graph);
    StmtPtr s = k.getCodeGenStmt();

    std::vector<at::Tensor> inputs = {input};
    std::vector<IValue> stack = fmap<IValue>(inputs);
    k.run(stack);
    auto o = stack[0].toTensor();
    auto ref = at::sign(input);
    ASSERT_TRUE(at::allclose(o, ref));
  };
  auto common_options = at::TensorOptions()
                            .layout(at::kStrided)
                            .device(at::kCPU)
                            .requires_grad(false);
  int default_input_size = 100;
  for (auto scalar_type : {ScalarType::Float, ScalarType::Double}) {
    at::Tensor corner_case_inputs;
    at::jit::TemplateEnv env;
    auto options = common_options;
    switch (scalar_type) {
      case ScalarType::Float: {
        env.s("dtype", "Float");
        options = options.dtype(at::kFloat);
        std::vector<float> input_float = {
            0.0f,
            -0.0f,
            std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
            std::nanf("1"),
            -std::nanf("1")};
        corner_case_inputs = at::from_blob(
            input_float.data(),
            {static_cast<long>(input_float.size())},
            options);
        auto rand_input = at::rand({default_input_size}, options);
        auto input = at::cat({rand_input, corner_case_inputs});
        env.d("size", at::numel(input));
        const auto graph_string = format(graph_template, env);
        run_test(graph_string, input);
        break;
      }
      case ScalarType::Double: {
        env.s("dtype", "Double");
        options = options.dtype(at::kDouble);
        std::vector<double> input_double = {
            0.0,
            -0.0,
            std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity(),
            std::nan("1"),
            -std::nan("1")};
        corner_case_inputs = at::from_blob(
            input_double.data(),
            {static_cast<long>(input_double.size())},
            options);
        auto rand_input = at::rand({default_input_size}, options);
        auto input = at::cat({rand_input, corner_case_inputs});
        env.d("size", at::numel(input));
        const auto graph_string = format(graph_template, env);
        run_test(graph_string, input);
        break;
      }
      default:
        throw unsupported_dtype();
    }
  }
}

TEST_F(Kernel, InlineProducerIntoReduction) {
  // Inline producer (mul) into reduction (sum).
  const auto graph_string = R"IR(
      graph(%0 : Float(5, 3, strides=[3, 1], device=cpu),
            %1 : Float(5, 3, strides=[3, 1], device=cpu)):
        %2 : Float(5, 3, strides=[3, 1], device=cpu) = aten::mul(%0, %1)
        %3 : int = prim::Constant[value=7]()
        %4 : Double(device=cpu) = aten::sum(%2, %3)
        return (%4))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  TensorExprKernel k(graph);
  StmtPtr s = k.getCodeGenStmt();
  std::ostringstream oss;
  oss << *s;

  // Check the IR we produced.
  // We should have only one loop in the end.
  const std::string& verification_pattern =
      R"IR(
        # CHECK: for (int64_t i_1 = 0ll; i_1 < 5
        # CHECK-NEXT: for (int64_t j_1 = 0ll; j_1 < 3
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

TEST_F(Kernel, InlineReductionIntoConsumer) {
  // Inline producer (mul %2) into reduction (sum %4) but DO NOT
  // inline the reduction into consumer (mul %4).
  const auto graph_string = R"IR(
      graph(%0 : Float(5, 3, strides=[3, 1], device=cpu),
            %1 : Float(5, 3, strides=[3, 1], device=cpu)):
        %2 : Float(5, 3, strides=[3, 1]) = aten::mul(%0, %1)
        %3 : int = prim::Constant[value=6]()
        %4 : Float(device=cpu) = aten::sum(%2, %3)
        %5 : Float(5, 3, strides=[3, 1], device=cpu) = aten::mul(%2, %4)
        return (%5))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  TensorExprKernel k(graph);
  StmtPtr s = k.getCodeGenStmt();
  std::ostringstream oss;
  oss << *s;

  // Check the IR we produced.
  // We should have two loops in the end.
  const std::string& verification_pattern =
      R"IR(
        # CHECK: for (int64_t i_1 = 0ll; i_1 < 5
        # CHECK-NEXT: for (int64_t j_1 = 0ll; j_1 < 3
        # CHECK-NEXT:   sum
        # CHECK: for (int64_t i_2 = 0ll; i_2 < 5
        # CHECK-NEXT: for (int64_t j_2 = 0ll; j_2 < 3
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

TEST_F(Kernel, SanitizeNames_CUDA) {
  const auto graph_string = R"IR(
      graph(%0 : Float(5, 3, strides=[3, 1], device=cuda:0),
            %1 : Float(5, 3, strides=[3, 1], device=cuda:0)):
        %2 : Float(5, 3, strides=[3, 1]) = aten::mul(%0, %1)
        %4 : Float(5, 3, strides=[3, 1]) = aten::mul(%0, %2)
        return (%4))IR";
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

TEST_F(Kernel, SanitizeConstants_CUDA) {
  const auto graph_string = R"IR(
        graph(%x : Float(16, 16, strides=[16, 1], device=cuda:0)):
          %none : NoneType = prim::Constant()
          %size : int = prim::Constant[value=16]()
          %sizes : int[] = prim::ListConstruct(%size, %size)
          %30 : Device = prim::Constant[value="cuda"]()
          %y : Float(16, 16, strides=[16, 1], device=cuda:0) = aten::ones(%sizes, %none, %none, %30, %none)
          %z : Float(16, 16, strides=[16, 1], device=cuda:0) = aten::mul(%x, %y)
          return (%z))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);
  // IRParser doesn't support tensor constants, so we insert a call to
  // aten::ones and then const-prop it
  ConstantPropagation(graph);

  // We set the name of the constant to include special characters that are
  // not allowed. This should be fixed by the sanitizer in TensorExprKernel.
  graph->nodes().front()->output()->setDebugName("illegal.name");

  // Check if we have a constant node with illegal name in the graph.
  auto const_node = graph->nodes().front();
  ASSERT_EQ(const_node->kind(), prim::Constant);
  ASSERT_NE(const_node->output()->debugName().find('.'), std::string::npos);

  TensorExprKernel k(graph);

  auto x = at::rand({16, 16}, TensorOptions(kCUDA).dtype(at::kFloat));
  std::vector<at::Tensor> inputs = {x};
  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  auto o = stack[0].toTensor();
  auto y = at::ones({16, 16}, TensorOptions(kCUDA).dtype(at::kFloat));
  auto ref = x * y;
  ASSERT_TRUE(at::allclose(o, ref));
}

TEST_F(Kernel, ConstantTensors) {
  const auto graph_string = R"IR(
        graph(%x : Float(16, 16, strides=[16, 1], device=cpu)):
          %none : NoneType = prim::Constant()
          %size : int = prim::Constant[value=16]()
          %sizes : int[] = prim::ListConstruct(%size, %size)
          %y : Float(16, 16, strides=[16, 1], device=cpu) = aten::ones(%sizes, %none, %none, %none, %none)
          %z : Float(16, 16, strides=[16, 1], device=cpu) = aten::mul(%x, %y)
          return (%z))IR";
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
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);
  // IRParser doesn't support tensor constants, so we generate several aten
  // calls to produce non-contiguous constant tensor and then const-prop it
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
    TORCH_CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
  }
#endif
}

TEST_F(Kernel, RunWithAllocatedOutputs) {
#ifdef TORCH_ENABLE_LLVM
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

  std::vector<at::Tensor> args = {o, a, b};
  std::vector<IValue> stack = fmap<IValue>(args);
  k.runWithAllocatedOutputs(stack);
  for (size_t i = 0; i < 5 * 3; i++) {
    TORCH_CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
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
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);
  // IRParser doesn't support tensor constants, so we generate several aten
  // calls to produce non-contiguous constant tensor and then const-prop it
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

Tensor lowerNanToNum(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device device) {
  auto input_buf = std::get<BufHandle>(inputs[0]);
  auto e = Compute(
      "custom_nan_to_num",
      outputShape,
      outputStrides,
      [&](const std::vector<VarHandle>& axes) {
        std::vector<ExprHandle> indices(axes.begin(), axes.end());
        auto load = input_buf.load(indices);
        return IfThenElse::make(Cast::make(kBool, isnan(load)), 0.0f, load);
      });
  return e;
}

TEST_F(Kernel, CustomLowering) {
  const auto graph_string = R"IR(
      graph(%x : Float(2, 2, strides=[2, 1], requires_grad=0, device=cpu)):
          %none : NoneType = prim::Constant()
          %y : Float(2, 2, strides=[2, 1], requires_grad=0, device=cpu) = aten::nan_to_num(%x, %none, %none, %none)
          return (%y)
)IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  std::unordered_map<c10::Symbol, NNCLoweringFunction> lowerings = {
      {aten::nan_to_num, lowerNanToNum}};
  TensorExprKernel k(graph, lowerings);

  auto stmt = k.getCodeGenStmt();
  std::ostringstream oss;
  oss << *stmt;

  // Check that our custom lowering is actually used
  torch::jit::testing::FileCheck().check("custom_nan_to_num")->run(oss.str());
  torch::jit::testing::FileCheck().check("isnan")->run(oss.str());
}

TEST_F(Kernel, Vectorize) {
#ifdef TORCH_ENABLE_LLVM
  const auto graph_string = R"IR(
      graph(%0 : Float(100, 16, strides=[16, 1], device=cpu),
            %1 : Float(100, 16, strides=[16, 1], device=cpu)):
        %2 : Float(100, 16, strides=[16, 1]) = aten::mul(%0, %1)
        %3 : Float(100, 16, strides=[16, 1]) = aten::mul(%0, %2)
        return (%3))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  auto a = at::rand({100, 16}, TensorOptions(kCPU).dtype(at::kFloat));
  auto b = at::rand({100, 16}, TensorOptions(kCPU).dtype(at::kFloat));
  auto o = at::zeros({100, 16}, TensorOptions(kCPU).dtype(at::kFloat));
  auto ref = a * (a * b);
  TensorExprKernel k(graph);
  std::vector<at::Tensor> inputs = {a, b};
  StmtPtr s = k.getCodeGenStmt();

  std::ostringstream oss;
  oss << *s;

  // Check the IR we produced
  const std::string& verification_pattern = R"IR(# CHECK: Ramp)IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  o = stack[0].toTensor();
  for (size_t i = 0; i < 100 * 16; i++) {
    TORCH_CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
  }
#endif
}

// TODO: To vectorize loopnest for 100x3 case, we need to flatten loops first.
TEST_F(Kernel, DISABLED_FlattenVectorize) {
#ifdef TORCH_ENABLE_LLVM
  const auto graph_string = R"IR(
      graph(%0 : Float(100, 3, strides=[3, 1], device=cpu),
            %1 : Float(100, 3, strides=[3, 1], device=cpu)):
        %2 : Float(100, 3, strides=[3, 1]) = aten::mul(%0, %1)
        %3 : Float(100, 3, strides=[3, 1]) = aten::mul(%0, %2)
        return (%3))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  auto a = at::rand({100, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto b = at::rand({100, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto o = at::zeros({100, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto ref = a * (a * b);
  TensorExprKernel k(graph);
  std::vector<at::Tensor> inputs = {a, b};
  StmtPtr s = k.getCodeGenStmt();

  std::ostringstream oss;
  oss << *s;

  // Check the IR we produced
  const std::string& verification_pattern = R"IR(# CHECK: Ramp)IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  o = stack[0].toTensor();
  for (size_t i = 0; i < 100 * 3; i++) {
    TORCH_CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
  }
#endif
}

TEST_F(Kernel, Strided1dWithinBounds) {
  auto ir = R"IR(
    graph(%0 : Float(3, strides=[1], device=cpu),
          %1 : Float(3, strides=[2], device=cpu)):
        %2 : int = prim::Constant[value=1]()
        %3 : Float(3, strides=[1]) = aten::add(%0, %1, %2)
        return (%3))IR";
  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> vmap;
  parseIR(ir, graph.get(), vmap);
  TensorExprKernel k(graph);

  auto a = at::rand({3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto b = at::rand({6}, TensorOptions(kCPU).dtype(at::kFloat))
               .index({Slice(None, None, 2)});
  auto expect = a + b;

  std::vector<at::Tensor> inputs = {a, b};

  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);

  auto output = stack[0].toTensor();

  for (size_t i = 0; i < 3; ++i) {
    TORCH_CHECK_EQ(
        ((float*)output.data_ptr())[i], ((float*)expect.data_ptr())[i]);
  }
}

TEST_F(Kernel, InputAsOutput) {
  const auto graph_string = R"IR(
      graph(%x : Float(5, 3, strides=[3, 1], device=cpu),
            %y : Float(5, 3, strides=[1, 5], device=cpu)):
        return (%x, %y))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  auto x = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto y =
      at::rand({3, 5}, TensorOptions(kCPU).dtype(at::kFloat)).transpose(0, 1);
  TensorExprKernel k(graph);
  std::vector<at::Tensor> inputs = {x, y};

  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  CHECK(at::allclose(x, stack[0].toTensor()));
  CHECK(at::allclose(y, stack[1].toTensor()));
}

TEST_F(Kernel, ScalarOut) {
  auto ir = R"IR(
graph(%x : int, %y : int):
  %z : int = aten::mul(%x, %y)
  %r : int = aten::mul(%z, %x)
  return (%r, %z))IR";
  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> vmap;
  parseIR(ir, graph.get(), vmap);
  TensorExprKernel k(graph);

  auto stmt = k.getCodeGenStmt();
  std::ostringstream oss;
  oss << *stmt;

  // Verify the generated IR. We expect to see a scalar variable (Let) followed
  // by a store to a 0-dim buffer.
  const std::string& verification_pattern = R"IR(
# CHECK: int64_t
# CHECK-NEXT: [0ll] =
# CHECK-NEXT: int64_t
# CHECK-NEXT: [0ll] =
)IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  int64_t x = 2, y = 3, r = 0, z = 0;

  // Verify that TEK::runFast works correctly with scalar outputs
  std::vector<void*> inputs = {&x, &y};
  std::vector<void*> outputs = {&r, &z};
  k.runFast(inputs, outputs);
  TORCH_CHECK_EQ(z, x * y);
  TORCH_CHECK_EQ(r, z * x);

  // Verify that TEK::run works correctly with scalar outputs
  std::vector<IValue> stack = {x, y};
  k.run(stack);
  TORCH_CHECK_EQ(stack[0], x * y * x);
  TORCH_CHECK_EQ(stack[1], x * y);
}

TEST_F(Kernel, ScalarTensorOut) {
  auto ir = R"IR(
graph(%x : int,
      %xt : Long(3, strides=[1], device=cpu),
      %y : int,
      %yt : Long(3, strides=[1], device=cpu)):
  %z : int = aten::mul(%x, %y)
  %r : int = aten::mul(%z, %x)
  %zt : Long(3, strides=[1], device=cpu) = aten::mul(%xt, %y)
  %rt : Long(3, strides=[1], device=cpu) = aten::mul(%zt, %xt)
  return (%r, %rt, %z, %zt))IR";
  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> vmap;
  parseIR(ir, graph.get(), vmap);
  TensorExprKernel k(graph);
  int64_t x = 2, y = 3, r = 0, z = 0;
  auto xt = at::ones({3}, TensorOptions(kCPU).dtype(at::kLong)) * 2;
  auto yt = at::ones({3}, TensorOptions(kCPU).dtype(at::kLong)) * 3;
  auto zt = at::zeros({3}, TensorOptions(kCPU).dtype(at::kLong));
  auto rt = at::zeros({3}, TensorOptions(kCPU).dtype(at::kLong));

  // Verify that TEK::runFast works correctly with mixed scalar and tensor
  // inputs/utputs
  std::vector<void*> inputs = {&x, xt.data_ptr(), &y, yt.data_ptr()};
  std::vector<void*> outputs = {&r, rt.data_ptr(), &z, zt.data_ptr()};
  k.runFast(inputs, outputs);
  TORCH_CHECK_EQ(z, x * y);
  TORCH_CHECK_EQ(r, z * x);
  ASSERT_TRUE(at::equal(zt, xt * yt));
  ASSERT_TRUE(at::equal(rt, zt * xt));

  // Verify that TEK::run works correctly with mixed scalar and tensor
  // inputs/utputs
  std::vector<IValue> stack = {x, xt, y, yt};
  k.run(stack);
  TORCH_CHECK_EQ(stack[0], x * y * x);
  ASSERT_TRUE(at::equal(stack[1].toTensor(), xt * yt * xt));
  TORCH_CHECK_EQ(stack[2], x * y);
  ASSERT_TRUE(at::equal(stack[3].toTensor(), xt * yt));
}

TEST_F(Kernel, FuseLoopsWithVariableBounds) {
#ifdef TORCH_ENABLE_LLVM
  bool old_cat_wo_conditionals = getCatWoConditionals();
  getCatWoConditionals() = true;
  const auto graph_string = R"IR(
      graph(%a : Float(SS(-2), 3, SS(-3), requires_grad=0, device=cpu),
            %b : Float(SS(-2), 7, SS(-3), requires_grad=0, device=cpu),
            %c : Float(SS(-2), 9, SS(-3), requires_grad=0, device=cpu),
            %SS_2 : int,
            %SS_3 : int):
        %dim : int = prim::Constant[value=1]()
        %inputs : Tensor[] = prim::ListConstruct(%a, %b, %c)
        %r : Float(SS(-2), 19, SS(-3), requires_grad=0, device=cpu) = aten::cat(%inputs, %dim)               # new size: [5,19,2]
        return (%r))IR";
  std::shared_ptr<Graph> graph = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, graph.get());

  std::vector<int64_t> symbolic_shape_inputs = {-2, -3};

  std::vector<torch::jit::StrideInput> input_desc = {
      torch::jit::StrideInput::TENSOR_CONT};
  std::unordered_map<
      const torch::jit::Value*,
      std::vector<torch::jit::StrideInput>>
      symbolic_strides;
  symbolic_strides[graph->inputs().at(0)] = input_desc;
  symbolic_strides[graph->inputs().at(1)] = input_desc;
  symbolic_strides[graph->inputs().at(2)] = input_desc;
  symbolic_strides[graph->outputs().at(0)] = input_desc;

  TensorExprKernel kernel(
      graph, {}, symbolic_shape_inputs, false, symbolic_strides);

  std::ostringstream oss;
  oss << *kernel.getCodeGenStmt();
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int64_t i
# CHECK-NEXT: for (int64_t j
# CHECK-NEXT: for (int64_t k
# CHECK: for (int64_t j
# CHECK-NEXT: for (int64_t k
# CHECK: for (int64_t j
# CHECK-NEXT: for (int64_t k
# CHECK-NOT: for (int64_t i
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  auto run_kernel = [&](int dim1, int dim2) {
    auto a =
        at::rand({dim1, 3, dim2}, at::TensorOptions(kCPU).dtype(at::kFloat));
    auto b =
        at::rand({dim1, 7, dim2}, at::TensorOptions(kCPU).dtype(at::kFloat));
    auto c =
        at::rand({dim1, 9, dim2}, at::TensorOptions(kCPU).dtype(at::kFloat));

    auto ref = at::cat({a, b, c}, 1);

    std::vector<IValue> stack =
        fmap<IValue>(std::vector<at::Tensor>({a, b, c}));
    stack.emplace_back(dim1);
    stack.emplace_back(dim2);
    kernel.run(stack);

    auto o = stack[0].toTensor();
    ASSERT_TRUE(at::allclose(o, ref));
  };

  run_kernel(10, 20);
  getCatWoConditionals() = old_cat_wo_conditionals;
#endif
}

TEST_F(Kernel, FuseLoopsWithVariableConcatDim) {
#ifdef TORCH_ENABLE_LLVM
  bool old_cat_wo_conditionals = getCatWoConditionals();
  getCatWoConditionals() = true;
  const auto graph_string = R"IR(
      graph(%a : Float(SS(-2), SS(-4), SS(-3), requires_grad=0, device=cpu),
            %b : Float(SS(-2), SS(-4), SS(-3), requires_grad=0, device=cpu),
            %c : Float(SS(-2), SS(-4), SS(-3), requires_grad=0, device=cpu),
            %SS_2 : int,
            %SS_3 : int,
            %SS_4 : int,
            %SS_5 : int):
        %dim : int = prim::Constant[value=1]()
        %inputs : Tensor[] = prim::ListConstruct(%a, %b, %c)
        %r : Float(SS(-2), SS(-5), SS(-3), requires_grad=0, device=cpu) = aten::cat(%inputs, %dim)               # new size: [5,19,2]
        return (%r))IR";
  std::shared_ptr<Graph> graph = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, graph.get());

  std::vector<int64_t> symbolic_shape_inputs = {-2, -3, -4, -5};

  std::vector<torch::jit::StrideInput> input_desc = {
      torch::jit::StrideInput::TENSOR_CONT};
  std::unordered_map<
      const torch::jit::Value*,
      std::vector<torch::jit::StrideInput>>
      symbolic_strides;
  symbolic_strides[graph->inputs().at(0)] = input_desc;
  symbolic_strides[graph->inputs().at(1)] = input_desc;
  symbolic_strides[graph->inputs().at(2)] = input_desc;
  symbolic_strides[graph->outputs().at(0)] = input_desc;

  TensorExprKernel kernel(
      graph, {}, symbolic_shape_inputs, false, symbolic_strides);

  std::ostringstream oss;
  oss << *kernel.getCodeGenStmt();
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int64_t i
# CHECK-NEXT: for (int64_t j
# CHECK-NEXT: for (int64_t k
# CHECK: for (int64_t j
# CHECK-NEXT: for (int64_t k
# CHECK: for (int64_t j
# CHECK-NEXT: for (int64_t k
# CHECK-NOT: for (int64_t i
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  auto run_kernel = [&](int dim1, int dim2, int dim3) {
    auto a =
        at::rand({dim1, dim3, dim2}, at::TensorOptions(kCPU).dtype(at::kFloat));
    auto b =
        at::rand({dim1, dim3, dim2}, at::TensorOptions(kCPU).dtype(at::kFloat));
    auto c =
        at::rand({dim1, dim3, dim2}, at::TensorOptions(kCPU).dtype(at::kFloat));

    auto ref = at::cat({a, b, c}, 1);

    std::vector<IValue> stack =
        fmap<IValue>(std::vector<at::Tensor>({a, b, c}));
    stack.emplace_back(dim1);
    stack.emplace_back(dim2);
    stack.emplace_back(dim3);
    stack.emplace_back(3 * dim3);
    kernel.run(stack);

    auto o = stack[0].toTensor();
    ASSERT_TRUE(at::allclose(o, ref));
  };

  run_kernel(10, 20, 15);
  getCatWoConditionals() = old_cat_wo_conditionals;
#endif
}

TEST_F(Kernel, DoNotFuseLoopsWithMismatchingVariableDims) {
#ifdef TORCH_ENABLE_LLVM
  bool old_cat_wo_conditionals = getCatWoConditionals();
  getCatWoConditionals() = true;
  const auto graph_string = R"IR(
      graph(%a : Float(SS(-2), SS(-4), SS(-3), requires_grad=0, device=cpu),
            %b : Float(SS(-2), SS(-5), SS(-3), requires_grad=0, device=cpu),
            %SS_2 : int,
            %SS_3 : int,
            %SS_4 : int,
            %SS_5 : int,
            %SS_6 : int):
        %dim : int = prim::Constant[value=1]()
        %inputs : Tensor[] = prim::ListConstruct(%a, %b)
        %r : Float(SS(-2), SS(-6), SS(-3), requires_grad=0, device=cpu) = aten::cat(%inputs, %dim)               # new size: [5,19,2]
        return (%r))IR";
  std::shared_ptr<Graph> graph = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, graph.get());

  std::vector<int64_t> symbolic_shape_inputs = {-2, -3, -4, -5, -6};

  std::vector<torch::jit::StrideInput> input_desc = {
      torch::jit::StrideInput::TENSOR_CONT};
  std::unordered_map<
      const torch::jit::Value*,
      std::vector<torch::jit::StrideInput>>
      symbolic_strides;
  symbolic_strides[graph->inputs().at(0)] = input_desc;
  symbolic_strides[graph->inputs().at(1)] = input_desc;
  symbolic_strides[graph->outputs().at(0)] = input_desc;

  TensorExprKernel kernel(
      graph, {}, symbolic_shape_inputs, false, symbolic_strides);

  std::ostringstream oss;
  oss << *kernel.getCodeGenStmt();
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int64_t i
# CHECK-NEXT: for (int64_t j
# CHECK-NEXT: for (int64_t k
# CHECK: for (int64_t j
# CHECK-NEXT: for (int64_t k
# CHECK-NOT: for (int64_t j
# CHECK-NOT: for (int64_t i
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  auto run_kernel = [&](int dim2, int dim3, int dim4, int dim5) {
    auto a =
        at::rand({dim2, dim4, dim3}, at::TensorOptions(kCPU).dtype(at::kFloat));
    auto b =
        at::rand({dim2, dim5, dim3}, at::TensorOptions(kCPU).dtype(at::kFloat));

    auto ref = at::cat({a, b}, 1);

    std::vector<IValue> stack = fmap<IValue>(std::vector<at::Tensor>({a, b}));
    stack.emplace_back(dim2);
    stack.emplace_back(dim3);
    stack.emplace_back(dim4);
    stack.emplace_back(dim5);
    stack.emplace_back(dim4 + dim5);
    kernel.run(stack);

    auto o = stack[0].toTensor();
    ASSERT_TRUE(at::allclose(o, ref));
  };

  run_kernel(10, 20, 15, 8);
  getCatWoConditionals() = old_cat_wo_conditionals;
#endif
}

} // namespace jit
} // namespace torch
