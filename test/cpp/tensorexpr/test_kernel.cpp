#include <test/cpp/tensorexpr/test_base.h>
#include <torch/csrc/jit/frontend/code_template.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/tensorexpr/buffer.h>
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

void testKernel_1() {
  KernelScope kernel_scope;

  const auto graph_string = R"IR(
      graph(%0 : Float(5:3,3:1, device=cpu),
            %1 : Float(5:3,3:1, device=cpu)):
        %2 : Float(5:3,3:1) = aten::mul(%0, %1)
        %3 : Float(5:3,3:1) = aten::mul(%0, %2)
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

void testKernel_2() {
  KernelScope kernel_scope;

  const auto graph_string = R"IR(
      graph(%0 : Float(5:3,3:1, device=cpu),
            %1 : Float(5:1,3:5, device=cpu)):
        %2 : Float(5:3,3:1) = aten::mul(%0, %1)
        %3 : Float(5:3,3:1) = aten::mul(%0, %2)
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

void testKernel_3() {
  KernelScope kernel_scope;

  const auto graph_string = R"IR(
      graph(%0 : Float(5:3,3:1, device=cpu),
            %1 : Float(5:12,3:2, device=cpu)):
        %2 : Float(5:3,3:1) = aten::mul(%0, %1)
        %3 : Float(5:3,3:1) = aten::mul(%0, %2)
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

void testKernel_4() {
  // Test TensorExpr shape inference capabilities: it should only require shapes
  // for the inputs
  {
    KernelScope kernel_scope;

    const auto graph_string = R"IR(
      graph(%0 : Float(5:3,  3:1, device=cpu),
            %1 : Float(5:12, 3:2, device=cpu)):
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
      graph(%0 : Float(8:8, 8:1, device=cpu),
            %1 : Float(8:8, 8:1, device=cpu)):
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
      graph(%a : Float(4:2, 2:1, device=cpu),
            %b : Float(4:6, 3:2, 2:1, device=cpu),
            %c : Float(3:4, 2:2, 2:1, device=cpu)):
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
      graph(%a : Float(5:6, 3:2, 2:1, device=cpu),
            %b : Float(5:14, 7:2, 2:1, device=cpu),
            %c : Float(5:18, 9:2, 2:1, device=cpu)):
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
}

namespace {

std::string dtypeConstant(ScalarType scalar_type) {
  if (scalar_type == ScalarType::None) {
    return "None = prim::Constant()";
  } else {
    TemplateEnv env_dtype;
    env_dtype.d("scalar_type", static_cast<int>(scalar_type));
    return format("int = prim::Constant[value=${scalar_type}]()", env_dtype);
  }
}

at::Tensor iotaTensor(IntArrayRef sizes, const at::TensorOptions& options) {
  int64_t numel = std::accumulate(
      sizes.begin(), sizes.end(), 1, std::multiplies<int64_t>());
  std::vector<float> values(numel);
  std::iota(values.begin(), values.end(), 0);
  auto a = at::tensor(values, options);
  return a.reshape(sizes);
}

} // namespace

void testKernelSumAllAxes() {
  // Test lowering of sum on all axes.
  const auto graph_template = R"IR(
      graph(%0 : Float(5:3,3:1, device=cpu)):
        %1 : ${dtype}
        %2 : Tensor = aten::sum(%0, %1)
        return (%2))IR";
  auto a = iotaTensor({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));

  for (auto scalar_type : {ScalarType::None, ScalarType::Double}) {
    KernelScope kernel_scope;
    TemplateEnv env;
    env.s("dtype", dtypeConstant(scalar_type));
    const auto graph_string = format(graph_template, env);

    auto graph = std::make_shared<Graph>();
    parseIR(graph_string, &*graph);

    auto o = at::empty({}, TensorOptions(kCPU));
    c10::optional<c10::ScalarType> dtype;
    if (scalar_type != ScalarType::None) {
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

void testKernelSumOneAxis() {
  // Test lowering of sum on one axis.
  const auto graph_template = R"IR(
      graph(%0 : Float(5:3,3:1, device=cpu)):
        %1 : int[] = prim::Constant[value=[${dim}]]()
        %2 : bool = prim::Constant[value=${keepdim}]()
        %3 : ${dtype}
        %4 : Tensor = aten::sum(%0, %1, %2, %3)
        return (%4))IR";
  auto a = iotaTensor({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));

  for (int dim = -a.dim(); dim < a.dim(); ++dim) {
    for (bool keepdim : {false, true}) {
      for (auto scalar_type : {ScalarType::None, ScalarType::Double}) {
        KernelScope kernel_scope;
        TemplateEnv env;
        env.d("dim", dim);
        env.d("keepdim", keepdim);
        env.s("dtype", dtypeConstant(scalar_type));
        const auto graph_string = format(graph_template, env);

        auto graph = std::make_shared<Graph>();
        parseIR(graph_string, &*graph);

        auto o = at::empty({}, TensorOptions(kCPU));
        c10::optional<c10::ScalarType> dtype;
        if (scalar_type != ScalarType::None) {
          dtype = static_cast<c10::ScalarType>(scalar_type);
        }
        auto ref = a.sum({dim}, /*keepdim=*/keepdim, /*dtype=*/dtype);
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
# CHECK: input1)IR";
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

void testKernelSumMultipleAxes() {
  // Test lowering of sum on multiple axes.
  const auto graph_template = R"IR(
      graph(%0 : Float(2:18,3:6,2:3,3:1, device=cpu)):
        %1 : int = prim::Constant[value=${dim1}]()
        %2 : int = prim::Constant[value=${dim2}]()
        %3 : int[] = prim::ListConstruct(%1, %2)
        %4 : bool = prim::Constant[value=${keepdim}]()
        %5 : ${dtype}
        %6 : Tensor = aten::sum(%0, %3, %4, %5)
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
        env.s("dtype", dtypeConstant(ScalarType::None));
        const auto graph_string = format(graph_template, env);

        auto graph = std::make_shared<Graph>();
        parseIR(graph_string, &*graph);

        auto o = at::empty({}, TensorOptions(kCPU));
        auto ref = a.sum(IntArrayRef{dim1, dim2}, /*keepdim=*/keepdim);
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
# CHECK: input1)IR";
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

} // namespace jit
} // namespace torch
