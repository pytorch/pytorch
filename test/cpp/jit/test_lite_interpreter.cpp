#include <gtest/gtest.h>

#include <c10/core/TensorOptions.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/resolver.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/custom_class.h>
#include <torch/torch.h>

#include <unordered_set>

#define ASSERT_THROWS_WITH(statement, substring)                         \
  try {                                                                  \
    (void)statement;                                                     \
    ASSERT_TRUE(false);                                                  \
  } catch (const std::exception& e) {                                    \
    ASSERT_NE(std::string(e.what()).find(substring), std::string::npos); \
  }

// Tests go in torch::jit
namespace torch {
namespace jit {

TEST(LiteInterpreterTest, UpsampleNearest2d) {
  Module m("m");
  m.define(R"(
    def forward(self, input: Tensor, scale:float):
      return torch.upsample_nearest2d(input, [1, 1], float(scale), float(scale))
  )");

  std::vector<IValue> inputs;
  inputs.emplace_back(torch::rand({1, 3, 128, 128}));
  inputs.emplace_back(at::Scalar(2.0));
  auto ref = m.forward(inputs);

  std::stringstream ss;
  m._save_for_mobile(ss);
  mobile::Module bc = _load_for_mobile(ss);
  IValue res;
  res = bc.forward(inputs);

  auto resd = res.toTensor();
  auto refd = ref.toTensor();
  ASSERT_TRUE(resd.equal(refd));
}

TEST(LiteInterpreterTest, CheckAttrAccess) {
  Module m("m");
  m.register_attribute("mobile_optimized", BoolType::get(), true);

  std::stringstream ss;
  m._save_for_mobile(ss);
  mobile::Module bc = _load_for_mobile(ss);
  bool mobile_optimized = bc.attr("mobile_optimized", false).toBool();

  AT_ASSERT(mobile_optimized);
  m.setattr("mobile_optimized", false);
  ss = std::stringstream();
  m._save_for_mobile(ss);
  bc = _load_for_mobile(ss);
  mobile_optimized = bc.attr("mobile_optimized", false).toBool();

  AT_ASSERT(!mobile_optimized);
}

TEST(LiteInterpreterTest, MethodInvocation) { // NOLINT (use =delete in gtest)
  const std::vector<std::string> test_programs{
      // test invoking a method with default parameter
      R"(
      def test_func(self, x, b : int = 4):
        return self.foo + x + b
      )",
      // inner method call with default parameter (gets inlined)
      R"(
      def add_with_default_arg(self, x, b : int = 4):
        return self.foo + x + b
      def test_func(self, x):
        return self.add_with_default_arg(x)  # invoke method w/ default arg
      )",
      // simple method call
      R"(
      def test_func(self, x):
        b = 4
        return self.foo + x + b
      )",
  };
  for (const auto& test_program : test_programs) {
    Module m("m");
    m.register_parameter("foo", torch::ones({}), false);
    m.define(test_program);

    const int fortyTwo = 42; // (keep linter happy)
    auto minput = fortyTwo * torch::ones({});
    auto ref = m.run_method("test_func", minput);

    std::stringstream ss;
    m._save_for_mobile(ss);
    mobile::Module bc = _load_for_mobile(ss);
    const auto& test_func = bc.get_method("test_func");
    IValue res;
    for (int i = 0; i < 3; ++i) {
      res = test_func({minput});
    }

    auto resd = res.toTensor().item<float>();
    auto refd = ref.toTensor().item<float>();
    AT_ASSERT(resd == refd);
  }
}

TEST(LiteInterpreterTest, Conv) {
  auto s = std::getenv("PYTORCH_TEST_WITH_TSAN");
  if (s && strcmp(s, "1") == 0)
    return;

  std::vector<torch::jit::IValue> inputs;

  Module m("m");
  m.register_parameter("weight", torch::ones({20, 1, 5, 5}), false);
  m.register_parameter("bias", torch::ones({20}), false);
  m.define(R"(
    def forward(self, input):
      return torch._convolution(input, self.weight, self.bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1, False, False, True, True)
  )");

  inputs.push_back(torch::ones({1, 1, 28, 28}));

  auto outputref = m.forward(inputs).toTensor();

  std::stringstream ss;
  m._save_for_mobile(ss);
  mobile::Module bc = _load_for_mobile(ss);
  IValue res;
  for (int i = 0; i < 3; ++i) {
    res = bc.get_method("forward")(inputs);
  }
  auto output = res.toTensor();
  AT_ASSERT(outputref.dim() == output.dim());
  AT_ASSERT(
      outputref[0][0][0][0].item<int>() == output[0][0][0][0].item<int>());
}

TEST(LiteInterpreterTest, Inline) {
  Module m("m");
  m.define(R"JIT(
  def foo1(self, x):
      return x + 1

  def foo2(self, x):
      return self.foo1(x) + 2

  def foo3(self, x):
      return self.foo2(x) + 3
  )JIT");
  std::stringstream ss;
  m._save_for_mobile(ss);
  mobile::Module bc = _load_for_mobile(ss);
  std::vector<torch::jit::IValue> inputs({torch::ones({})});
  auto output = bc.get_method("foo3")(inputs);
  AT_ASSERT(output.toTensor().item<float>() == 7.0);
}

TEST(LiteInterpreterTest, Tuple) {
  Module m("m");
  m.define(R"JIT(
  def foo(self, x):
      return (1, 2, x + 3)

  def forward(self, x):
      tuple = self.foo(x)
      return tuple
  )JIT");
  std::stringstream ss;
  m._save_for_mobile(ss);
  mobile::Module bc = _load_for_mobile(ss);
  std::vector<torch::jit::IValue> inputs({torch::ones({})});
  auto output = bc.get_method("forward")(inputs);
  AT_ASSERT(output.toTuple()->elements()[1].toInt() == 2);
}

TEST(LiteInterpreterTest, Dict) {
  Module m("m");
  m.define(R"JIT(
  def foo(self, x):
      return {"result": x + 1}

  def forward(self, x):
      d = self.foo(x)
      return d
  )JIT");
  std::stringstream ss;
  m._save_for_mobile(ss);
  mobile::Module bc = _load_for_mobile(ss);
  std::vector<torch::jit::IValue> inputs({torch::ones({})});
  auto output = bc.get_method("forward")(inputs);
  AT_ASSERT(output.toGenericDict().at("result").toTensor().item().toInt() == 2);
}

TEST(LiteInterpreterTest, PrimOverload) {
  /*
  // temporarily disabled
  script::Module m("m");
  m.define(R"JIT(
  def forward(self, x):
      result = [1, 2]
      result.append(3)
      return result
  )JIT");
  std::stringstream ss;
  m._save_for_mobile(ss);
  mobile::Module bc = _load_for_mobile(ss);
  std::vector<torch::jit::IValue> inputs({torch::ones({})});
  auto output = bc.get_method("forward")(inputs);
  AT_ASSERT(output.toIntList()[2] == 3);
  */
}

TEST(LiteInterpreterTest, Prim) {
  Module m("m");
  m.define(R"JIT(
        def forward(self, x):
            return int(x)
  )JIT");

  std::vector<IValue> inputs;
  auto minput = 3.5 * torch::ones({});
  inputs.emplace_back(minput);
  auto ref = m.run_method("forward", minput);

  std::stringstream ss;
  m._save_for_mobile(ss);
  mobile::Module bc = _load_for_mobile(ss);
  IValue res;
  for (int i = 0; i < 3; ++i) {
    auto bcinputs = inputs;
    res = bc.get_method("forward")(bcinputs);
  }

  auto resi = res.toInt();
  auto refi = ref.toInt();
  AT_ASSERT(resi == refi);
}

TEST(LiteInterpreterTest, PrimScalar) {
  Module m("m");
  m.define(R"JIT(
        def forward(self, x):
            return int(x.item())
  )JIT");

  std::vector<IValue> inputs;
  auto minput = 3.5 * torch::ones({});
  inputs.emplace_back(minput);
  auto ref = m.run_method("forward", minput);

  std::stringstream ss;
  m._save_for_mobile(ss);
  mobile::Module bc = _load_for_mobile(ss);
  IValue res;
  for (int i = 0; i < 3; ++i) {
    auto bcinputs = inputs;
    res = bc.get_method("forward")(bcinputs);
  }

  auto resi = res.toInt();
  auto refi = ref.toInt();
  AT_ASSERT(resi == refi);
}

TEST(LiteInterpreterTest, LoadOrigJit) {
  Module m("m");
  m.register_parameter("foo", torch::ones({}), false);
  m.define(R"(
    def forward(self, x):
      b = 4
      return self.foo + x + b
  )");
  std::stringstream ss;
  m.save(ss);
  ASSERT_THROWS_WITH(_load_for_mobile(ss), "file not found");
}

TEST(LiteInterpreterTest, WrongMethodName) {
  Module m("m");
  m.register_parameter("foo", torch::ones({}), false);
  m.define(R"(
    def add(self, x):
      b = 4
      return self.foo + x + b
  )");
  std::stringstream ss;
  m._save_for_mobile(ss);
  mobile::Module bc = _load_for_mobile(ss);
  std::vector<IValue> inputs;
  auto minput = 5 * torch::ones({});
  inputs.emplace_back(minput);
  ASSERT_THROWS_WITH(bc.get_method("forward")(inputs), "is not defined");
}

TEST(LiteInterpreterTest, SetState) {
  Module m("m");
  m.register_parameter("foo", torch::ones({}), false);
  m.define(R"(
    def __getstate__(self):
      return self.foo + self.foo
    def __setstate__(self, a):
      self.foo = a
    def forward(self, x):
      b = 4
      return self.foo + x + b
  )");

  std::vector<IValue> inputs;
  auto minput = 5 * torch::ones({});
  inputs.emplace_back(minput);

  std::stringstream ms;
  m.save(ms);
  auto loaded_m = load(ms);
  auto ref = loaded_m.run_method("forward", minput);

  std::stringstream ss;
  m._save_for_mobile(ss);
  mobile::Module bc = _load_for_mobile(ss);
  IValue res;
  for (int i = 0; i < 3; ++i) {
    auto bcinputs = inputs;
    res = bc.get_method("forward")(bcinputs);
  }

  auto resd = res.toTensor().item<float>();
  auto refd = ref.toTensor().item<float>();
  AT_ASSERT(resd == refd);
}

class TorchBindLiteInterpreterTestStruct
    : public torch::jit::CustomClassHolder {
 public:
  std::string get(at::Tensor t) {
    std::stringstream ss;
    ss << "Hello! Your tensor has ";
    ss << t.numel();
    ss << " elements!";
    return ss.str();
  }
};

namespace {
struct ClassNamespaceValue : public SugaredValue {
  explicit ClassNamespaceValue(c10::QualifiedName name)
      : basename_(std::move(name)) {}

  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      Function& m,
      const std::string& name) override {
    const auto fullName = c10::QualifiedName(basename_, name);

    // Check to see if it is a custom class.
    if (auto custom_class = getCustomClass(fullName.qualifiedName())) {
      return std::make_shared<ClassValue>(custom_class);
    }

    // If it's not a custom class, assume it's another namespace
    return std::make_shared<ClassNamespaceValue>(std::move(fullName));
  }

  std::string kind() const override {
    return "Class Namespace";
  }

 private:
  c10::QualifiedName basename_;
};

struct TestModuleResolver : public Resolver {
  std::shared_ptr<SugaredValue> resolveValue(
      const std::string& name,
      Function& m,
      const SourceRange& loc) override {
    if (name == "torch") {
      return std::make_shared<BuiltinModule>("aten");
    } else if (name == "__torch__") {
      return std::make_shared<ClassNamespaceValue>(c10::QualifiedName(name));
    }

    return nullptr;
  }

  TypePtr resolveType(const std::string& name, const SourceRange& loc)
      override {
    return nullptr;
  }
};
} // namespace

TEST(LiteInterpreterTest, BuiltinClass) {
  script::Module m("m");

  auto cls = getCustomClass(
      "__torch__.torch.classes._TorchScriptTesting._LiteInterpreterTest");
  TORCH_INTERNAL_ASSERT(cls);
  c10::intrusive_ptr<torch::CustomClassHolder> obj_holder;
  m.register_attribute("my_obj", cls, IValue::make_capsule(obj_holder));

  m.register_parameter("foo", torch::ones({}), false);
  m.define(
      R"(
    def __getstate__(self):
      return 1
    def __setstate__(self, a):
      self.my_obj = __torch__.torch.classes._TorchScriptTesting._LiteInterpreterTest()

    def forward(self, x) -> str:
      return self.my_obj.get(x)
  )",
      std::make_shared<TestModuleResolver>());

  std::stringstream ss;
  m._save_for_mobile(ss);
  mobile::Module bc = _load_for_mobile(ss);
  auto res =
      bc.get_method("forward")(std::vector<IValue>{torch::zeros({3, 4})});
  const auto& str = res.toStringRef();
  std::string expected = "Hello! Your tensor has 12 elements!";
  AT_ASSERT(str == expected);
}

TEST(LiteInterpreterTest, BuiltinFunction) {
  script::Module m("m");
  auto custom_class_obj =
      make_custom_class<TorchBindLiteInterpreterTestStruct>();
  m.register_attribute("my_obj", custom_class_obj.type(), custom_class_obj);
  m.define(R"(
    def forward(self, x) -> str:
      return self.my_obj.get(x)
  )");

  std::stringstream ss;
  m._save_for_mobile(ss);
  mobile::Module bc = _load_for_mobile(ss);
  auto res =
      bc.get_method("forward")(std::vector<IValue>{torch::zeros({3, 4})});
  auto str = res.toStringRef();
  std::string expected = "Hello! Your tensor has 12 elements!";
  AT_ASSERT(str == expected);
}

TEST(LiteInterpreterTest, ModuleInfoBasic) {
  Module m("M");
  m.define(R"JIT(
    def forward(self, x):
      return 2 * x
  )JIT");

  std::stringstream ss;
  m._save_for_mobile(ss, {}, true);
  mobile::Module bc = _load_for_mobile(ss);

  std::unordered_set<std::string> module_debug_info_set;
  size_t pc = 0;
  while (true) {
    try {
      std::string module_info = bc.get_forward_method_debug_info(pc);
      if (!module_info.empty() && module_info != "<no module info>") {
        module_debug_info_set.insert(module_info);
      }
      ++pc;
    } catch (const std::exception& e) {
      break;
    }
  }

  std::unordered_set<std::string> expected_result({"top(M).forward"});
  AT_ASSERT(module_debug_info_set == expected_result);
}

TEST(LiteInterpreterTest, NotSaveModuleInfo) {
  Module m("M");
  m.define(R"JIT(
    def forward(self, x):
      return x + 5
  )JIT");

  std::stringstream ss;
  m._save_for_mobile(ss);
  mobile::Module bc = _load_for_mobile(ss);

  size_t pc = 0;
  while (true) {
    try {
      std::string module_info = bc.get_forward_method_debug_info(pc);
      AT_ASSERT(module_info.empty() || module_info == "<no module info>");
      ++pc;
    } catch (const std::exception& e) {
      break;
    }
  }
}

TEST(LiteInterpreterTest, OneSubmoduleModuleInfo) {
  Module a("A");
  a.define(R"JIT(
    def forward(self, x):
      return 2 * x + 5
  )JIT");
  Module b("B");
  b.register_module("A0", a);
  b.define(R"JIT(
    def forward(self, x):
      return self.A0.forward(x) + 1
  )JIT");

  std::stringstream ss;
  b._save_for_mobile(ss, {}, true);
  mobile::Module bc = _load_for_mobile(ss);

  std::unordered_set<std::string> module_debug_info_set;
  size_t pc = 0;
  while (true) {
    try {
      std::string module_info = bc.get_forward_method_debug_info(pc);
      if (!module_info.empty() && module_info != "<no module info>") {
        module_debug_info_set.insert(module_info);
      }
      ++pc;
    } catch (const std::exception& e) {
      break;
    }
  }

  std::unordered_set<std::string> expected_result(
      {"top(B).forward", "top(B).A0(A).forward"});
  AT_ASSERT(module_debug_info_set == expected_result);
}

TEST(LiteInterpreterTest, TwoSubmodulesModuleInfo) {
  Module a("A");
  a.define(R"JIT(
    def forward(self, x):
      return x + 1
  )JIT");
  Module b("B");
  b.define(R"JIT(
    def forward(self, x):
      return x + 2
  )JIT");
  Module c("C");
  c.register_module("A0", a);
  c.register_module("B0", b);
  c.define(R"JIT(
    def forward(self, x):
      return self.A0.forward(x) + self.B0.forward(x)
  )JIT");

  std::stringstream ss;
  c._save_for_mobile(ss, {}, true);
  mobile::Module bc = _load_for_mobile(ss);

  std::unordered_set<std::string> module_debug_info_set;
  size_t pc = 0;
  while (true) {
    try {
      std::string module_info = bc.get_forward_method_debug_info(pc);
      if (!module_info.empty() && module_info != "<no module info>") {
        module_debug_info_set.insert(module_info);
      }
      ++pc;
    } catch (const std::exception& e) {
      break;
    }
  }

  std::unordered_set<std::string> expected_result(
      {"top(C).forward", "top(C).A0(A).forward", "top(C).B0(B).forward"});
  AT_ASSERT(module_debug_info_set == expected_result);
}

TEST(LiteInterpreterTest, SequentialModuleInfo) {
  Module a("A");
  a.define(R"JIT(
    def forward(self, x):
      return x + 1
  )JIT");
  Module b("B");
  b.define(R"JIT(
    def forward(self, x):
      return x + 2
  )JIT");
  Module c("C");
  c.register_module("A0", a);
  c.register_module("B0", b);
  c.define(R"JIT(
    def forward(self, x):
      return self.A0.forward(self.B0.forward(x))
  )JIT");

  std::stringstream ss;
  c._save_for_mobile(ss, {}, true);
  mobile::Module bc = _load_for_mobile(ss);

  std::unordered_set<std::string> module_debug_info_set;
  size_t pc = 0;
  while (true) {
    try {
      std::string module_info = bc.get_forward_method_debug_info(pc);
      if (!module_info.empty() && module_info != "<no module info>") {
        module_debug_info_set.insert(module_info);
      }
      ++pc;
    } catch (const std::exception& e) {
      break;
    }
  }

  // class A(nn.Module):
  //   def __init__(self):
  //     super(A, self).__init__()

  //   def forward(self, x):
  //     return x + 1

  // class B(nn.Module):
  //   def __init__(self):
  //     super(B, self).__init__()

  //   def forward(self, x):
  //     return x + 2

  // class C(nn.Module):
  //   def __init__(self):
  //     super(C, self).__init__()
  //     self.A0 = A()
  //     self.B0 = B()

  //   def forward(self, x):
  //     return self.A0.forward(self.B0.forward(x))

  std::unordered_set<std::string> expected_result(
      {"top(C).A0(A).forward", "top(C).B0(B).forward"});
  AT_ASSERT(module_debug_info_set == expected_result);
}

TEST(LiteInterpreterTest, HierarchyModuleInfo) {
  Module a("A");
  a.define(R"JIT(
    def forward(self, x):
      return x + 1
  )JIT");
  Module b("B");
  b.register_module("A0", a);
  b.define(R"JIT(
    def forward(self, x):
      return self.A0.forward(x) + 1
  )JIT");
  Module c("C");
  c.register_module("B0", b);
  c.define(R"JIT(
    def forward(self, x):
      return self.B0.forward(x) + 1
  )JIT");

  std::stringstream ss;
  c._save_for_mobile(ss, {}, true);
  mobile::Module bc = _load_for_mobile(ss);

  std::unordered_set<std::string> module_debug_info_set;
  size_t pc = 0;
  while (true) {
    try {
      std::string module_info = bc.get_forward_method_debug_info(pc);
      if (!module_info.empty() && module_info != "<no module info>") {
        module_debug_info_set.insert(module_info);
      }
      ++pc;
    } catch (const std::exception& e) {
      break;
    }
  }

  // There are 3 module information strings here.
  // "top(C).forward": for the add operator in top.
  // "top(C).B0(B).forward": for the add operator in B0.
  // "top(C).B0(B).forward.A0(A).forward": for the add operator in A0.
  std::unordered_set<std::string> expected_result(
      {"top(C).forward",
       "top(C).B0(B).forward",
       "top(C).B0(B).forward.A0(A).forward"});
  AT_ASSERT(module_debug_info_set == expected_result);
}

TEST(LiteInterpreterTest, DuplicatedClassTypeModuleInfo) {
  Module a("A");
  a.define(R"JIT(
    def forward(self, x):
      return x + 5
  )JIT");
  Module b("B");
  b.register_module("A0", a);
  b.register_module("A1", a);
  b.define(R"JIT(
    def forward(self, x):
      return self.A0.forward(x) + self.A1.forward(x)
  )JIT");

  std::stringstream ss;
  b._save_for_mobile(ss, {}, true);
  mobile::Module bc = _load_for_mobile(ss);

  std::unordered_set<std::string> module_debug_info_set;
  size_t pc = 0;
  while (true) {
    try {
      std::string module_info = bc.get_forward_method_debug_info(pc);
      if (!module_info.empty() && module_info != "<no module info>") {
        module_debug_info_set.insert(module_info);
      }
      ++pc;
    } catch (const std::exception& e) {
      break;
    }
  }

  // class A(nn.Module):
  //   def __init__(self):
  //     super(A, self).__init__()

  //   def forward(self, x):
  //     return x + 5

  // class B(nn.Module):
  //   def __init__(self):
  //     super(B, self).__init__()
  //     self.A0 = A()
  //     self.A1 = A()

  //   def forward(self, x):
  //     return self.A0.forward(x) + self.A1.forward(x)

  // There are 3 module information strings here.
  // "top(B).forward": for the add operator in top.
  // "top(B).A0(A).forward": for the add operator in A0.
  // "top(B).A1(A).forward": for the add operator in A1.

  std::unordered_set<std::string> expected_result(
      {"top(B).forward", "top(B).A0(A).forward", "top(B).A1(A).forward"});
  AT_ASSERT(module_debug_info_set == expected_result);
}

TEST(LiteInterpreterTest, Eval) {
  std::vector<torch::jit::IValue> inputs;

  Module m("m");
  m.define(R"(
    def __init__(self, x):
      self.training = True

    def forward(self, input):
      return torch.dropout(input, 1.0, self.training)
  )");

  inputs.push_back(torch::ones({1, 1, 28, 28}));
  m.eval();
  auto outputref = m.forward(inputs).toTensor();

  // save m in training mode to make sure that mobile eval() will correctly
  // change back to eval mode
  m.train();
  std::stringstream ss;
  m._save_for_mobile(ss);
  mobile::Module bc = _load_for_mobile(ss);
  bc.eval();
  IValue res;
  for (int i = 0; i < 3; ++i) {
    res = bc.get_method("forward")(inputs);
  }
  auto output = res.toTensor();
  AT_ASSERT(outputref.dim() == output.dim());
  AT_ASSERT(
      outputref[0][0][0][0].item<int>() == output[0][0][0][0].item<int>());
}

TEST(LiteInterpreterTest, FindWrongMethodName) {
  Module m("m");
  m.register_parameter("foo", torch::ones({}), false);
  m.define(R"(
    def add(self, x):
      b = 4
      return self.foo + x + b
  )");
  std::stringstream ss;
  m._save_for_mobile(ss);
  mobile::Module bc = _load_for_mobile(ss);
  ASSERT_TRUE(bc.find_method("forward") == c10::nullopt);
}

TEST(LiteInterpreterTest, FindAndRunMethod) {
  Module m("m");
  m.register_parameter("foo", torch::ones({}), false);
  m.define(R"(
    def add_it(self, x):
      b = 4
      return self.foo + x + b
  )");

  std::vector<IValue> inputs;
  auto minput = 5 * torch::ones({});
  inputs.emplace_back(minput);
  auto ref = m.get_method("add_it")(inputs);

  std::stringstream ss;
  m._save_for_mobile(ss);
  mobile::Module bc = _load_for_mobile(ss);
  IValue res;
  for (int i = 0; i < 3; ++i) {
    auto bcinputs = inputs;
    auto method = bc.find_method("add_it");
    AT_ASSERT(method != c10::nullopt);
    res = (*method)(std::move(bcinputs));
  }

  auto resd = res.toTensor().item<float>();
  auto refd = ref.toTensor().item<float>();
  AT_ASSERT(resd == refd);
}

TEST(LiteInterpreterTest, RunMethodVariadic) {
  Module m("m");
  m.register_parameter("foo", torch::ones({}), false);
  m.define(R"(
    def add_three(self, x, y):
      return self.foo + x + y
  )");

  std::vector<IValue> inputs;
  auto inputx = 5 * torch::ones({});
  auto inputy = 4 * torch::ones({});
  auto ref = m.run_method("add_three", inputx, inputy);

  std::stringstream ss;
  m._save_for_mobile(ss);
  mobile::Module bc = _load_for_mobile(ss);
  IValue res = bc.run_method("add_three", inputx, inputy);

  auto resd = res.toTensor().item<float>();
  auto refd = ref.toTensor().item<float>();
  AT_ASSERT(resd == refd);
}

TEST(LiteInterpreterTest, ExtraFiles) {
  const auto script = R"JIT(
    def forward(self):
        x = torch.rand(5, 5)
        x = x.mm(x)
        return x
  )JIT";

  auto module =
      std::make_shared<Module>("Module", std::make_shared<CompilationUnit>());
  module->define(script);
  std::ostringstream oss;
  std::unordered_map<std::string, std::string> extra_files;
  extra_files["metadata.json"] = "abc";
  module->_save_for_mobile(oss, extra_files);

  std::istringstream iss(oss.str());
  caffe2::serialize::IStreamAdapter adapter{&iss};
  std::unordered_map<std::string, std::string> loaded_extra_files;
  loaded_extra_files["metadata.json"] = "";
  auto loaded_module =
      torch::jit::_load_for_mobile(iss, torch::kCPU, loaded_extra_files);
  ASSERT_EQ(loaded_extra_files["metadata.json"], "abc");
}

TEST(LiteInterpreterTest, OpNameExportFetchRootOperators) {
  torch::jit::Module m("m");
  m.register_parameter("weight", torch::ones({20, 1, 5, 5}), false);
  m.register_parameter("bias", torch::ones({20}), false);
  m.define(R"(
    def forward(self, input):
      x1 = torch.zeros(2, 2)
      x2 = torch.empty_like(torch.empty(2, 2))
      x3 = torch._convolution(input, self.weight, self.bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1, False, False, True, True)
      return (x1, x2, x3)
  )");
  m.eval();

  std::stringstream ss;
  m._save_for_mobile(ss);

  torch::jit::mobile::Module ptl_model = torch::jit::_load_for_mobile(ss);
  std::set<std::string> operator_names =
      torch::jit::mobile::_export_operator_list(ptl_model);
  std::set<std::string> expected_operator_names = {
      "aten::_convolution",
      "aten::empty.memory_format",
      "aten::empty_like",
      "aten::zeros",
  };
  EXPECT_EQ(operator_names, expected_operator_names)
      << "Expected the root operator lists to be the same";
}

namespace {
static auto reg =
    torch::class_<TorchBindLiteInterpreterTestStruct>(
        "_TorchScriptTesting",
        "_LiteInterpreterTest")
        .def(torch::init<>())
        .def("get", &TorchBindLiteInterpreterTestStruct::get)
        .def_pickle(
            // __getattr__
            [](const c10::intrusive_ptr<TorchBindLiteInterpreterTestStruct>&
                   self) -> int64_t { return 0; },
            // __setattr__
            [](int64_t state) {
              return c10::make_intrusive<TorchBindLiteInterpreterTestStruct>();
            });

} // namespace

} // namespace jit
} // namespace torch
