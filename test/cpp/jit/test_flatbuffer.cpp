#include <test/cpp/jit/test_utils.h>

#include <gtest/gtest.h>

#include <c10/core/TensorOptions.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/resolver.h>
#include <torch/csrc/jit/mobile/compatibility/backport.h>
#include <torch/csrc/jit/mobile/compatibility/backport_manager.h>
#include <torch/csrc/jit/mobile/compatibility/model_compatibility.h>
#include <torch/csrc/jit/mobile/compatibility/runtime_compatibility.h>
#include <torch/csrc/jit/mobile/flatbuffer_loader.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/interpreter.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/mobile/parse_bytecode.h>
#include <torch/csrc/jit/mobile/parse_operators.h>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/csrc/jit/serialization/export_bytecode.h>
#include <torch/csrc/jit/serialization/flatbuffer_serializer.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/custom_class.h>
#include <torch/torch.h>

#include <torch/csrc/jit/serialization/import_export_functions.h>
#include <unordered_set>
// Tests go in torch::jit
namespace torch {
namespace jit {

mobile::Module parse_mobile_module(void* data, size_t) {
  auto* flatbuffer_module = mobile::serialization::GetMutableModule(data);
  return initialize_mobile_module(flatbuffer_module);
}

TEST(FlatbufferTest, UpsampleNearest2d) {
  Module m("m");
  m.define(R"(
    def forward(self, input: Tensor, scale:float):
      return torch.upsample_nearest2d(input, [1, 1], float(scale), float(scale))
  )");

  std::vector<IValue> inputs;
  inputs.emplace_back(torch::rand({1, 3, 128, 128}));
  inputs.emplace_back(at::Scalar(2.0));
  auto ref = m.forward(inputs);

  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
  IValue res;
  res = bc.forward(inputs);

  auto resd = res.toTensor();
  auto refd = ref.toTensor();
  ASSERT_TRUE(resd.equal(refd));

  auto buff = save_mobile_module_to_bytes(bc);
  mobile::Module bc2 = parse_mobile_module(buff.data(), buff.size());
  auto res2 = bc2.forward(inputs);
  auto resd2 = res2.toTensor();
  ASSERT_TRUE(resd2.equal(refd));
}

TEST(FlatbufferTest, CheckAttrAccess) {
  Module m("m");
  m.register_attribute("mobile_optimized", BoolType::get(), true);

  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
  bool mobile_optimized = bc.attr("mobile_optimized", false).toBool();

  AT_ASSERT(mobile_optimized);
  m.setattr("mobile_optimized", false);
  bc = jitModuleToMobile(m, options);
  mobile_optimized = bc.attr("mobile_optimized", false).toBool();

  AT_ASSERT(!mobile_optimized);

  auto buff = save_mobile_module_to_bytes(bc);
  mobile::Module bc2 = parse_mobile_module(buff.data(), buff.size());
  auto mobile_optimized2 = bc2.attr("mobile_optimized", false).toBool();
  AT_ASSERT(!mobile_optimized2);
}

TEST(FlatbufferTest, MethodInvocation) { // NOLINT (use =delete in gtest)
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

    CompilationOptions options;
    mobile::Module bc = jitModuleToMobile(m, options);
    const auto& test_func = bc.get_method("test_func");
    IValue res;
    for (int i = 0; i < 3; ++i) {
      res = test_func({minput});
    }

    auto resd = res.toTensor().item<float>();
    auto refd = ref.toTensor().item<float>();
    AT_ASSERT(resd == refd);

    auto buff = save_mobile_module_to_bytes(bc);
    mobile::Module bc2 = parse_mobile_module(buff.data(), buff.size());
    const auto& test_func2 = bc2.get_method("test_func");
    IValue res2;
    for (int i = 0; i < 3; ++i) {
      res2 = test_func2({minput});
    }
    auto resd2 = res2.toTensor().item<float>();
    AT_ASSERT(resd2 == refd);
  }
}

TEST(FlatbufferTest, ExtraFiles) {
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
  extra_files["mobile_info.json"] = "{\"key\": 23}";

  std::unordered_map<std::string, std::string> loaded_extra_files;
#if defined ENABLE_FLATBUFFER
  std::stringstream ss;
  module->_save_for_mobile(ss, extra_files, true, /*use_flatbuffer=*/true);

  loaded_extra_files["metadata.json"] = "";
  auto mobile_module = _load_for_mobile(ss, c10::nullopt, loaded_extra_files);

  ASSERT_EQ(loaded_extra_files["metadata.json"], "abc");
  ASSERT_EQ(loaded_extra_files["mobile_info.json"], "{\"key\": 23}");

  // load it twice using the same stream
  auto mobile_module2 = _load_for_mobile(ss, c10::nullopt, loaded_extra_files);
#else
  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(*module, options);
  auto buff = save_mobile_module_to_bytes(bc, extra_files);

  loaded_extra_files["metadata.json"] = "";
  auto* flatbuffer_module =
      mobile::serialization::GetMutableModule(buff.data());

  parseExtraFiles(flatbuffer_module, loaded_extra_files);
#endif

  ASSERT_EQ(loaded_extra_files["metadata.json"], "abc");
  ASSERT_EQ(loaded_extra_files["mobile_info.json"], "{\"key\": 23}");
}

TEST(FlatbufferTest, Conv) {
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

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,modernize-use-emplace)
  inputs.push_back(torch::ones({1, 1, 28, 28}));

  auto outputref = m.forward(inputs).toTensor();

  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
  IValue res;
  for (int i = 0; i < 3; ++i) {
    res = bc.get_method("forward")(inputs);
  }
  auto output = res.toTensor();
  AT_ASSERT(outputref.dim() == output.dim());
  AT_ASSERT(
      outputref[0][0][0][0].item<int>() == output[0][0][0][0].item<int>());

  auto buff = save_mobile_module_to_bytes(bc);
  mobile::Module bc2 = parse_mobile_module(buff.data(), buff.size());
  for (int i = 0; i < 3; ++i) {
    res = bc2.get_method("forward")(inputs);
  }
  output = res.toTensor();
  AT_ASSERT(outputref.dim() == output.dim());
  AT_ASSERT(
      outputref[0][0][0][0].item<int>() == output[0][0][0][0].item<int>());
}

TEST(FlatbufferTest, Inline) {
  Module m("m");
  m.define(R"JIT(
  def foo1(self, x):
      return x + 1

  def foo2(self, x):
      return self.foo1(x) + 2

  def foo3(self, x):
      return self.foo2(x) + 3
  )JIT");
  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
  std::vector<torch::jit::IValue> inputs({torch::ones({})});
  auto output = bc.get_method("foo3")(inputs);
  AT_ASSERT(output.toTensor().item<float>() == 7.0);

  auto buff = save_mobile_module_to_bytes(bc);
  mobile::Module bc2 = parse_mobile_module(buff.data(), buff.size());
  std::vector<torch::jit::IValue> inputs2({torch::ones({})});
  output = bc2.get_method("foo3")(inputs2);
  AT_ASSERT(output.toTensor().item<float>() == 7.0);
}

TEST(FlatbufferTest, Tuple) {
  Module m("m");
  m.define(R"JIT(
  def foo(self, x):
      return (1, 2, x + 3)

  def forward(self, x):
      tuple = self.foo(x)
      return tuple
  )JIT");
  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
  std::vector<torch::jit::IValue> inputs({torch::ones({})});
  auto output = bc.get_method("forward")(inputs);
  AT_ASSERT(output.toTupleRef().elements()[1].toInt() == 2);

  auto buff = save_mobile_module_to_bytes(bc);
  mobile::Module bc2 = parse_mobile_module(buff.data(), buff.size());
  output = bc2.get_method("forward")(inputs);
  AT_ASSERT(output.toTuple()->elements()[1].toInt() == 2);
}

TEST(FlatbufferTest, Dict) {
  Module m("m");
  m.define(R"JIT(
  def foo(self, x):
      return {"result": x + 1}

  def forward(self, x):
      d = self.foo(x)
      return d
  )JIT");
  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
  std::vector<torch::jit::IValue> inputs({torch::ones({})});
  auto output = bc.get_method("forward")(inputs);
  AT_ASSERT(output.toGenericDict().at("result").toTensor().item().toInt() == 2);

  auto buff = save_mobile_module_to_bytes(bc);
  mobile::Module bc2 = parse_mobile_module(buff.data(), buff.size());
  output = bc2.get_method("forward")(inputs);
  AT_ASSERT(output.toGenericDict().at("result").toTensor().item().toInt() == 2);
}

TEST(FlatbufferTest, Prim) {
  Module m("m");
  m.define(R"JIT(
        def forward(self, x):
            return int(x)
  )JIT");

  std::vector<IValue> inputs;
  auto minput = 3.5 * torch::ones({});
  inputs.emplace_back(minput);
  auto ref = m.run_method("forward", minput);

  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
  IValue res;
  for (int i = 0; i < 3; ++i) {
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    auto bcinputs = inputs;
    res = bc.get_method("forward")(bcinputs);
  }

  auto resi = res.toInt();
  auto refi = ref.toInt();
  AT_ASSERT(resi == refi);

  auto buff = save_mobile_module_to_bytes(bc);
  mobile::Module bc2 = parse_mobile_module(buff.data(), buff.size());
  for (int i = 0; i < 3; ++i) {
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    auto bcinputs = inputs;
    res = bc2.get_method("forward")(bcinputs);
  }
  auto resi2 = res.toInt();
  AT_ASSERT(resi2 == refi);
}

TEST(FlatbufferTest, PrimScalar) {
  Module m("m");
  m.define(R"JIT(
        def forward(self, x):
            return int(x.item())
  )JIT");

  std::vector<IValue> inputs;
  auto minput = 3.5 * torch::ones({});
  inputs.emplace_back(minput);
  auto ref = m.run_method("forward", minput);

  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
  IValue res;
  for (int i = 0; i < 3; ++i) {
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    auto bcinputs = inputs;
    res = bc.get_method("forward")(bcinputs);
  }

  auto resi = res.toInt();
  auto refi = ref.toInt();
  AT_ASSERT(resi == refi);

  auto buff = save_mobile_module_to_bytes(bc);
  mobile::Module bc2 = parse_mobile_module(buff.data(), buff.size());
  for (int i = 0; i < 3; ++i) {
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    auto bcinputs = inputs;
    res = bc2.get_method("forward")(bcinputs);
  }
  auto resi2 = res.toInt();
  AT_ASSERT(resi2 == refi);
}

TEST(FlatbufferTest, WrongMethodName) {
  Module m("m");
  m.register_parameter("foo", torch::ones({}), false);
  m.define(R"(
    def add(self, x):
      b = 4
      return self.foo + x + b
  )");
  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
  std::vector<IValue> inputs;
  auto minput = 5 * torch::ones({});
  inputs.emplace_back(minput);
  ASSERT_THROWS_WITH_MESSAGE(
      bc.get_method("forward")(inputs), "is not defined");

  auto buff = save_mobile_module_to_bytes(bc);
  mobile::Module bc2 = parse_mobile_module(buff.data(), buff.size());
  ASSERT_THROWS_WITH_MESSAGE(
      bc2.get_method("forward")(inputs), "is not defined");
}

TEST(FlatbufferTest, SetState) {
  Module m("m");
  m.register_parameter("foo", torch::ones({}), false);
  m.define(R"(
    def __getstate__(self):
      return self.foo
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

  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
  IValue res;
  for (int i = 0; i < 3; ++i) {
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    auto bcinputs = inputs;
    res = bc.get_method("forward")(bcinputs);
  }

  auto resd = res.toTensor().item<float>();
  auto refd = ref.toTensor().item<float>();
  AT_ASSERT(resd == refd);

  auto buff = save_mobile_module_to_bytes(bc);
  mobile::Module bc2 = parse_mobile_module(buff.data(), buff.size());
  for (int i = 0; i < 3; ++i) {
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    auto bcinputs = inputs;
    res = bc2.get_method("forward")(bcinputs);
  }

  auto resd2 = res.toTensor().item<float>();
  AT_ASSERT(resd2 == refd);
}

class TorchBindFlatbufferTestStruct : public torch::jit::CustomClassHolder {
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
      GraphFunction& m,
      const std::string& name) override {
    const auto fullName = c10::QualifiedName(basename_, name);

    // Check to see if it is a custom class.
    if (auto custom_class = getCustomClass(fullName.qualifiedName())) {
      return std::make_shared<ClassValue>(custom_class);
    }

    // If it's not a custom class, assume it's another namespace
    // NOLINTNEXTLINE(performance-move-const-arg)
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
      GraphFunction& m,
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

TEST(FlatbufferTest, BuiltinClass) {
  script::Module m("m");

  auto cls = getCustomClass(
      "__torch__.torch.classes._TorchScriptTesting._FlatbufferTest");
  TORCH_INTERNAL_ASSERT(cls);
  c10::intrusive_ptr<torch::CustomClassHolder> obj_holder;
  m.register_attribute("my_obj", cls, IValue::make_capsule(obj_holder));

  m.register_parameter("foo", torch::ones({}), false);
  m.define(
      R"(
    def __getstate__(self):
      return 1
    def __setstate__(self, a):
      self.my_obj = __torch__.torch.classes._TorchScriptTesting._FlatbufferTest()

    def forward(self, x) -> str:
      return self.my_obj.get(x)
  )",
      std::make_shared<TestModuleResolver>());

  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
  auto buff = save_mobile_module_to_bytes(bc);
  mobile::Module bc2 = parse_mobile_module(buff.data(), buff.size());
  std::string expected = "Hello! Your tensor has 12 elements!";
  auto res =
      bc2.get_method("forward")(std::vector<IValue>{torch::zeros({3, 4})});
  const auto& str2 = res.toStringRef();
  AT_ASSERT(str2 == expected);
}

TEST(FlatbufferTest, BuiltinFunction) {
  script::Module m("m");
  auto custom_class_obj = make_custom_class<TorchBindFlatbufferTestStruct>();
  m.register_attribute("my_obj", custom_class_obj.type(), custom_class_obj);
  m.define(R"(
    def forward(self, x) -> str:
      return self.my_obj.get(x)
  )");

  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
  auto res =
      bc.get_method("forward")(std::vector<IValue>{torch::zeros({3, 4})});
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  auto str = res.toStringRef();
  std::string expected = "Hello! Your tensor has 12 elements!";
  AT_ASSERT(str == expected);

  auto buff = save_mobile_module_to_bytes(bc);
  mobile::Module bc2 = parse_mobile_module(buff.data(), buff.size());
  res = bc2.get_method("forward")(std::vector<IValue>{torch::zeros({3, 4})});
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  str = res.toStringRef();
  AT_ASSERT(str == expected);
}

TEST(FlatbufferTest, Eval) {
  std::vector<torch::jit::IValue> inputs;

  Module m("m");
  m.define(R"(
    def __init__(self, x):
      self.training = True

    def forward(self, input):
      return torch.dropout(input, 1.0, self.training)
  )");

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,modernize-use-emplace)
  inputs.push_back(torch::ones({1, 1, 28, 28}));
  m.eval();
  auto outputref = m.forward(inputs).toTensor();

  // save m in training mode to make sure that mobile eval() will correctly
  // change back to eval mode
  m.train();
  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
  bc.eval();
  IValue res;
  for (int i = 0; i < 3; ++i) {
    res = bc.get_method("forward")(inputs);
  }
  auto output = res.toTensor();
  AT_ASSERT(outputref.dim() == output.dim());
  AT_ASSERT(
      outputref[0][0][0][0].item<int>() == output[0][0][0][0].item<int>());

  auto buff = save_mobile_module_to_bytes(bc);
  mobile::Module bc2 = parse_mobile_module(buff.data(), buff.size());
  bc2.eval();
  for (int i = 0; i < 3; ++i) {
    res = bc2.get_method("forward")(inputs);
  }
  output = res.toTensor();
  AT_ASSERT(outputref.dim() == output.dim());
  AT_ASSERT(
      outputref[0][0][0][0].item<int>() == output[0][0][0][0].item<int>());
}

TEST(FlatbufferTest, FindWrongMethodName) {
  Module m("m");
  m.register_parameter("foo", torch::ones({}), false);
  m.define(R"(
    def add(self, x):
      b = 4
      return self.foo + x + b
  )");
  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
  ASSERT_TRUE(bc.find_method("forward") == c10::nullopt);

  auto buff = save_mobile_module_to_bytes(bc);
  mobile::Module bc2 = parse_mobile_module(buff.data(), buff.size());
  ASSERT_TRUE(bc2.find_method("forward") == c10::nullopt);
}

TEST(FlatbufferTest, FindAndRunMethod) {
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

  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
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

  auto buff = save_mobile_module_to_bytes(bc);
  mobile::Module bc2 = parse_mobile_module(buff.data(), buff.size());

  for (int i = 0; i < 3; ++i) {
    auto bcinputs = inputs;
    auto method = bc2.find_method("add_it");
    AT_ASSERT(method != c10::nullopt);
    res = (*method)(std::move(bcinputs));
  }

  resd = res.toTensor().item<float>();
  AT_ASSERT(resd == refd);
}

TEST(FlatbufferTest, RunMethodVariadic) {
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

  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
  IValue res = bc.run_method("add_three", inputx, inputy);

  auto resd = res.toTensor().item<float>();
  auto refd = ref.toTensor().item<float>();
  AT_ASSERT(resd == refd);

  auto buff = save_mobile_module_to_bytes(bc);
  mobile::Module bc2 = parse_mobile_module(buff.data(), buff.size());
  res = bc.run_method("add_three", inputx, inputy);
  resd = res.toTensor().item<float>();
  AT_ASSERT(resd == refd);
}

TEST(FlatbufferTest, DuplicateSetState) {
  Module m("M");
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

  Module b("B");
  b.register_module("M0", m);
  b.register_module("M1", m);
  b.define(R"(
    def forward(self, x):
      return self.M0.forward(x) + self.M1.forward(x)
  )");

  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
  const auto methods = bc.get_methods();
  const size_t expected_n = 3;
  ASSERT_EQ(methods.size(), expected_n);

  auto buff = save_mobile_module_to_bytes(bc);
  mobile::Module bc2 = parse_mobile_module(buff.data(), buff.size());
  const auto methods2 = bc.get_methods();
  ASSERT_EQ(methods2.size(), expected_n);
}

TEST(FlatbufferTest, OpNameExportFetchRootOperators) {
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

  CompilationOptions options;
  mobile::Module ptl_model = jitModuleToMobile(m, options);
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

  auto buff = save_mobile_module_to_bytes(ptl_model);
  mobile::Module bc2 = parse_mobile_module(buff.data(), buff.size());
  operator_names = torch::jit::mobile::_export_operator_list(bc2);
  EXPECT_EQ(operator_names, expected_operator_names)
      << "Expected the root operator lists to be the same";
}

TEST(FlatbufferTest, DefaultArgsConv) {
  auto s = std::getenv("PYTORCH_TEST_WITH_TSAN");
  if (s && strcmp(s, "1") == 0)
    return;

  std::vector<torch::jit::IValue> inputs;

  Module m("m");
  m.register_parameter("weight", torch::ones({20, 1, 5, 5}), false);
  m.register_parameter("bias", torch::ones({20}), false);
  m.define(R"(
    def forward(self, input):
      return torch.conv2d(input, self.weight, self.bias, [1, 1], [0, 0], [1, 1], 1)
  )");

  inputs.emplace_back(torch::ones({1, 1, 28, 28}));

  auto outputref = m.forward(inputs).toTensor();

  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
  IValue res;
  for (int i = 0; i < 1; ++i) {
    res = bc.get_method("forward")(inputs);
  }
  auto output = res.toTensor();
  AT_ASSERT(outputref.dim() == output.dim());
  AT_ASSERT(output.equal(outputref));

  auto buff = save_mobile_module_to_bytes(bc);
  mobile::Module bc2 = parse_mobile_module(buff.data(), buff.size());
  for (int i = 0; i < 1; ++i) {
    res = bc2.get_method("forward")(inputs);
  }
  output = res.toTensor();
  AT_ASSERT(outputref.dim() == output.dim());
  AT_ASSERT(output.equal(outputref));
}

namespace {
void testLiteModuleCompareResultTensors(
    Module& m,
    const std::vector<torch::jit::IValue>& inputs,
    const std::string& method_name = "forward") {
  auto outputref = m.get_method(method_name)(inputs).toTensor();

  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
  IValue res;
  for (int i = 0; i < 3; ++i) {
    res = bc.get_method(method_name)(inputs);
  }
  auto output = res.toTensor();
  AT_ASSERT(outputref.dim() == output.dim());
  AT_ASSERT(output.equal(outputref));

  auto buff = save_mobile_module_to_bytes(bc);
  mobile::Module bc2 = parse_mobile_module(buff.data(), buff.size());
  for (int i = 0; i < 3; ++i) {
    res = bc2.get_method(method_name)(inputs);
  }
  output = res.toTensor();
  AT_ASSERT(outputref.dim() == output.dim());
  AT_ASSERT(output.equal(outputref));
}

static void testDefaultArgsPinv(int num_args) {
  Module m("m");
  if (num_args == 1) {
    m.define(R"(
      def forward(self, input):
        return torch.linalg_pinv(input)
    )");
  } else if (num_args == 2) {
    m.define(R"(
      def forward(self, input):
        return torch.linalg_pinv(input, 1e-5)
    )");
  } else if (num_args == 3) {
    m.define(R"(
      def forward(self, input):
        return torch.linalg_pinv(input, 1e-5, True)
    )");
  }

  std::vector<torch::jit::IValue> inputs;
  const int N = 28;
  auto input = torch::range(1, N * N, 1);
  input[0] = 1; // a more stable matrix
  input = input.view({N, N});
  inputs.emplace_back(input);
  testLiteModuleCompareResultTensors(m, inputs);
}
} // namespace

#if !defined FB_XPLAT_BUILD
TEST(FlatbufferTest, DefaultArgsPinv) {
  // Test with different number of specified arguments.
  // Arguments not specified take default value.
  for (int num_args = 1; num_args <= 3; ++num_args) {
    testDefaultArgsPinv(num_args);
  }

  //  bytecode with one specified argument:
  //  (6,
  //      ('__torch__.m.forward',
  //          (('instructions',
  //              (('STOREN', 1, 2),
  //                  ('DROPR', 1, 0),
  //                  ('MOVE', 2, 0),
  //                  ('OP', 0, 0),
  //                  ('RET', 0, 0))),
  //              ('operators', (('aten::linalg_pinv', '', 1),)),
  //              ('constants', (False, 1e-15)), # default constants are not
  //              used
  //              ('types', ()),
  //              ('register_size', 2)),
  //          (('arguments',
  //              ((('name', 'self'), ('type', '__torch__.m'), ('default_value',
  //              None)),
  //                  (('name', 'input'), ('type', 'Tensor'), ('default_value',
  //                  None)))),
  //              ('returns',
  //                  ((('name', ''), ('type', 'Tensor'), ('default_value',
  //                  None)),)))))

  //  bytecode with 2 specified argument:
  //  (6,
  //      ('__torch__.m.forward',
  //          (('instructions',
  //              (('STOREN', 1, 2),
  //                  ('DROPR', 1, 0),
  //                  ('MOVE', 2, 0),
  //                  ('LOADC', 1, 0), # added LOADC for specified argument
  //                  ('OP', 0, 0),
  //                  ('RET', 0, 0))),
  //              ('operators', (('aten::linalg_pinv', '', 2),)),
  //              ('constants', (False, 1e-05)), # updated constant table
  //              ('types', ()),
  //              ('register_size', 2)),
  //          (('arguments',
  //              ((('name', 'self'), ('type', '__torch__.m'), ('default_value',
  //              None)),
  //                  (('name', 'input'), ('type', 'Tensor'), ('default_value',
  //                  None)))),
  //              ('returns',
  //                  ((('name', ''), ('type', 'Tensor'), ('default_value',
  //                  None)),)))))

  //  bytecode with 3 specified arguments:
  //  (6,
  //      ('__torch__.m.forward',
  //          (('instructions',
  //              (('STOREN', 1, 2),
  //                  ('DROPR', 1, 0),
  //                  ('MOVE', 2, 0),
  //                  ('LOADC', 1, 0),
  //                  ('LOADC', 0, 0),
  //                  ('OP', 0, 0),
  //                  ('RET', 0, 0))),
  //              ('operators', (('aten::linalg_pinv', '', 3),)),
  //              ('constants', (True, 1e-05)),
  //              ('types', ()),
  //              ('register_size', 2)),
  //          (('arguments',
  //              ((('name', 'self'), ('type', '__torch__.m'), ('default_value',
  //              None)),
  //                  (('name', 'input'), ('type', 'Tensor'), ('default_value',
  //                  None)))),
  //              ('returns',
  //                  ((('name', ''), ('type', 'Tensor'), ('default_value',
  //                  None)),)))))
}

TEST(FlatbufferTest, DefaultArgsTensorinvSpecifyDefault) {
  // The second argument is specified, but the value is the same as the default
  // value. It's treated as "not specified" since the value can be fetched from
  // schema.
  Module m("m");
  m.define(R"(
    def forward(self, input):
      return torch.linalg_tensorinv(input, 2)
  )");
  torch::jit::MobileCode code(m.get_method("forward").graph(), "forward");
  auto arg_nums = code.op_to_num_specified_args();
  ASSERT_EQ(arg_nums.size(), 1);
  ASSERT_EQ(arg_nums["aten::linalg_tensorinv"], 1);
  std::vector<torch::jit::IValue> inputs;
  const int N = 4;
  auto input = torch::rand({N, N, N, N});
  inputs.emplace_back(input);
  testLiteModuleCompareResultTensors(m, inputs);
}

static void testDefaultArgsPinvWithOutArg(int num_args) {
  Module m("m");
  if (num_args == 1) {
    m.define(R"(
      def forward(self, input):
        return torch.linalg_pinv(input, out=input)
    )");
  } else if (num_args == 2) {
    m.define(R"(
      def forward(self, input):
        return torch.linalg_pinv(input, 1e-5, out=input)
    )");
  } else if (num_args == 3) {
    m.define(R"(
      def forward(self, input):
        return torch.linalg_pinv(input, 1e-5, True, out=input)
    )");
  }

  const int N = 28;
  auto input = torch::range(1, N * N, 1);
  input[0] = 10000; // a more stable matrix
  input = input.view({N, N});
  auto ref = m.run_method("forward", input);
  TORCH_CHECK(!input.equal(torch::range(1, N * N, 1)));
  TORCH_CHECK(input.equal(ref.toTensor()));
}

TEST(FlatbufferTest, DefaultArgsPinvWithOutArg) {
  // Test with different number of specified arguments + out arg.
  // Arguments not specified take default value.
  for (int num_args = 1; num_args <= 3; ++num_args) {
    testDefaultArgsPinvWithOutArg(num_args);
  }
}

TEST(FlatbufferTest, DefaultArgsWithOutArg) {
  Module m("m");
  m.define(R"(
    def forward(self, x, h):
      torch.add(x, h, out=x)
  )");

  std::vector<IValue> inputs;
  auto input_x = 2 * torch::ones({});
  auto input_h = torch::ones({});
  auto ref = m.run_method("forward", input_x, input_h);

  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
  bc.run_method("forward", input_x, input_h);
  AT_ASSERT(input_x.equal(4 * torch::ones({})));

  auto buff = save_mobile_module_to_bytes(bc);
  mobile::Module bc2 = parse_mobile_module(buff.data(), buff.size());
  auto input_x2 = 2 * torch::ones({});
  auto input_h2 = torch::ones({});
  m.run_method("forward", input_x2, input_h2);
  bc2.run_method("forward", input_x2, input_h2);
  AT_ASSERT(input_x2.equal(4 * torch::ones({})));
}

#endif // !defined(FB_XPLAT_BUILD)

namespace {
static auto reg =
    torch::class_<TorchBindFlatbufferTestStruct>(
        "_TorchScriptTesting",
        "_FlatbufferTest")
        .def(torch::init<>())
        .def("get", &TorchBindFlatbufferTestStruct::get)
        .def_pickle(
            // __getattr__
            [](const c10::intrusive_ptr<TorchBindFlatbufferTestStruct>& self)
                -> int64_t { return 0; },
            // __setattr__
            [](int64_t state) {
              return c10::make_intrusive<TorchBindFlatbufferTestStruct>();
            });

} // namespace

TEST(FlatbufferTest, OperatorCacheDifferentiatesDefaultArgs) {
  // Create 3 methods:
  //
  // 1. forward() returns a tensor with dtype=torch.int64 (4)
  // 2. forward2() returns a tensor with dtype=torch.float32 (6)
  // 3. forward3() returns a tensor with dtype=torch.float32 but
  //    the dtype is inferred by the input tensor's dtype
  //
  // If caching works correctly, then the result from the full-jit
  // module and the lite module will be the same. Otherwise, it
  // will be different if we don't correctly ignore the cache
  // entry for an operator that has a different number of
  // arguments.
  Module m("m");
  m.define(R"(
    def forward(self):
      ret1 = torch.new_empty(torch.zeros(10), [10], dtype=4)
      return ret1.fill_(25)
  )");
  m.define(R"(
    def forward2(self):
      ret1 = torch.new_empty(torch.zeros(10), [10], dtype=6)
      return ret1.fill_(32.0)
  )");
  m.define(R"(
    def forward3(self):
      ret1 = torch.new_empty(torch.zeros(10), [10])
      return ret1.fill_(12.0)
  )");

  std::vector<torch::jit::IValue> inputs;
  testLiteModuleCompareResultTensors(m, inputs, "forward");
  testLiteModuleCompareResultTensors(m, inputs, "forward2");
  testLiteModuleCompareResultTensors(m, inputs, "forward3");
}

TEST(FlatbufferTest, OperatorSize1) {
  Module m("m");
  m.define(R"(
    def forward(self, input: Tensor, scale:float):
      return torch.upsample_nearest2d(input, [1, 1], float(scale), float(scale))
  )");

  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
  const auto& func = bc.get_method("forward").function();
  ASSERT_EQ(
      func.get_code().operator_input_sizes_.size(),
      func.get_code().operators_.size());

  auto buff = save_mobile_module_to_bytes(bc);
  mobile::Module bc2 = parse_mobile_module(buff.data(), buff.size());
  const auto& func2 = bc.get_method("forward").function();
  ASSERT_EQ(
      func2.get_code().operator_input_sizes_.size(),
      func2.get_code().operators_.size());
}

TEST(FlatbufferTest, BoolAndDoubleList) {
  Module m("m");
  c10::List<bool> boollist;
  boollist.push_back(false);
  IValue boollist_ival = boollist;
  IValue doublelist = std::vector<double>{2.0};
  m.register_attribute("bool_list", boollist_ival.type(), boollist_ival);
  m.register_attribute("double_list", doublelist.type(), doublelist);

  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
  auto buff = save_mobile_module_to_bytes(bc);
  mobile::Module bc2 = parse_mobile_module(buff.data(), buff.size());

  // if the variables read are wrong type the conversion will raise exception
  auto boolval = bc2.attr("bool_list", {}).toBoolList().get(0);
  auto doubleval = bc2.attr("double_list", {}).toDoubleList().get(0);

  ASSERT_EQ(boolval, false);
  ASSERT_EQ(doubleval, 2.0);
}

TEST(FlatbufferTest, OperatorTest2) { // NOLINT (use =delete in gtest)
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

    CompilationOptions options;
    mobile::Module bc = jitModuleToMobile(m, options);
    const auto& func = bc.get_method("test_func").function();
    ASSERT_EQ(
        func.get_code().operator_input_sizes_.size(),
        func.get_code().operators_.size());

    auto buff = save_mobile_module_to_bytes(bc);
    mobile::Module bc2 = parse_mobile_module(buff.data(), buff.size());
    const auto& func2 = bc.get_method("test_func").function();
    ASSERT_EQ(
        func2.get_code().operator_input_sizes_.size(),
        func2.get_code().operators_.size());
  }
}

Module jitModuleFromBuffer(void* data) {
  auto* flatbuffer_module = mobile::serialization::GetMutableModule(data);
  FlatbufferLoader loader;
  mobile::Module mobilem = loader.parseModule(flatbuffer_module);
  ExtraFilesMap files;
  std::vector<IValue> constants;
  loader.extractJitSourceAndConstants(&files, &constants);
  return jitModuleFromSourceAndConstants(
      mobilem._ivalue(), files, constants, 8);
}

TEST(TestSourceFlatbuffer, UpsampleNearest2d) {
  Module m("m");
  m.define(R"(
    def forward(self, input: Tensor, scale:float):
      return torch.upsample_nearest2d(input, [1, 1], float(scale), float(scale))
  )");

  std::vector<IValue> inputs;
  inputs.emplace_back(torch::rand({1, 3, 128, 128}));
  inputs.emplace_back(at::Scalar(2.0));
  auto ref = m.forward(inputs);

  auto data = save_jit_module_to_bytes(m);
  Module m2 = jitModuleFromBuffer(data.data());
  auto res = m2.forward(inputs);

  auto resd = res.toTensor();
  auto refd = ref.toTensor();
  ASSERT_TRUE(resd.equal(refd));

  mobile::Module m3 = parse_mobile_module(data.data(), data.size());
  res = m3.forward(inputs);
  resd = res.toTensor();
  refd = ref.toTensor();
  ASSERT_TRUE(resd.equal(refd));
}

TEST(TestSourceFlatbuffer, CheckAttrAccess) {
  Module m("m");
  m.register_attribute("mobile_optimized", BoolType::get(), true);
  auto data = save_jit_module_to_bytes(m);
  Module m2 = jitModuleFromBuffer(data.data());
  bool mobile_optimized = m2.attr("mobile_optimized", false).toBool();
  AT_ASSERT(mobile_optimized);
  mobile::Module m3 = parse_mobile_module(data.data(), data.size());
  mobile_optimized = m3.attr("mobile_optimized", false).toBool();
  AT_ASSERT(mobile_optimized);
}

TEST(TestSourceFlatbuffer,
     MethodInvocation) { // NOLINT (use =delete in gtest)
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

    auto data = save_jit_module_to_bytes(m);
    Module m2 = jitModuleFromBuffer(data.data());
    const auto& test_func = m2.get_method("test_func");
    IValue res;
    for (int i = 0; i < 3; ++i) {
      res = test_func({minput});
    }
    auto resd = res.toTensor().item<float>();
    auto refd = ref.toTensor().item<float>();
    AT_ASSERT(resd == refd);

    mobile::Module m3 = parse_mobile_module(data.data(), data.size());
    const auto& test_func3 = m3.get_method("test_func");
    for (int i = 0; i < 3; ++i) {
      res = test_func3({minput});
    }
    resd = res.toTensor().item<float>();
    refd = ref.toTensor().item<float>();
    AT_ASSERT(resd == refd);
  }
}

} // namespace jit
} // namespace torch
