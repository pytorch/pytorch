#include <test/cpp/jit/test_utils.h>

#include <c10/core/TensorOptions.h>
#include <gtest/gtest.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/resolver.h>
#include <torch/csrc/jit/mobile/compatibility/backport.h>
#include <torch/csrc/jit/mobile/compatibility/backport_manager.h>
#include <torch/csrc/jit/mobile/compatibility/model_compatibility.h>
#include <torch/csrc/jit/mobile/compatibility/runtime_compatibility.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/interpreter.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/mobile/parse_bytecode.h>
#include <torch/csrc/jit/mobile/parse_operators.h>
#include <torch/csrc/jit/mobile/upgrader_mobile.h>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/custom_class.h>
#include <torch/torch.h>

#include <torch/csrc/jit/serialization/import_export_functions.h>
#include <unordered_set>

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

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,modernize-use-emplace)
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
  AT_ASSERT(output.toTupleRef().elements()[1].toInt() == 2);
}

TEST(LiteInterpreterTest, AtenFormat) {
  Module m("m");
  m.define(R"""(
  def forward(self, fmt:str="first {} {}", num:str="abc"):
    x = 2
    x = x * x
    return fmt.format(num, x)
  )""");
  std::stringstream ss;
  m._save_for_mobile(ss);
  mobile::Module bc = _load_for_mobile(ss);
  std::vector<torch::jit::IValue> inputs;
  auto output_bc = bc.get_method("forward")(inputs);
  auto output_m = m.get_method("forward")(inputs);
  // std::cout << output_m.toStringRef() << "\n"
  //           << output_bc.toStringRef() << std::endl;
  AT_ASSERT(output_m.toStringRef() == output_bc.toStringRef());
}

TEST(LiteInterpreterTest, PrimDevice) {
  Module m("m");
  m.define(R"""(
  def forward(self, x:torch.Tensor):
    return x.device
  )""");
  std::stringstream ss;
  m._save_for_mobile(ss);
  mobile::Module bc = _load_for_mobile(ss);
  std::vector<torch::jit::IValue> inputs;
  auto minput = 3.5 * torch::ones({});
  inputs.emplace_back(minput);
  auto output_bc = bc.get_method("forward")(inputs);
  auto output_m = m.get_method("forward")(inputs);
  AT_ASSERT(output_bc.toDevice().str() == output_m.toDevice().str());
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

TEST(LiteInterpreterTest, List) {
  Module m("m");
  m.define(R"JIT(
  def foo(self, x):
      return [x + 2]

  def forward(self, x):
      d = self.foo(x)
      return d
  )JIT");
  std::stringstream ss;
  m._save_for_mobile(ss);
  mobile::Module bc = _load_for_mobile(ss);
  std::vector<torch::jit::IValue> inputs({torch::ones({})});
  auto output = bc.get_method("forward")(inputs);
  auto server_output = m.forward(inputs);
  EXPECT_EQ(output.toList().get(0).toTensor().item().toInt(), 3);
  EXPECT_EQ(output, server_output);
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
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
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
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
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
  ASSERT_THROWS_WITH_MESSAGE(_load_for_mobile(ss), "file not found");
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
  ASSERT_THROWS_WITH_MESSAGE(
      bc.get_method("forward")(inputs), "is not defined");
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
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
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
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  auto str = res.toStringRef();
  std::string expected = "Hello! Your tensor has 12 elements!";
  AT_ASSERT(str == expected);
}

#if !defined FB_XPLAT_BUILD
TEST(LiteInterpreterTest, GetRuntimeByteCodeVersion) {
  auto runtime_bytecode_version = _get_runtime_bytecode_version();
  AT_ASSERT(
      runtime_bytecode_version ==
      caffe2::serialize::kMaxSupportedBytecodeVersion);
}

TEST(LiteInterpreterTest, GetRuntimeOperatorsVersion) {
  auto runtime_operators_version = _get_runtime_operators_min_max_versions();
  AT_ASSERT(
      runtime_operators_version.first ==
          caffe2::serialize::kMinSupportedFileFormatVersion &&
      runtime_operators_version.second ==
          caffe2::serialize::kMaxSupportedFileFormatVersion);
}

/**
 * The test below is disarmed for FB internal xplat builds since
 * BUCK requires us to pass in the script_module_v4.ptl file in
 * as a resource dependency of the build rule for this file, and
 * we would need to access it via the C++ Resources API instead
 * of directly reading from disk (which is what the open source
 * build/run does).
 */
TEST(LiteInterpreterTest, GetByteCodeVersion) {
  std::string filePath(__FILE__);
  auto test_model_file_v4 =
      filePath.substr(0, filePath.find_last_of("/\\") + 1);
  test_model_file_v4.append("script_module_v4.ptl");

  auto version_v4 = _get_model_bytecode_version(test_model_file_v4);
  AT_ASSERT(version_v4 == 4);
}

#endif // !defined(FB_XPLAT_BUILD)

TEST(LiteInterpreterTest, GetContainTypes) {
  Module m("m");
  m.define(R"(
    def forward(self):
      return 3
  )");

  std::stringstream ss;
  m._save_for_mobile(ss, {}, true);

  _get_mobile_model_contained_types(ss);
}

namespace {

void compareModelOutput(
    c10::ArrayRef<IValue> actual_result_list,
    const std::vector<IValue>& expect_result_list) {
  AT_ASSERT(actual_result_list.size() == expect_result_list.size());
  AT_ASSERT(
      actual_result_list[0].toTensor().equal(expect_result_list[0].toTensor()));
  AT_ASSERT(
      actual_result_list[1].toTensor().dim() ==
      expect_result_list[1].toTensor().dim());
  AT_ASSERT(
      actual_result_list[2].toTensor().equal(expect_result_list[2].toTensor()));
  AT_ASSERT(
      actual_result_list[3].toTensor().equal(expect_result_list[3].toTensor()));
  ASSERT_EQ(
      actual_result_list[4].toStringRef(), expect_result_list[4].toStringRef());
  ASSERT_EQ(actual_result_list[5].toBool(), expect_result_list[5].toBool());
  ASSERT_EQ(actual_result_list[6].toBool(), expect_result_list[6].toBool());
  ASSERT_EQ(actual_result_list[7].toBool(), expect_result_list[7].toBool());
  AT_ASSERT(
      actual_result_list[8].toTensor().equal(expect_result_list[8].toTensor()));
  ASSERT_EQ(
      actual_result_list[9].toStringRef(), expect_result_list[9].toStringRef());
  ASSERT_EQ(actual_result_list[10].toInt(), expect_result_list[10].toInt());
  ASSERT_EQ(actual_result_list[11].toBool(), expect_result_list[11].toBool());
}

void runAndCheckTorchScriptModel(
    std::stringstream& input_model_stream,
    const std::vector<IValue>& input_data,
    const std::vector<IValue>& expect_result_list,
    const uint64_t expect_version) {
  auto actual_version = _get_model_bytecode_version(input_model_stream);
  AT_ASSERT(actual_version == expect_version);

  // Load and run the backport model, then compare the result with expect
  // result
  Module m_mobile = load(input_model_stream);

  auto actual_result = m_mobile.forward(input_data);
  const auto& actual_result_list = actual_result.toTupleRef().elements();
  compareModelOutput(actual_result_list, expect_result_list);
}

void runAndCheckBytecodeModel(
    std::stringstream& input_model_stream,
    const std::vector<IValue>& input_data,
    const std::vector<IValue>& expect_result_list,
    const uint64_t expect_version) {
  auto actual_version = _get_model_bytecode_version(input_model_stream);
  AT_ASSERT(actual_version == expect_version);

  // Load and run the backport model, then compare the result with expect
  // result
  Module m_mobile = load(input_model_stream);

  auto actual_result = m_mobile.forward(input_data);
  const auto& actual_result_list = actual_result.toTupleRef().elements();

  compareModelOutput(actual_result_list, expect_result_list);
}

void backportAllVersionCheck(
    std::stringstream& test_model_file_stream,
    std::vector<IValue>& input_data,
    std::vector<IValue>& expect_result_list,
    const uint64_t expect_from_version) {
  auto from_version = _get_model_bytecode_version(test_model_file_stream);
  EXPECT_EQ(from_version, expect_from_version);
  AT_ASSERT(from_version > 0);

  // Backport script_module_v5.ptl to an older version
  constexpr int64_t minimum_to_version = 4;
  auto current_to_version = from_version - 1;

  // Verify all candidate to_version work as expected. All backport to version
  // larger than minimum_to_version should success.
  while (current_to_version >= minimum_to_version) {
    // Do not declare std::stringstream oss outside of the while loop as
    // oss.clear() doesn't reset the stream content, only clears out error state
    // flag in stringstream causing a problematic stream. Instead, it's cleaner
    // and safer to just declare a new std::stringstream one and swap them.
    std::stringstream oss;
    bool backPortSuccess =
        _backport_for_mobile(test_model_file_stream, oss, current_to_version);
    AT_ASSERT(backPortSuccess);

    // Check backport model version
    auto backport_version = _get_model_bytecode_version(oss);
    backport_version = _get_model_bytecode_version(oss);
    AT_ASSERT(backport_version == current_to_version);

    // Load and run the backport model, then compare the result with expect
    // result
    runAndCheckBytecodeModel(
        oss, input_data, expect_result_list, current_to_version);
    oss.seekg(0, oss.beg);
    runAndCheckTorchScriptModel(
        oss, input_data, expect_result_list, current_to_version);

    current_to_version--;
  }
  //  backport to minimum version - 1 should fail
  std::stringstream oss;
  bool backPortSuccess =
      _backport_for_mobile(test_model_file_stream, oss, minimum_to_version - 1);
  AT_ASSERT(!backPortSuccess);
}
} // namespace

#if !defined FB_XPLAT_BUILD
TEST(LiteInterpreterTest, BackPortByteCodeModelAllVersions) {
  torch::jit::Module module("m");
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  module.register_parameter("weight", torch::ones({20, 1, 5, 5}), false);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  module.register_parameter("bias", torch::ones({20}), false);
  module.define(R"(
    def fn(self, x:float=1.0):
      return x

    def forward(self, input):
      x1 = torch.zeros(2, 2)
      x2 = torch.empty_like(torch.empty(2, 2))
      x3 = torch._convolution(input, self.weight, self.bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1, False, False, True, True)
      # Add torch.add operator to cover bytecode version bump from 6 to 7
      # for bytecode version 7, the main change is to support defaults arguments with out arguments
      x = 2 * torch.ones(1)
      h = torch.ones(1)
      torch.add(x, h, out=x)
      device = torch.ones(1, 1).cpu().device.type
      is_cuda = x1.is_cuda
      bool_val = True
      check_is = [] is None
      check_is_not = [1] is not None
      check_not = not bool_val
      num_to_tensor = torch.tensor([self.fn()])
      d = {"a": "abc"}
      check_dict_index = d["a"]
      check_dim = x1.dim()
      return (
        x1, x2, x3, x, device, is_cuda, check_is,
        check_is_not, num_to_tensor, check_dict_index,
        check_dim, check_not
        )
      )");

  torch::jit::Module module_freeze = freeze(module);

  std::stringstream input_model_stream;
  module_freeze._save_for_mobile(
      input_model_stream,
      /*extra_files=*/{},
      /*save_mobile_debug_info=*/false,
      /*use_flatbuffer=*/true);
  std::vector<IValue> input_data =
      std::vector<IValue>({torch::ones({1, 1, 28, 28})});
  std::vector<IValue> expect_result_list;
  expect_result_list.emplace_back(at::ones({2, 2}, ScalarType::Float) * 0);
  expect_result_list.emplace_back(at::ones({2, 2}, ScalarType::Float));
  expect_result_list.emplace_back(
      at::ones({1, 20, 24, 24}, ScalarType::Float) * 26);
  expect_result_list.emplace_back(3 * at::ones({1}));
  // "cpu" False, False, True, tensor(1), "abc", 2, False)
  expect_result_list.emplace_back(c10::IValue("cpu"));
  expect_result_list.emplace_back(c10::IValue(false));
  expect_result_list.emplace_back(c10::IValue(false));
  expect_result_list.emplace_back(c10::IValue(true));
  expect_result_list.emplace_back(c10::IValue(at::ones({1})));
  expect_result_list.emplace_back(c10::IValue("abc"));
  expect_result_list.emplace_back(c10::IValue(2));
  expect_result_list.emplace_back(c10::IValue(false));

  backportAllVersionCheck(
      input_model_stream,
      input_data,
      expect_result_list,
      9); // flatbuffer starts at 9
}
#endif // !defined(FB_XPLAT_BUILD)

TEST(LiteInterpreterTest, GetRuntimeOpsAndInfo) {
  auto runtime_ops = _get_runtime_ops_and_info();
  // Ballpark estimate of the minimal number of ops; just used to
  // verify API returns a reasonably large number.
  AT_ASSERT(runtime_ops.size() > 2900);
}

TEST(LiteInterpreterTest, isCompatibleSuccess) {
  // test trivial success case
  auto runtime_info = RuntimeCompatibilityInfo::get();
  std::unordered_map<std::string, OperatorInfo> model_ops;
  model_ops["aten::add.Scalar"] = OperatorInfo{2};

  std::unordered_set<std::string> types = {"List", "int", "NamedTuple"};
  auto model_info = ModelCompatibilityInfo{
      caffe2::serialize::kMaxSupportedBytecodeVersion,
      model_ops,
      types,
      _get_runtime_bytecode_min_max_versions().first};

  AT_ASSERT(
      is_compatible(runtime_info, model_info).status ==
      ModelCompatibilityStatus::OK);
}

TEST(LiteInterpreterTest, isCompatibleFail) {
  // test trivial failure due to ops
  std::unordered_map<std::string, OperatorInfo> model_ops;
  model_ops["aten::add.Scalar"] = OperatorInfo{2};
  auto model_info = ModelCompatibilityInfo{
      caffe2::serialize::kMaxSupportedBytecodeVersion, model_ops};
  std::unordered_map<std::string, OperatorInfo> runtime_ops;
  runtime_ops["aten::add.Int"] = OperatorInfo{2};
  auto runtime_info = RuntimeCompatibilityInfo{
      std::pair<uint64_t, uint64_t>(
          caffe2::serialize::kMinSupportedBytecodeVersion,
          caffe2::serialize::kMaxSupportedBytecodeVersion),
      runtime_ops,
      _get_mobile_supported_types()};

  auto result = is_compatible(runtime_info, model_info);
  AT_ASSERT(result.status = ModelCompatibilityStatus::ERROR);
  AT_ASSERT(
      result.errors[0] ==
      "Operator 'aten::add.Scalar' missing from runtime (not found)");

  // test trivial failure due to bytecode greater than max supported bytecode
  // version
  runtime_ops["aten::add.Scalar"] = OperatorInfo{2};
  runtime_info = RuntimeCompatibilityInfo{
      std::pair<uint64_t, uint64_t>(
          caffe2::serialize::kMinSupportedBytecodeVersion,
          caffe2::serialize::kMaxSupportedBytecodeVersion),
      runtime_ops,
      _get_mobile_supported_types()};
  model_info.bytecode_version =
      caffe2::serialize::kMaxSupportedBytecodeVersion + 1;

  result = is_compatible(runtime_info, model_info);
  AT_ASSERT(result.status = ModelCompatibilityStatus::ERROR);

  // test trivial failure due to bytecode less than min supported bytecode
  // version
  runtime_ops["aten::add.Scalar"] = OperatorInfo{2};
  runtime_info = RuntimeCompatibilityInfo{
      std::pair<uint64_t, uint64_t>(
          caffe2::serialize::kMinSupportedBytecodeVersion,
          caffe2::serialize::kMaxSupportedBytecodeVersion),
      runtime_ops,
      _get_mobile_supported_types()};
  model_info.bytecode_version =
      caffe2::serialize::kMinSupportedBytecodeVersion - 1;

  result = is_compatible(runtime_info, model_info);
  AT_ASSERT(result.status = ModelCompatibilityStatus::ERROR);

  // test trivial failure due to type
  runtime_info = RuntimeCompatibilityInfo::get();
  std::unordered_set<std::string> types = {"List", "int", "Sequence"};

  model_info = ModelCompatibilityInfo{
      caffe2::serialize::kMaxSupportedBytecodeVersion,
      model_ops,
      types,
      _get_runtime_bytecode_min_max_versions().first};

  AT_ASSERT(
      is_compatible(runtime_info, model_info).status ==
      ModelCompatibilityStatus::ERROR);

  // test trivial failure due to operator version
  runtime_info = RuntimeCompatibilityInfo::get();

  model_info = ModelCompatibilityInfo{
      caffe2::serialize::kMaxSupportedBytecodeVersion, model_ops, {}, 0};

  AT_ASSERT(
      is_compatible(runtime_info, model_info).status ==
      ModelCompatibilityStatus::ERROR);
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

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,modernize-use-emplace)
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
  ASSERT_TRUE(bc.find_method("forward") == std::nullopt);
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
    AT_ASSERT(method != std::nullopt);
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

TEST(LiteInterpreterTest, DuplicateSetState) {
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

  std::stringstream ss;
  m._save_for_mobile(ss);
  mobile::Module bc = _load_for_mobile(ss);
  const auto methods = bc.get_methods();
  const size_t expected_n = 3;
  ASSERT_EQ(methods.size(), expected_n);
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
  extra_files["mobile_info.json"] = "{\"key\": 23}";
  module->_save_for_mobile(oss, extra_files);

  std::istringstream iss(oss.str());
  std::unordered_map<std::string, std::string> loaded_extra_files;
  loaded_extra_files["metadata.json"] = "";
  torch::jit::_load_for_mobile(iss, torch::kCPU, loaded_extra_files);
  ASSERT_EQ(loaded_extra_files["metadata.json"], "abc");

  loaded_extra_files.clear();
  std::vector<std::string> all_files =
      caffe2::serialize::PyTorchStreamReader(&iss).getAllRecords();

  for (auto& file_name : all_files) {
    if (file_name.find("extra/") == 0) {
      loaded_extra_files[file_name.substr(6)] = "";
    }
  }
  iss.seekg(0, iss.beg);
  torch::jit::_load_for_mobile(iss, torch::kCPU, loaded_extra_files);
  ASSERT_EQ(loaded_extra_files["metadata.json"], "abc");
  ASSERT_EQ(loaded_extra_files["mobile_info.json"], "{\"key\": 23}");

  std::unordered_map<std::string, std::string>
      loaded_extra_files_without_explicit_mapping;
  iss.seekg(0, iss.beg);
  torch::jit::_load_for_mobile(
      iss,
      torch::kCPU,
      loaded_extra_files_without_explicit_mapping,
      MobileModuleLoadOptions::PARSE_ALL_EXTRA_FILE_MAPS);
  ASSERT_EQ(
      loaded_extra_files_without_explicit_mapping["metadata.json"], "abc");
  ASSERT_EQ(
      loaded_extra_files_without_explicit_mapping["mobile_info.json"],
      "{\"key\": 23}");
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

TEST(LiteInterpreterTest, DefaultArgsConv) {
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

  inputs.push_back(torch::ones({1, 1, 28, 28}));

  auto outputref = m.forward(inputs).toTensor();

  std::stringstream ss;
  m._save_for_mobile(ss);
  mobile::Module bc = _load_for_mobile(ss);
  IValue res;
  for (int i = 0; i < 1; ++i) {
    res = bc.get_method("forward")(inputs);
  }
  auto output = res.toTensor();
  AT_ASSERT(outputref.dim() == output.dim());
  AT_ASSERT(output.equal(outputref));
}

TEST(RunTimeTest, ParseBytecode) {
  // A simple example to show a simple bytecode that can be used independent of
  // PyTorch TorchScript serialization (unpickler, etc) and operator library.
  // It has basic control flow (if, else) and basic data orchestration (list
  // construction). The original PyTorch program:

  //  class Module(torch.nn.Module):
  //
  //    def __init__(self) -> None:
  //      super().__init__()
  //
  //    def forward(self, x: int, h: int, xfirst: bool):
  //      if xfirst:
  //        return [x, h]
  //      else:
  //        return [h, x]

  // 1. Prepare for the bytecode. In reality it can be from a customized
  // deserializer.
  std::vector<IValue> instructions{
      to_tuple({"STOREN", 1, 4}),
      to_tuple({"DROPR", 1, 0}),
      to_tuple({"MOVE", 4, 0}),
      to_tuple({"JF", 5, 0}),
      to_tuple({"LOAD", 2, 0}),
      to_tuple({"LOAD", 3, 0}),
      to_tuple({"LIST_CONSTRUCT", 0, 2}),
      to_tuple({"JMP", 4, 0}),
      to_tuple({"LOAD", 3, 0}),
      to_tuple({"LOAD", 2, 0}),
      to_tuple({"LIST_CONSTRUCT", 1, 2}),
      to_tuple({"STORE", 5, 0}),
      to_tuple({"DROPR", 3, 0}),
      to_tuple({"DROPR", 2, 0}),
      to_tuple({"MOVE", 5, 0}),
      to_tuple({"RET", 0, 0}),
  };
  std::vector<IValue> operators; // empty for this example
  std::vector<IValue> constants; // empty for this example

  std::vector<IValue> types{"List[int]", "List[int]"};
  // 2. Parse the function
  std::string function_name("test_function");
  auto function = std::unique_ptr<mobile::Function>(
      new mobile::Function(c10::QualifiedName(function_name)));
  c10::ivalue::TupleElements debug_handles_m_tuple;
  parseInstructions(
      function_name,
      std::move(*c10::ivalue::Tuple::create(instructions)).elements(),
      debug_handles_m_tuple,
      function.get());
  parseTypes(c10::ivalue::Tuple::create(types)->elements(), function.get());
  const size_t rsize = 5;
  parseRegisterSize(rsize, function.get());

  // 3. Prepare for inputs and run the function
  // Note that the first input is reserved for Module object.
  // Since this is a function test and Module object is not required,
  // a dummy IValue (0) is added here.
  std::vector<IValue> inputs{0, 1, 2, true};
  function->run(inputs);
  auto output = inputs[0].toList();
  ASSERT_EQ(output[0], 1);
  ASSERT_EQ(output[1], 2);

  std::vector<IValue> inputs1{0, 1, 2, false};
  function->run(inputs1);
  auto output1 = inputs1[0].toList();
  ASSERT_EQ(output1[0], 2);
  ASSERT_EQ(output1[1], 1);
}

TEST(RunTimeTest, ParseOperator) {
  // A simple example to show a simple bytecode that can be used independent of
  // PyTorch TorchScript serialization (unpickler, etc) and operator library.
  // It has one operator and we should be able to register it. The original
  // PyTorch program:

  // class Add(torch.nn.Module):
  //     def __init__(self) -> None:
  //         super().__init__()

  //     def forward(self, a, b):
  //         return a + b

  // 1. Prepare for the bytecode. In reality it can be from a customized
  // deserializer.
  std::vector<IValue> instructions{
      to_tuple({"STOREN", 1, 3}),
      to_tuple({"DROPR", 1, 0}),
      to_tuple({"MOVE", 2, 0}),
      to_tuple({"MOVE", 3, 0}),
      to_tuple({"OP", 0, 0}),
      to_tuple({"RET", 0, 0}),
  };
  std::vector<IValue> operators{
      to_tuple({"aten::add", "Tensor", 2}),
  };
  std::vector<IValue> constants{
      to_tuple({1}),
  };
  // 2. Parse the function
  std::string function_name("test_function");
  auto function = std::unique_ptr<mobile::Function>(
      new mobile::Function(c10::QualifiedName(function_name)));
  c10::ivalue::TupleElements debug_handles_m_tuple;
  parseInstructions(
      function_name,
      std::move(*c10::ivalue::Tuple::create(instructions)).elements(),
      debug_handles_m_tuple,
      function.get());
  parseOperators(
      std::move(*c10::ivalue::Tuple::create(operators)).elements(),
      1,
      function.get());
  const size_t rsize = 5;
  parseRegisterSize(rsize, function.get());

  // 3. Prepare for inputs and run the function
  // Note that the first input is reserved for Module object.
  // Since this is a function test and Module object is not required,
  // a dummy IValue (0) is added here.
  std::vector<IValue> inputs{0, at::tensor(1), at::tensor(2)};
  function->run(inputs);
  auto output = inputs[0];
  ASSERT_EQ(output, at::tensor(3));
}

namespace {
void testLiteModuleCompareResultTensors(
    Module& m,
    const std::vector<torch::jit::IValue>& inputs,
    const std::string& method_name = "forward") {
  auto outputref = m.get_method(method_name)(inputs).toTensor();

  std::stringstream ss;
  m._save_for_mobile(ss);
  mobile::Module bc = _load_for_mobile(ss);
  IValue res;
  for (int i = 0; i < 3; ++i) {
    res = bc.get_method(method_name)(inputs);
  }
  auto output = res.toTensor();
  AT_ASSERT(outputref.dim() == output.dim());
  AT_ASSERT(output.equal(outputref));
}

void testDefaultArgsPinv(int num_args) {
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
  inputs.push_back(input);
  testLiteModuleCompareResultTensors(m, inputs);
}
} // namespace

#if !defined FB_XPLAT_BUILD
TEST(LiteInterpreterTest, DefaultArgsPinv) {
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

TEST(LiteInterpreterTest, DefaultArgsTensorinvSpecifyDefault) {
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
  inputs.push_back(input);
  testLiteModuleCompareResultTensors(m, inputs);
}

void testDefaultArgsPinvWithOutArg(int num_args) {
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

TEST(LiteInterpreterTest, DefaultArgsPinvWithOutArg) {
  // Test with different number of specified arguments + out arg.
  // Arguments not specified take default value.
  for (int num_args = 1; num_args <= 3; ++num_args) {
    testDefaultArgsPinvWithOutArg(num_args);
  }
}

TEST(LiteInterpreterTest, DefaultArgsWithOutArg) {
  Module m("m");
  m.define(R"(
    def forward(self, x, h):
      torch.add(x, h, out=x)
  )");

  std::vector<IValue> inputs;
  auto input_x = 2 * torch::ones({});
  auto input_h = torch::ones({});
  auto ref = m.run_method("forward", input_x, input_h);

  std::stringstream ss;

  m._save_for_mobile(ss, {}, true);
  mobile::Module bc = _load_for_mobile(ss);
  bc.run_method("forward", input_x, input_h);
  AT_ASSERT(input_x.equal(4 * torch::ones({})));

  auto ops = _get_model_ops_and_info(ss);
  auto op = ops.find("aten::add.out");
  TORCH_CHECK(
      op != ops.end() && op->second.num_schema_args.has_value() &&
      op->second.num_schema_args.value() == 3);
}

TEST(LiteInterpreterTest, TestExceptionStackWithTwoLevelModuleHierarchy) {
  Module a("A");
  a.define(R"(
    def bar(self, x, y):
      return x + y
  )");
  Module b("B");
  b.register_module("A0", a);
  b.define(R"(
    def foo(self, x, y):
      return self.A0.bar(x, y) + 2
  )");
  Module c("C");
  c.register_module("B0", b);
  c.define(R"(
    def forward(self, x, y):
      return self.B0.foo(x, y) + 3
  )");

  std::vector<IValue> inputs;
  inputs.emplace_back(torch::rand({2, 4}));
  inputs.emplace_back(torch::rand({13, 9}));

  std::stringstream ss;
  c._save_for_mobile(ss, ExtraFilesMap(), true);
  auto lite_m = _load_for_mobile(ss);
  std::string error_pattern = R"(
  Module hierarchy:top(C)::<unknown>.B0(B)::foo.A0(A)::bar.aten::add
Traceback of TorchScript (most recent call last):
  File "<string>", line 3, in <unknown>

    def forward(self, x, y):
      return self.B0.foo(x, y) + 3
             ~~~~~~~~~~~ <--- HERE

  File "<string>", line 3, in foo

    def foo(self, x, y):
      return self.A0.bar(x, y) + 2
             ~~~~~~~~~~~ <--- HERE

  File "<string>", line 3, in bar

    def bar(self, x, y):
      return x + y
             ~~~~~ <--- HERE
  )";
  ASSERT_THROWS_WITH_MESSAGE(lite_m.forward(inputs), error_pattern);
}
#endif // !defined(FB_XPLAT_BUILD)

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

TEST(LiteInterpreterTest, OperatorCacheDifferentiatesDefaultArgs) {
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

TEST(RunTimeTest, RuntimeCall) {
  //     def call(x):
  //         return x + x
  //
  //     def forward(a):
  //         x = a + call(a)
  //         y = a + call(x)
  //         return y

  std::vector<IValue> instructionsCall{
      to_tuple({"STORE", 1, 0}),
      to_tuple({"LOAD", 1, 0}),
      to_tuple({"MOVE", 1, 0}),
      to_tuple({"LOADC", 0, 0}),
      to_tuple({"OP", 0, 0}),
      to_tuple({"RET", 0, 0}),
  };
  std::vector<IValue> instructionsFoo{
      to_tuple({"STORE", 1, 0}),
      to_tuple({"LOAD", 1, 0}),
      to_tuple({"LOAD", 1, 0}),
      to_tuple({"MOVE", 1, 0}),
      to_tuple({"CALL", 0, 0}),
      to_tuple({"LOADC", 0, 0}),
      to_tuple({"OP", 0, 0}),
      to_tuple({"CALL", 0, 0}),
      to_tuple({"LOADC", 0, 0}),
      to_tuple({"OP", 0, 0}),
      to_tuple({"RET", 0, 0}),
  };
  std::vector<IValue> operatorsFoo{
      to_tuple({"aten::add", "Tensor", 3}),
  };
  std::vector<IValue> constantsFoo{
      1,
  };
  std::vector<IValue> operatorsCall{
      to_tuple({"aten::add", "Tensor", 3}),
  };
  std::vector<IValue> constantsCall{
      1,
  };

  auto foo = std::make_unique<mobile::Function>(c10::QualifiedName("foo"));
  c10::ivalue::TupleElements debug_handles_m_tuple;
  parseInstructions(
      "foo",
      std::move(*c10::ivalue::Tuple::create(instructionsFoo)).elements(),
      debug_handles_m_tuple,
      foo.get());
  parseOperators(
      std::move(*c10::ivalue::Tuple::create(operatorsFoo)).elements(),
      1,
      foo.get());
  parseConstants(
      std::move(*c10::ivalue::Tuple::create(constantsFoo)).elements(),
      foo.get());
  const size_t rsize = 5;
  parseRegisterSize(rsize, foo.get());

  auto call = std::make_unique<mobile::Function>(c10::QualifiedName("call"));
  parseInstructions(
      "call",
      std::move(*c10::ivalue::Tuple::create(instructionsCall)).elements(),
      debug_handles_m_tuple,
      call.get());
  parseOperators(
      std::move(*c10::ivalue::Tuple::create(operatorsCall)).elements(),
      1,
      call.get());
  parseConstants(
      std::move(*c10::ivalue::Tuple::create(constantsCall)).elements(),
      call.get());
  parseRegisterSize(rsize, call.get());

  foo->append_function(*call);

  std::vector<IValue> inputs{at::tensor(1)};
  foo->run(inputs);
  auto output = inputs[0];
  ASSERT_EQ(output, at::tensor(7));
}

TEST(LiteInterpreterTest, OperatorSize1) {
  Module m("m");
  m.define(R"(
    def forward(self, input: Tensor, scale:float):
      return torch.upsample_nearest2d(input, [1, 1], float(scale), float(scale))
  )");

  std::stringstream ss;
  m._save_for_mobile(ss);
  mobile::Module bc = _load_for_mobile(ss);
  const auto& func = bc.get_method("forward").function();
  ASSERT_EQ(
      func.get_code().operator_input_sizes_.size(),
      func.get_code().operators_.size());
}

TEST(LiteInterpreterTest, OperatorTest2) { // NOLINT (use =delete in gtest)
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

    std::stringstream ss;
    m._save_for_mobile(ss);
    mobile::Module bc = _load_for_mobile(ss);
    const auto& func = bc.get_method("test_func").function();
    ASSERT_EQ(
        func.get_code().operator_input_sizes_.size(),
        func.get_code().operators_.size());
  }
}

#if !defined FB_XPLAT_BUILD
// The following test run in fbcode only
TEST(LiteInterpreterUpgraderTest, DivTensorV2) {
  std::string filePath(__FILE__);
  auto test_model_file = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  test_model_file.append("upgrader_models/test_versioned_div_tensor_v2.ptl");
  /*
  (('__torch__.MyModule.forward',
    (('instructions',
      (('STOREN', 1, 3),
       ('DROPR', 1, 0),
       ('LOAD', 2, 0),
       ('LOAD', 3, 0),
       ('OP', 0, 0),
       ('LOAD', 2, 0),
       ('LOAD', 3, 0),
       ('OP', 1, 0),
       ('MOVE', 2, 0),
       ('MOVE', 3, 0),
       ('OP', 2, 0),
       ('TUPLE_CONSTRUCT', 3, 0),
       ('RET', 0, 0))),
     ('operators',
      (('aten::div', 'Tensor'),
       ('aten::div', 'Tensor'),
       ('aten::div', 'Tensor'))),
     ('constants', ()),
     ('types', ()),
     ('register_size', 3))),)

  */
  mobile::Module m_module = _load_for_mobile(test_model_file);
  auto intrsuction_list =
      m_module.get_method("forward").function().get_code().instructions_;
  uint64_t number_of_call_instruction = 0;
  for (auto& instruction : intrsuction_list) {
    number_of_call_instruction += (instruction.op == OpCode::CALL);
  }
  // 3 operators will use upgrader
  ASSERT_EQ(number_of_call_instruction, 3);

  std::vector<IValue> inputs = {
      IValue(6 * torch::ones({1})), IValue(3 * torch::ones({1}))};
  auto actual_output = m_module.forward(inputs);
  auto expect_output = 2.0 * torch::ones({1});
  auto actual_output_list = actual_output.toTuple()->elements();
  ASSERT_TRUE(actual_output_list[0].toTensor().equal(expect_output));
}

TEST(LiteInterpreterUpgraderTest, DivTensorOutV2) {
  std::string filePath(__FILE__);
  auto test_model_file = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  test_model_file.append(
      "upgrader_models/test_versioned_div_tensor_out_v2.ptl");
  /*
  (('__torch__.MyModule.forward',
    (('instructions',
      (('STOREN', 1, 4),
       ('DROPR', 1, 0),
       ('MOVE', 2, 0),
       ('MOVE', 3, 0),
       ('MOVE', 4, 0),
       ('OP', 0, 0),
       ('RET', 0, 0))),
     ('operators', (('aten::div', 'out'),)),
     ('constants', ()),
     ('types', ()),
     ('register_size', 4))),)
  */
  mobile::Module m_module = _load_for_mobile(test_model_file);

  auto intrsuction_list =
      m_module.get_method("forward").function().get_code().instructions_;
  uint64_t number_of_call_instruction = 0;
  for (auto& instruction : intrsuction_list) {
    number_of_call_instruction += (instruction.op == OpCode::CALL);
  }
  // One operator will use upgrader
  ASSERT_EQ(number_of_call_instruction, 1);

  std::vector<IValue> inputs{
      IValue(6 * torch::ones({1})),
      IValue(3 * torch::ones({1})),
      IValue(torch::empty({1}))};
  m_module.forward(inputs);
  auto expect_output = 2.0 * torch::ones({1});
  auto actual_output = inputs[2].toTensor();
  // The out argument will be overwritten with the output
  ASSERT_TRUE(actual_output.equal(expect_output));
}

TEST(LiteInterpreterUpgraderTest, DivTensorInplaceV2) {
  std::string filePath(__FILE__);
  auto test_model_file = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  test_model_file.append(
      "upgrader_models/test_versioned_div_tensor_inplace_v2.ptl");
  /*
  (('__torch__.MyModule.forward',
    (('instructions',
      (('STOREN', 1, 3),
       ('DROPR', 1, 0),
       ('MOVE', 2, 0),
       ('MOVE', 3, 0),
       ('OP', 0, 0),
       ('RET', 0, 0))),
     ('operators', (('aten::div_', 'Tensor'),)),
     ('constants', ()),
     ('types', ()),
     ('register_size', 3))),)
  */
  mobile::Module m_module = _load_for_mobile(test_model_file);

  auto intrsuction_list =
      m_module.get_method("forward").function().get_code().instructions_;
  uint64_t number_of_call_instruction = 0;
  for (auto& instruction : intrsuction_list) {
    number_of_call_instruction += (instruction.op == OpCode::CALL);
  }
  // One operator will use upgrader
  ASSERT_EQ(number_of_call_instruction, 1);

  std::vector<IValue> inputs{
      IValue(6 * torch::ones({1})), IValue(3 * torch::ones({1}))};
  m_module.forward(inputs);
  auto expect_output = 2.0 * torch::ones({1});
  auto actual_output = inputs[0].toTensor();
  // The out argument will be overwritten with the output
  ASSERT_TRUE(actual_output.equal(expect_output));
}

TEST(LiteInterpreterUpgraderTest, DivScalarFloatV2) {
  std::string filePath(__FILE__);
  auto test_model_file = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  test_model_file.append(
      "upgrader_models/test_versioned_div_scalar_float_v2.ptl");
  /*
  (('__torch__.MyModuleFloat.forward',
    (('instructions',
    (('STOREN', 1, 3),
    ('DROPR', 1, 0),
    ('MOVE', 2, 0),
    ('MOVE', 3, 0),
    ('OP', 0, 0),
    ('RET', 0, 0))),
    ('operators', (('aten::div', 'Scalar'),)),
    ('constants', ()),
    ('types', ()),
    ('register_size', 3))),)
  */

  mobile::Module m_module = _load_for_mobile(test_model_file);

  auto intrsuction_list =
      m_module.get_method("forward").function().get_code().instructions_;
  uint64_t number_of_call_instruction = 0;
  for (auto& instruction : intrsuction_list) {
    number_of_call_instruction += (instruction.op == OpCode::CALL);
  }
  // One operator will use upgrader
  ASSERT_EQ(number_of_call_instruction, 1);

  std::vector<IValue> inputs{IValue(6 * torch::ones({1})), IValue(3.0)};
  auto output = m_module.forward(inputs);
  auto expect_output = 2.0 * torch::ones({1});
  auto actual_output = output.toTensor();

  // The out argument will be overwritten with the output
  ASSERT_TRUE(actual_output.equal(expect_output));
}

TEST(LiteInterpreterUpgraderTest, DivScalarReciprocalFloatV2) {
  std::string filePath(__FILE__);
  auto test_model_file = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  test_model_file.append(
      "upgrader_models/test_versioned_div_scalar_reciprocal_float_v2.ptl");
  /*
  (('__torch__.MyModuleFloat.forward',
    (('instructions',
      (('STOREN', 1, 3),
      ('DROPR', 1, 0),
      ('MOVE', 2, 0),
      ('OP', 0, 0),
      ('MOVE', 3, 0),
      ('OP', 1, 0),
      ('RET', 0, 0))),
    ('operators', (('aten::reciprocal', ''), ('aten::mul', 'Scalar'))),
    ('constants', ()),
    ('types', ()),
    ('register_size', 3))),)
  */
  mobile::Module m_module = _load_for_mobile(test_model_file);

  auto intrsuction_list =
      m_module.get_method("forward").function().get_code().instructions_;
  uint64_t number_of_call_instruction = 0;
  for (auto& instruction : intrsuction_list) {
    number_of_call_instruction += (instruction.op == OpCode::CALL);
  }
  // No operator will use upgrader
  ASSERT_EQ(number_of_call_instruction, 0);

  std::vector<IValue> inputs{IValue(6 * torch::ones({1})), IValue(3.0)};
  auto output = m_module.forward(inputs);
  auto expect_output = 0.5 * torch::ones({1});
  auto actual_output = output.toTensor();
  std::cout << "expect output: " << expect_output;
  std::cout << "actual output: " << actual_output;
  // The out argument will be overwritten with the output
  ASSERT_TRUE(actual_output.equal(expect_output));
}

TEST(LiteInterpreterUpgraderTest, DivScalarReciprocalIntV2) {
  std::string filePath(__FILE__);
  auto test_model_file = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  test_model_file.append(
      "upgrader_models/test_versioned_div_scalar_reciprocal_int_v2.ptl");
  /*
  (('__torch__.MyModuleInt.forward',
  (('instructions',
    (('STOREN', 1, 3),
     ('DROPR', 1, 0),
     ('MOVE', 2, 0),
     ('OP', 0, 0),
     ('MOVE', 3, 0),
     ('OP', 1, 0),
     ('RET', 0, 0))),
   ('operators', (('aten::reciprocal', ''), ('aten::mul', 'Scalar'))),
   ('constants', ()),
   ('types', ()),
   ('register_size', 3))),)
  */
  mobile::Module m_module = _load_for_mobile(test_model_file);

  auto intrsuction_list =
      m_module.get_method("forward").function().get_code().instructions_;
  uint64_t number_of_call_instruction = 0;
  for (auto& instruction : intrsuction_list) {
    number_of_call_instruction += (instruction.op == OpCode::CALL);
  }
  // No operator will use upgrader
  ASSERT_EQ(number_of_call_instruction, 0);

  std::vector<IValue> inputs{IValue(6 * torch::ones({1})), IValue(3.0)};
  auto output = m_module.forward(inputs);
  auto expect_output = 0.5 * torch::ones({1});
  auto actual_output = output.toTensor();

  // The out argument will be overwritten with the output
  ASSERT_TRUE(actual_output.equal(expect_output));
}

TEST(LiteInterpreterUpgraderTest, DivScalarScalarV2) {
  std::string filePath(__FILE__);
  auto test_model_file = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  test_model_file.append(
      "upgrader_models/test_versioned_div_scalar_scalar_v2.ptl");
  /*
  (('__torch__.MyModule.forward',
    (('instructions',
      (('STOREN', 1, 5),
      ('DROPR', 1, 0),
      ('LOAD', 2, 0),
      ('LOAD', 3, 0),
      ('OP', 0, 0),
      ('MOVE', 2, 0),
      ('LOAD', 4, 0),
      ('OP', 1, 0),
      ('LOAD', 3, 0),
      ('MOVE', 4, 0),
      ('OP', 2, 0),
      ('MOVE', 3, 0),
      ('MOVE', 5, 0),
      ('OP', 3, 0),
      ('TUPLE_CONSTRUCT', 4, 0),
      ('RET', 0, 0))),
    ('operators',
      (('aten::div', ''),
      ('aten::div', 'float'),
      ('aten::div', ''),
      ('aten::div', 'int'))),
    ('constants', ()),
    ('types', ()),
    ('register_size', 5))),)
  */
  mobile::Module m_module = _load_for_mobile(test_model_file);
  auto intrsuction_list =
      m_module.get_method("forward").function().get_code().instructions_;
  uint64_t number_of_call_instruction = 0;
  for (auto& instruction : intrsuction_list) {
    number_of_call_instruction += (instruction.op == OpCode::CALL);
  }
  // No operator will use upgrader
  ASSERT_EQ(number_of_call_instruction, 0);

  std::vector<IValue> inputs{IValue(20.0), IValue(10), IValue(2.0), IValue(5)};
  auto output = m_module.forward(inputs);
  auto output_list = output.toTupleRef().elements();
  auto expect_output = std::vector<IValue>(
      {IValue(2.0), IValue(10.0), IValue(5.0), IValue(2.0)});
  // auto actual_output = output.toTensor();
  for (size_t i = 0; i < expect_output.size(); i++) {
    ASSERT_EQ(output_list[i], expect_output[i]);
  }
}

TEST(LiteInterpreterUpgraderTest, DivScalarIntV2) {
  std::string filePath(__FILE__);
  auto test_model_file = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  test_model_file.append(
      "upgrader_models/test_versioned_div_scalar_int_v2.ptl");
  /*
  (('__torch__.MyModuleInt.forward',
    (('instructions',
      (('STOREN', 1, 3),
      ('DROPR', 1, 0),
      ('MOVE', 2, 0),
      ('MOVE', 3, 0),
      ('OP', 0, 0),
      ('RET', 0, 0))),
    ('operators', (('aten::div', 'Scalar'),)),
    ('constants', ()),
    ('types', ()),
    ('register_size', 3))),)
  */
  mobile::Module m_module = _load_for_mobile(test_model_file);

  auto intrsuction_list =
      m_module.get_method("forward").function().get_code().instructions_;
  uint64_t number_of_call_instruction = 0;
  for (auto& instruction : intrsuction_list) {
    number_of_call_instruction += (instruction.op == OpCode::CALL);
  }
  // One operator will use upgrader
  ASSERT_EQ(number_of_call_instruction, 1);

  std::vector<IValue> inputs{IValue(6 * torch::ones({1})), IValue(3)};
  auto output = m_module.forward(inputs);
  auto expect_output = 2.0 * torch::ones({1});
  auto actual_output = output.toTensor();

  // The out argument will be overwritten with the output
  ASSERT_TRUE(actual_output.equal(expect_output));
}

TEST(LiteInterpreterUpgraderTest, DivScalarInplaceFloatV2) {
  std::string filePath(__FILE__);
  auto test_model_file = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  test_model_file.append(
      "upgrader_models/test_versioned_div_scalar_inplace_float_v2.ptl");
  /*
  (('__torch__.MyModuleFloat.forward',
    (('instructions',
      (('STOREN', 1, 3),
      ('DROPR', 1, 0),
      ('MOVE', 2, 0),
      ('MOVE', 3, 0),
      ('OP', 0, 0),
      ('RET', 0, 0))),
    ('operators', (('aten::div_', 'Scalar'),)),
    ('constants', ()),
    ('types', ()),
    ('register_size', 3))),)
  */

  mobile::Module m_module = _load_for_mobile(test_model_file);

  auto intrsuction_list =
      m_module.get_method("forward").function().get_code().instructions_;
  uint64_t number_of_call_instruction = 0;
  for (auto& instruction : intrsuction_list) {
    number_of_call_instruction += (instruction.op == OpCode::CALL);
  }
  // One operator will use upgrader
  ASSERT_EQ(number_of_call_instruction, 1);

  std::vector<IValue> inputs{IValue(6 * torch::ones({1})), IValue(3.0)};
  auto output = m_module.forward(inputs);
  auto expect_output = 2.0 * torch::ones({1});
  auto actual_output = output.toTensor();

  // The out argument will be overwritten with the output
  ASSERT_TRUE(actual_output.equal(expect_output));
}

TEST(LiteInterpreterUpgraderTest, DivScalarInplaceIntV2) {
  std::string filePath(__FILE__);
  auto test_model_file = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  test_model_file.append(
      "upgrader_models/test_versioned_div_scalar_inplace_int_v2.ptl");
  /*
  (('__torch__.MyModuleInt.forward',
    (('instructions',
      (('STOREN', 1, 3),
       ('DROPR', 1, 0),
       ('MOVE', 2, 0),
       ('MOVE', 3, 0),
       ('OP', 0, 0),
       ('RET', 0, 0))),
     ('operators', (('aten::div_', 'Scalar'),)),
     ('constants', ()),
     ('types', ()),
     ('register_size', 3))),)
  */

  mobile::Module m_module = _load_for_mobile(test_model_file);

  auto intrsuction_list =
      m_module.get_method("forward").function().get_code().instructions_;
  uint64_t number_of_call_instruction = 0;
  for (auto& instruction : intrsuction_list) {
    number_of_call_instruction += (instruction.op == OpCode::CALL);
  }
  // One operator will use upgrader
  ASSERT_EQ(number_of_call_instruction, 1);

  std::vector<IValue> inputs{IValue(6 * torch::ones({1})), IValue(3)};
  auto output = m_module.forward(inputs);
  auto expect_output = 2.0 * torch::ones({1});
  auto actual_output = output.toTensor();

  // The out argument will be overwritten with the output
  ASSERT_TRUE(actual_output.equal(expect_output));
}

#endif // !defined(FB_XPLAT_BUILD)

TEST(LiteInterpreterUpgraderTest, Upgrader) {
  std::vector<mobile::Function> upgrader_functions;

  for (auto& byteCodeFunctionWithOperator : getUpgraderBytecodeList()) {
    byteCodeFunctionWithOperator.function.initialize_operators(true);
    ASSERT_EQ(
        byteCodeFunctionWithOperator.function.get_code().operators_.size(),
        byteCodeFunctionWithOperator.function.get_code().op_names_.size());
    if (byteCodeFunctionWithOperator.function.get_code().operators_.empty()) {
      for (const auto& op : byteCodeFunctionWithOperator.operators) {
        byteCodeFunctionWithOperator.function.append_operator(
            op.name, op.overload_name, op.num_specified_args);
      }
    }
    upgrader_functions.push_back(byteCodeFunctionWithOperator.function);
  }

  ASSERT_EQ(getUpgraderBytecodeList().size(), upgrader_functions.size());
}

void enumerateTupleType(
    size_t depth,
    std::vector<TypePtr>& current,
    const std::vector<TypePtr>& candidates,
    std::vector<TypePtr>& out) {
  static std::vector<std::string> fieldNames;
  if (depth > fieldNames.size()) {
    fieldNames.reserve(depth);
    for (size_t i = fieldNames.size(); i < depth; i++) {
      fieldNames.push_back("field" + std::to_string(i));
    }
  }
  if (depth == 0) {
    out.push_back(TupleType::create(current));
    while (fieldNames.size() > current.size()) {
      fieldNames.pop_back();
    }
    out.push_back(TupleType::createNamed("NamedTuple", fieldNames, current));
    return;
  }
  for (const auto& type : candidates) {
    if (containsAnyType(type)) {
      continue;
    }
    current.push_back(type);
    enumerateTupleType(depth - 1, current, candidates, out);
    current.pop_back();
  }
}

class LiteInterpreterDynamicTypeTestFixture
    : public ::testing::TestWithParam<size_t> {
 protected:
  void SetUp() override {
    cu = std::make_shared<CompilationUnit>();
    std::vector<TypePtr> keyTypes = {
        AnyType::get(),
        IntType::get(),
        BoolType::get(),
        FloatType::get(),
        ComplexType::get(),
        StringType::get(),
        TensorType::get(),
        DeviceObjType::get(),
    };
    types = {
        NoneType::get(),
        NumberType::get(),
        ClassType::create("__torch__.TestClass1", cu),
        ClassType::create("__torch__.TestClass2", cu),
        AnyListType::get(),
        AnyTupleType::get(),
        StreamObjType::get(),
        CapsuleType::get(),
        GeneratorType::get(),
        StorageType::get(),
        VarType::create("t"),
        VarType::create("v"),
        AnyClassType::get()};
    std::copy(keyTypes.begin(), keyTypes.end(), back_inserter(types));
    auto expandTypes = [&](size_t tupleSize) {
      std::vector<TypePtr> nested;
      for (const auto& type : types) {
        if (!(type == AnyType::get())) {
          nested.emplace_back(ListType::create(type));
          if (!(type == NoneType::get() ||
                type->kind() == OptionalType::Kind)) {
            nested.emplace_back(OptionalType::create(type));
          }
        }
        for (const auto& keyType : keyTypes) {
          nested.emplace_back(DictType::create(keyType, type));
        }
      }
      std::vector<TypePtr> tmp;
      enumerateTupleType(tupleSize, tmp, types, nested);
      std::move(
          std::begin(nested), std::end(nested), std::back_inserter(types));
    };
    expandTypes(1);
    expandTypes(1);
  }
  std::shared_ptr<CompilationUnit> cu;
  std::vector<TypePtr> types;

 public:
  static constexpr size_t kNumSplits = 10;
};

/**
 * Enumerate all possible JIT types appearing in mobile runtime, and test
 * whether subtyping relation is preserved after one of the JIT types is
 * converted to DynamicType.
 *
 * We firstly enumerate all "base" types in a vector, and implement
 * expandTypes() to enumerate container types one "level" up for a given set
 * of types. We call expandTypes() twice to test types nested less or equal
 * to two levels. e.g. List[Optional[Tensor]], Optional[Dict[Int, Bool]], etc.
 */
TEST_P(LiteInterpreterDynamicTypeTestFixture, Conformance) {
  size_t num = types.size() / LiteInterpreterDynamicTypeTestFixture::kNumSplits;
  size_t begin = num * GetParam();
  size_t end = std::min(types.size(), begin + num);
  for (const auto& a : types) {
    auto da = DynamicType::create(*a);
    for (size_t i = begin; i < end; i++) {
      const auto& b = types[i];
      bool result = a->isSubtypeOf(*b);
      EXPECT_EQ(result, da->isSubtypeOf(*b));
      result = b->isSubtypeOf(*a);
      EXPECT_EQ(result, b->isSubtypeOf(*da));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    PyTorch,
    LiteInterpreterDynamicTypeTestFixture,
    ::testing::Range(
        static_cast<size_t>(0),
        LiteInterpreterDynamicTypeTestFixture::kNumSplits));

} // namespace jit
} // namespace torch
