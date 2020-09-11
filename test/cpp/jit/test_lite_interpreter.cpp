#include <c10/core/TensorOptions.h>
#include <test/cpp/jit/test_base.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/custom_class.h>
#include <torch/torch.h>

#include <unordered_set>

// Tests go in torch::jit
namespace torch {
namespace jit {

void testLiteInterpreterUpsampleNearest2d() {
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

void testLiteInterpreterAdd() {
  Module m("m");
  m.register_parameter("foo", torch::ones({}), false);
  // TODO: support default param val, which was pushed in
  // function schema's checkAndNormalizeInputs()
  //  m.define(R"(
  //    def add_it(self, x, b : int = 4):
  //      return self.foo + x + b
  //  )");
  m.define(R"(
    def add_it(self, x):
      b = 4
      return self.foo + x + b
  )");

  std::vector<IValue> inputs;
  auto minput = 5 * torch::ones({});
  inputs.emplace_back(minput);
  auto ref = m.run_method("add_it", minput);

  std::stringstream ss;
  m._save_for_mobile(ss);
  mobile::Module bc = _load_for_mobile(ss);
  IValue res;
  for (int i = 0; i < 3; ++i) {
    auto bcinputs = inputs;
    res = bc.get_method("add_it")(bcinputs);
  }

  auto resd = res.toTensor().item<float>();
  auto refd = ref.toTensor().item<float>();
  AT_ASSERT(resd == refd);
}

void testLiteInterpreterConv() {
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

void testLiteInterpreterInline() {
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

void testLiteInterpreterTuple() {
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

void testLiteInterpreterDict() {
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

void testLiteInterpreterPrimOverload() {
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

void testLiteInterpreterPrim() {
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

void testLiteInterpreterLoadOrigJit() {
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

void testLiteInterpreterWrongMethodName() {
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

void testLiteInterpreterSetState() {
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

void testLiteInterpreterBuiltinFunction() {
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

void testLiteInterpreterModuleInfoBasic() {
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

void testLiteInterpreterNotSavingModuleInfo() {
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

void testLiteInterpreterOneSubmoduleModuleInfo() {
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

void testLiteInterpreterTwoSubmodulesModuleInfo() {
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

void testLiteInterpreterSequentialModuleInfo() {
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

  std::unordered_set<std::string> expected_result(
      {"top(C).A0(A).forward", "top(C).B0(B).forward"});
  AT_ASSERT(module_debug_info_set == expected_result);
}

void testLiteInterpreterHierarchyModuleInfo() {
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
  // "top(C).B0(B).A0(A).forward": for the add operator in A0.
  std::unordered_set<std::string> expected_result(
      {"top(C).forward", "top(C).B0(B).forward", "top(C).B0(B).A0(A).forward"});
  AT_ASSERT(module_debug_info_set == expected_result);
}

void testLiteInterpreterDuplicatedClassTypeModuleInfo() {
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

  // The current approach is not able to distinguish between A0 and A1,
  // which have the same class type. Hence, it only records module
  // information for A1.
  std::unordered_set<std::string> expected_result(
      {"top(B).forward", "top(B).A1(A).forward"});
  AT_ASSERT(module_debug_info_set == expected_result);
}

void testLiteInterpreterEval() {
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

void testLiteInterpreterFindWrongMethodName() {
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

void testLiteInterpreterFindAndRunMethod() {
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

namespace {
static auto reg =
    torch::class_<TorchBindLiteInterpreterTestStruct>(
        "_TorchScriptTesting",
        "_LiteInterpreterTest")
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
