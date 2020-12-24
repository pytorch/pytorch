#include <gtest/gtest.h>

#include <test/cpp/jit/test_custom_class_registrations.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/custom_class.h>
#include <torch/script.h>

#include <iostream>
#include <string>
#include <vector>

namespace torch {
namespace jit {

TEST(CustomClassTest, TorchbindIValueAPI) {
  script::Module m("m");

  // test make_custom_class API
  auto custom_class_obj = make_custom_class<MyStackClass<std::string>>(
      std::vector<std::string>{"foo", "bar"});
  m.define(R"(
    def forward(self, s : __torch__.torch.classes._TorchScriptTesting._StackString):
      return s.pop(), s
  )");

  auto test_with_obj = [&m](IValue obj, std::string expected) {
    auto res = m.run_method("forward", obj);
    auto tup = res.toTuple();
    AT_ASSERT(tup->elements().size() == 2);
    auto str = tup->elements()[0].toStringRef();
    auto other_obj =
        tup->elements()[1].toCustomClass<MyStackClass<std::string>>();
    AT_ASSERT(str == expected);
    auto ref_obj = obj.toCustomClass<MyStackClass<std::string>>();
    AT_ASSERT(other_obj.get() == ref_obj.get());
  };

  test_with_obj(custom_class_obj, "bar");

  // test IValue() API
  auto my_new_stack = c10::make_intrusive<MyStackClass<std::string>>(
      std::vector<std::string>{"baz", "boo"});
  auto new_stack_ivalue = c10::IValue(my_new_stack);

  test_with_obj(new_stack_ivalue, "boo");
}

class TorchBindTestClass : public torch::jit::CustomClassHolder {
 public:
  std::string get() {
    return "Hello, I am your test custom class";
  }
};

constexpr char class_doc_string[] = R"(
  I am docstring for TorchBindTestClass
  Args:
      What is an argument? Oh never mind, I don't take any.

  Return:
      How would I know? I am just a holder of some meaningless test methods.
  )";
constexpr char method_doc_string[] =
    "I am docstring for TorchBindTestClass get_with_docstring method";

namespace {
static auto reg =
    torch::class_<TorchBindTestClass>(
        "_TorchBindTest",
        "_TorchBindTestClass",
        class_doc_string)
        .def("get", &TorchBindTestClass::get)
        .def("get_with_docstring", &TorchBindTestClass::get, method_doc_string);

} // namespace

// Tests DocString is properly propagated when defining CustomClasses.
TEST(CustomClassTest, TestDocString) {
  auto class_type = getCustomClass(
      "__torch__.torch.classes._TorchBindTest._TorchBindTestClass");
  AT_ASSERT(class_type);
  AT_ASSERT(class_type->doc_string() == class_doc_string);

  AT_ASSERT(class_type->getMethod("get").doc_string().empty());
  AT_ASSERT(
      class_type->getMethod("get_with_docstring").doc_string() ==
      method_doc_string);
}

TEST(CustomClassTest, Serialization) {
  script::Module m("m");

  // test make_custom_class API
  auto custom_class_obj = make_custom_class<MyStackClass<std::string>>(
      std::vector<std::string>{"foo", "bar"});
  m.register_attribute(
      "s",
      custom_class_obj.type(),
      custom_class_obj,
      /*is_parameter=*/false);
  m.define(R"(
    def forward(self):
      return self.s.return_a_tuple()
  )");

  auto test_with_obj = [](script::Module& mod) {
    auto res = mod.run_method("forward");
    auto tup = res.toTuple();
    AT_ASSERT(tup->elements().size() == 2);
    auto i = tup->elements()[1].toInt();
    AT_ASSERT(i == 123);
  };

  auto frozen_m = torch::jit::freeze_module(m.clone());

  test_with_obj(m);
  test_with_obj(frozen_m);

  std::ostringstream oss;
  m.save(oss);
  std::istringstream iss(oss.str());
  caffe2::serialize::IStreamAdapter adapter{&iss};
  auto loaded_module = torch::jit::load(iss, torch::kCPU);

  std::ostringstream oss_frozen;
  frozen_m.save(oss_frozen);
  std::istringstream iss_frozen(oss_frozen.str());
  caffe2::serialize::IStreamAdapter adapter_frozen{&iss_frozen};
  auto loaded_frozen_module = torch::jit::load(iss_frozen, torch::kCPU);
}

} // namespace jit
} // namespace torch
