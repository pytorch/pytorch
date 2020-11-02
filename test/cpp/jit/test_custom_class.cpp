#include <gtest/gtest.h>

#include <test/cpp/jit/test_custom_class_registrations.h>
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

} // namespace jit
} // namespace torch
