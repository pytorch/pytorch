/*
 * We have a python unit test for exceptions in test/jit/test_exception.py .
 * Add a CPP version here to verify that excepted exception types thrown from
 * C++. This is hard to test in python code since C++ exceptions will be
 * translated to python exceptions.
 */
#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <torch/csrc/jit/frontend/parser.h>
#include <torch/csrc/jit/frontend/resolver.h>
#include <torch/csrc/jit/runtime/jit_exception.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/jit.h>
#include <iostream>
#include <stdexcept>

namespace torch {
namespace jit {

namespace py = pybind11;

TEST(TestException, TestAssertion) {
  std::string pythonCode = R"PY(
  def foo():
    raise AssertionError("An assertion failed")
  )PY";
  auto cu_ptr = torch::jit::compile(pythonCode);
  torch::jit::GraphFunction* gf =
      (torch::jit::GraphFunction*)&cu_ptr->get_function("foo");
  std::cerr << "Graph is\n" << *gf->graph() << std::endl;

  bool is_jit_exception = false;
  std::string message;
  c10::optional<std::string> exception_class;
  try {
    cu_ptr->run_method("foo");
  } catch (JITException& e) {
    is_jit_exception = true;
    message = e.what();
    exception_class = e.getPythonClassName();
  }
  EXPECT_TRUE(is_jit_exception);
  EXPECT_FALSE(exception_class);
  EXPECT_TRUE(
      message.find("RuntimeError: AssertionError: An assertion failed") !=
      std::string::npos);
}

struct MyPythonExceptionValue : public torch::jit::SugaredValue {
  explicit MyPythonExceptionValue(const py::object& exception_class) {
    qualified_name_ =
        (py::str(py::getattr(exception_class, "__module__", py::str(""))) +
         py::str(".") +
         py::str(py::getattr(exception_class, "__name__", py::str(""))))
            .cast<std::string>();
  }

  std::string kind() const override {
    return "My Python exception";
  }

  // Simplified from PythonExceptionValue::call
  std::shared_ptr<torch::jit::SugaredValue> call(
      const torch::jit::SourceRange& loc,
      torch::jit::GraphFunction& caller,
      at::ArrayRef<torch::jit::NamedValue> args,
      at::ArrayRef<torch::jit::NamedValue> kwargs,
      size_t n_binders) override {
    TORCH_CHECK(args.size() == 1);
    Value* error_message = args.at(0).value(*caller.graph());
    Value* qualified_class_name =
        insertConstant(*caller.graph(), qualified_name_, loc);
    return std::make_shared<ExceptionMessageValue>(
        error_message, qualified_class_name);
  }

 private:
  std::string qualified_name_;
};

class SimpleResolver : public torch::jit::Resolver {
 public:
  explicit SimpleResolver() {}

  std::shared_ptr<torch::jit::SugaredValue> resolveValue(
      const std::string& name,
      torch::jit::GraphFunction& m,
      const torch::jit::SourceRange& loc) override {
    // follows toSugaredValue (toSugaredValue is defined in caffe2:_C which is
    // a python extension. We can not add that as a cpp_binary's dep)
    if (name == "SimpleValueError") {
      py::object obj = py::globals()["SimpleValueError"];
      return std::make_shared<MyPythonExceptionValue>(obj);
    }
    TORCH_CHECK(false, "resolveValue: can not resolve '", name, "{}'");
  }

  torch::jit::TypePtr resolveType(
      const std::string& name,
      const torch::jit::SourceRange& loc) override {
    return nullptr;
  }
};

/*
 * - The python source code parsing for TorchScript here is learned from
 * torch::jit::compile.
 * - The code only parses one Def. If there are multiple in the code, those
 * except the first one are skipped.
 */
TEST(TestException, TestCustomException) {
  py::scoped_interpreter guard{};
  py::exec(R"PY(
  class SimpleValueError(ValueError):
    def __init__(self, message):
      super().__init__(message)
  )PY");

  std::string pythonCode = R"PY(
  def foo():
    raise SimpleValueError("An assertion failed")
  )PY";

  torch::jit::Parser p(
      std::make_shared<torch::jit::Source>(pythonCode, "<string>", 1));
  auto def = torch::jit::Def(p.parseFunction(/*is_method=*/false));
  std::cerr << "Def is:\n" << def << std::endl;
  auto cu = std::make_shared<torch::jit::CompilationUnit>();
  (void)cu->define(
      c10::nullopt,
      {},
      {},
      {def},
      // class PythonResolver is defined in
      // torch/csrc/jit/python/script_init.cpp. It's not in a header file so I
      // can not use it. Create a SimpleResolver instead
      {std::make_shared<SimpleResolver>()},
      nullptr);
  torch::jit::GraphFunction* gf =
      (torch::jit::GraphFunction*)&cu->get_function("foo");
  std::cerr << "Graph is\n" << *gf->graph() << std::endl;
  bool is_jit_exception = false;
  c10::optional<std::string> exception_class;
  std::string message;
  try {
    cu->run_method("foo");
  } catch (JITException& e) {
    is_jit_exception = true;
    exception_class = e.getPythonClassName();
    message = e.what();
  }
  EXPECT_TRUE(is_jit_exception);
  EXPECT_EQ("__main__.SimpleValueError", *exception_class);
  EXPECT_TRUE(
      message.find("__main__.SimpleValueError: An assertion failed") !=
      std::string::npos);
}

} // namespace jit
} // namespace torch
