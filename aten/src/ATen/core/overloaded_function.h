#pragma once

#include <ATen/core/function.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <c10/util/Exception.h>
#include <c10/util/intrusive_ptr.h>
#include <pybind11/detail/common.h>
#include <pybind11/pytypes.h>
#include <torch/csrc/jit/frontend/error_report.h>
#include <functional>
#include <utility>

namespace c10 {
struct FunctionSchema;
struct IValue;
}; // namespace c10

namespace torch {
namespace jit {

struct SourceRange;

struct OverloadedFunction : public Function {
  OverloadedFunction(
      c10::QualifiedName qualname,
      c10::FunctionSchema schema,
      std::function<void(Stack&)> callable,
      std::string doc_string = "")
      : name_(std::move(qualname)),
        callable_(std::move(callable)),
        schema_(std::move(schema)),
        doc_string_(std::move(doc_string)) {
    TORCH_INTERNAL_ASSERT(schema_.returns().size() == 1);
  }

  const std::string& doc_string() const override {
    return doc_string_;
  }

  bool isGraphFunction() const override {
    return false;
  }

  void run(Stack& stack) override {
    callable_(stack);
  }

  void run(Stack&& stack) override {
    callable_(stack);
  }

  std::vector<const at::FunctionSchema*> loadPossibleSchemas(
      const at::ClassTypePtr& owner_class,
      const SourceRange& loc) {
    auto overloadedMethods = owner_class->findOverloadedMethod(name());
    std::vector<const at::FunctionSchema*> schemas;

    for (auto method : overloadedMethods) {
      try {
        method->ensure_defined();
      } catch (const RecursiveMethodCallError&) {
        throw ErrorReport(loc)
            << " method '" << method->name() << "' is called recursively. "
            << "Recursive calls are not supported";
      }
      schemas.push_back(&(method->getSchema()));
    }
    return schemas;
  }

  c10::intrusive_ptr<c10::ivalue::Future> runAsync(
      Stack& stack,
      TaskLauncher /* not used */) override {
    run(stack);
    auto res = c10::make_intrusive<c10::ivalue::Future>(stack.front().type());
    res->markCompleted(std::move(stack.front()));
    return res;
  }

  at::IValue operator()(std::vector<at::IValue> stack, const Kwargs& kwargs)
      override {
    getSchema().checkAndNormalizeInputs(stack, kwargs);
    callable_(stack);
    return stack.front();
  }

  const c10::QualifiedName& qualname() const override {
    return name_;
  }

  const std::string& name() const override {
    return name_.name();
  }

  // if this isn't yet defined, run its method_creator function
  void ensure_defined() override {
    // nop
  }

  std::shared_ptr<Graph> graph() const override {
    TORCH_INTERNAL_ASSERT(
        false,
        "OverloadedFunction had a graph requested "
        "from it. This probably indicates that the JIT calling context needs a "
        "special case on Function::isGraphFunction()");
  }

  std::shared_ptr<Graph> optimized_graph() const override {
    TORCH_INTERNAL_ASSERT(
        false,
        "OverloadedFunction had a graph requested "
        "from it. This probably indicates that the JIT calling context needs a "
        "special case on Function::isGraphFunction()");
  }

  void clear_execution_info() override {
    TORCH_INTERNAL_ASSERT(
        false,
        "OverloadedFunction had a graph requested "
        "from it. This probably indicates that the JIT calling context needs a "
        "special case on Function::isGraphFunction()");
  }

  GraphExecutor& get_executor() override {
    TORCH_INTERNAL_ASSERT(
        false,
        "OverloadedFunction had a GraphExecutor requested "
        "from it. This probably indicates that the JIT calling context needs a "
        "special case on Function::isGraphFunction()");
  }

  const c10::FunctionSchema& getSchema() const override {
    return schema_;
  }

  size_t num_inputs() const override {
    return schema_.arguments().size();
  }

  void check_single_output() override {
    TORCH_CHECK(schema_.returns().size() == 1);
  }

  std::string pretty_print_schema() const override {
#ifdef __NVCC__
// Disable the "statement is unreachable" warning
#pragma diag_suppress code_is_unreachable
#endif

    TORCH_INTERNAL_ASSERT(false);
    return "";

#ifdef __NVCC__
#pragma diag_default code_is_unreachable
#endif
  }

  Function& setSchema(c10::FunctionSchema schema) override {
    schema_ = std::move(schema);
    return *this;
  }

 private:
  c10::QualifiedName name_;

  std::function<void(Stack&)> callable_;

  c10::FunctionSchema schema_;

  std::string doc_string_;
};

} // namespace jit
} // namespace torch
