#pragma once

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/function.h>

namespace torch {
namespace jit {

struct BuiltinOpFunction : public Function {
  BuiltinOpFunction(
      c10::QualifiedName qualname,
      c10::FunctionSchema schema,
      std::function<void(Stack&)> callable)
      : name_(std::move(qualname)),
        callable_(std::move(callable)),
        schema_(std::move(schema)) {
    TORCH_INTERNAL_ASSERT(schema_.returns().size() == 1);
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
    TORCH_INTERNAL_ASSERT(false , "BuiltinFunction had a graph requested "
      "from it. This probably indicates that the JIT calling context needs a "
      "special case on Function::isGraphFunction()");
  }

  std::shared_ptr<Graph> optimized_graph() const override {
    TORCH_INTERNAL_ASSERT(false , "BuiltinFunction had a graph requested "
      "from it. This probably indicates that the JIT calling context needs a "
      "special case on Function::isGraphFunction()");
  }

  GraphExecutor& get_executor() override {
    TORCH_INTERNAL_ASSERT(false , "BuiltinFunction had a GraphExecutor requested "
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
    TORCH_INTERNAL_ASSERT(false);
    std::stringstream ss;
    ss << getSchema();
    return ss.str();
  }

  Function& setSchema(c10::FunctionSchema schema) override {
    schema_ = std::move(schema);
    return *this;
  }

  ~BuiltinOpFunction() {}

 private:
  c10::QualifiedName name_;

  std::function<void(Stack&)> callable_;

  c10::FunctionSchema schema_;
};

} // namespace jit
} // namespace torch
