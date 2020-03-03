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
        schema_(std::move(schema)) {}

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
    TORCH_INTERNAL_ASSERT(false);
  }

  std::shared_ptr<Graph> optimized_graph() const override {
    TORCH_INTERNAL_ASSERT(false);
  }

  GraphExecutor& get_executor() override {
    TORCH_INTERNAL_ASSERT(false);
  }

  const c10::FunctionSchema& getSchema() const override {
    if (!adapted_schema_) {
      std::unique_lock<std::mutex> lk(adapted_schema_mutex_);
      if (schema_.returns().size() == 0) {
        adapted_schema_ = schema_.cloneWithReturns(
            {c10::Argument("", c10::NoneType::get())});
      } else if (schema_.returns().size() > 1) {
        std::vector<c10::TypePtr> return_types;
        for (const auto& arg : schema_.returns()) {
          return_types.emplace_back(arg.type());
        }
        adapted_schema_ = schema_.cloneWithReturns({c10::Argument(
            "", c10::TupleType::create(std::move(return_types)))});
      } else {
        adapted_schema_ = schema_;
      }
    }
    return *adapted_schema_;
  }

  size_t num_inputs() const override {
    TORCH_INTERNAL_ASSERT(false);
  }

  void check_single_output() override {
    TORCH_INTERNAL_ASSERT(false);
  }

  std::string pretty_print_schema() const override {
    TORCH_INTERNAL_ASSERT(false);
    std::stringstream ss;
    ss << getSchema();
    return ss.str();
  }

  Function& setSchema(c10::FunctionSchema schema) override {
    schema_ = std::move(schema);
    std::unique_lock<std::mutex> lk(adapted_schema_mutex_);
    adapted_schema_ = c10::nullopt;
    return *this;
  }

  ~BuiltinOpFunction() {}

 private:
  c10::QualifiedName name_;

  std::function<void(Stack&)> callable_;

  c10::FunctionSchema schema_;

  // We maintain a separate schema with the return values adapted to
  // the Method calling convention (single return value with None, single
  // value, or Tuple for 0, 1, or 2+ returns, respectively)
  mutable std::mutex adapted_schema_mutex_;
  mutable c10::optional<c10::FunctionSchema> adapted_schema_;
};

} // namespace jit
} // namespace torch
