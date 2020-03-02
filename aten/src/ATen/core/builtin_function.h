#pragma once

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/function.h>

namespace torch {
namespace jit {

struct BuiltinOpFunction : public Function {
  BuiltinOpFunction(c10::QualifiedName qualname, c10::Symbol op_symbol)
      : name_(std::move(qualname)), symbol_(std::move(op_symbol)) {}

  bool isGraphFunction() const override {
    return false;
  }

  void run(Stack& stack) override {
    auto handle = c10::Dispatcher::singleton().findSchemaOrThrow(
        symbol_.toQualString(), "");
    handle.callBoxed(&stack);
    if (handle.schema().returns().size() == 0) {
      stack.push_back(IValue());
    } else if (handle.schema().returns().size() > 1) {
      auto ivalue_tup = c10::ivalue::Tuple::create(stack);
      stack.clear();
      stack.emplace_back(std::move(ivalue_tup));
    }
  }

  void run(Stack&& stack) override {
    auto handle = c10::Dispatcher::singleton().findSchemaOrThrow(
        symbol_.toQualString(), "");
    handle.callBoxed(&stack);
  }

  at::IValue operator()(std::vector<at::IValue> stack, const Kwargs& kwargs)
      override {
    getSchema().checkAndNormalizeInputs(stack, kwargs);
    run(stack);
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
    if (!cached_schema_) {
      std::unique_lock<std::mutex> lk(schema_mutex_);
      auto handle = c10::Dispatcher::singleton().findSchemaOrThrow(
          symbol_.toQualString(), "");
      cached_schema_ = handle.schema();
      if (cached_schema_->returns().size() == 0) {
        cached_schema_ = cached_schema_->cloneWithReturns(
            {c10::Argument("", c10::NoneType::get())});
      } else if (cached_schema_->returns().size() > 1) {
        std::vector<c10::TypePtr> return_types;
        for (const auto& arg : cached_schema_->returns()) {
          return_types.emplace_back(arg.type());
        }
        cached_schema_ = cached_schema_->cloneWithReturns({c10::Argument(
            "", c10::TupleType::create(std::move(return_types)))});
      }
    }
    return *cached_schema_;
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
    TORCH_INTERNAL_ASSERT(false);
  }

  c10::Symbol op_symbol() const {
    return symbol_;
  }

  ~BuiltinOpFunction() {}

 private:
  c10::QualifiedName name_;

  c10::Symbol symbol_;

  mutable std::mutex schema_mutex_;
  mutable c10::optional<c10::FunctionSchema> cached_schema_;
};

} // namespace jit
} // namespace torch
