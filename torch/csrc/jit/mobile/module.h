#pragma once
//#include <ATen/core/function_schema.h>
#include <torch/csrc/jit/mobile/function.h>
#include <torch/csrc/jit/mobile/method.h>

namespace torch {
namespace jit {
namespace mobile {
using Stack = std::vector<c10::IValue>;

class CompilationUnit {
 public:
  void register_function(std::unique_ptr<Function> fn);
  std::vector<std::unique_ptr<Function>>& methods() {
    return methods_;
  }
  Function* find_function(const c10::QualifiedName& qn);

 private:
  std::vector<std::unique_ptr<Function>> methods_;
};

class TORCH_API Module {
 public:
  Module(
      c10::intrusive_ptr<c10::ivalue::Object> object,
      std::shared_ptr<CompilationUnit> cu)
      : object_(object),
        metadata_(std::unordered_map<std::string, std::string>()),
        cu_(std::move(cu)) {}
  Module(
      c10::intrusive_ptr<c10::ivalue::Object> object,
      std::unordered_map<std::string, std::string> metadata,
      std::shared_ptr<CompilationUnit> cu)
      : object_(object), metadata_(std::move(metadata)), cu_(std::move(cu)) {}
  Module() = default;
  Method get_method(const std::string& method_name) const;
  template <typename... Types>
  c10::IValue run_method(const std::string& method_name, Types&&... args) {
    return get_method(method_name)({IValue(std::forward<Types>(args))...});
  }
  c10::IValue forward(std::vector<c10::IValue> inputs) {
    return get_method("forward")(std::move(inputs));
  }
  c10::optional<Method> find_method(const std::string& basename) const;
  const std::string name() const {
    return object_->name();
  }
  const std::vector<at::IValue>& slots() const {
    return object_->slots();
  }
  const c10::intrusive_ptr<c10::ivalue::Object> _ivalue() const {
    return object_;
  }
  const std::vector<at::Tensor> parameters() const;
  const std::map<std::string, at::Tensor> named_parameters() const;
  std::string get_forward_method_debug_info(size_t pc) const;
  /// Enables "training" mode.
  void train(bool on = true);
  /// Calls train(false) to enable "eval" mode.
  void eval() {
    train(/*on=*/false);
  }
  /// True if the module is in training mode.
  bool is_training() const;
  const std::unordered_map<std::string, std::string> metadata() const {
    return metadata_;
  }

  c10::IValue attr(const std::string& name, c10::IValue or_else) const {
    if (auto r = object_->type()->findAttributeSlot(name)) {
      return object_->getSlot(*r);
    }
    if (auto r = object_->type()->findConstantSlot(name)) {
      return object_->type()->getConstant(*r);
    }
    return or_else;
  }

 private:
  c10::intrusive_ptr<c10::ivalue::Object> object_;
  std::unordered_map<std::string, std::string> metadata_;
  std::shared_ptr<CompilationUnit> cu_;
};
} // namespace mobile
} // namespace jit
} // namespace torch
