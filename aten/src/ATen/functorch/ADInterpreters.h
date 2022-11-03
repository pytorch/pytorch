#pragma once
#include <ATen/functorch/Interpreter.h>

namespace at { namespace functorch {

// These are the interpreters for our AD transforms
// (grad, vjp and jvp).
// See NOTE: [functorch interpreter stack] for more details.

struct GradInterpreterPtr {
  explicit GradInterpreterPtr(const Interpreter* base): base_(base) { TORCH_INTERNAL_ASSERT(base->key() == TransformType::Grad); }
  TransformType key() const { return base_->key(); }
  int64_t level() const { return base_->level(); }
  void processImpl(const c10::OperatorHandle& op, torch::jit::Stack* stack);
  void sendToNextInterpreterImpl(const c10::OperatorHandle& op, torch::jit::Stack* stack, bool grad_special_case);
  bool prevGradMode() const {
    return c10::get<GradInterpreterMeta>(base_->meta()).prevGradMode_;
  }
 private:
  const Interpreter* base_;
};

struct JvpInterpreterPtr {
  explicit JvpInterpreterPtr(const Interpreter* base): base_(base) { TORCH_INTERNAL_ASSERT(base->key() == TransformType::Jvp); }
  TransformType key() const { return base_->key(); }
  int64_t level() const { return base_->level(); }
  void processImpl(const c10::OperatorHandle& op, torch::jit::Stack* stack);
  void sendToNextInterpreterImpl(const c10::OperatorHandle& op, torch::jit::Stack* stack, bool grad_special_case);
  bool prevFwdGradMode() const {
    return c10::get<JvpInterpreterMeta>(base_->meta()).prevFwdGradMode_;
  }
 private:
  const Interpreter* base_;
};

}} // namespace at::functorch
