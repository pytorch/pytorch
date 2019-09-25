#include "torch/csrc/autograd/VariableTypeUtils.h"

#include <ATen/TypeDefault.h>
#include <ATen/core/op_registration/op_registration.h>

// @generated from tools/autograd/templates/VariableType.cpp

// NOTE [Sharded File]: on this file's split-into-shards state
//
// Back in the good old days, VariableType.cpp was generated as one
// file with every function in it, and everything was great and
// simple.
//
// However, this file was also very large (over 36,000 lines), and
// compiling it was very slow, and in fact was a significant
// bottleneck for incremental rebuilds. To address this, we now
// generate the file split across multiple shards, named
// VariableType_0.cpp and so on, which can be compiled in parallel.
//
// For ease of inspection and debugging, so that it's not necessary to
// go rooting around in multiple files, we also generate all the
// functions together in VariableTypeEverything.cpp. This generated
// file is only for convenience; it's not actually used in the
// build. If the file you're looking at now is one of the shards, you
// may want to switch over to the Everything variant to make you
// grepping smoother.

using namespace at;
using namespace torch::autograd::generated;

namespace torch { namespace autograd {

Tensor & VariableType::__ilshift__(Tensor & self, Scalar other) {
  RECORD_FUNCTION("__ilshift__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__ilshift__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.__ilshift__(other);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::__ilshift__(Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("__ilshift__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__ilshift__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.__ilshift__(other_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::__ior__(Tensor & self, Scalar other) {
  RECORD_FUNCTION("__ior__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__ior__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.__ior__(other);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::__ior__(Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("__ior__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__ior__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.__ior__(other_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::__ixor__(Tensor & self, Scalar other) {
  RECORD_FUNCTION("__ixor__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__ixor__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.__ixor__(other);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::__ixor__(Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("__ixor__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__ixor__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.__ixor__(other_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::__lshift__(const Tensor & self, Scalar other) {
  RECORD_FUNCTION("__lshift__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__lshift__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::__lshift__(self_, other);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::__lshift__(const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("__lshift__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__lshift__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::__lshift__(self_, other_);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::__or__(const Tensor & self, Scalar other) {
  RECORD_FUNCTION("__or__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__or__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::__or__(self_, other);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::__or__(const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("__or__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__or__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::__or__(self_, other_);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::__xor__(const Tensor & self, Scalar other) {
  RECORD_FUNCTION("__xor__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__xor__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::__xor__(self_, other);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::__xor__(const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("__xor__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__xor__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::__xor__(self_, other_);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_addr_out(Tensor & out, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) {
  RECORD_FUNCTION("_addr_out", std::vector<c10::IValue>({out, self, vec1, vec2, beta, alpha}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& vec1_ = unpack(vec1, "vec1", 2);
  auto& vec2_ = unpack(vec2, "vec2", 3);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, vec1, vec2 )) {
    throw_error_out_requires_grad("_addr");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("_addr");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_addr");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "vec1", vec1);
    jit::tracer::addInputs(node, "vec2", vec2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_addr_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> vec1__storage_saved =
    vec1_.has_storage() ? c10::optional<Storage>(vec1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> vec1__impl_saved;
  if (vec1_.defined()) vec1__impl_saved = vec1_.getIntrusivePtr();
  c10::optional<Storage> vec2__storage_saved =
    vec2_.has_storage() ? c10::optional<Storage>(vec2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> vec2__impl_saved;
  if (vec2_.defined()) vec2__impl_saved = vec2_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::_addr_out(out_, self_, vec1_, vec2_, beta, alpha);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (vec1__storage_saved.has_value())
    AT_ASSERT(vec1__storage_saved.value().is_alias_of(vec1_.storage()));
  if (vec1__impl_saved) AT_ASSERT(vec1__impl_saved == vec1_.getIntrusivePtr());
  if (vec2__storage_saved.has_value())
    AT_ASSERT(vec2__storage_saved.value().is_alias_of(vec2_.storage()));
  if (vec2__impl_saved) AT_ASSERT(vec2__impl_saved == vec2_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::_baddbmm_mkl_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
  RECORD_FUNCTION("_baddbmm_mkl_", std::vector<c10::IValue>({self, batch1, batch2, beta, alpha}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_baddbmm_mkl");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_baddbmm_mkl_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "batch1", batch1);
    jit::tracer::addInputs(node, "batch2", batch2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_baddbmm_mkl_", self);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::_baddbmm_mkl_(self, batch1, batch2, beta, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::_cast_Double(const Tensor & self, bool non_blocking) {
  RECORD_FUNCTION("_cast_Double", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cast_Double");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::_cast_Double(self, non_blocking);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_cast_Short(const Tensor & self, bool non_blocking) {
  RECORD_FUNCTION("_cast_Short", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cast_Short");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::_cast_Short(self, non_blocking);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> VariableType::_cudnn_rnn(const Tensor & input, TensorList weight, int64_t weight_stride0, const Tensor & weight_buf, const Tensor & hx, const Tensor & cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntArrayRef batch_sizes, const Tensor & dropout_state) {
  RECORD_FUNCTION("_cudnn_rnn", std::vector<c10::IValue>({input, weight_buf, hx, cx, dropout_state}), Node::peek_at_next_sequence_nr());
  auto& input_ = unpack(input, "input", 0);
  auto weight_ = unpack(weight, "weight", 1);
  auto weight_buf_ = unpack_opt(weight_buf, "weight_buf", 3);
  auto& hx_ = unpack(hx, "hx", 4);
  auto cx_ = unpack_opt(cx, "cx", 5);
  auto dropout_state_ = unpack_opt(dropout_state, "dropout_state", 14);
  check_no_requires_grad(weight_buf, "weight_buf");
  std::shared_ptr<CudnnRnnBackward> grad_fn;
  if (compute_requires_grad( input, weight, hx, cx )) {
    grad_fn = std::shared_ptr<CudnnRnnBackward>(new CudnnRnnBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, weight, hx, cx ));
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->weight_ = make_saved_variable_list(weight);
    grad_fn->weight_stride0 = weight_stride0;
    grad_fn->hx_ = SavedVariable(hx, false);
    grad_fn->cx_ = SavedVariable(cx, false);
    grad_fn->mode = mode;
    grad_fn->hidden_size = hidden_size;
    grad_fn->num_layers = num_layers;
    grad_fn->batch_first = batch_first;
    grad_fn->dropout = dropout;
    grad_fn->train = train;
    grad_fn->bidirectional = bidirectional;
    grad_fn->batch_sizes = batch_sizes.vec();
    grad_fn->dropout_state_ = SavedVariable(dropout_state, false);
    grad_fn->weight_size_ = weight.size();
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  Tensor result3;
  Tensor result4;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cudnn_rnn");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "weight_stride0", weight_stride0);
    jit::tracer::addInputs(node, "weight_buf", weight_buf);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "cx", cx);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "hidden_size", hidden_size);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    jit::tracer::addInputs(node, "batch_sizes", batch_sizes);
    jit::tracer::addInputs(node, "dropout_state", dropout_state);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  std::vector<c10::optional<Storage>> weight__storage_saved(weight_.size());
  for (Tensor tensor : weight_)
    weight__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> weight__impl_saved(weight_.size());
  for (size_t i=0; i<weight_.size(); i++)
    if (weight_[i].defined()) weight__impl_saved[i] = weight_[i].getIntrusivePtr();
  c10::optional<Storage> weight_buf__storage_saved =
    weight_buf_.has_storage() ? c10::optional<Storage>(weight_buf_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight_buf__impl_saved;
  if (weight_buf_.defined()) weight_buf__impl_saved = weight_buf_.getIntrusivePtr();
  c10::optional<Storage> hx__storage_saved =
    hx_.has_storage() ? c10::optional<Storage>(hx_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> hx__impl_saved;
  if (hx_.defined()) hx__impl_saved = hx_.getIntrusivePtr();
  c10::optional<Storage> cx__storage_saved =
    cx_.has_storage() ? c10::optional<Storage>(cx_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> cx__impl_saved;
  if (cx_.defined()) cx__impl_saved = cx_.getIntrusivePtr();
  c10::optional<Storage> dropout_state__storage_saved =
    dropout_state_.has_storage() ? c10::optional<Storage>(dropout_state_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> dropout_state__impl_saved;
  if (dropout_state_.defined()) dropout_state__impl_saved = dropout_state_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_cudnn_rnn(input_, weight_, weight_stride0, weight_buf_, hx_, cx_, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state_);
  })();
  std::tie(result0, result1, result2, result3, result4) = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  for (size_t i=0; i<weight_.size(); i++) {
    if (weight__storage_saved[i].has_value())
      AT_ASSERT(weight__storage_saved[i].value().is_alias_of(weight_[i].storage()));
  }
  for (size_t i=0; i<weight_.size(); i++) {
    if (weight__impl_saved[i])
      AT_ASSERT(weight__impl_saved[i] == weight_[i].getIntrusivePtr());
  }
  if (weight_buf__storage_saved.has_value())
    AT_ASSERT(weight_buf__storage_saved.value().is_alias_of(weight_buf_.storage()));
  if (weight_buf__impl_saved) AT_ASSERT(weight_buf__impl_saved == weight_buf_.getIntrusivePtr());
  if (hx__storage_saved.has_value())
    AT_ASSERT(hx__storage_saved.value().is_alias_of(hx_.storage()));
  if (hx__impl_saved) AT_ASSERT(hx__impl_saved == hx_.getIntrusivePtr());
  if (cx__storage_saved.has_value())
    AT_ASSERT(cx__storage_saved.value().is_alias_of(cx_.storage()));
  if (cx__impl_saved) AT_ASSERT(cx__impl_saved == cx_.getIntrusivePtr());
  if (dropout_state__storage_saved.has_value())
    AT_ASSERT(dropout_state__storage_saved.value().is_alias_of(dropout_state_.storage()));
  if (dropout_state__impl_saved) AT_ASSERT(dropout_state__impl_saved == dropout_state_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1, result2 ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
    jit::tracer::addOutput(node, result4);
  }
  if (grad_fn) {
    grad_fn->result0_ = SavedVariable(result0, true);
    grad_fn->result3_ = SavedVariable(result3, true);
    grad_fn->result4_ = SavedVariable(result4, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3), std::move(result4));
}
Tensor & VariableType::_cumsum_out(Tensor & out, const Tensor & self, int64_t dim) {
  RECORD_FUNCTION("_cumsum_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_cumsum");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("_cumsum");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cumsum");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_cumsum_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::_cumsum_out(out_, self_, dim);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
int64_t VariableType::_dimV(const Tensor & self) {
  RECORD_FUNCTION("_dimV", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return self_._dimV();
  })();
  auto result = tmp;
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  return result;
}
std::tuple<Tensor,Tensor,Tensor,Tensor> VariableType::_embedding_bag(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const Tensor & per_sample_weights) {
  RECORD_FUNCTION("_embedding_bag", std::vector<c10::IValue>({weight, indices, offsets, per_sample_weights}), Node::peek_at_next_sequence_nr());
  auto& weight_ = unpack(weight, "weight", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& offsets_ = unpack(offsets, "offsets", 2);
  auto per_sample_weights_ = unpack_opt(per_sample_weights, "per_sample_weights", 6);
  std::shared_ptr<EmbeddingBagBackward> grad_fn;
  if (compute_requires_grad( weight, per_sample_weights )) {
    grad_fn = std::shared_ptr<EmbeddingBagBackward>(new EmbeddingBagBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( weight, per_sample_weights ));
    if (grad_fn->should_compute_output(1)) {
      grad_fn->weight_ = SavedVariable(weight, false);
    }
    grad_fn->indices_ = SavedVariable(indices, false);
    grad_fn->offsets_ = SavedVariable(offsets, false);
    grad_fn->mode = mode;
    grad_fn->weight_argsize_0 = weight.size(0);
    grad_fn->scale_grad_by_freq = scale_grad_by_freq;
    grad_fn->sparse = sparse;
    if (grad_fn->should_compute_output(0)) {
      grad_fn->per_sample_weights_ = SavedVariable(per_sample_weights, false);
    }
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  Tensor result3;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_embedding_bag");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "offsets", offsets);
    jit::tracer::addInputs(node, "scale_grad_by_freq", scale_grad_by_freq);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "sparse", sparse);
    jit::tracer::addInputs(node, "per_sample_weights", per_sample_weights);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  c10::optional<Storage> offsets__storage_saved =
    offsets_.has_storage() ? c10::optional<Storage>(offsets_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> offsets__impl_saved;
  if (offsets_.defined()) offsets__impl_saved = offsets_.getIntrusivePtr();
  c10::optional<Storage> per_sample_weights__storage_saved =
    per_sample_weights_.has_storage() ? c10::optional<Storage>(per_sample_weights_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> per_sample_weights__impl_saved;
  if (per_sample_weights_.defined()) per_sample_weights__impl_saved = per_sample_weights_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_embedding_bag(weight_, indices_, offsets_, scale_grad_by_freq, mode, sparse, per_sample_weights_);
  })();
  std::tie(result0, result1, result2, result3) = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  if (offsets__storage_saved.has_value())
    AT_ASSERT(offsets__storage_saved.value().is_alias_of(offsets_.storage()));
  if (offsets__impl_saved) AT_ASSERT(offsets__impl_saved == offsets_.getIntrusivePtr());
  if (per_sample_weights__storage_saved.has_value())
    AT_ASSERT(per_sample_weights__storage_saved.value().is_alias_of(per_sample_weights_.storage()));
  if (per_sample_weights__impl_saved) AT_ASSERT(per_sample_weights__impl_saved == per_sample_weights_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0 ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
  }
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
    grad_fn->result2_ = SavedVariable(result2, true);
    grad_fn->result3_ = SavedVariable(result3, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3));
}
Tensor VariableType::_embedding_bag_sparse_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, const Tensor & per_sample_weights) {
  RECORD_FUNCTION("_embedding_bag_sparse_backward", std::vector<c10::IValue>({grad, indices, offsets, offset2bag, bag_size, per_sample_weights}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_embedding_bag_sparse_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "offsets", offsets);
    jit::tracer::addInputs(node, "offset2bag", offset2bag);
    jit::tracer::addInputs(node, "bag_size", bag_size);
    jit::tracer::addInputs(node, "num_weights", num_weights);
    jit::tracer::addInputs(node, "scale_grad_by_freq", scale_grad_by_freq);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "per_sample_weights", per_sample_weights);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::_embedding_bag_sparse_backward(grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode, per_sample_weights);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_gather_sparse_backward(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & grad) {
  RECORD_FUNCTION("_gather_sparse_backward", std::vector<c10::IValue>({self, index, grad}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_gather_sparse_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "grad", grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::_gather_sparse_backward(self, dim, index, grad);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_index_put_impl_(Tensor & self, TensorList indices, const Tensor & values, bool accumulate, bool unsafe) {
  RECORD_FUNCTION("_index_put_impl_", std::vector<c10::IValue>({self, values}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto indices_ = unpack(indices, "indices", 1);
  auto& values_ = unpack(values, "values", 2);
  check_inplace(self);
  check_no_requires_grad(indices, "indices");
  std::shared_ptr<IndexPutImplBackward> grad_fn;
  if (compute_requires_grad( self, values )) {
    grad_fn = std::shared_ptr<IndexPutImplBackward>(new IndexPutImplBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, values ));
    grad_fn->indices_ = make_saved_variable_list(indices);
    grad_fn->values_info = values;
    grad_fn->accumulate = accumulate;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_index_put_impl");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_index_put_impl_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices, true);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "accumulate", accumulate);
    jit::tracer::addInputs(node, "unsafe", unsafe);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_index_put_impl_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  std::vector<c10::optional<Storage>> indices__storage_saved(indices_.size());
  for (Tensor tensor : indices_)
    indices__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> indices__impl_saved(indices_.size());
  for (size_t i=0; i<indices_.size(); i++)
    if (indices_[i].defined()) indices__impl_saved[i] = indices_[i].getIntrusivePtr();
  c10::optional<Storage> values__storage_saved =
    values_.has_storage() ? c10::optional<Storage>(values_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> values__impl_saved;
  if (values_.defined()) values__impl_saved = values_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::_index_put_impl_(self_, indices_, values_, accumulate, unsafe);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  for (size_t i=0; i<indices_.size(); i++) {
    if (indices__storage_saved[i].has_value())
      AT_ASSERT(indices__storage_saved[i].value().is_alias_of(indices_[i].storage()));
  }
  for (size_t i=0; i<indices_.size(); i++) {
    if (indices__impl_saved[i])
      AT_ASSERT(indices__impl_saved[i] == indices_[i].getIntrusivePtr());
  }
  if (values__storage_saved.has_value())
    AT_ASSERT(values__storage_saved.value().is_alias_of(values_.storage()));
  if (values__impl_saved) AT_ASSERT(values__impl_saved == values_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::_indices(const Tensor & self) {
  RECORD_FUNCTION("_indices", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_indices");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return self_._indices();
  })();
  auto result = as_view(self, tmp, false);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Scalar VariableType::_local_scalar_dense(const Tensor & self) {
  RECORD_FUNCTION("_local_scalar_dense", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_local_scalar_dense(self_);
  })();
  auto result = tmp;
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  return result;
}
std::tuple<Tensor &,Tensor &> VariableType::_max_out(Tensor & max, Tensor & max_indices, const Tensor & self, int64_t dim, bool keepdim) {
  RECORD_FUNCTION("_max_out", std::vector<c10::IValue>({max, max_indices, self}), Node::peek_at_next_sequence_nr());
  auto& max_ = unpack(max, "max", 0);
  auto& max_indices_ = unpack(max_indices, "max_indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_max");
  }
  if (compute_requires_grad( max, max_indices )) {
    throw_error_out_requires_grad("_max");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_max");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "max_indices", max_indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "max", max);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_max_out", max);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> max__storage_saved =
    max_.has_storage() ? c10::optional<Storage>(max_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> max__impl_saved;
  if (max_.defined()) max__impl_saved = max_.getIntrusivePtr();
  c10::optional<Storage> max_indices__storage_saved =
    max_indices_.has_storage() ? c10::optional<Storage>(max_indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> max_indices__impl_saved;
  if (max_indices_.defined()) max_indices__impl_saved = max_indices_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::_max_out(max_, max_indices_, self_, dim, keepdim);
  }
  #ifndef NDEBUG
  if (max__storage_saved.has_value())
    AT_ASSERT(max__storage_saved.value().is_alias_of(max_.storage()));
  if (max__impl_saved) AT_ASSERT(max__impl_saved == max_.getIntrusivePtr());
  if (max_indices__storage_saved.has_value())
    AT_ASSERT(max_indices__storage_saved.value().is_alias_of(max_indices_.storage()));
  if (max_indices__impl_saved) AT_ASSERT(max_indices__impl_saved == max_indices_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(max);
  increment_version(max_indices);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( max, max_indices ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, max);
    jit::tracer::addOutput(node, max_indices);
  }
  return std::forward_as_tuple(max, max_indices);
}
Tensor VariableType::_mkldnn_transpose(const Tensor & self, int64_t dim0, int64_t dim1) {
  RECORD_FUNCTION("_mkldnn_transpose", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_mkldnn_transpose"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_mkldnn_transpose");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim0", dim0);
    jit::tracer::addInputs(node, "dim1", dim1);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_mkldnn_transpose(self_, dim0, dim1);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_mkldnn_transpose_(Tensor & self, int64_t dim0, int64_t dim1) {
  RECORD_FUNCTION("_mkldnn_transpose_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_mkldnn_transpose_"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_mkldnn_transpose");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_mkldnn_transpose_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim0", dim0);
    jit::tracer::addInputs(node, "dim1", dim1);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_mkldnn_transpose_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::_mkldnn_transpose_(self_, dim0, dim1);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
std::tuple<Tensor &,Tensor &> VariableType::_mode_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) {
  RECORD_FUNCTION("_mode_out", std::vector<c10::IValue>({values, indices, self}), Node::peek_at_next_sequence_nr());
  auto& values_ = unpack(values, "values", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_mode");
  }
  if (compute_requires_grad( values, indices )) {
    throw_error_out_requires_grad("_mode");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_mode");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "values", values);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_mode_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> values__storage_saved =
    values_.has_storage() ? c10::optional<Storage>(values_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> values__impl_saved;
  if (values_.defined()) values__impl_saved = values_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::_mode_out(values_, indices_, self_, dim, keepdim);
  }
  #ifndef NDEBUG
  if (values__storage_saved.has_value())
    AT_ASSERT(values__storage_saved.value().is_alias_of(values_.storage()));
  if (values__impl_saved) AT_ASSERT(values__impl_saved == values_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(values);
  increment_version(indices);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( values, indices ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(values, indices);
}
Tensor VariableType::_nnpack_spatial_convolution_backward_weight(const Tensor & input, IntArrayRef weightsize, const Tensor & grad_output, IntArrayRef padding) {
  RECORD_FUNCTION("_nnpack_spatial_convolution_backward_weight", std::vector<c10::IValue>({input, grad_output}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_nnpack_spatial_convolution_backward_weight");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weightsize", weightsize);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::_nnpack_spatial_convolution_backward_weight(input, weightsize, grad_output, padding);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_pdist_forward(const Tensor & self, double p) {
  RECORD_FUNCTION("_pdist_forward", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<PdistBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<PdistBackward>(new PdistBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->p = p;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_pdist_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_pdist_forward(self_, p);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor VariableType::_softmax(const Tensor & self, int64_t dim, bool half_to_float) {
  RECORD_FUNCTION("_softmax", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SoftmaxBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SoftmaxBackward>(new SoftmaxBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_softmax");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "half_to_float", half_to_float);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_softmax(self_, dim, half_to_float);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor VariableType::_sparse_coo_tensor_with_dims(int64_t sparse_dim, int64_t dense_dim, IntArrayRef size, const TensorOptions & options) {
  RECORD_FUNCTION("_sparse_coo_tensor_with_dims", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  auto options_ = TensorOptions(options).is_variable(false);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_coo_tensor_with_dims");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "sparse_dim", sparse_dim);
    jit::tracer::addInputs(node, "dense_dim", dense_dim);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_sparse_coo_tensor_with_dims(sparse_dim, dense_dim, size, options_);
  })();
  auto result = as_variable(std::move(tmp));
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_sparse_sum(const Tensor & self) {
  RECORD_FUNCTION("_sparse_sum", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_sum");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::_sparse_sum(self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_sparse_sum(const Tensor & self, ScalarType dtype) {
  RECORD_FUNCTION("_sparse_sum", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_sum");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::_sparse_sum(self, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_sparse_sum(const Tensor & self, IntArrayRef dim) {
  RECORD_FUNCTION("_sparse_sum", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SparseSumBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SparseSumBackward>(new SparseSumBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim.vec();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_sum");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_sparse_sum(self_, dim);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_sparse_sum(const Tensor & self, IntArrayRef dim, ScalarType dtype) {
  RECORD_FUNCTION("_sparse_sum", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_sum");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::_sparse_sum(self, dim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::_thnn_fused_gru_cell(const Tensor & input_gates, const Tensor & hidden_gates, const Tensor & hx, const Tensor & input_bias, const Tensor & hidden_bias) {
  RECORD_FUNCTION("_thnn_fused_gru_cell", std::vector<c10::IValue>({input_gates, hidden_gates, hx, input_bias, hidden_bias}), Node::peek_at_next_sequence_nr());
  auto& input_gates_ = unpack(input_gates, "input_gates", 0);
  auto& hidden_gates_ = unpack(hidden_gates, "hidden_gates", 1);
  auto& hx_ = unpack(hx, "hx", 2);
  auto input_bias_ = unpack_opt(input_bias, "input_bias", 3);
  auto hidden_bias_ = unpack_opt(hidden_bias, "hidden_bias", 4);
  std::shared_ptr<ThnnFusedGruCellBackward> grad_fn;
  if (compute_requires_grad( input_gates, hidden_gates, hx, input_bias, hidden_bias )) {
    grad_fn = std::shared_ptr<ThnnFusedGruCellBackward>(new ThnnFusedGruCellBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input_gates, hidden_gates, hx, input_bias, hidden_bias ));
    grad_fn->input_bias_ = SavedVariable(input_bias, false);
  }
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_fused_gru_cell");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input_gates", input_gates);
    jit::tracer::addInputs(node, "hidden_gates", hidden_gates);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "input_bias", input_bias);
    jit::tracer::addInputs(node, "hidden_bias", hidden_bias);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> input_gates__storage_saved =
    input_gates_.has_storage() ? c10::optional<Storage>(input_gates_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input_gates__impl_saved;
  if (input_gates_.defined()) input_gates__impl_saved = input_gates_.getIntrusivePtr();
  c10::optional<Storage> hidden_gates__storage_saved =
    hidden_gates_.has_storage() ? c10::optional<Storage>(hidden_gates_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> hidden_gates__impl_saved;
  if (hidden_gates_.defined()) hidden_gates__impl_saved = hidden_gates_.getIntrusivePtr();
  c10::optional<Storage> hx__storage_saved =
    hx_.has_storage() ? c10::optional<Storage>(hx_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> hx__impl_saved;
  if (hx_.defined()) hx__impl_saved = hx_.getIntrusivePtr();
  c10::optional<Storage> input_bias__storage_saved =
    input_bias_.has_storage() ? c10::optional<Storage>(input_bias_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input_bias__impl_saved;
  if (input_bias_.defined()) input_bias__impl_saved = input_bias_.getIntrusivePtr();
  c10::optional<Storage> hidden_bias__storage_saved =
    hidden_bias_.has_storage() ? c10::optional<Storage>(hidden_bias_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> hidden_bias__impl_saved;
  if (hidden_bias_.defined()) hidden_bias__impl_saved = hidden_bias_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_thnn_fused_gru_cell(input_gates_, hidden_gates_, hx_, input_bias_, hidden_bias_);
  })();
  std::tie(result0, result1) = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (input_gates__storage_saved.has_value())
    AT_ASSERT(input_gates__storage_saved.value().is_alias_of(input_gates_.storage()));
  if (input_gates__impl_saved) AT_ASSERT(input_gates__impl_saved == input_gates_.getIntrusivePtr());
  if (hidden_gates__storage_saved.has_value())
    AT_ASSERT(hidden_gates__storage_saved.value().is_alias_of(hidden_gates_.storage()));
  if (hidden_gates__impl_saved) AT_ASSERT(hidden_gates__impl_saved == hidden_gates_.getIntrusivePtr());
  if (hx__storage_saved.has_value())
    AT_ASSERT(hx__storage_saved.value().is_alias_of(hx_.storage()));
  if (hx__impl_saved) AT_ASSERT(hx__impl_saved == hx_.getIntrusivePtr());
  if (input_bias__storage_saved.has_value())
    AT_ASSERT(input_bias__storage_saved.value().is_alias_of(input_bias_.storage()));
  if (input_bias__impl_saved) AT_ASSERT(input_bias__impl_saved == input_bias_.getIntrusivePtr());
  if (hidden_bias__storage_saved.has_value())
    AT_ASSERT(hidden_bias__storage_saved.value().is_alias_of(hidden_bias_.storage()));
  if (hidden_bias__impl_saved) AT_ASSERT(hidden_bias__impl_saved == hidden_bias_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0 ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> VariableType::_thnn_fused_lstm_cell_backward(const Tensor & grad_hy, const Tensor & grad_cy, const Tensor & cx, const Tensor & cy, const Tensor & workspace, bool has_bias) {
  RECORD_FUNCTION("_thnn_fused_lstm_cell_backward", std::vector<c10::IValue>({grad_hy, grad_cy, cx, cy, workspace}), Node::peek_at_next_sequence_nr());
  auto grad_hy_ = unpack_opt(grad_hy, "grad_hy", 0);
  auto grad_cy_ = unpack_opt(grad_cy, "grad_cy", 1);
  auto& cx_ = unpack(cx, "cx", 2);
  auto& cy_ = unpack(cy, "cy", 3);
  auto& workspace_ = unpack(workspace, "workspace", 4);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_hy, grad_cy, cx, cy, workspace )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_fused_lstm_cell_backward"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_hy, grad_cy, cx, cy, workspace ));
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  Tensor result3;
  Tensor result4;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_fused_lstm_cell_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_hy", grad_hy);
    jit::tracer::addInputs(node, "grad_cy", grad_cy);
    jit::tracer::addInputs(node, "cx", cx);
    jit::tracer::addInputs(node, "cy", cy);
    jit::tracer::addInputs(node, "workspace", workspace);
    jit::tracer::addInputs(node, "has_bias", has_bias);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_hy__storage_saved =
    grad_hy_.has_storage() ? c10::optional<Storage>(grad_hy_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_hy__impl_saved;
  if (grad_hy_.defined()) grad_hy__impl_saved = grad_hy_.getIntrusivePtr();
  c10::optional<Storage> grad_cy__storage_saved =
    grad_cy_.has_storage() ? c10::optional<Storage>(grad_cy_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_cy__impl_saved;
  if (grad_cy_.defined()) grad_cy__impl_saved = grad_cy_.getIntrusivePtr();
  c10::optional<Storage> cx__storage_saved =
    cx_.has_storage() ? c10::optional<Storage>(cx_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> cx__impl_saved;
  if (cx_.defined()) cx__impl_saved = cx_.getIntrusivePtr();
  c10::optional<Storage> cy__storage_saved =
    cy_.has_storage() ? c10::optional<Storage>(cy_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> cy__impl_saved;
  if (cy_.defined()) cy__impl_saved = cy_.getIntrusivePtr();
  c10::optional<Storage> workspace__storage_saved =
    workspace_.has_storage() ? c10::optional<Storage>(workspace_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> workspace__impl_saved;
  if (workspace_.defined()) workspace__impl_saved = workspace_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_thnn_fused_lstm_cell_backward(grad_hy_, grad_cy_, cx_, cy_, workspace_, has_bias);
  })();
  std::tie(result0, result1, result2, result3, result4) = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (grad_hy__storage_saved.has_value())
    AT_ASSERT(grad_hy__storage_saved.value().is_alias_of(grad_hy_.storage()));
  if (grad_hy__impl_saved) AT_ASSERT(grad_hy__impl_saved == grad_hy_.getIntrusivePtr());
  if (grad_cy__storage_saved.has_value())
    AT_ASSERT(grad_cy__storage_saved.value().is_alias_of(grad_cy_.storage()));
  if (grad_cy__impl_saved) AT_ASSERT(grad_cy__impl_saved == grad_cy_.getIntrusivePtr());
  if (cx__storage_saved.has_value())
    AT_ASSERT(cx__storage_saved.value().is_alias_of(cx_.storage()));
  if (cx__impl_saved) AT_ASSERT(cx__impl_saved == cx_.getIntrusivePtr());
  if (cy__storage_saved.has_value())
    AT_ASSERT(cy__storage_saved.value().is_alias_of(cy_.storage()));
  if (cy__impl_saved) AT_ASSERT(cy__impl_saved == cy_.getIntrusivePtr());
  if (workspace__storage_saved.has_value())
    AT_ASSERT(workspace__storage_saved.value().is_alias_of(workspace_.storage()));
  if (workspace__impl_saved) AT_ASSERT(workspace__impl_saved == workspace_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1, result2, result3, result4 ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
    jit::tracer::addOutput(node, result4);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3), std::move(result4));
}
Tensor VariableType::_trilinear(const Tensor & i1, const Tensor & i2, const Tensor & i3, IntArrayRef expand1, IntArrayRef expand2, IntArrayRef expand3, IntArrayRef sumdim, int64_t unroll_dim) {
  RECORD_FUNCTION("_trilinear", std::vector<c10::IValue>({i1, i2, i3}), Node::peek_at_next_sequence_nr());
  auto& i1_ = unpack(i1, "i1", 0);
  auto& i2_ = unpack(i2, "i2", 1);
  auto& i3_ = unpack(i3, "i3", 2);
  std::shared_ptr<TrilinearBackward> grad_fn;
  if (compute_requires_grad( i1, i2, i3 )) {
    grad_fn = std::shared_ptr<TrilinearBackward>(new TrilinearBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( i1, i2, i3 ));
    grad_fn->i1_ = SavedVariable(i1, false);
    grad_fn->i2_ = SavedVariable(i2, false);
    grad_fn->i3_ = SavedVariable(i3, false);
    grad_fn->expand1 = expand1.vec();
    grad_fn->expand2 = expand2.vec();
    grad_fn->expand3 = expand3.vec();
    grad_fn->sumdim = sumdim.vec();
    grad_fn->unroll_dim = unroll_dim;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_trilinear");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "i1", i1);
    jit::tracer::addInputs(node, "i2", i2);
    jit::tracer::addInputs(node, "i3", i3);
    jit::tracer::addInputs(node, "expand1", expand1);
    jit::tracer::addInputs(node, "expand2", expand2);
    jit::tracer::addInputs(node, "expand3", expand3);
    jit::tracer::addInputs(node, "sumdim", sumdim);
    jit::tracer::addInputs(node, "unroll_dim", unroll_dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> i1__storage_saved =
    i1_.has_storage() ? c10::optional<Storage>(i1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> i1__impl_saved;
  if (i1_.defined()) i1__impl_saved = i1_.getIntrusivePtr();
  c10::optional<Storage> i2__storage_saved =
    i2_.has_storage() ? c10::optional<Storage>(i2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> i2__impl_saved;
  if (i2_.defined()) i2__impl_saved = i2_.getIntrusivePtr();
  c10::optional<Storage> i3__storage_saved =
    i3_.has_storage() ? c10::optional<Storage>(i3_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> i3__impl_saved;
  if (i3_.defined()) i3__impl_saved = i3_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_trilinear(i1_, i2_, i3_, expand1, expand2, expand3, sumdim, unroll_dim);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (i1__storage_saved.has_value())
    AT_ASSERT(i1__storage_saved.value().is_alias_of(i1_.storage()));
  if (i1__impl_saved) AT_ASSERT(i1__impl_saved == i1_.getIntrusivePtr());
  if (i2__storage_saved.has_value())
    AT_ASSERT(i2__storage_saved.value().is_alias_of(i2_.storage()));
  if (i2__impl_saved) AT_ASSERT(i2__impl_saved == i2_.getIntrusivePtr());
  if (i3__storage_saved.has_value())
    AT_ASSERT(i3__storage_saved.value().is_alias_of(i3_.storage()));
  if (i3__impl_saved) AT_ASSERT(i3__impl_saved == i3_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_unsafe_view(const Tensor & self, IntArrayRef size) {
  RECORD_FUNCTION("_unsafe_view", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<UnsafeViewBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<UnsafeViewBackward>(new UnsafeViewBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_unsafe_view");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_unsafe_view(self_, size);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::_weight_norm_cuda_interface(const Tensor & v, const Tensor & g, int64_t dim) {
  RECORD_FUNCTION("_weight_norm_cuda_interface", std::vector<c10::IValue>({v, g}), Node::peek_at_next_sequence_nr());
  auto& v_ = unpack(v, "v", 0);
  auto& g_ = unpack(g, "g", 1);
  std::shared_ptr<WeightNormCudaInterfaceBackward> grad_fn;
  if (compute_requires_grad( v, g )) {
    grad_fn = std::shared_ptr<WeightNormCudaInterfaceBackward>(new WeightNormCudaInterfaceBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( v, g ));
    grad_fn->v_ = SavedVariable(v, false);
    grad_fn->g_ = SavedVariable(g, false);
    grad_fn->dim = dim;
  }
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_weight_norm_cuda_interface");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "v", v);
    jit::tracer::addInputs(node, "g", g);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> v__storage_saved =
    v_.has_storage() ? c10::optional<Storage>(v_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> v__impl_saved;
  if (v_.defined()) v__impl_saved = v_.getIntrusivePtr();
  c10::optional<Storage> g__storage_saved =
    g_.has_storage() ? c10::optional<Storage>(g_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> g__impl_saved;
  if (g_.defined()) g__impl_saved = g_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_weight_norm_cuda_interface(v_, g_, dim);
  })();
  std::tie(result0, result1) = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (v__storage_saved.has_value())
    AT_ASSERT(v__storage_saved.value().is_alias_of(v_.storage()));
  if (v__impl_saved) AT_ASSERT(v__impl_saved == v_.getIntrusivePtr());
  if (g__storage_saved.has_value())
    AT_ASSERT(g__storage_saved.value().is_alias_of(g_.storage()));
  if (g__impl_saved) AT_ASSERT(g__impl_saved == g_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0 ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor VariableType::abs(const Tensor & self) {
  RECORD_FUNCTION("abs", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AbsBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AbsBackward>(new AbsBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::abs");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::abs(self_);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::abs_(Tensor & self) {
  RECORD_FUNCTION("abs_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<AbsBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AbsBackward>(new AbsBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::abs");
    } else {
      op_name = jit::Symbol::fromQualString("aten::abs_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("abs_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::abs_(self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::adaptive_avg_pool1d(const Tensor & self, IntArrayRef output_size) {
  RECORD_FUNCTION("adaptive_avg_pool1d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::adaptive_avg_pool1d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::adaptive_avg_pool1d(self, output_size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::adaptive_avg_pool2d_out(Tensor & out, const Tensor & self, IntArrayRef output_size) {
  RECORD_FUNCTION("adaptive_avg_pool2d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("adaptive_avg_pool2d");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("adaptive_avg_pool2d");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::adaptive_avg_pool2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("adaptive_avg_pool2d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::adaptive_avg_pool2d_out(out_, self_, output_size);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::adaptive_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices) {
  RECORD_FUNCTION("adaptive_max_pool2d_backward", std::vector<c10::IValue>({grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 2);
  check_no_requires_grad(indices, "indices");
  std::shared_ptr<AdaptiveMaxPool2DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<AdaptiveMaxPool2DBackwardBackward>(new AdaptiveMaxPool2DBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->indices_ = SavedVariable(indices, false);
    grad_fn->self_info = self;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::adaptive_max_pool2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::adaptive_max_pool2d_backward(grad_output_, self_, indices_);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::adaptive_max_pool3d(const Tensor & self, IntArrayRef output_size) {
  RECORD_FUNCTION("adaptive_max_pool3d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AdaptiveMaxPool3DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AdaptiveMaxPool3DBackward>(new AdaptiveMaxPool3DBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::adaptive_max_pool3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::adaptive_max_pool3d(self_, output_size);
  })();
  std::tie(result0, result1) = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0 ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor & VariableType::adaptive_max_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices) {
  RECORD_FUNCTION("adaptive_max_pool3d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& indices_ = unpack(indices, "indices", 3);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self, indices )) {
    throw_error_out_requires_grad("adaptive_max_pool3d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("adaptive_max_pool3d_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::adaptive_max_pool3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("adaptive_max_pool3d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::adaptive_max_pool3d_backward_out(grad_input_, grad_output_, self_, indices_);
  }
  #ifndef NDEBUG
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  #endif
  increment_version(grad_input);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
  RECORD_FUNCTION("addcmul", std::vector<c10::IValue>({self, tensor1, tensor2, value}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& tensor1_ = unpack(tensor1, "tensor1", 1);
  auto& tensor2_ = unpack(tensor2, "tensor2", 2);
  std::shared_ptr<AddcmulBackward> grad_fn;
  if (compute_requires_grad( self, tensor1, tensor2 )) {
    grad_fn = std::shared_ptr<AddcmulBackward>(new AddcmulBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, tensor1, tensor2 ));
    if (grad_fn->should_compute_output(1)) {
      grad_fn->tensor2_ = SavedVariable(tensor2, false);
    }
    grad_fn->value = value;
    if (grad_fn->should_compute_output(2)) {
      grad_fn->tensor1_ = SavedVariable(tensor1, false);
    }
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::addcmul");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "tensor1", tensor1);
    jit::tracer::addInputs(node, "tensor2", tensor2);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> tensor1__storage_saved =
    tensor1_.has_storage() ? c10::optional<Storage>(tensor1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> tensor1__impl_saved;
  if (tensor1_.defined()) tensor1__impl_saved = tensor1_.getIntrusivePtr();
  c10::optional<Storage> tensor2__storage_saved =
    tensor2_.has_storage() ? c10::optional<Storage>(tensor2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> tensor2__impl_saved;
  if (tensor2_.defined()) tensor2__impl_saved = tensor2_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::addcmul(self_, tensor1_, tensor2_, value);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (tensor1__storage_saved.has_value())
    AT_ASSERT(tensor1__storage_saved.value().is_alias_of(tensor1_.storage()));
  if (tensor1__impl_saved) AT_ASSERT(tensor1__impl_saved == tensor1_.getIntrusivePtr());
  if (tensor2__storage_saved.has_value())
    AT_ASSERT(tensor2__storage_saved.value().is_alias_of(tensor2_.storage()));
  if (tensor2__impl_saved) AT_ASSERT(tensor2__impl_saved == tensor2_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
  RECORD_FUNCTION("addcmul_", std::vector<c10::IValue>({self, tensor1, tensor2, value}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& tensor1_ = unpack(tensor1, "tensor1", 1);
  auto& tensor2_ = unpack(tensor2, "tensor2", 2);
  check_inplace(self);
  std::shared_ptr<AddcmulBackward> grad_fn;
  if (compute_requires_grad( self, tensor1, tensor2 )) {
    grad_fn = std::shared_ptr<AddcmulBackward>(new AddcmulBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, tensor1, tensor2 ));
    if (grad_fn->should_compute_output(1)) {
      grad_fn->tensor2_ = SavedVariable(tensor2, false);
    }
    grad_fn->value = value;
    if (grad_fn->should_compute_output(2)) {
      grad_fn->tensor1_ = SavedVariable(tensor1, false);
    }
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::addcmul");
    } else {
      op_name = jit::Symbol::fromQualString("aten::addcmul_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "tensor1", tensor1);
    jit::tracer::addInputs(node, "tensor2", tensor2);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("addcmul_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> tensor1__storage_saved =
    tensor1_.has_storage() ? c10::optional<Storage>(tensor1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> tensor1__impl_saved;
  if (tensor1_.defined()) tensor1__impl_saved = tensor1_.getIntrusivePtr();
  c10::optional<Storage> tensor2__storage_saved =
    tensor2_.has_storage() ? c10::optional<Storage>(tensor2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> tensor2__impl_saved;
  if (tensor2_.defined()) tensor2__impl_saved = tensor2_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.addcmul_(tensor1_, tensor2_, value);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (tensor1__storage_saved.has_value())
    AT_ASSERT(tensor1__storage_saved.value().is_alias_of(tensor1_.storage()));
  if (tensor1__impl_saved) AT_ASSERT(tensor1__impl_saved == tensor1_.getIntrusivePtr());
  if (tensor2__storage_saved.has_value())
    AT_ASSERT(tensor2__storage_saved.value().is_alias_of(tensor2_.storage()));
  if (tensor2__impl_saved) AT_ASSERT(tensor2__impl_saved == tensor2_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {
  RECORD_FUNCTION("addmm", std::vector<c10::IValue>({self, mat1, mat2, beta, alpha}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& mat1_ = unpack(mat1, "mat1", 1);
  auto& mat2_ = unpack(mat2, "mat2", 2);
  std::shared_ptr<AddmmBackward> grad_fn;
  if (compute_requires_grad( self, mat1, mat2 )) {
    grad_fn = std::shared_ptr<AddmmBackward>(new AddmmBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, mat1, mat2 ));
    grad_fn->mat1_ = SavedVariable(mat1, false);
    grad_fn->mat2_ = SavedVariable(mat2, false);
    grad_fn->alpha = alpha;
    grad_fn->mat2_sizes = mat2.sizes().vec();
    grad_fn->beta = beta;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::addmm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat1", mat1);
    jit::tracer::addInputs(node, "mat2", mat2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> mat1__storage_saved =
    mat1_.has_storage() ? c10::optional<Storage>(mat1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mat1__impl_saved;
  if (mat1_.defined()) mat1__impl_saved = mat1_.getIntrusivePtr();
  c10::optional<Storage> mat2__storage_saved =
    mat2_.has_storage() ? c10::optional<Storage>(mat2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mat2__impl_saved;
  if (mat2_.defined()) mat2__impl_saved = mat2_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::addmm(self_, mat1_, mat2_, beta, alpha);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (mat1__storage_saved.has_value())
    AT_ASSERT(mat1__storage_saved.value().is_alias_of(mat1_.storage()));
  if (mat1__impl_saved) AT_ASSERT(mat1__impl_saved == mat1_.getIntrusivePtr());
  if (mat2__storage_saved.has_value())
    AT_ASSERT(mat2__storage_saved.value().is_alias_of(mat2_.storage()));
  if (mat2__impl_saved) AT_ASSERT(mat2__impl_saved == mat2_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {
  RECORD_FUNCTION("addmm_", std::vector<c10::IValue>({self, mat1, mat2, beta, alpha}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& mat1_ = unpack(mat1, "mat1", 1);
  auto& mat2_ = unpack(mat2, "mat2", 2);
  check_inplace(self);
  std::shared_ptr<AddmmBackward> grad_fn;
  if (compute_requires_grad( self, mat1, mat2 )) {
    grad_fn = std::shared_ptr<AddmmBackward>(new AddmmBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, mat1, mat2 ));
    grad_fn->mat1_ = SavedVariable(mat1, false);
    grad_fn->mat2_ = SavedVariable(mat2, false);
    grad_fn->alpha = alpha;
    grad_fn->mat2_sizes = mat2.sizes().vec();
    grad_fn->beta = beta;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::addmm");
    } else {
      op_name = jit::Symbol::fromQualString("aten::addmm_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat1", mat1);
    jit::tracer::addInputs(node, "mat2", mat2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("addmm_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> mat1__storage_saved =
    mat1_.has_storage() ? c10::optional<Storage>(mat1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mat1__impl_saved;
  if (mat1_.defined()) mat1__impl_saved = mat1_.getIntrusivePtr();
  c10::optional<Storage> mat2__storage_saved =
    mat2_.has_storage() ? c10::optional<Storage>(mat2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mat2__impl_saved;
  if (mat2_.defined()) mat2__impl_saved = mat2_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.addmm_(mat1_, mat2_, beta, alpha);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (mat1__storage_saved.has_value())
    AT_ASSERT(mat1__storage_saved.value().is_alias_of(mat1_.storage()));
  if (mat1__impl_saved) AT_ASSERT(mat1__impl_saved == mat1_.getIntrusivePtr());
  if (mat2__storage_saved.has_value())
    AT_ASSERT(mat2__storage_saved.value().is_alias_of(mat2_.storage()));
  if (mat2__impl_saved) AT_ASSERT(mat2__impl_saved == mat2_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::addr_out(Tensor & out, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) {
  RECORD_FUNCTION("addr_out", std::vector<c10::IValue>({out, self, vec1, vec2, beta, alpha}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& vec1_ = unpack(vec1, "vec1", 2);
  auto& vec2_ = unpack(vec2, "vec2", 3);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, vec1, vec2 )) {
    throw_error_out_requires_grad("addr");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("addr");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::addr");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "vec1", vec1);
    jit::tracer::addInputs(node, "vec2", vec2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("addr_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> vec1__storage_saved =
    vec1_.has_storage() ? c10::optional<Storage>(vec1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> vec1__impl_saved;
  if (vec1_.defined()) vec1__impl_saved = vec1_.getIntrusivePtr();
  c10::optional<Storage> vec2__storage_saved =
    vec2_.has_storage() ? c10::optional<Storage>(vec2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> vec2__impl_saved;
  if (vec2_.defined()) vec2__impl_saved = vec2_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::addr_out(out_, self_, vec1_, vec2_, beta, alpha);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (vec1__storage_saved.has_value())
    AT_ASSERT(vec1__storage_saved.value().is_alias_of(vec1_.storage()));
  if (vec1__impl_saved) AT_ASSERT(vec1__impl_saved == vec1_.getIntrusivePtr());
  if (vec2__storage_saved.has_value())
    AT_ASSERT(vec2__storage_saved.value().is_alias_of(vec2_.storage()));
  if (vec2__impl_saved) AT_ASSERT(vec2__impl_saved == vec2_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::align_as(const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("align_as", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::align_as");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::align_as(self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::vector<Tensor> VariableType::align_tensors(TensorList tensors) {
  RECORD_FUNCTION("align_tensors", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::align_tensors");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::align_tensors(tensors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::align_to(const Tensor & self, DimnameList names) {
  RECORD_FUNCTION("align_to", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::align_to");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "names", names);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::align_to(self, names);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::argmax(const Tensor & self, c10::optional<int64_t> dim, bool keepdim) {
  RECORD_FUNCTION("argmax", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::argmax");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::argmax(self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::argsort(const Tensor & self, int64_t dim, bool descending) {
  RECORD_FUNCTION("argsort", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::argsort");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "descending", descending);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::argsort(self, dim, descending);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::argsort(const Tensor & self, Dimname dim, bool descending) {
  RECORD_FUNCTION("argsort", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::argsort");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "descending", descending);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::argsort(self, dim, descending);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::atan(const Tensor & self) {
  RECORD_FUNCTION("atan", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AtanBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AtanBackward>(new AtanBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::atan");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::atan(self_);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::atan2(const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("atan2", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  std::shared_ptr<Atan2Backward> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<Atan2Backward>(new Atan2Backward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->other_ = SavedVariable(other, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::atan2");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::atan2(self_, other_);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::atan2_(Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("atan2_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<Atan2Backward> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<Atan2Backward>(new Atan2Backward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
    grad_fn->other_ = SavedVariable(other, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::atan2");
    } else {
      op_name = jit::Symbol::fromQualString("aten::atan2_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("atan2_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.atan2_(other_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::atan_(Tensor & self) {
  RECORD_FUNCTION("atan_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<AtanBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AtanBackward>(new AtanBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::atan");
    } else {
      op_name = jit::Symbol::fromQualString("aten::atan_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("atan_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::atan_(self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::avg_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  RECORD_FUNCTION("avg_pool2d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AvgPool2DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AvgPool2DBackward>(new AvgPool2DBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->ceil_mode = ceil_mode;
    grad_fn->count_include_pad = count_include_pad;
    grad_fn->divisor_override = divisor_override;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::avg_pool2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    jit::tracer::addInputs(node, "count_include_pad", count_include_pad);
    jit::tracer::addInputs(node, "divisor_override", divisor_override);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::avg_pool2d(self_, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::avg_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  RECORD_FUNCTION("avg_pool2d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self}), Node::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("avg_pool2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("avg_pool2d_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::avg_pool2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    jit::tracer::addInputs(node, "count_include_pad", count_include_pad);
    jit::tracer::addInputs(node, "divisor_override", divisor_override);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("avg_pool2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::avg_pool2d_backward_out(grad_input_, grad_output_, self_, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  }
  #ifndef NDEBUG
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(grad_input);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::avg_pool3d_out(Tensor & out, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  RECORD_FUNCTION("avg_pool3d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("avg_pool3d");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("avg_pool3d");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::avg_pool3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    jit::tracer::addInputs(node, "count_include_pad", count_include_pad);
    jit::tracer::addInputs(node, "divisor_override", divisor_override);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("avg_pool3d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::avg_pool3d_out(out_, self_, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::baddbmm_out(Tensor & out, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
  RECORD_FUNCTION("baddbmm_out", std::vector<c10::IValue>({out, self, batch1, batch2, beta, alpha}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& batch1_ = unpack(batch1, "batch1", 2);
  auto& batch2_ = unpack(batch2, "batch2", 3);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, batch1, batch2 )) {
    throw_error_out_requires_grad("baddbmm");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("baddbmm");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::baddbmm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "batch1", batch1);
    jit::tracer::addInputs(node, "batch2", batch2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("baddbmm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> batch1__storage_saved =
    batch1_.has_storage() ? c10::optional<Storage>(batch1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> batch1__impl_saved;
  if (batch1_.defined()) batch1__impl_saved = batch1_.getIntrusivePtr();
  c10::optional<Storage> batch2__storage_saved =
    batch2_.has_storage() ? c10::optional<Storage>(batch2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> batch2__impl_saved;
  if (batch2_.defined()) batch2__impl_saved = batch2_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::baddbmm_out(out_, self_, batch1_, batch2_, beta, alpha);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (batch1__storage_saved.has_value())
    AT_ASSERT(batch1__storage_saved.value().is_alias_of(batch1_.storage()));
  if (batch1__impl_saved) AT_ASSERT(batch1__impl_saved == batch1_.getIntrusivePtr());
  if (batch2__storage_saved.has_value())
    AT_ASSERT(batch2__storage_saved.value().is_alias_of(batch2_.storage()));
  if (batch2__impl_saved) AT_ASSERT(batch2__impl_saved == batch2_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::bartlett_window(int64_t window_length, const TensorOptions & options) {
  RECORD_FUNCTION("bartlett_window", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bartlett_window");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::bartlett_window(window_length, options);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::bartlett_window(int64_t window_length, bool periodic, const TensorOptions & options) {
  RECORD_FUNCTION("bartlett_window", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bartlett_window");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "periodic", periodic);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::bartlett_window(window_length, periodic, options);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::batch_norm_update_stats(const Tensor & input, const Tensor & running_mean, const Tensor & running_var, double momentum) {
  RECORD_FUNCTION("batch_norm_update_stats", std::vector<c10::IValue>({input, running_mean, running_var}), Node::peek_at_next_sequence_nr());
  auto& input_ = unpack(input, "input", 0);
  auto running_mean_ = unpack_opt(running_mean, "running_mean", 1);
  auto running_var_ = unpack_opt(running_var, "running_var", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( input, running_mean, running_var )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("batch_norm_update_stats"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, running_mean, running_var ));
  }
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::batch_norm_update_stats");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "momentum", momentum);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  c10::optional<Storage> running_mean__storage_saved =
    running_mean_.has_storage() ? c10::optional<Storage>(running_mean_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> running_mean__impl_saved;
  if (running_mean_.defined()) running_mean__impl_saved = running_mean_.getIntrusivePtr();
  c10::optional<Storage> running_var__storage_saved =
    running_var_.has_storage() ? c10::optional<Storage>(running_var_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> running_var__impl_saved;
  if (running_var_.defined()) running_var__impl_saved = running_var_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::batch_norm_update_stats(input_, running_mean_, running_var_, momentum);
  })();
  std::tie(result0, result1) = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (running_mean__storage_saved.has_value())
    AT_ASSERT(running_mean__storage_saved.value().is_alias_of(running_mean_.storage()));
  if (running_mean__impl_saved) AT_ASSERT(running_mean__impl_saved == running_mean_.getIntrusivePtr());
  if (running_var__storage_saved.has_value())
    AT_ASSERT(running_var__storage_saved.value().is_alias_of(running_var_.storage()));
  if (running_var__impl_saved) AT_ASSERT(running_var__impl_saved == running_var_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor VariableType::binary_cross_entropy_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) {
  RECORD_FUNCTION("binary_cross_entropy_backward", std::vector<c10::IValue>({grad_output, self, target, weight}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  auto weight_ = unpack_opt(weight, "weight", 3);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, self, target, weight )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("binary_cross_entropy_backward"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, target, weight ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::binary_cross_entropy_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::binary_cross_entropy_backward(grad_output_, self_, target_, weight_, reduction);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::bitwise_not_out(Tensor & out, const Tensor & self) {
  RECORD_FUNCTION("bitwise_not_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("bitwise_not");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("bitwise_not");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bitwise_not");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_not_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::bitwise_not_out(out_, self_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::bmm_out(Tensor & out, const Tensor & self, const Tensor & mat2) {
  RECORD_FUNCTION("bmm_out", std::vector<c10::IValue>({out, self, mat2}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& mat2_ = unpack(mat2, "mat2", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, mat2 )) {
    throw_error_out_requires_grad("bmm");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("bmm");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bmm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat2", mat2);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bmm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> mat2__storage_saved =
    mat2_.has_storage() ? c10::optional<Storage>(mat2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mat2__impl_saved;
  if (mat2_.defined()) mat2__impl_saved = mat2_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::bmm_out(out_, self_, mat2_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (mat2__storage_saved.has_value())
    AT_ASSERT(mat2__storage_saved.value().is_alias_of(mat2_.storage()));
  if (mat2__impl_saved) AT_ASSERT(mat2__impl_saved == mat2_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::cdist(const Tensor & x1, const Tensor & x2, double p) {
  RECORD_FUNCTION("cdist", std::vector<c10::IValue>({x1, x2}), Node::peek_at_next_sequence_nr());
  auto& x1_ = unpack(x1, "x1", 0);
  auto& x2_ = unpack(x2, "x2", 1);
  std::shared_ptr<CdistBackward> grad_fn;
  if (compute_requires_grad( x1, x2 )) {
    grad_fn = std::shared_ptr<CdistBackward>(new CdistBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( x1, x2 ));
    grad_fn->x1_ = SavedVariable(x1, false);
    grad_fn->x2_ = SavedVariable(x2, false);
    grad_fn->p = p;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cdist");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x1", x1);
    jit::tracer::addInputs(node, "x2", x2);
    jit::tracer::addInputs(node, "p", p);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> x1__storage_saved =
    x1_.has_storage() ? c10::optional<Storage>(x1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> x1__impl_saved;
  if (x1_.defined()) x1__impl_saved = x1_.getIntrusivePtr();
  c10::optional<Storage> x2__storage_saved =
    x2_.has_storage() ? c10::optional<Storage>(x2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> x2__impl_saved;
  if (x2_.defined()) x2__impl_saved = x2_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::cdist(x1_, x2_, p);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (x1__storage_saved.has_value())
    AT_ASSERT(x1__storage_saved.value().is_alias_of(x1_.storage()));
  if (x1__impl_saved) AT_ASSERT(x1__impl_saved == x1_.getIntrusivePtr());
  if (x2__storage_saved.has_value())
    AT_ASSERT(x2__storage_saved.value().is_alias_of(x2_.storage()));
  if (x2__impl_saved) AT_ASSERT(x2__impl_saved == x2_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor VariableType::celu(const Tensor & self, Scalar alpha) {
  RECORD_FUNCTION("celu", std::vector<c10::IValue>({self, alpha}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::celu");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::celu(self, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::celu_(Tensor & self, Scalar alpha) {
  RECORD_FUNCTION("celu_", std::vector<c10::IValue>({self, alpha}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::celu");
    } else {
      op_name = jit::Symbol::fromQualString("aten::celu_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("celu_", self);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::celu_(self, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::cholesky_inverse(const Tensor & self, bool upper) {
  RECORD_FUNCTION("cholesky_inverse", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<CholeskyInverseBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<CholeskyInverseBackward>(new CholeskyInverseBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->upper = upper;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cholesky_inverse");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "upper", upper);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::cholesky_inverse(self_, upper);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor & VariableType::cholesky_out(Tensor & out, const Tensor & self, bool upper) {
  RECORD_FUNCTION("cholesky_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("cholesky");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("cholesky");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cholesky");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "upper", upper);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cholesky_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::cholesky_out(out_, self_, upper);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::clamp(const Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max) {
  RECORD_FUNCTION("clamp", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ClampBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ClampBackward>(new ClampBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->min = min;
    grad_fn->max = max;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::clamp");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min", min);
    jit::tracer::addInputs(node, "max", max);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::clamp(self_, min, max);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::clamp_(Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max) {
  RECORD_FUNCTION("clamp_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<ClampBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ClampBackward>(new ClampBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
    grad_fn->min = min;
    grad_fn->max = max;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::clamp");
    } else {
      op_name = jit::Symbol::fromQualString("aten::clamp_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min", min);
    jit::tracer::addInputs(node, "max", max);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("clamp_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::clamp_(self_, min, max);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::clamp_max_out(Tensor & out, const Tensor & self, Scalar max) {
  RECORD_FUNCTION("clamp_max_out", std::vector<c10::IValue>({out, self, max}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("clamp_max");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("clamp_max");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::clamp_max");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "max", max);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("clamp_max_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::clamp_max_out(out_, self_, max);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::conv_transpose3d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, int64_t groups, IntArrayRef dilation) {
  RECORD_FUNCTION("conv_transpose3d", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  auto result = TypeDefault::conv_transpose3d(input, weight, bias, stride, padding, output_padding, groups, dilation);
  return result;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::convolution_backward_overrideable(const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups, std::array<bool,3> output_mask) {
  RECORD_FUNCTION("convolution_backward_overrideable", std::vector<c10::IValue>({grad_output, input, weight}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& input_ = unpack(input, "input", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  std::shared_ptr<ConvolutionBackwardOverrideableBackward> grad_fn;
  if (compute_requires_grad( grad_output, input, weight )) {
    grad_fn = std::shared_ptr<ConvolutionBackwardOverrideableBackward>(new ConvolutionBackwardOverrideableBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, input, weight ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->dilation = dilation.vec();
    grad_fn->output_padding = output_padding.vec();
    grad_fn->groups = groups;
  }
  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::convolution_backward_overrideable");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "transposed", transposed);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::convolution_backward_overrideable(grad_output_, input_, weight_, stride, padding, dilation, transposed, output_padding, groups, output_mask);
  })();
  std::tie(grad_input, grad_weight, grad_bias) = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( grad_input, grad_weight, grad_bias ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
    jit::tracer::addOutput(node, grad_weight);
    jit::tracer::addOutput(node, grad_bias);
  }
  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}
Tensor VariableType::cos(const Tensor & self) {
  RECORD_FUNCTION("cos", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<CosBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<CosBackward>(new CosBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cos");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::cos(self_);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::cos_(Tensor & self) {
  RECORD_FUNCTION("cos_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<CosBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<CosBackward>(new CosBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::cos");
    } else {
      op_name = jit::Symbol::fromQualString("aten::cos_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cos_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::cos_(self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::cudnn_batch_norm_backward(const Tensor & input, const Tensor & grad_output, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_var, double epsilon) {
  RECORD_FUNCTION("cudnn_batch_norm_backward", std::vector<c10::IValue>({input, grad_output, weight, running_mean, running_var, save_mean, save_var}), Node::peek_at_next_sequence_nr());
  auto& input_ = unpack(input, "input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto running_mean_ = unpack_opt(running_mean, "running_mean", 3);
  auto running_var_ = unpack_opt(running_var, "running_var", 4);
  auto save_mean_ = unpack_opt(save_mean, "save_mean", 5);
  auto save_var_ = unpack_opt(save_var, "save_var", 6);
  check_no_requires_grad(running_mean, "running_mean");
  check_no_requires_grad(running_var, "running_var");
  std::shared_ptr<CudnnBatchNormBackwardBackward> grad_fn;
  if (compute_requires_grad( input, grad_output, weight, save_mean, save_var )) {
    grad_fn = std::shared_ptr<CudnnBatchNormBackwardBackward>(new CudnnBatchNormBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, grad_output, weight, save_mean, save_var ));
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->running_mean_ = SavedVariable(running_mean, false);
    grad_fn->running_var_ = SavedVariable(running_var, false);
    grad_fn->save_mean_ = SavedVariable(save_mean, false);
    grad_fn->save_var_ = SavedVariable(save_var, false);
    grad_fn->epsilon = epsilon;
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cudnn_batch_norm_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "save_mean", save_mean);
    jit::tracer::addInputs(node, "save_var", save_var);
    jit::tracer::addInputs(node, "epsilon", epsilon);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> running_mean__storage_saved =
    running_mean_.has_storage() ? c10::optional<Storage>(running_mean_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> running_mean__impl_saved;
  if (running_mean_.defined()) running_mean__impl_saved = running_mean_.getIntrusivePtr();
  c10::optional<Storage> running_var__storage_saved =
    running_var_.has_storage() ? c10::optional<Storage>(running_var_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> running_var__impl_saved;
  if (running_var_.defined()) running_var__impl_saved = running_var_.getIntrusivePtr();
  c10::optional<Storage> save_mean__storage_saved =
    save_mean_.has_storage() ? c10::optional<Storage>(save_mean_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> save_mean__impl_saved;
  if (save_mean_.defined()) save_mean__impl_saved = save_mean_.getIntrusivePtr();
  c10::optional<Storage> save_var__storage_saved =
    save_var_.has_storage() ? c10::optional<Storage>(save_var_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> save_var__impl_saved;
  if (save_var_.defined()) save_var__impl_saved = save_var_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::cudnn_batch_norm_backward(input_, grad_output_, weight_, running_mean_, running_var_, save_mean_, save_var_, epsilon);
  })();
  std::tie(result0, result1, result2) = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (running_mean__storage_saved.has_value())
    AT_ASSERT(running_mean__storage_saved.value().is_alias_of(running_mean_.storage()));
  if (running_mean__impl_saved) AT_ASSERT(running_mean__impl_saved == running_mean_.getIntrusivePtr());
  if (running_var__storage_saved.has_value())
    AT_ASSERT(running_var__storage_saved.value().is_alias_of(running_var_.storage()));
  if (running_var__impl_saved) AT_ASSERT(running_var__impl_saved == running_var_.getIntrusivePtr());
  if (save_mean__storage_saved.has_value())
    AT_ASSERT(save_mean__storage_saved.value().is_alias_of(save_mean_.storage()));
  if (save_mean__impl_saved) AT_ASSERT(save_mean__impl_saved == save_mean_.getIntrusivePtr());
  if (save_var__storage_saved.has_value())
    AT_ASSERT(save_var__storage_saved.value().is_alias_of(save_var_.storage()));
  if (save_var__impl_saved) AT_ASSERT(save_var__impl_saved == save_var_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1, result2 ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
std::tuple<Tensor,Tensor> VariableType::cudnn_grid_sampler_backward(const Tensor & self, const Tensor & grid, const Tensor & grad_output) {
  RECORD_FUNCTION("cudnn_grid_sampler_backward", std::vector<c10::IValue>({self, grid, grad_output}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& grid_ = unpack(grid, "grid", 1);
  auto& grad_output_ = unpack(grad_output, "grad_output", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, grid, grad_output )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("cudnn_grid_sampler_backward"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, grid, grad_output ));
  }
  Tensor grad_self;
  Tensor grad_grid;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cudnn_grid_sampler_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "grid", grid);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> grid__storage_saved =
    grid_.has_storage() ? c10::optional<Storage>(grid_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grid__impl_saved;
  if (grid_.defined()) grid__impl_saved = grid_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::cudnn_grid_sampler_backward(self_, grid_, grad_output_);
  })();
  std::tie(grad_self, grad_grid) = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (grid__storage_saved.has_value())
    AT_ASSERT(grid__storage_saved.value().is_alias_of(grid_.storage()));
  if (grid__impl_saved) AT_ASSERT(grid__impl_saved == grid_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( grad_self, grad_grid ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_self);
    jit::tracer::addOutput(node, grad_grid);
  }
  return std::make_tuple(std::move(grad_self), std::move(grad_grid));
}
Tensor & VariableType::cumsum_out(Tensor & out, const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) {
  RECORD_FUNCTION("cumsum_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("cumsum");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("cumsum");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cumsum");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cumsum_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::cumsum_out(out_, self_, dim, dtype);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::cumsum_out(Tensor & out, const Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) {
  RECORD_FUNCTION("cumsum_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cumsum");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cumsum_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::cumsum_out(out, self, dim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::diag(const Tensor & self, int64_t diagonal) {
  RECORD_FUNCTION("diag", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<DiagBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<DiagBackward>(new DiagBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->diagonal = diagonal;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::diag");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "diagonal", diagonal);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::diag(self_, diagonal);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::digamma(const Tensor & self) {
  RECORD_FUNCTION("digamma", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<DigammaBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<DigammaBackward>(new DigammaBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::digamma");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::digamma(self_);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::digamma_(Tensor & self) {
  RECORD_FUNCTION("digamma_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<DigammaBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<DigammaBackward>(new DigammaBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::digamma");
    } else {
      op_name = jit::Symbol::fromQualString("aten::digamma_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("digamma_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.digamma_();
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::elu_out(Tensor & out, const Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale) {
  RECORD_FUNCTION("elu_out", std::vector<c10::IValue>({out, self, alpha, scale, input_scale}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("elu");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("elu");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::elu");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "input_scale", input_scale);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("elu_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::elu_out(out_, self_, alpha, scale, input_scale);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
std::tuple<Tensor,Tensor,Tensor,Tensor> VariableType::embedding_bag(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const Tensor & per_sample_weights) {
  RECORD_FUNCTION("embedding_bag", std::vector<c10::IValue>({weight, indices, offsets, per_sample_weights}), Node::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  Tensor result2;
  Tensor result3;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::embedding_bag");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "offsets", offsets);
    jit::tracer::addInputs(node, "scale_grad_by_freq", scale_grad_by_freq);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "sparse", sparse);
    jit::tracer::addInputs(node, "per_sample_weights", per_sample_weights);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1, result2, result3) = TypeDefault::embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3));
}
Tensor VariableType::embedding_dense_backward(const Tensor & grad_output, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) {
  RECORD_FUNCTION("embedding_dense_backward", std::vector<c10::IValue>({grad_output, indices}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  std::shared_ptr<EmbeddingDenseBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output )) {
    grad_fn = std::shared_ptr<EmbeddingDenseBackwardBackward>(new EmbeddingDenseBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
    grad_fn->indices_ = SavedVariable(indices, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::embedding_dense_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "num_weights", num_weights);
    jit::tracer::addInputs(node, "padding_idx", padding_idx);
    jit::tracer::addInputs(node, "scale_grad_by_freq", scale_grad_by_freq);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::embedding_dense_backward(grad_output_, indices_, num_weights, padding_idx, scale_grad_by_freq);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::empty_like(const Tensor & self) {
  RECORD_FUNCTION("empty_like", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::empty_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::empty_like(self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::empty_like(const Tensor & self, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  RECORD_FUNCTION("empty_like", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::empty_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "options", options);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::empty_like(self, options, memory_format);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::empty_strided(IntArrayRef size, IntArrayRef stride, const TensorOptions & options) {
  RECORD_FUNCTION("empty_strided", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  auto options_ = TensorOptions(options).is_variable(false);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::empty_strided");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::empty_strided(size, stride, options_);
  })();
  auto result = as_variable(std::move(tmp));
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::erfc_out(Tensor & out, const Tensor & self) {
  RECORD_FUNCTION("erfc_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("erfc");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("erfc");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::erfc");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("erfc_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::erfc_out(out_, self_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::erfinv(const Tensor & self) {
  RECORD_FUNCTION("erfinv", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ErfinvBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ErfinvBackward>(new ErfinvBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::erfinv");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::erfinv(self_);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::erfinv_(Tensor & self) {
  RECORD_FUNCTION("erfinv_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<ErfinvBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ErfinvBackward>(new ErfinvBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::erfinv");
    } else {
      op_name = jit::Symbol::fromQualString("aten::erfinv_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("erfinv_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.erfinv_();
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::expand(const Tensor & self, IntArrayRef size, bool implicit) {
  RECORD_FUNCTION("expand", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ExpandBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ExpandBackward>(new ExpandBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::expand");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "implicit", implicit);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return self_.expand(size, implicit);
  })();
  auto result = as_view(self, tmp, true);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::expm1_out(Tensor & out, const Tensor & self) {
  RECORD_FUNCTION("expm1_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("expm1");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("expm1");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::expm1");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("expm1_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::expm1_out(out_, self_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::fake_quantize_per_tensor_affine_backward(const Tensor & grad, const Tensor & self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max) {
  RECORD_FUNCTION("fake_quantize_per_tensor_affine_backward", std::vector<c10::IValue>({grad, self}), Node::peek_at_next_sequence_nr());
  auto& grad_ = unpack(grad, "grad", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad, self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("fake_quantize_per_tensor_affine_backward"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad, self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fake_quantize_per_tensor_affine_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);
    jit::tracer::addInputs(node, "quant_min", quant_min);
    jit::tracer::addInputs(node, "quant_max", quant_max);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad__storage_saved =
    grad_.has_storage() ? c10::optional<Storage>(grad_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad__impl_saved;
  if (grad_.defined()) grad__impl_saved = grad_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::fake_quantize_per_tensor_affine_backward(grad_, self_, scale, zero_point, quant_min, quant_max);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (grad__storage_saved.has_value())
    AT_ASSERT(grad__storage_saved.value().is_alias_of(grad_.storage()));
  if (grad__impl_saved) AT_ASSERT(grad__impl_saved == grad_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::fft(const Tensor & self, int64_t signal_ndim, bool normalized) {
  RECORD_FUNCTION("fft", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fft");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "signal_ndim", signal_ndim);
    jit::tracer::addInputs(node, "normalized", normalized);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::fft(self, signal_ndim, normalized);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::flatten(const Tensor & self, int64_t start_dim, int64_t end_dim) {
  RECORD_FUNCTION("flatten", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::flatten");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "start_dim", start_dim);
    jit::tracer::addInputs(node, "end_dim", end_dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::flatten(self, start_dim, end_dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::flatten(const Tensor & self, int64_t start_dim, int64_t end_dim, Dimname out_dim) {
  RECORD_FUNCTION("flatten", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::flatten");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "start_dim", start_dim);
    jit::tracer::addInputs(node, "end_dim", end_dim);
    jit::tracer::addInputs(node, "out_dim", out_dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::flatten(self, start_dim, end_dim, out_dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::flatten(const Tensor & self, Dimname start_dim, Dimname end_dim, Dimname out_dim) {
  RECORD_FUNCTION("flatten", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::flatten");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "start_dim", start_dim);
    jit::tracer::addInputs(node, "end_dim", end_dim);
    jit::tracer::addInputs(node, "out_dim", out_dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::flatten(self, start_dim, end_dim, out_dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::flatten(const Tensor & self, DimnameList dims, Dimname out_dim) {
  RECORD_FUNCTION("flatten", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::flatten");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dims", dims);
    jit::tracer::addInputs(node, "out_dim", out_dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::flatten(self, dims, out_dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::floor_out(Tensor & out, const Tensor & self) {
  RECORD_FUNCTION("floor_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("floor");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("floor");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::floor");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("floor_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::floor_out(out_, self_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::full(IntArrayRef size, Scalar fill_value, c10::optional<DimnameList> names, const TensorOptions & options) {
  RECORD_FUNCTION("full", std::vector<c10::IValue>({fill_value}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::full");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "fill_value", fill_value);
    jit::tracer::addInputs(node, "names", names);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::full(size, fill_value, names, options);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::full(IntArrayRef size, Scalar fill_value, const TensorOptions & options) {
  RECORD_FUNCTION("full", std::vector<c10::IValue>({fill_value}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::full");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "fill_value", fill_value);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::full(size, fill_value, options);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::gather(const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
  RECORD_FUNCTION("gather", std::vector<c10::IValue>({self, index}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  std::shared_ptr<GatherBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<GatherBackward>(new GatherBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim;
    grad_fn->index_ = SavedVariable(index, false);
    grad_fn->sparse_grad = sparse_grad;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::gather");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "sparse_grad", sparse_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> index__storage_saved =
    index_.has_storage() ? c10::optional<Storage>(index_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> index__impl_saved;
  if (index_.defined()) index__impl_saved = index_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::gather(self_, dim, index_, sparse_grad);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (index__storage_saved.has_value())
    AT_ASSERT(index__storage_saved.value().is_alias_of(index_.storage()));
  if (index__impl_saved) AT_ASSERT(index__impl_saved == index_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::gather(const Tensor & self, Dimname dim, const Tensor & index, bool sparse_grad) {
  RECORD_FUNCTION("gather", std::vector<c10::IValue>({self, index}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::gather");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "sparse_grad", sparse_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::gather(self, dim, index, sparse_grad);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::gelu_backward(const Tensor & grad, const Tensor & self) {
  RECORD_FUNCTION("gelu_backward", std::vector<c10::IValue>({grad, self}), Node::peek_at_next_sequence_nr());
  auto& grad_ = unpack(grad, "grad", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad, self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("gelu_backward"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad, self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::gelu_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad__storage_saved =
    grad_.has_storage() ? c10::optional<Storage>(grad_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad__impl_saved;
  if (grad_.defined()) grad__impl_saved = grad_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::gelu_backward(grad_, self_);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (grad__storage_saved.has_value())
    AT_ASSERT(grad__storage_saved.value().is_alias_of(grad_.storage()));
  if (grad__impl_saved) AT_ASSERT(grad__impl_saved == grad_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::grid_sampler_3d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  RECORD_FUNCTION("grid_sampler_3d_backward", std::vector<c10::IValue>({grad_output, input, grid}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& input_ = unpack(input, "input", 1);
  auto& grid_ = unpack(grid, "grid", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, input, grid )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("grid_sampler_3d_backward"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, input, grid ));
  }
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::grid_sampler_3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "grid", grid);
    jit::tracer::addInputs(node, "interpolation_mode", interpolation_mode);
    jit::tracer::addInputs(node, "padding_mode", padding_mode);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  c10::optional<Storage> grid__storage_saved =
    grid_.has_storage() ? c10::optional<Storage>(grid_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grid__impl_saved;
  if (grid_.defined()) grid__impl_saved = grid_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::grid_sampler_3d_backward(grad_output_, input_, grid_, interpolation_mode, padding_mode, align_corners);
  })();
  std::tie(result0, result1) = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (grid__storage_saved.has_value())
    AT_ASSERT(grid__storage_saved.value().is_alias_of(grid_.storage()));
  if (grid__impl_saved) AT_ASSERT(grid__impl_saved == grid_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor VariableType::gru_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh) {
  RECORD_FUNCTION("gru_cell", std::vector<c10::IValue>({input, hx, w_ih, w_hh, b_ih, b_hh}), Node::peek_at_next_sequence_nr());
  auto result = TypeDefault::gru_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
  return result;
}
Tensor VariableType::hann_window(int64_t window_length, const TensorOptions & options) {
  RECORD_FUNCTION("hann_window", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hann_window");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::hann_window(window_length, options);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::hann_window(int64_t window_length, bool periodic, const TensorOptions & options) {
  RECORD_FUNCTION("hann_window", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hann_window");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "periodic", periodic);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::hann_window(window_length, periodic, options);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::hardshrink(const Tensor & self, Scalar lambd) {
  RECORD_FUNCTION("hardshrink", std::vector<c10::IValue>({self, lambd}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<HardshrinkBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<HardshrinkBackward>(new HardshrinkBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->lambd = lambd;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hardshrink");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "lambd", lambd);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::hardshrink(self_, lambd);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::ifft(const Tensor & self, int64_t signal_ndim, bool normalized) {
  RECORD_FUNCTION("ifft", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ifft");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "signal_ndim", signal_ndim);
    jit::tracer::addInputs(node, "normalized", normalized);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::ifft(self, signal_ndim, normalized);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::index_select_out(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index) {
  RECORD_FUNCTION("index_select_out", std::vector<c10::IValue>({out, self, index}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& index_ = unpack(index, "index", 3);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("index_select");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("index_select");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::index_select");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("index_select_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> index__storage_saved =
    index_.has_storage() ? c10::optional<Storage>(index_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> index__impl_saved;
  if (index_.defined()) index__impl_saved = index_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::index_select_out(out_, self_, dim, index_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (index__storage_saved.has_value())
    AT_ASSERT(index__storage_saved.value().is_alias_of(index_.storage()));
  if (index__impl_saved) AT_ASSERT(index__impl_saved == index_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::index_select_out(Tensor & out, const Tensor & self, Dimname dim, const Tensor & index) {
  RECORD_FUNCTION("index_select_out", std::vector<c10::IValue>({out, self, index}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::index_select");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("index_select_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::index_select_out(out, self, dim, index);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::indices(const Tensor & self) {
  RECORD_FUNCTION("indices", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::indices");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return self_.indices();
  })();
  auto result = as_view(self, tmp, false);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
bool VariableType::is_complex(const Tensor & self) {
  RECORD_FUNCTION("is_complex", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto result = TypeDefault::is_complex(self);
  return result;
}
bool VariableType::is_same_size(const Tensor & self, const Tensor & other) {
  auto result = TypeDefault::is_same_size(self, other);
  return result;
}
Tensor & VariableType::l1_loss_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction) {
  RECORD_FUNCTION("l1_loss_out", std::vector<c10::IValue>({out, self, target}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, target )) {
    throw_error_out_requires_grad("l1_loss");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("l1_loss");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::l1_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("l1_loss_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::l1_loss_out(out_, self_, target_, reduction);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::layer_norm(const Tensor & input, IntArrayRef normalized_shape, const Tensor & weight, const Tensor & bias, double eps, bool cudnn_enable) {
  RECORD_FUNCTION("layer_norm", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::layer_norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "normalized_shape", normalized_shape);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "cudnn_enable", cudnn_enable);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::layer_norm(input, normalized_shape, weight, bias, eps, cudnn_enable);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::leaky_relu_backward(const Tensor & grad_output, const Tensor & self, Scalar negative_slope) {
  RECORD_FUNCTION("leaky_relu_backward", std::vector<c10::IValue>({grad_output, self, negative_slope}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<LeakyReluBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<LeakyReluBackwardBackward>(new LeakyReluBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->negative_slope = negative_slope;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::leaky_relu_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "negative_slope", negative_slope);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::leaky_relu_backward(grad_output_, self_, negative_slope);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::lerp(const Tensor & self, const Tensor & end, Scalar weight) {
  RECORD_FUNCTION("lerp", std::vector<c10::IValue>({self, end, weight}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& end_ = unpack(end, "end", 1);
  std::shared_ptr<LerpBackward0> grad_fn;
  if (compute_requires_grad( self, end )) {
    grad_fn = std::shared_ptr<LerpBackward0>(new LerpBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, end ));
    grad_fn->weight = weight;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::lerp");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "weight", weight);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> end__storage_saved =
    end_.has_storage() ? c10::optional<Storage>(end_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> end__impl_saved;
  if (end_.defined()) end__impl_saved = end_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::lerp(self_, end_, weight);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (end__storage_saved.has_value())
    AT_ASSERT(end__storage_saved.value().is_alias_of(end_.storage()));
  if (end__impl_saved) AT_ASSERT(end__impl_saved == end_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::lerp(const Tensor & self, const Tensor & end, const Tensor & weight) {
  RECORD_FUNCTION("lerp", std::vector<c10::IValue>({self, end, weight}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& end_ = unpack(end, "end", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  check_no_requires_grad(weight, "weight");
  std::shared_ptr<LerpBackward1> grad_fn;
  if (compute_requires_grad( self, end )) {
    grad_fn = std::shared_ptr<LerpBackward1>(new LerpBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, end ));
    grad_fn->weight_ = SavedVariable(weight, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::lerp");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "weight", weight);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> end__storage_saved =
    end_.has_storage() ? c10::optional<Storage>(end_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> end__impl_saved;
  if (end_.defined()) end__impl_saved = end_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::lerp(self_, end_, weight_);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (end__storage_saved.has_value())
    AT_ASSERT(end__storage_saved.value().is_alias_of(end_.storage()));
  if (end__impl_saved) AT_ASSERT(end__impl_saved == end_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::lerp_(Tensor & self, const Tensor & end, Scalar weight) {
  RECORD_FUNCTION("lerp_", std::vector<c10::IValue>({self, end, weight}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& end_ = unpack(end, "end", 1);
  check_inplace(self);
  std::shared_ptr<LerpBackward0> grad_fn;
  if (compute_requires_grad( self, end )) {
    grad_fn = std::shared_ptr<LerpBackward0>(new LerpBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, end ));
    grad_fn->weight = weight;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::lerp");
    } else {
      op_name = jit::Symbol::fromQualString("aten::lerp_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "weight", weight);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("lerp_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> end__storage_saved =
    end_.has_storage() ? c10::optional<Storage>(end_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> end__impl_saved;
  if (end_.defined()) end__impl_saved = end_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.lerp_(end_, weight);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (end__storage_saved.has_value())
    AT_ASSERT(end__storage_saved.value().is_alias_of(end_.storage()));
  if (end__impl_saved) AT_ASSERT(end__impl_saved == end_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::lerp_(Tensor & self, const Tensor & end, const Tensor & weight) {
  RECORD_FUNCTION("lerp_", std::vector<c10::IValue>({self, end, weight}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& end_ = unpack(end, "end", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  check_inplace(self);
  check_no_requires_grad(weight, "weight");
  std::shared_ptr<LerpBackward1> grad_fn;
  if (compute_requires_grad( self, end )) {
    grad_fn = std::shared_ptr<LerpBackward1>(new LerpBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, end ));
    grad_fn->weight_ = SavedVariable(weight, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::lerp");
    } else {
      op_name = jit::Symbol::fromQualString("aten::lerp_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "weight", weight);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("lerp_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> end__storage_saved =
    end_.has_storage() ? c10::optional<Storage>(end_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> end__impl_saved;
  if (end_.defined()) end__impl_saved = end_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.lerp_(end_, weight_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (end__storage_saved.has_value())
    AT_ASSERT(end__storage_saved.value().is_alias_of(end_.storage()));
  if (end__impl_saved) AT_ASSERT(end__impl_saved == end_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::linear(const Tensor & input, const Tensor & weight, const Tensor & bias) {
  RECORD_FUNCTION("linear", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  auto result = TypeDefault::linear(input, weight, bias);
  return result;
}
Tensor VariableType::log_sigmoid(const Tensor & self) {
  RECORD_FUNCTION("log_sigmoid", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::log_sigmoid");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::log_sigmoid(self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::log_sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & buffer) {
  RECORD_FUNCTION("log_sigmoid_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, buffer}), Node::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& buffer_ = unpack(buffer, "buffer", 3);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self, buffer )) {
    throw_error_out_requires_grad("log_sigmoid_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("log_sigmoid_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::log_sigmoid_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "buffer", buffer);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("log_sigmoid_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> buffer__storage_saved =
    buffer_.has_storage() ? c10::optional<Storage>(buffer_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> buffer__impl_saved;
  if (buffer_.defined()) buffer__impl_saved = buffer_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::log_sigmoid_backward_out(grad_input_, grad_output_, self_, buffer_);
  }
  #ifndef NDEBUG
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (buffer__storage_saved.has_value())
    AT_ASSERT(buffer__storage_saved.value().is_alias_of(buffer_.storage()));
  if (buffer__impl_saved) AT_ASSERT(buffer__impl_saved == buffer_.getIntrusivePtr());
  #endif
  increment_version(grad_input);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::logical_xor_out(Tensor & out, const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("logical_xor_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("logical_xor");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("logical_xor");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::logical_xor");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logical_xor_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::logical_xor_out(out_, self_, other_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::logspace_out(Tensor & out, Scalar start, Scalar end, int64_t steps, double base) {
  RECORD_FUNCTION("logspace_out", std::vector<c10::IValue>({out, start, end}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::logspace");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "steps", steps);
    jit::tracer::addInputs(node, "base", base);
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", out.options());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logspace_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::logspace_out(out_, start, end, steps, base);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::logsumexp_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim) {
  RECORD_FUNCTION("logsumexp_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("logsumexp");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("logsumexp");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::logsumexp");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logsumexp_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::logsumexp_out(out_, self_, dim, keepdim);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::logsumexp_out(Tensor & out, const Tensor & self, DimnameList dim, bool keepdim) {
  RECORD_FUNCTION("logsumexp_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::logsumexp");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logsumexp_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::logsumexp_out(out, self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::matmul_out(Tensor & out, const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("matmul_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::matmul");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("matmul_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::matmul_out(out, self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
std::tuple<Tensor &,Tensor &> VariableType::max_out(Tensor & max, Tensor & max_values, const Tensor & self, int64_t dim, bool keepdim) {
  RECORD_FUNCTION("max_out", std::vector<c10::IValue>({max, max_values, self}), Node::peek_at_next_sequence_nr());
  auto& max_ = unpack(max, "max", 0);
  auto& max_values_ = unpack(max_values, "max_values", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("max");
  }
  if (compute_requires_grad( max )) {
    throw_error_out_requires_grad("max");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "max_values", max_values);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "max", max);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("max_out", max);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> max__storage_saved =
    max_.has_storage() ? c10::optional<Storage>(max_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> max__impl_saved;
  if (max_.defined()) max__impl_saved = max_.getIntrusivePtr();
  c10::optional<Storage> max_values__storage_saved =
    max_values_.has_storage() ? c10::optional<Storage>(max_values_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> max_values__impl_saved;
  if (max_values_.defined()) max_values__impl_saved = max_values_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::max_out(max_, max_values_, self_, dim, keepdim);
  }
  #ifndef NDEBUG
  if (max__storage_saved.has_value())
    AT_ASSERT(max__storage_saved.value().is_alias_of(max_.storage()));
  if (max__impl_saved) AT_ASSERT(max__impl_saved == max_.getIntrusivePtr());
  if (max_values__storage_saved.has_value())
    AT_ASSERT(max_values__storage_saved.value().is_alias_of(max_values_.storage()));
  if (max_values__impl_saved) AT_ASSERT(max_values__impl_saved == max_values_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(max);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( max ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, max);
    jit::tracer::addOutput(node, max_values);
  }
  return std::forward_as_tuple(max, max_values);
}
std::tuple<Tensor &,Tensor &> VariableType::max_out(Tensor & max, Tensor & max_values, const Tensor & self, Dimname dim, bool keepdim) {
  RECORD_FUNCTION("max_out", std::vector<c10::IValue>({max, max_values, self}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "max_values", max_values);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "max", max);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("max_out", max);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::max_out(max, max_values, self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, max);
    jit::tracer::addOutput(node, max_values);
  }
  return std::forward_as_tuple(max, max_values);
}
Tensor & VariableType::max_out(Tensor & out, const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("max_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("max");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("max");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("max_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::max_out(out_, self_, other_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::max_unpool2d(const Tensor & self, const Tensor & indices, IntArrayRef output_size) {
  RECORD_FUNCTION("max_unpool2d", std::vector<c10::IValue>({self, indices}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  std::shared_ptr<MaxUnpool2DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MaxUnpool2DBackward>(new MaxUnpool2DBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->indices_ = SavedVariable(indices, false);
    grad_fn->output_size = output_size.vec();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max_unpool2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::max_unpool2d(self_, indices_, output_size);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::max_unpool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size) {
  RECORD_FUNCTION("max_unpool2d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& indices_ = unpack(indices, "indices", 3);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("max_unpool2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("max_unpool2d_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max_unpool2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "output_size", output_size);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("max_unpool2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::max_unpool2d_backward_out(grad_input_, grad_output_, self_, indices_, output_size);
  }
  #ifndef NDEBUG
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  #endif
  increment_version(grad_input);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::max_unpool3d_out(Tensor & out, const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) {
  RECORD_FUNCTION("max_unpool3d_out", std::vector<c10::IValue>({out, self, indices}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("max_unpool3d");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("max_unpool3d");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max_unpool3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("max_unpool3d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::max_unpool3d_out(out_, self_, indices_, output_size, stride, padding);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::min_values(const Tensor & self, IntArrayRef dim, bool keepdim) {
  RECORD_FUNCTION("min_values", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::min_values");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::min_values(self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::min_values(const Tensor & self, DimnameList dim, bool keepdim) {
  RECORD_FUNCTION("min_values", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::min_values");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::min_values(self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::miopen_convolution_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask) {
  RECORD_FUNCTION("miopen_convolution_backward", std::vector<c10::IValue>({self, grad_output, weight}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  std::shared_ptr<MiopenConvolutionBackwardBackward> grad_fn;
  if (compute_requires_grad( self, grad_output, weight )) {
    grad_fn = std::shared_ptr<MiopenConvolutionBackwardBackward>(new MiopenConvolutionBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, grad_output, weight ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->padding = padding.vec();
    grad_fn->stride = stride.vec();
    grad_fn->dilation = dilation.vec();
    grad_fn->groups = groups;
    grad_fn->benchmark = benchmark;
    grad_fn->deterministic = deterministic;
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::miopen_convolution_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "benchmark", benchmark);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::miopen_convolution_backward(self_, grad_output_, weight_, padding, stride, dilation, groups, benchmark, deterministic, output_mask);
  })();
  std::tie(result0, result1, result2) = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1, result2 ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor VariableType::miopen_convolution_backward_bias(const Tensor & grad_output) {
  RECORD_FUNCTION("miopen_convolution_backward_bias", std::vector<c10::IValue>({grad_output}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("miopen_convolution_backward_bias"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::miopen_convolution_backward_bias");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::miopen_convolution_backward_bias(grad_output_);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::miopen_convolution_backward_input(IntArrayRef self_size, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  RECORD_FUNCTION("miopen_convolution_backward_input", std::vector<c10::IValue>({grad_output, weight}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, weight )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("miopen_convolution_backward_input"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, weight ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::miopen_convolution_backward_input");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self_size", self_size);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "benchmark", benchmark);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::miopen_convolution_backward_input(self_size, grad_output_, weight_, padding, stride, dilation, groups, benchmark, deterministic);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::miopen_convolution_transpose(const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  RECORD_FUNCTION("miopen_convolution_transpose", std::vector<c10::IValue>({self, weight, bias}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 2);
  std::shared_ptr<MiopenConvolutionTransposeBackward> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::shared_ptr<MiopenConvolutionTransposeBackward>(new MiopenConvolutionTransposeBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->padding = padding.vec();
    grad_fn->output_padding = output_padding.vec();
    grad_fn->stride = stride.vec();
    grad_fn->dilation = dilation.vec();
    grad_fn->groups = groups;
    grad_fn->benchmark = benchmark;
    grad_fn->deterministic = deterministic;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::miopen_convolution_transpose");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "benchmark", benchmark);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> bias__storage_saved =
    bias_.has_storage() ? c10::optional<Storage>(bias_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> bias__impl_saved;
  if (bias_.defined()) bias__impl_saved = bias_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::miopen_convolution_transpose(self_, weight_, bias_, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (bias__storage_saved.has_value())
    AT_ASSERT(bias__storage_saved.value().is_alias_of(bias_.storage()));
  if (bias__impl_saved) AT_ASSERT(bias__impl_saved == bias_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::mkldnn_adaptive_avg_pool2d(const Tensor & self, IntArrayRef output_size) {
  RECORD_FUNCTION("mkldnn_adaptive_avg_pool2d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("mkldnn_adaptive_avg_pool2d"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mkldnn_adaptive_avg_pool2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::mkldnn_adaptive_avg_pool2d(self_, output_size);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::mkldnn_convolution(const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups) {
  RECORD_FUNCTION("mkldnn_convolution", std::vector<c10::IValue>({self, weight, bias}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 2);
  std::shared_ptr<MkldnnConvolutionBackward> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::shared_ptr<MkldnnConvolutionBackward>(new MkldnnConvolutionBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->padding = padding.vec();
    grad_fn->stride = stride.vec();
    grad_fn->dilation = dilation.vec();
    grad_fn->groups = groups;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mkldnn_convolution");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> bias__storage_saved =
    bias_.has_storage() ? c10::optional<Storage>(bias_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> bias__impl_saved;
  if (bias_.defined()) bias__impl_saved = bias_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::mkldnn_convolution(self_, weight_, bias_, padding, stride, dilation, groups);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (bias__storage_saved.has_value())
    AT_ASSERT(bias__storage_saved.value().is_alias_of(bias_.storage()));
  if (bias__impl_saved) AT_ASSERT(bias__impl_saved == bias_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::mkldnn_reorder_conv2d_weight(const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups) {
  RECORD_FUNCTION("mkldnn_reorder_conv2d_weight", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("mkldnn_reorder_conv2d_weight"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mkldnn_reorder_conv2d_weight");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::mkldnn_reorder_conv2d_weight(self_, padding, stride, dilation, groups);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor &,Tensor &> VariableType::mode_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) {
  RECORD_FUNCTION("mode_out", std::vector<c10::IValue>({values, indices, self}), Node::peek_at_next_sequence_nr());
  auto& values_ = unpack(values, "values", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("mode");
  }
  if (compute_requires_grad( values )) {
    throw_error_out_requires_grad("mode");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mode");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "values", values);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mode_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> values__storage_saved =
    values_.has_storage() ? c10::optional<Storage>(values_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> values__impl_saved;
  if (values_.defined()) values__impl_saved = values_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::mode_out(values_, indices_, self_, dim, keepdim);
  }
  #ifndef NDEBUG
  if (values__storage_saved.has_value())
    AT_ASSERT(values__storage_saved.value().is_alias_of(values_.storage()));
  if (values__impl_saved) AT_ASSERT(values__impl_saved == values_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(values);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( values ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(values, indices);
}
std::tuple<Tensor &,Tensor &> VariableType::mode_out(Tensor & values, Tensor & indices, const Tensor & self, Dimname dim, bool keepdim) {
  RECORD_FUNCTION("mode_out", std::vector<c10::IValue>({values, indices, self}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mode");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "values", values);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mode_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::mode_out(values, indices, self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(values, indices);
}
Tensor VariableType::mse_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  RECORD_FUNCTION("mse_loss_backward", std::vector<c10::IValue>({grad_output, self, target}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  check_no_requires_grad(target, "target");
  std::shared_ptr<MseLossBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<MseLossBackwardBackward>(new MseLossBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->reduction = reduction;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mse_loss_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::mse_loss_backward(grad_output_, self_, target_, reduction);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::multilabel_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, const Tensor & is_target) {
  RECORD_FUNCTION("multilabel_margin_loss_backward", std::vector<c10::IValue>({grad_output, self, target, is_target}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  auto& is_target_ = unpack(is_target, "is_target", 4);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, self, target, is_target )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("multilabel_margin_loss_backward"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, target, is_target ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::multilabel_margin_loss_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "is_target", is_target);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  c10::optional<Storage> is_target__storage_saved =
    is_target_.has_storage() ? c10::optional<Storage>(is_target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> is_target__impl_saved;
  if (is_target_.defined()) is_target__impl_saved = is_target_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::multilabel_margin_loss_backward(grad_output_, self_, target_, reduction, is_target_);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  if (is_target__storage_saved.has_value())
    AT_ASSERT(is_target__storage_saved.value().is_alias_of(is_target_.storage()));
  if (is_target__impl_saved) AT_ASSERT(is_target__impl_saved == is_target_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor &,Tensor &> VariableType::multilabel_margin_loss_forward_out(Tensor & output, Tensor & is_target, const Tensor & self, const Tensor & target, int64_t reduction) {
  RECORD_FUNCTION("multilabel_margin_loss_forward_out", std::vector<c10::IValue>({output, is_target, self, target}), Node::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& is_target_ = unpack(is_target, "is_target", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& target_ = unpack(target, "target", 3);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("multilabel_margin_loss_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("multilabel_margin_loss_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::multilabel_margin_loss_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "is_target", is_target);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("multilabel_margin_loss_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  c10::optional<Storage> is_target__storage_saved =
    is_target_.has_storage() ? c10::optional<Storage>(is_target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> is_target__impl_saved;
  if (is_target_.defined()) is_target__impl_saved = is_target_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::multilabel_margin_loss_forward_out(output_, is_target_, self_, target_, reduction);
  }
  #ifndef NDEBUG
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  if (is_target__storage_saved.has_value())
    AT_ASSERT(is_target__storage_saved.value().is_alias_of(is_target_.storage()));
  if (is_target__impl_saved) AT_ASSERT(is_target__impl_saved == is_target_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  #endif
  increment_version(output);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( output ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, is_target);
  }
  return std::forward_as_tuple(output, is_target);
}
Tensor VariableType::multinomial(const Tensor & self, int64_t num_samples, bool replacement, Generator * generator) {
  RECORD_FUNCTION("multinomial", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::multinomial");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "num_samples", num_samples);
    jit::tracer::addInputs(node, "replacement", replacement);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::multinomial(self_, num_samples, replacement, generator);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::mvlgamma(const Tensor & self, int64_t p) {
  RECORD_FUNCTION("mvlgamma", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<MvlgammaBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MvlgammaBackward>(new MvlgammaBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->p = p;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mvlgamma");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::mvlgamma(self_, p);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::mvlgamma_(Tensor & self, int64_t p) {
  RECORD_FUNCTION("mvlgamma_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<MvlgammaBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MvlgammaBackward>(new MvlgammaBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
    grad_fn->p = p;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::mvlgamma");
    } else {
      op_name = jit::Symbol::fromQualString("aten::mvlgamma_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mvlgamma_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.mvlgamma_(p);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::narrow(const Tensor & self, int64_t dim, int64_t start, int64_t length) {
  RECORD_FUNCTION("narrow", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::narrow");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "length", length);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::narrow(self, dim, start, length);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::native_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) {
  RECORD_FUNCTION("native_batch_norm", std::vector<c10::IValue>({input, weight, bias, running_mean, running_var}), Node::peek_at_next_sequence_nr());
  auto& input_ = unpack(input, "input", 0);
  auto weight_ = unpack_opt(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 2);
  auto running_mean_ = unpack_opt(running_mean, "running_mean", 3);
  auto running_var_ = unpack_opt(running_var, "running_var", 4);
  check_no_requires_grad(running_mean, "running_mean");
  check_no_requires_grad(running_var, "running_var");
  std::shared_ptr<NativeBatchNormBackward> grad_fn;
  if (compute_requires_grad( input, weight, bias )) {
    grad_fn = std::shared_ptr<NativeBatchNormBackward>(new NativeBatchNormBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, weight, bias ));
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->running_mean_ = SavedVariable(running_mean, false);
    grad_fn->running_var_ = SavedVariable(running_var, false);
    grad_fn->training = training;
    grad_fn->eps = eps;
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::native_batch_norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "momentum", momentum);
    jit::tracer::addInputs(node, "eps", eps);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> bias__storage_saved =
    bias_.has_storage() ? c10::optional<Storage>(bias_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> bias__impl_saved;
  if (bias_.defined()) bias__impl_saved = bias_.getIntrusivePtr();
  c10::optional<Storage> running_mean__storage_saved =
    running_mean_.has_storage() ? c10::optional<Storage>(running_mean_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> running_mean__impl_saved;
  if (running_mean_.defined()) running_mean__impl_saved = running_mean_.getIntrusivePtr();
  c10::optional<Storage> running_var__storage_saved =
    running_var_.has_storage() ? c10::optional<Storage>(running_var_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> running_var__impl_saved;
  if (running_var_.defined()) running_var__impl_saved = running_var_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::native_batch_norm(input_, weight_, bias_, running_mean_, running_var_, training, momentum, eps);
  })();
  std::tie(result0, result1, result2) = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (bias__storage_saved.has_value())
    AT_ASSERT(bias__storage_saved.value().is_alias_of(bias_.storage()));
  if (bias__impl_saved) AT_ASSERT(bias__impl_saved == bias_.getIntrusivePtr());
  if (running_mean__storage_saved.has_value())
    AT_ASSERT(running_mean__storage_saved.value().is_alias_of(running_mean_.storage()));
  if (running_mean__impl_saved) AT_ASSERT(running_mean__impl_saved == running_mean_.getIntrusivePtr());
  if (running_var__storage_saved.has_value())
    AT_ASSERT(running_var__storage_saved.value().is_alias_of(running_var_.storage()));
  if (running_var__impl_saved) AT_ASSERT(running_var__impl_saved == running_var_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0 ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
    grad_fn->result2_ = SavedVariable(result2, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor & VariableType::ne_out(Tensor & out, const Tensor & self, Scalar other) {
  RECORD_FUNCTION("ne_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ne");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ne_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::ne_out(out_, self_, other);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::ne_out(Tensor & out, const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("ne_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ne");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ne_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::ne_out(out_, self_, other_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::new_full(const Tensor & self, IntArrayRef size, Scalar fill_value, const TensorOptions & options) {
  RECORD_FUNCTION("new_full", std::vector<c10::IValue>({self, fill_value}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::new_full");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "fill_value", fill_value);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::new_full(self, size, fill_value, options);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::nll_loss(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {
  RECORD_FUNCTION("nll_loss", std::vector<c10::IValue>({self, target, weight}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::nll_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "ignore_index", ignore_index);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::nll_loss(self, target, weight, reduction, ignore_index);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::nll_loss2d(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {
  RECORD_FUNCTION("nll_loss2d", std::vector<c10::IValue>({self, target, weight}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::nll_loss2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "ignore_index", ignore_index);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::nll_loss2d(self, target, weight, reduction, ignore_index);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::nll_loss2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) {
  RECORD_FUNCTION("nll_loss2d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, target, weight, total_weight}), Node::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& target_ = unpack(target, "target", 3);
  auto weight_ = unpack_opt(weight, "weight", 4);
  auto& total_weight_ = unpack(total_weight, "total_weight", 7);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self, weight, total_weight )) {
    throw_error_out_requires_grad("nll_loss2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("nll_loss2d_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::nll_loss2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "ignore_index", ignore_index);
    jit::tracer::addInputs(node, "total_weight", total_weight);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("nll_loss2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> total_weight__storage_saved =
    total_weight_.has_storage() ? c10::optional<Storage>(total_weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> total_weight__impl_saved;
  if (total_weight_.defined()) total_weight__impl_saved = total_weight_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::nll_loss2d_backward_out(grad_input_, grad_output_, self_, target_, weight_, reduction, ignore_index, total_weight_);
  }
  #ifndef NDEBUG
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (total_weight__storage_saved.has_value())
    AT_ASSERT(total_weight__storage_saved.value().is_alias_of(total_weight_.storage()));
  if (total_weight__impl_saved) AT_ASSERT(total_weight__impl_saved == total_weight_.getIntrusivePtr());
  #endif
  increment_version(grad_input);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::nll_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) {
  RECORD_FUNCTION("nll_loss_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, target, weight, total_weight}), Node::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& target_ = unpack(target, "target", 3);
  auto weight_ = unpack_opt(weight, "weight", 4);
  auto& total_weight_ = unpack(total_weight, "total_weight", 7);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self, weight, total_weight )) {
    throw_error_out_requires_grad("nll_loss_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("nll_loss_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::nll_loss_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "ignore_index", ignore_index);
    jit::tracer::addInputs(node, "total_weight", total_weight);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("nll_loss_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> total_weight__storage_saved =
    total_weight_.has_storage() ? c10::optional<Storage>(total_weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> total_weight__impl_saved;
  if (total_weight_.defined()) total_weight__impl_saved = total_weight_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::nll_loss_backward_out(grad_input_, grad_output_, self_, target_, weight_, reduction, ignore_index, total_weight_);
  }
  #ifndef NDEBUG
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (total_weight__storage_saved.has_value())
    AT_ASSERT(total_weight__storage_saved.value().is_alias_of(total_weight_.storage()));
  if (total_weight__impl_saved) AT_ASSERT(total_weight__impl_saved == total_weight_.getIntrusivePtr());
  #endif
  increment_version(grad_input);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::nuclear_norm(const Tensor & self, bool keepdim) {
  RECORD_FUNCTION("nuclear_norm", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::nuclear_norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::nuclear_norm(self, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::nuclear_norm(const Tensor & self, IntArrayRef dim, bool keepdim) {
  RECORD_FUNCTION("nuclear_norm", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::nuclear_norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::nuclear_norm(self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
int64_t VariableType::numel(const Tensor & self) {
  auto result = TypeDefault::numel(self);
  return result;
}
Tensor VariableType::orgqr(const Tensor & self, const Tensor & input2) {
  RECORD_FUNCTION("orgqr", std::vector<c10::IValue>({self, input2}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& input2_ = unpack(input2, "input2", 1);
  std::shared_ptr<OrgqrBackward> grad_fn;
  if (compute_requires_grad( self, input2 )) {
    grad_fn = std::shared_ptr<OrgqrBackward>(new OrgqrBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, input2 ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::orgqr");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "input2", input2);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> input2__storage_saved =
    input2_.has_storage() ? c10::optional<Storage>(input2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input2__impl_saved;
  if (input2_.defined()) input2__impl_saved = input2_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::orgqr(self_, input2_);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (input2__storage_saved.has_value())
    AT_ASSERT(input2__storage_saved.value().is_alias_of(input2_.storage()));
  if (input2__impl_saved) AT_ASSERT(input2__impl_saved == input2_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::ormqr_out(Tensor & out, const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) {
  RECORD_FUNCTION("ormqr_out", std::vector<c10::IValue>({out, self, input2, input3}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& input2_ = unpack(input2, "input2", 2);
  auto& input3_ = unpack(input3, "input3", 3);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, input2, input3 )) {
    throw_error_out_requires_grad("ormqr");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("ormqr");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ormqr");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "input2", input2);
    jit::tracer::addInputs(node, "input3", input3);
    jit::tracer::addInputs(node, "left", left);
    jit::tracer::addInputs(node, "transpose", transpose);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ormqr_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> input2__storage_saved =
    input2_.has_storage() ? c10::optional<Storage>(input2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input2__impl_saved;
  if (input2_.defined()) input2__impl_saved = input2_.getIntrusivePtr();
  c10::optional<Storage> input3__storage_saved =
    input3_.has_storage() ? c10::optional<Storage>(input3_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input3__impl_saved;
  if (input3_.defined()) input3__impl_saved = input3_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::ormqr_out(out_, self_, input2_, input3_, left, transpose);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (input2__storage_saved.has_value())
    AT_ASSERT(input2__storage_saved.value().is_alias_of(input2_.storage()));
  if (input2__impl_saved) AT_ASSERT(input2__impl_saved == input2_.getIntrusivePtr());
  if (input3__storage_saved.has_value())
    AT_ASSERT(input3__storage_saved.value().is_alias_of(input3_.storage()));
  if (input3__impl_saved) AT_ASSERT(input3__impl_saved == input3_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::permute(const Tensor & self, IntArrayRef dims) {
  RECORD_FUNCTION("permute", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<PermuteBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<PermuteBackward>(new PermuteBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->dims = dims.vec();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::permute");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dims", dims);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return self_.permute(dims);
  })();
  auto result = as_view(self, tmp, true);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::pixel_shuffle(const Tensor & self, int64_t upscale_factor) {
  RECORD_FUNCTION("pixel_shuffle", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::pixel_shuffle");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "upscale_factor", upscale_factor);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::pixel_shuffle(self, upscale_factor);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::put_(Tensor & self, const Tensor & index, const Tensor & source, bool accumulate) {
  RECORD_FUNCTION("put_", std::vector<c10::IValue>({self, index, source}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 1);
  auto& source_ = unpack(source, "source", 2);
  check_inplace(self);
  std::shared_ptr<PutBackward> grad_fn;
  if (compute_requires_grad( self, source )) {
    grad_fn = std::shared_ptr<PutBackward>(new PutBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, source ));
    grad_fn->index_ = SavedVariable(index, false);
    grad_fn->source_info = source;
    grad_fn->accumulate = accumulate;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::put");
    } else {
      op_name = jit::Symbol::fromQualString("aten::put_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "source", source);
    jit::tracer::addInputs(node, "accumulate", accumulate);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("put_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> index__storage_saved =
    index_.has_storage() ? c10::optional<Storage>(index_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> index__impl_saved;
  if (index_.defined()) index__impl_saved = index_.getIntrusivePtr();
  c10::optional<Storage> source__storage_saved =
    source_.has_storage() ? c10::optional<Storage>(source_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> source__impl_saved;
  if (source_.defined()) source__impl_saved = source_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.put_(index_, source_, accumulate);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (index__storage_saved.has_value())
    AT_ASSERT(index__storage_saved.value().is_alias_of(index_.storage()));
  if (index__impl_saved) AT_ASSERT(index__impl_saved == index_.getIntrusivePtr());
  if (source__storage_saved.has_value())
    AT_ASSERT(source__storage_saved.value().is_alias_of(source_.storage()));
  if (source__impl_saved) AT_ASSERT(source__impl_saved == source_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
int64_t VariableType::q_zero_point(const Tensor & self) {
  RECORD_FUNCTION("q_zero_point", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::q_zero_point(self_);
  })();
  auto result = tmp;
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  return result;
}
Tensor VariableType::quantize_per_tensor(const Tensor & self, double scale, int64_t zero_point, ScalarType dtype) {
  RECORD_FUNCTION("quantize_per_tensor", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::quantize_per_tensor");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::quantize_per_tensor(self_, scale, zero_point, dtype);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::quantized_gru(const Tensor & input, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
  RECORD_FUNCTION("quantized_gru", std::vector<c10::IValue>({input, hx}), Node::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::quantized_gru");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "params", params);
    jit::tracer::addInputs(node, "has_biases", has_biases);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = TypeDefault::quantized_gru(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
std::tuple<Tensor,Tensor> VariableType::quantized_gru(const Tensor & data, const Tensor & batch_sizes, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) {
  RECORD_FUNCTION("quantized_gru", std::vector<c10::IValue>({data, batch_sizes, hx}), Node::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::quantized_gru");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "data", data);
    jit::tracer::addInputs(node, "batch_sizes", batch_sizes);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "params", params);
    jit::tracer::addInputs(node, "has_biases", has_biases);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = TypeDefault::quantized_gru(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
std::tuple<Tensor,Tensor> VariableType::quantized_lstm_cell(const Tensor & input, TensorList hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh, const Tensor & packed_ih, const Tensor & packed_hh, const Tensor & col_offsets_ih, const Tensor & col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) {
  RECORD_FUNCTION("quantized_lstm_cell", std::vector<c10::IValue>({input, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh}), Node::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::quantized_lstm_cell");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "w_ih", w_ih);
    jit::tracer::addInputs(node, "w_hh", w_hh);
    jit::tracer::addInputs(node, "b_ih", b_ih);
    jit::tracer::addInputs(node, "b_hh", b_hh);
    jit::tracer::addInputs(node, "packed_ih", packed_ih);
    jit::tracer::addInputs(node, "packed_hh", packed_hh);
    jit::tracer::addInputs(node, "col_offsets_ih", col_offsets_ih);
    jit::tracer::addInputs(node, "col_offsets_hh", col_offsets_hh);
    jit::tracer::addInputs(node, "scale_ih", scale_ih);
    jit::tracer::addInputs(node, "scale_hh", scale_hh);
    jit::tracer::addInputs(node, "zero_point_ih", zero_point_ih);
    jit::tracer::addInputs(node, "zero_point_hh", zero_point_hh);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = TypeDefault::quantized_lstm_cell(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor & VariableType::rand_out(Tensor & out, IntArrayRef size) {
  RECORD_FUNCTION("rand_out", std::vector<c10::IValue>({out}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rand");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", out.options());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("rand_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::rand_out(out, size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::rand_out(Tensor & out, IntArrayRef size, Generator * generator) {
  RECORD_FUNCTION("rand_out", std::vector<c10::IValue>({out}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rand");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "generator", generator);
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", out.options());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("rand_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::rand_out(out, size, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::randn_out(Tensor & out, IntArrayRef size) {
  RECORD_FUNCTION("randn_out", std::vector<c10::IValue>({out}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randn");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", out.options());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("randn_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::randn_out(out, size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::randn_out(Tensor & out, IntArrayRef size, Generator * generator) {
  RECORD_FUNCTION("randn_out", std::vector<c10::IValue>({out}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randn");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "generator", generator);
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", out.options());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("randn_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::randn_out(out, size, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::range(Scalar start, Scalar end, Scalar step, const TensorOptions & options) {
  RECORD_FUNCTION("range", std::vector<c10::IValue>({start, end, step}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::range");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "step", step);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::range(start, end, step, options);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::range(Scalar start, Scalar end, const TensorOptions & options) {
  RECORD_FUNCTION("range", std::vector<c10::IValue>({start, end}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::range");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::range(start, end, options);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::reciprocal(const Tensor & self) {
  RECORD_FUNCTION("reciprocal", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ReciprocalBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ReciprocalBackward>(new ReciprocalBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::reciprocal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::reciprocal(self_);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor & VariableType::reciprocal_(Tensor & self) {
  RECORD_FUNCTION("reciprocal_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<ReciprocalBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ReciprocalBackward>(new ReciprocalBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::reciprocal");
    } else {
      op_name = jit::Symbol::fromQualString("aten::reciprocal_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("reciprocal_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::reciprocal_(self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(self, true, as_variable_ref(self).is_view());
  }
  return self;
}
Tensor VariableType::refine_names(const Tensor & self, DimnameList names) {
  RECORD_FUNCTION("refine_names", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::refine_names");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "names", names);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::refine_names(self, names);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::reflection_pad1d(const Tensor & self, IntArrayRef padding) {
  RECORD_FUNCTION("reflection_pad1d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ReflectionPad1DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ReflectionPad1DBackward>(new ReflectionPad1DBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->padding = padding.vec();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::reflection_pad1d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::reflection_pad1d(self_, padding);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::reflection_pad1d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
  RECORD_FUNCTION("reflection_pad1d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self}), Node::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("reflection_pad1d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("reflection_pad1d_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::reflection_pad1d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("reflection_pad1d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::reflection_pad1d_backward_out(grad_input_, grad_output_, self_, padding);
  }
  #ifndef NDEBUG
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(grad_input);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::reflection_pad2d_out(Tensor & out, const Tensor & self, IntArrayRef padding) {
  RECORD_FUNCTION("reflection_pad2d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("reflection_pad2d");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("reflection_pad2d");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::reflection_pad2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("reflection_pad2d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::reflection_pad2d_out(out_, self_, padding);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::relu(const Tensor & self) {
  RECORD_FUNCTION("relu", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ReluBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ReluBackward0>(new ReluBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::relu");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::relu(self_);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::relu_(Tensor & self) {
  RECORD_FUNCTION("relu_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<ReluBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ReluBackward1>(new ReluBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::relu");
    } else {
      op_name = jit::Symbol::fromQualString("aten::relu_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("relu_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::relu_(self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(self, true, as_variable_ref(self).is_view());
  }
  return self;
}
Tensor & VariableType::remainder_out(Tensor & out, const Tensor & self, Scalar other) {
  RECORD_FUNCTION("remainder_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("remainder");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("remainder");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::remainder");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("remainder_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::remainder_out(out_, self_, other);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::remainder_out(Tensor & out, const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("remainder_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("remainder");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("remainder");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::remainder");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("remainder_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::remainder_out(out_, self_, other_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::replication_pad1d_out(Tensor & out, const Tensor & self, IntArrayRef padding) {
  RECORD_FUNCTION("replication_pad1d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("replication_pad1d");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("replication_pad1d");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::replication_pad1d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("replication_pad1d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::replication_pad1d_out(out_, self_, padding);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::rnn_relu_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh) {
  RECORD_FUNCTION("rnn_relu_cell", std::vector<c10::IValue>({input, hx, w_ih, w_hh, b_ih, b_hh}), Node::peek_at_next_sequence_nr());
  auto result = TypeDefault::rnn_relu_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
  return result;
}
Tensor VariableType::rrelu_with_noise(const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) {
  RECORD_FUNCTION("rrelu_with_noise", std::vector<c10::IValue>({self, noise, lower, upper}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& noise_ = unpack(noise, "noise", 1);
  check_no_requires_grad(noise, "noise");
  std::shared_ptr<RreluWithNoiseBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<RreluWithNoiseBackward0>(new RreluWithNoiseBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->noise_ = SavedVariable(noise, false);
    grad_fn->lower = lower;
    grad_fn->upper = upper;
    grad_fn->training = training;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rrelu_with_noise");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "noise", noise);
    jit::tracer::addInputs(node, "lower", lower);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> noise__storage_saved =
    noise_.has_storage() ? c10::optional<Storage>(noise_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> noise__impl_saved;
  if (noise_.defined()) noise__impl_saved = noise_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::rrelu_with_noise(self_, noise_, lower, upper, training, generator);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (noise__storage_saved.has_value())
    AT_ASSERT(noise__storage_saved.value().is_alias_of(noise_.storage()));
  if (noise__impl_saved) AT_ASSERT(noise__impl_saved == noise_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::rrelu_with_noise_(Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) {
  RECORD_FUNCTION("rrelu_with_noise_", std::vector<c10::IValue>({self, noise, lower, upper}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& noise_ = unpack(noise, "noise", 1);
  check_inplace(self);
  check_no_requires_grad(noise, "noise");
  std::shared_ptr<RreluWithNoiseBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<RreluWithNoiseBackward1>(new RreluWithNoiseBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->noise_ = SavedVariable(noise, false);
    grad_fn->lower = lower;
    grad_fn->upper = upper;
    grad_fn->training = training;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::rrelu_with_noise");
    } else {
      op_name = jit::Symbol::fromQualString("aten::rrelu_with_noise_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "noise", noise);
    jit::tracer::addInputs(node, "lower", lower);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("rrelu_with_noise_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> noise__storage_saved =
    noise_.has_storage() ? c10::optional<Storage>(noise_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> noise__impl_saved;
  if (noise_.defined()) noise__impl_saved = noise_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::rrelu_with_noise_(self_, noise_, lower, upper, training, generator);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (noise__storage_saved.has_value())
    AT_ASSERT(noise__storage_saved.value().is_alias_of(noise_.storage()));
  if (noise__impl_saved) AT_ASSERT(noise__impl_saved == noise_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(self, true, as_variable_ref(self).is_view());
  }
  return self;
}
Tensor & VariableType::rrelu_with_noise_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training) {
  RECORD_FUNCTION("rrelu_with_noise_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, noise, lower, upper}), Node::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& noise_ = unpack(noise, "noise", 3);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self, noise )) {
    throw_error_out_requires_grad("rrelu_with_noise_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("rrelu_with_noise_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rrelu_with_noise_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "noise", noise);
    jit::tracer::addInputs(node, "lower", lower);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "training", training);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("rrelu_with_noise_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> noise__storage_saved =
    noise_.has_storage() ? c10::optional<Storage>(noise_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> noise__impl_saved;
  if (noise_.defined()) noise__impl_saved = noise_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::rrelu_with_noise_backward_out(grad_input_, grad_output_, self_, noise_, lower, upper, training);
  }
  #ifndef NDEBUG
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (noise__storage_saved.has_value())
    AT_ASSERT(noise__storage_saved.value().is_alias_of(noise_.storage()));
  if (noise__impl_saved) AT_ASSERT(noise__impl_saved == noise_.getIntrusivePtr());
  #endif
  increment_version(grad_input);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::scatter_add(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) {
  RECORD_FUNCTION("scatter_add", std::vector<c10::IValue>({self, index, src}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::scatter_add");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "src", src);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::scatter_add(self, dim, index, src);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::scatter_add(const Tensor & self, Dimname dim, const Tensor & index, const Tensor & src) {
  RECORD_FUNCTION("scatter_add", std::vector<c10::IValue>({self, index, src}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::scatter_add");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "src", src);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::scatter_add(self, dim, index, src);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::scatter_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) {
  RECORD_FUNCTION("scatter_add_", std::vector<c10::IValue>({self, index, src}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  auto& src_ = unpack(src, "src", 3);
  check_inplace(self);
  std::shared_ptr<ScatterAddBackward> grad_fn;
  if (compute_requires_grad( self, src )) {
    grad_fn = std::shared_ptr<ScatterAddBackward>(new ScatterAddBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, src ));
    grad_fn->dim = dim;
    if (grad_fn->should_compute_output(1)) {
      grad_fn->index_ = SavedVariable(index, false);
    }
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::scatter_add");
    } else {
      op_name = jit::Symbol::fromQualString("aten::scatter_add_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "src", src);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("scatter_add_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> index__storage_saved =
    index_.has_storage() ? c10::optional<Storage>(index_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> index__impl_saved;
  if (index_.defined()) index__impl_saved = index_.getIntrusivePtr();
  c10::optional<Storage> src__storage_saved =
    src_.has_storage() ? c10::optional<Storage>(src_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> src__impl_saved;
  if (src_.defined()) src__impl_saved = src_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.scatter_add_(dim, index_, src_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (index__storage_saved.has_value())
    AT_ASSERT(index__storage_saved.value().is_alias_of(index_.storage()));
  if (index__impl_saved) AT_ASSERT(index__impl_saved == index_.getIntrusivePtr());
  if (src__storage_saved.has_value())
    AT_ASSERT(src__storage_saved.value().is_alias_of(src_.storage()));
  if (src__impl_saved) AT_ASSERT(src__impl_saved == src_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::select(const Tensor & self, Dimname dim, int64_t index) {
  RECORD_FUNCTION("select", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::select");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::select(self, dim, index);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::select(const Tensor & self, int64_t dim, int64_t index) {
  RECORD_FUNCTION("select", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SelectBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SelectBackward>(new SelectBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->dim = dim;
    grad_fn->index = index;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::select");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::select(self_, dim, index);
  })();
  auto result = as_view(self, tmp, true);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::sin(const Tensor & self) {
  RECORD_FUNCTION("sin", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SinBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SinBackward>(new SinBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sin");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::sin(self_);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::sin_(Tensor & self) {
  RECORD_FUNCTION("sin_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<SinBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SinBackward>(new SinBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::sin");
    } else {
      op_name = jit::Symbol::fromQualString("aten::sin_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sin_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::sin_(self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::slow_conv_dilated3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, std::array<bool,3> output_mask) {
  RECORD_FUNCTION("slow_conv_dilated3d_backward", std::vector<c10::IValue>({grad_output, self, weight}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  std::shared_ptr<SlowConvDilated3DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self, weight )) {
    grad_fn = std::shared_ptr<SlowConvDilated3DBackwardBackward>(new SlowConvDilated3DBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, weight ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->dilation = dilation.vec();
  }
  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::slow_conv_dilated3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::slow_conv_dilated3d_backward(grad_output_, self_, weight_, kernel_size, stride, padding, dilation, output_mask);
  })();
  std::tie(grad_input, grad_weight, grad_bias) = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( grad_input, grad_weight, grad_bias ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
    jit::tracer::addOutput(node, grad_weight);
    jit::tracer::addOutput(node, grad_bias);
  }
  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}
Tensor & VariableType::soft_margin_loss_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction) {
  RECORD_FUNCTION("soft_margin_loss_out", std::vector<c10::IValue>({out, self, target}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, target )) {
    throw_error_out_requires_grad("soft_margin_loss");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("soft_margin_loss");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::soft_margin_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("soft_margin_loss_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::soft_margin_loss_out(out_, self_, target_, reduction);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::softmax(const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) {
  RECORD_FUNCTION("softmax", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::softmax");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::softmax(self, dim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::softmax(const Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) {
  RECORD_FUNCTION("softmax", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::softmax");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::softmax(self, dim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::softplus_out(Tensor & out, const Tensor & self, Scalar beta, Scalar threshold) {
  RECORD_FUNCTION("softplus_out", std::vector<c10::IValue>({out, self, beta, threshold}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("softplus");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("softplus");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::softplus");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "threshold", threshold);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("softplus_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::softplus_out(out_, self_, beta, threshold);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::softshrink_backward(const Tensor & grad_output, const Tensor & self, Scalar lambd) {
  RECORD_FUNCTION("softshrink_backward", std::vector<c10::IValue>({grad_output, self, lambd}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<SoftshrinkBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<SoftshrinkBackwardBackward>(new SoftshrinkBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->lambd = lambd;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::softshrink_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "lambd", lambd);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::softshrink_backward(grad_output_, self_, lambd);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor &,Tensor &> VariableType::sort_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool descending) {
  RECORD_FUNCTION("sort_out", std::vector<c10::IValue>({values, indices, self}), Node::peek_at_next_sequence_nr());
  auto& values_ = unpack(values, "values", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("sort");
  }
  if (compute_requires_grad( values )) {
    throw_error_out_requires_grad("sort");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sort");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "descending", descending);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "values", values);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sort_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> values__storage_saved =
    values_.has_storage() ? c10::optional<Storage>(values_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> values__impl_saved;
  if (values_.defined()) values__impl_saved = values_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::sort_out(values_, indices_, self_, dim, descending);
  }
  #ifndef NDEBUG
  if (values__storage_saved.has_value())
    AT_ASSERT(values__storage_saved.value().is_alias_of(values_.storage()));
  if (values__impl_saved) AT_ASSERT(values__impl_saved == values_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(values);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( values ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(values, indices);
}
std::tuple<Tensor &,Tensor &> VariableType::sort_out(Tensor & values, Tensor & indices, const Tensor & self, Dimname dim, bool descending) {
  RECORD_FUNCTION("sort_out", std::vector<c10::IValue>({values, indices, self}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sort");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "descending", descending);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "values", values);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sort_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::sort_out(values, indices, self, dim, descending);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(values, indices);
}
Tensor VariableType::squeeze(const Tensor & self) {
  RECORD_FUNCTION("squeeze", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SqueezeBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SqueezeBackward0>(new SqueezeBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::squeeze");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::squeeze(self_);
  })();
  auto result = as_view(self, tmp, true);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::squeeze(const Tensor & self, int64_t dim) {
  RECORD_FUNCTION("squeeze", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SqueezeBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SqueezeBackward1>(new SqueezeBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->dim = dim;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::squeeze");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::squeeze(self_, dim);
  })();
  auto result = as_view(self, tmp, true);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::squeeze(const Tensor & self, Dimname dim) {
  RECORD_FUNCTION("squeeze", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::squeeze");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::squeeze(self, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::squeeze_(Tensor & self) {
  RECORD_FUNCTION("squeeze_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<SqueezeBackward2> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SqueezeBackward2>(new SqueezeBackward2(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::squeeze");
    } else {
      op_name = jit::Symbol::fromQualString("aten::squeeze_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("squeeze_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.squeeze_();
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      set_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::squeeze_(Tensor & self, int64_t dim) {
  RECORD_FUNCTION("squeeze_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<SqueezeBackward3> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SqueezeBackward3>(new SqueezeBackward3(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->dim = dim;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::squeeze");
    } else {
      op_name = jit::Symbol::fromQualString("aten::squeeze_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("squeeze_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.squeeze_(dim);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      set_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::squeeze_(Tensor & self, Dimname dim) {
  RECORD_FUNCTION("squeeze_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::squeeze");
    } else {
      op_name = jit::Symbol::fromQualString("aten::squeeze_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("squeeze_", self);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::squeeze_(self, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::std_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim) {
  RECORD_FUNCTION("std_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("std");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("std");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::std");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "unbiased", unbiased);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("std_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::std_out(out_, self_, dim, unbiased, keepdim);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::std_out(Tensor & out, const Tensor & self, DimnameList dim, bool unbiased, bool keepdim) {
  RECORD_FUNCTION("std_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::std");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "unbiased", unbiased);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("std_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::std_out(out, self, dim, unbiased, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::sub(const Tensor & self, const Tensor & other, Scalar alpha) {
  RECORD_FUNCTION("sub", std::vector<c10::IValue>({self, other, alpha}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  std::shared_ptr<SubBackward0> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<SubBackward0>(new SubBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->alpha = alpha;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sub");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::sub(self_, other_, alpha);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::sub(const Tensor & self, Scalar other, Scalar alpha) {
  RECORD_FUNCTION("sub", std::vector<c10::IValue>({self, other, alpha}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SubBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SubBackward1>(new SubBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sub");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::sub(self_, other, alpha);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::sub_(Tensor & self, const Tensor & other, Scalar alpha) {
  RECORD_FUNCTION("sub_", std::vector<c10::IValue>({self, other, alpha}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<SubBackward0> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<SubBackward0>(new SubBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->alpha = alpha;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::sub");
    } else {
      op_name = jit::Symbol::fromQualString("aten::sub_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sub_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.sub_(other_, alpha);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::sub_(Tensor & self, Scalar other, Scalar alpha) {
  RECORD_FUNCTION("sub_", std::vector<c10::IValue>({self, other, alpha}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<SubBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SubBackward1>(new SubBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::sub");
    } else {
      op_name = jit::Symbol::fromQualString("aten::sub_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sub_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.sub_(other, alpha);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::sum_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) {
  RECORD_FUNCTION("sum_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("sum");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("sum");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sum");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "dtype", dtype);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sum_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::sum_out(out_, self_, dim, keepdim, dtype);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::sum_out(Tensor & out, const Tensor & self, DimnameList dim, bool keepdim, c10::optional<ScalarType> dtype) {
  RECORD_FUNCTION("sum_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sum");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "dtype", dtype);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sum_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::sum_out(out, self, dim, keepdim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::take_out(Tensor & out, const Tensor & self, const Tensor & index) {
  RECORD_FUNCTION("take_out", std::vector<c10::IValue>({out, self, index}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& index_ = unpack(index, "index", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("take");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("take");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::take");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "index", index);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("take_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> index__storage_saved =
    index_.has_storage() ? c10::optional<Storage>(index_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> index__impl_saved;
  if (index_.defined()) index__impl_saved = index_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::take_out(out_, self_, index_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (index__storage_saved.has_value())
    AT_ASSERT(index__storage_saved.value().is_alias_of(index_.storage()));
  if (index__impl_saved) AT_ASSERT(index__impl_saved == index_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::thnn_conv2d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) {
  RECORD_FUNCTION("thnn_conv2d_forward", std::vector<c10::IValue>({self, weight, bias}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 3);
  std::shared_ptr<ThnnConv2DBackward> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::shared_ptr<ThnnConv2DBackward>(new ThnnConv2DBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
  }
  Tensor output;
  Tensor finput;
  Tensor fgrad_input;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_conv2d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> bias__storage_saved =
    bias_.has_storage() ? c10::optional<Storage>(bias_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> bias__impl_saved;
  if (bias_.defined()) bias__impl_saved = bias_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::thnn_conv2d_forward(self_, weight_, kernel_size, bias_, stride, padding);
  })();
  std::tie(output, finput, fgrad_input) = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (bias__storage_saved.has_value())
    AT_ASSERT(bias__storage_saved.value().is_alias_of(bias_.storage()));
  if (bias__impl_saved) AT_ASSERT(bias__impl_saved == bias_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( output ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, finput);
    jit::tracer::addOutput(node, fgrad_input);
  }
  if (grad_fn) {
    grad_fn->finput_ = SavedVariable(finput, true);
    grad_fn->fgrad_input_ = SavedVariable(fgrad_input, true);
  }
  return std::make_tuple(std::move(output), std::move(finput), std::move(fgrad_input));
}
std::tuple<Tensor,Tensor,Tensor> VariableType::thnn_conv3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) {
  RECORD_FUNCTION("thnn_conv3d_backward", std::vector<c10::IValue>({grad_output, self, weight, finput, fgrad_input}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto& finput_ = unpack(finput, "finput", 6);
  auto& fgrad_input_ = unpack(fgrad_input, "fgrad_input", 7);
  check_no_requires_grad(finput, "finput");
  check_no_requires_grad(fgrad_input, "fgrad_input");
  std::shared_ptr<ThnnConv3DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self, weight )) {
    grad_fn = std::shared_ptr<ThnnConv3DBackwardBackward>(new ThnnConv3DBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, weight ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
  }
  Tensor grad_input_return;
  Tensor grad_weight;
  Tensor grad_bias;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_conv3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "finput", finput);
    jit::tracer::addInputs(node, "fgrad_input", fgrad_input);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> finput__storage_saved =
    finput_.has_storage() ? c10::optional<Storage>(finput_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> finput__impl_saved;
  if (finput_.defined()) finput__impl_saved = finput_.getIntrusivePtr();
  c10::optional<Storage> fgrad_input__storage_saved =
    fgrad_input_.has_storage() ? c10::optional<Storage>(fgrad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> fgrad_input__impl_saved;
  if (fgrad_input_.defined()) fgrad_input__impl_saved = fgrad_input_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::thnn_conv3d_backward(grad_output_, self_, weight_, kernel_size, stride, padding, finput_, fgrad_input_, output_mask);
  })();
  std::tie(grad_input_return, grad_weight, grad_bias) = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (finput__storage_saved.has_value())
    AT_ASSERT(finput__storage_saved.value().is_alias_of(finput_.storage()));
  if (finput__impl_saved) AT_ASSERT(finput__impl_saved == finput_.getIntrusivePtr());
  if (fgrad_input__storage_saved.has_value())
    AT_ASSERT(fgrad_input__storage_saved.value().is_alias_of(fgrad_input_.storage()));
  if (fgrad_input__impl_saved) AT_ASSERT(fgrad_input__impl_saved == fgrad_input_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( grad_input_return, grad_weight, grad_bias ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input_return);
    jit::tracer::addOutput(node, grad_weight);
    jit::tracer::addOutput(node, grad_bias);
  }
  return std::make_tuple(std::move(grad_input_return), std::move(grad_weight), std::move(grad_bias));
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::thnn_conv3d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) {
  RECORD_FUNCTION("thnn_conv3d_forward_out", std::vector<c10::IValue>({output, finput, fgrad_input, self, weight, bias}), Node::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& finput_ = unpack(finput, "finput", 1);
  auto& fgrad_input_ = unpack(fgrad_input, "fgrad_input", 2);
  auto& self_ = unpack(self, "self", 3);
  auto& weight_ = unpack(weight, "weight", 4);
  auto bias_ = unpack_opt(bias, "bias", 6);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    throw_error_out_requires_grad("thnn_conv3d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("thnn_conv3d_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_conv3d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "finput", finput);
    jit::tracer::addInputs(node, "fgrad_input", fgrad_input);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("thnn_conv3d_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  c10::optional<Storage> finput__storage_saved =
    finput_.has_storage() ? c10::optional<Storage>(finput_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> finput__impl_saved;
  if (finput_.defined()) finput__impl_saved = finput_.getIntrusivePtr();
  c10::optional<Storage> fgrad_input__storage_saved =
    fgrad_input_.has_storage() ? c10::optional<Storage>(fgrad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> fgrad_input__impl_saved;
  if (fgrad_input_.defined()) fgrad_input__impl_saved = fgrad_input_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> bias__storage_saved =
    bias_.has_storage() ? c10::optional<Storage>(bias_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> bias__impl_saved;
  if (bias_.defined()) bias__impl_saved = bias_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::thnn_conv3d_forward_out(output_, finput_, fgrad_input_, self_, weight_, kernel_size, bias_, stride, padding);
  }
  #ifndef NDEBUG
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  if (finput__storage_saved.has_value())
    AT_ASSERT(finput__storage_saved.value().is_alias_of(finput_.storage()));
  if (finput__impl_saved) AT_ASSERT(finput__impl_saved == finput_.getIntrusivePtr());
  if (fgrad_input__storage_saved.has_value())
    AT_ASSERT(fgrad_input__storage_saved.value().is_alias_of(fgrad_input_.storage()));
  if (fgrad_input__impl_saved) AT_ASSERT(fgrad_input__impl_saved == fgrad_input_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (bias__storage_saved.has_value())
    AT_ASSERT(bias__storage_saved.value().is_alias_of(bias_.storage()));
  if (bias__impl_saved) AT_ASSERT(bias__impl_saved == bias_.getIntrusivePtr());
  #endif
  increment_version(output);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( output ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, finput);
    jit::tracer::addOutput(node, fgrad_input);
  }
  return std::forward_as_tuple(output, finput, fgrad_input);
}
Tensor & VariableType::thnn_conv_depthwise2d_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  RECORD_FUNCTION("thnn_conv_depthwise2d_out", std::vector<c10::IValue>({out, self, weight, bias}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_conv_depthwise2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("thnn_conv_depthwise2d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::thnn_conv_depthwise2d_out(out, self, weight, kernel_size, bias, stride, padding, dilation);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::to_dense_backward(const Tensor & grad, const Tensor & input) {
  RECORD_FUNCTION("to_dense_backward", std::vector<c10::IValue>({grad, input}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::to_dense_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "input", input);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::to_dense_backward(grad, input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor &,Tensor &> VariableType::topk_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) {
  RECORD_FUNCTION("topk_out", std::vector<c10::IValue>({values, indices, self}), Node::peek_at_next_sequence_nr());
  auto& values_ = unpack(values, "values", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("topk");
  }
  if (compute_requires_grad( values )) {
    throw_error_out_requires_grad("topk");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::topk");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "k", k);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "largest", largest);
    jit::tracer::addInputs(node, "sorted", sorted);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "values", values);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("topk_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> values__storage_saved =
    values_.has_storage() ? c10::optional<Storage>(values_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> values__impl_saved;
  if (values_.defined()) values__impl_saved = values_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::topk_out(values_, indices_, self_, k, dim, largest, sorted);
  }
  #ifndef NDEBUG
  if (values__storage_saved.has_value())
    AT_ASSERT(values__storage_saved.value().is_alias_of(values_.storage()));
  if (values__impl_saved) AT_ASSERT(values__impl_saved == values_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(values);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( values ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(values, indices);
}
Tensor & VariableType::trunc_out(Tensor & out, const Tensor & self) {
  RECORD_FUNCTION("trunc_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("trunc");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("trunc");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::trunc");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("trunc_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::trunc_out(out_, self_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
std::vector<Tensor> VariableType::unbind(const Tensor & self, int64_t dim) {
  RECORD_FUNCTION("unbind", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<UnbindBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<UnbindBackward>(new UnbindBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->dim = dim;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::unbind");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::unbind(self_, dim);
  })();
  auto result = as_view(self, tmp, true);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::vector<Tensor> VariableType::unbind(const Tensor & self, Dimname dim) {
  RECORD_FUNCTION("unbind", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::unbind");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::unbind(self, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::unique_consecutive(const Tensor & self, bool return_inverse, bool return_counts, c10::optional<int64_t> dim) {
  RECORD_FUNCTION("unique_consecutive", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("unique_consecutive"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::unique_consecutive");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "return_inverse", return_inverse);
    jit::tracer::addInputs(node, "return_counts", return_counts);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::unique_consecutive(self_, return_inverse, return_counts, dim);
  })();
  std::tie(result0, result1, result2) = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1, result2 ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor VariableType::upsample_bilinear2d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) {
  RECORD_FUNCTION("upsample_bilinear2d_backward", std::vector<c10::IValue>({grad_output}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  std::shared_ptr<UpsampleBilinear2DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output )) {
    grad_fn = std::shared_ptr<UpsampleBilinear2DBackwardBackward>(new UpsampleBilinear2DBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
    grad_fn->output_size = output_size.vec();
    grad_fn->align_corners = align_corners;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_bilinear2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::upsample_bilinear2d_backward(grad_output_, output_size, input_size, align_corners);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::upsample_linear1d(const Tensor & self, IntArrayRef output_size, bool align_corners) {
  RECORD_FUNCTION("upsample_linear1d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<UpsampleLinear1DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<UpsampleLinear1DBackward>(new UpsampleLinear1DBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->output_size = output_size.vec();
    grad_fn->align_corners = align_corners;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_linear1d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::upsample_linear1d(self_, output_size, align_corners);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::upsample_linear1d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) {
  RECORD_FUNCTION("upsample_linear1d_backward_out", std::vector<c10::IValue>({grad_input, grad_output}), Node::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output )) {
    throw_error_out_requires_grad("upsample_linear1d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("upsample_linear1d_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_linear1d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_linear1d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::upsample_linear1d_backward_out(grad_input_, grad_output_, output_size, input_size, align_corners);
  }
  #ifndef NDEBUG
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  #endif
  increment_version(grad_input);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::upsample_nearest1d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size) {
  RECORD_FUNCTION("upsample_nearest1d_backward", std::vector<c10::IValue>({grad_output}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  std::shared_ptr<UpsampleNearest1DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output )) {
    grad_fn = std::shared_ptr<UpsampleNearest1DBackwardBackward>(new UpsampleNearest1DBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
    grad_fn->output_size = output_size.vec();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_nearest1d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::upsample_nearest1d_backward(grad_output_, output_size, input_size);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::upsample_nearest2d(const Tensor & self, IntArrayRef output_size) {
  RECORD_FUNCTION("upsample_nearest2d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<UpsampleNearest2DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<UpsampleNearest2DBackward>(new UpsampleNearest2DBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->output_size = output_size.vec();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_nearest2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::upsample_nearest2d(self_, output_size);
  })();
  auto result = as_variable(std::move(tmp));
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::upsample_nearest2d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size) {
  RECORD_FUNCTION("upsample_nearest2d_backward_out", std::vector<c10::IValue>({grad_input, grad_output}), Node::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output )) {
    throw_error_out_requires_grad("upsample_nearest2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("upsample_nearest2d_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_nearest2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_nearest2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::upsample_nearest2d_backward_out(grad_input_, grad_output_, output_size, input_size);
  }
  #ifndef NDEBUG
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  #endif
  increment_version(grad_input);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::upsample_nearest3d_out(Tensor & out, const Tensor & self, IntArrayRef output_size) {
  RECORD_FUNCTION("upsample_nearest3d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("upsample_nearest3d");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("upsample_nearest3d");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_nearest3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_nearest3d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::upsample_nearest3d_out(out_, self_, output_size);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::view_as(const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("view_as", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::view_as");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::view_as(self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

namespace {

static auto registerer = torch::RegisterOperators()
  .op(torch::RegisterOperators::options()
    .schema("aten::__ilshift__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, Scalar), &VariableType::__ilshift__>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::__ilshift__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &), &VariableType::__ilshift__>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::__ior__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, Scalar), &VariableType::__ior__>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::__ior__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &), &VariableType::__ior__>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::__ixor__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, Scalar), &VariableType::__ixor__>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::__ixor__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &), &VariableType::__ixor__>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::__lshift__.Scalar(Tensor self, Scalar other) -> Tensor")
    .kernel<Tensor (const Tensor &, Scalar), &VariableType::__lshift__>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::__lshift__.Tensor(Tensor self, Tensor other) -> Tensor")
    .kernel<Tensor (const Tensor &, const Tensor &), &VariableType::__lshift__>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::__or__.Scalar(Tensor self, Scalar other) -> Tensor")
    .kernel<Tensor (const Tensor &, Scalar), &VariableType::__or__>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::__or__.Tensor(Tensor self, Tensor other) -> Tensor")
    .kernel<Tensor (const Tensor &, const Tensor &), &VariableType::__or__>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::__xor__.Scalar(Tensor self, Scalar other) -> Tensor")
    .kernel<Tensor (const Tensor &, Scalar), &VariableType::__xor__>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::__xor__.Tensor(Tensor self, Tensor other) -> Tensor")
    .kernel<Tensor (const Tensor &, const Tensor &), &VariableType::__xor__>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_addr.out(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar), &VariableType::_addr_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_baddbmm_mkl_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar), &VariableType::_baddbmm_mkl_>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_cast_Double(Tensor self, bool non_blocking=False) -> Tensor")
    .kernel<Tensor (const Tensor &, bool), &VariableType::_cast_Double>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_cast_Short(Tensor self, bool non_blocking=False) -> Tensor")
    .kernel<Tensor (const Tensor &, bool), &VariableType::_cast_Short>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_cudnn_rnn(Tensor input, Tensor[] weight, int weight_stride0, Tensor? weight_buf, Tensor hx, Tensor? cx, int mode, int hidden_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state) -> (Tensor, Tensor, Tensor, Tensor, Tensor)")
    .impl_unboxedOnlyKernel<std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> (const Tensor &, TensorList, int64_t, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, bool, double, bool, bool, IntArrayRef, const Tensor &), &VariableType::_cudnn_rnn>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_cumsum.out(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, int64_t), &VariableType::_cumsum_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_dimV(Tensor self) -> int")
    .kernel<int64_t (const Tensor &), &VariableType::_dimV>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None) -> (Tensor, Tensor, Tensor, Tensor)")
    .impl_unboxedOnlyKernel<std::tuple<Tensor,Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, bool, int64_t, bool, const Tensor &), &VariableType::_embedding_bag>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_embedding_bag_sparse_backward(Tensor grad, Tensor indices, Tensor offsets, Tensor offset2bag, Tensor bag_size, int num_weights, bool scale_grad_by_freq, int mode, Tensor? per_sample_weights) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, bool, int64_t, const Tensor &), &VariableType::_embedding_bag_sparse_backward>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_gather_sparse_backward(Tensor self, int dim, Tensor index, Tensor grad) -> Tensor")
    .kernel<Tensor (const Tensor &, int64_t, const Tensor &, const Tensor &), &VariableType::_gather_sparse_backward>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_index_put_impl_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False, bool unsafe=False) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, TensorList, const Tensor &, bool, bool), &VariableType::_index_put_impl_>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_indices(Tensor(a) self) -> Tensor(a)")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &), &VariableType::_indices>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_local_scalar_dense(Tensor self) -> Scalar")
    .kernel<Scalar (const Tensor &), &VariableType::_local_scalar_dense>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_max.max(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_indices) -> (Tensor(a!), Tensor(b!))")
    .impl_unboxedOnlyKernel<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, int64_t, bool), &VariableType::_max_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_mkldnn_transpose(Tensor self, int dim0, int dim1) -> Tensor")
    .kernel<Tensor (const Tensor &, int64_t, int64_t), &VariableType::_mkldnn_transpose>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_mkldnn_transpose_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, int64_t, int64_t), &VariableType::_mkldnn_transpose_>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_mode.values(Tensor self, int dim=-1, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))")
    .impl_unboxedOnlyKernel<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, int64_t, bool), &VariableType::_mode_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_nnpack_spatial_convolution_backward_weight(Tensor input, int[] weightsize, Tensor grad_output, int[2] padding) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, IntArrayRef, const Tensor &, IntArrayRef), &VariableType::_nnpack_spatial_convolution_backward_weight>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_pdist_forward(Tensor self, float p=2) -> Tensor")
    .kernel<Tensor (const Tensor &, double), &VariableType::_pdist_forward>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_softmax(Tensor self, int dim, bool half_to_float) -> Tensor")
    .kernel<Tensor (const Tensor &, int64_t, bool), &VariableType::_softmax>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_sparse_coo_tensor_with_dims(int sparse_dim, int dense_dim, int[] size, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (int64_t, int64_t, IntArrayRef, const TensorOptions &), &VariableType::_sparse_coo_tensor_with_dims>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_sparse_sum(Tensor self) -> Tensor")
    .kernel<Tensor (const Tensor &), &VariableType::_sparse_sum>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_sparse_sum.dtype(Tensor self, *, ScalarType dtype) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, ScalarType), &VariableType::_sparse_sum>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_sparse_sum.dim(Tensor self, int[1] dim) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, IntArrayRef), &VariableType::_sparse_sum>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_sparse_sum.dim_dtype(Tensor self, int[1] dim, *, ScalarType dtype) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, IntArrayRef, ScalarType), &VariableType::_sparse_sum>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_thnn_fused_gru_cell(Tensor input_gates, Tensor hidden_gates, Tensor hx, Tensor? input_bias=None, Tensor? hidden_bias=None) -> (Tensor, Tensor)")
    .impl_unboxedOnlyKernel<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &), &VariableType::_thnn_fused_gru_cell>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_thnn_fused_lstm_cell_backward(Tensor? grad_hy, Tensor? grad_cy, Tensor cx, Tensor cy, Tensor workspace, bool has_bias) -> (Tensor, Tensor, Tensor, Tensor, Tensor)")
    .impl_unboxedOnlyKernel<std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool), &VariableType::_thnn_fused_lstm_cell_backward>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_trilinear(Tensor i1, Tensor i2, Tensor i3, int[] expand1, int[] expand2, int[] expand3, int[] sumdim, int unroll_dim=1) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t), &VariableType::_trilinear>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_unsafe_view(Tensor self, int[] size) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, IntArrayRef), &VariableType::_unsafe_view>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_weight_norm_cuda_interface(Tensor v, Tensor g, int dim=0) -> (Tensor, Tensor)")
    .impl_unboxedOnlyKernel<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, int64_t), &VariableType::_weight_norm_cuda_interface>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::abs(Tensor self) -> Tensor")
    .kernel<Tensor (const Tensor &), &VariableType::abs>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::abs_(Tensor(a!) self) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &), &VariableType::abs_>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::adaptive_avg_pool1d(Tensor self, int[1] output_size) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, IntArrayRef), &VariableType::adaptive_avg_pool1d>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::adaptive_avg_pool2d.out(Tensor self, int[2] output_size, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, IntArrayRef), &VariableType::adaptive_avg_pool2d_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::adaptive_max_pool2d_backward(Tensor grad_output, Tensor self, Tensor indices) -> Tensor")
    .kernel<Tensor (const Tensor &, const Tensor &, const Tensor &), &VariableType::adaptive_max_pool2d_backward>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::adaptive_max_pool3d(Tensor self, int[3] output_size) -> (Tensor, Tensor)")
    .impl_unboxedOnlyKernel<std::tuple<Tensor,Tensor> (const Tensor &, IntArrayRef), &VariableType::adaptive_max_pool3d>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::adaptive_max_pool3d_backward.grad_input(Tensor grad_output, Tensor self, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &), &VariableType::adaptive_max_pool3d_backward_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor")
    .kernel<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar), &VariableType::addcmul>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::addcmul_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar), &VariableType::addcmul_>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor")
    .kernel<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar), &VariableType::addmm>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::addmm_(Tensor(a!) self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar), &VariableType::addmm_>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::addr.out(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar), &VariableType::addr_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::align_as(Tensor self, Tensor other) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const Tensor &), &VariableType::align_as>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::align_tensors(Tensor[] tensors) -> Tensor[]")
    .impl_unboxedOnlyKernel<std::vector<Tensor> (TensorList), &VariableType::align_tensors>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::align_to(Tensor(a) self, DimnameList names) -> Tensor(a)")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, DimnameList), &VariableType::align_to>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor")
    .kernel<Tensor (const Tensor &, c10::optional<int64_t>, bool), &VariableType::argmax>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::argsort(Tensor self, int dim=-1, bool descending=False) -> Tensor")
    .kernel<Tensor (const Tensor &, int64_t, bool), &VariableType::argsort>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::argsort.dimname(Tensor self, Dimname dim, bool descending=False) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, Dimname, bool), &VariableType::argsort>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::atan(Tensor self) -> Tensor")
    .kernel<Tensor (const Tensor &), &VariableType::atan>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::atan2(Tensor self, Tensor other) -> Tensor")
    .kernel<Tensor (const Tensor &, const Tensor &), &VariableType::atan2>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::atan2_(Tensor(a!) self, Tensor other) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &), &VariableType::atan2_>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::atan_(Tensor(a!) self) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &), &VariableType::atan_>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>), &VariableType::avg_pool2d>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::avg_pool2d_backward.grad_input(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override, *, Tensor(a!) grad_input) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>), &VariableType::avg_pool2d_backward_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::avg_pool3d.out(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>), &VariableType::avg_pool3d_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::baddbmm.out(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar), &VariableType::baddbmm_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::bartlett_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (int64_t, const TensorOptions &), &VariableType::bartlett_window>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::bartlett_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (int64_t, bool, const TensorOptions &), &VariableType::bartlett_window>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::batch_norm_update_stats(Tensor input, Tensor? running_mean, Tensor? running_var, float momentum) -> (Tensor, Tensor)")
    .impl_unboxedOnlyKernel<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, double), &VariableType::batch_norm_update_stats>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::binary_cross_entropy_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t), &VariableType::binary_cross_entropy_backward>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::bitwise_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &), &VariableType::bitwise_not_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::bmm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &), &VariableType::bmm_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::cdist(Tensor x1, Tensor x2, float p=2) -> Tensor")
    .kernel<Tensor (const Tensor &, const Tensor &, double), &VariableType::cdist>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::celu(Tensor self, Scalar alpha=1.0) -> Tensor")
    .kernel<Tensor (const Tensor &, Scalar), &VariableType::celu>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::celu_(Tensor(a!) self, Scalar alpha=1.0) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, Scalar), &VariableType::celu_>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::cholesky_inverse(Tensor self, bool upper=False) -> Tensor")
    .kernel<Tensor (const Tensor &, bool), &VariableType::cholesky_inverse>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::cholesky.out(Tensor self, bool upper=False, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, bool), &VariableType::cholesky_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor")
    .kernel<Tensor (const Tensor &, c10::optional<Scalar>, c10::optional<Scalar>), &VariableType::clamp>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::clamp_(Tensor(a!) self, Scalar? min=None, Scalar? max=None) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, c10::optional<Scalar>, c10::optional<Scalar>), &VariableType::clamp_>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::clamp_max.out(Tensor self, Scalar max, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, Scalar), &VariableType::clamp_max_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::conv_transpose3d.input(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] output_padding=0, int groups=1, int[3] dilation=1) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef), &VariableType::conv_transpose3d>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::convolution_backward_overrideable(Tensor grad_output, Tensor input, Tensor weight, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)")
    .impl_unboxedOnlyKernel<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, std::array<bool,3>), &VariableType::convolution_backward_overrideable>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::cos(Tensor self) -> Tensor")
    .kernel<Tensor (const Tensor &), &VariableType::cos>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::cos_(Tensor(a!) self) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &), &VariableType::cos_>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::cudnn_batch_norm_backward(Tensor input, Tensor grad_output, Tensor weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var, float epsilon) -> (Tensor, Tensor, Tensor)")
    .impl_unboxedOnlyKernel<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double), &VariableType::cudnn_batch_norm_backward>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::cudnn_grid_sampler_backward(Tensor self, Tensor grid, Tensor grad_output) -> (Tensor grad_self, Tensor grad_grid)")
    .impl_unboxedOnlyKernel<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &), &VariableType::cudnn_grid_sampler_backward>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::cumsum.out(Tensor self, int dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, int64_t, c10::optional<ScalarType>), &VariableType::cumsum_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::cumsum.dimname_out(Tensor self, Dimname dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, Dimname, c10::optional<ScalarType>), &VariableType::cumsum_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::data(Tensor self) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &), &VariableType::data>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::diag(Tensor self, int diagonal=0) -> Tensor")
    .kernel<Tensor (const Tensor &, int64_t), &VariableType::diag>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::digamma(Tensor self) -> Tensor")
    .kernel<Tensor (const Tensor &), &VariableType::digamma>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::digamma_(Tensor(a!) self) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &), &VariableType::digamma_>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::elu.out(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, Scalar, Scalar, Scalar), &VariableType::elu_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None) -> (Tensor, Tensor, Tensor, Tensor)")
    .impl_unboxedOnlyKernel<std::tuple<Tensor,Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, bool, int64_t, bool, const Tensor &), &VariableType::embedding_bag>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::embedding_dense_backward(Tensor grad_output, Tensor indices, int num_weights, int padding_idx, bool scale_grad_by_freq) -> Tensor")
    .kernel<Tensor (const Tensor &, const Tensor &, int64_t, int64_t, bool), &VariableType::embedding_dense_backward>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::empty_like(Tensor self) -> Tensor")
    .kernel<Tensor (const Tensor &), &VariableType::empty_like>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::empty_like.dtype(Tensor self, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False, MemoryFormat? memory_format=contiguous_format) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const TensorOptions &, c10::optional<MemoryFormat>), &VariableType::empty_like>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::empty_strided(int[] size, int[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (IntArrayRef, IntArrayRef, const TensorOptions &), &VariableType::empty_strided>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::erfc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &), &VariableType::erfc_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::erfinv(Tensor self) -> Tensor")
    .kernel<Tensor (const Tensor &), &VariableType::erfinv>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::erfinv_(Tensor(a!) self) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &), &VariableType::erfinv_>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::expand(Tensor(a) self, int[] size, *, bool implicit=False) -> Tensor(a)")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, IntArrayRef, bool), &VariableType::expand>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::expm1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &), &VariableType::expm1_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::fake_quantize_per_tensor_affine_backward(Tensor grad, Tensor self, float scale, int zero_point, int quant_min, int quant_max) -> Tensor")
    .kernel<Tensor (const Tensor &, const Tensor &, double, int64_t, int64_t, int64_t), &VariableType::fake_quantize_per_tensor_affine_backward>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::fft(Tensor self, int signal_ndim, bool normalized=False) -> Tensor")
    .kernel<Tensor (const Tensor &, int64_t, bool), &VariableType::fft>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::flatten.using_ints(Tensor self, int start_dim=0, int end_dim=-1) -> Tensor")
    .kernel<Tensor (const Tensor &, int64_t, int64_t), &VariableType::flatten>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::flatten.named_out_dim(Tensor self, int start_dim, int end_dim, Dimname out_dim) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, int64_t, int64_t, Dimname), &VariableType::flatten>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::flatten.using_names(Tensor self, Dimname start_dim, Dimname end_dim, Dimname out_dim) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, Dimname, Dimname, Dimname), &VariableType::flatten>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::flatten.DimnameList(Tensor self, DimnameList dims, Dimname out_dim) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, DimnameList, Dimname), &VariableType::flatten>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::floor.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &), &VariableType::floor_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::full.names(int[] size, Scalar fill_value, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (IntArrayRef, Scalar, c10::optional<DimnameList>, const TensorOptions &), &VariableType::full>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::full(int[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (IntArrayRef, Scalar, const TensorOptions &), &VariableType::full>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor")
    .kernel<Tensor (const Tensor &, int64_t, const Tensor &, bool), &VariableType::gather>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::gather.dimname(Tensor self, Dimname dim, Tensor index, *, bool sparse_grad=False) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, Dimname, const Tensor &, bool), &VariableType::gather>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::gelu_backward(Tensor grad, Tensor self) -> Tensor")
    .kernel<Tensor (const Tensor &, const Tensor &), &VariableType::gelu_backward>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::grid_sampler_3d_backward(Tensor grad_output, Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> (Tensor, Tensor)")
    .impl_unboxedOnlyKernel<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, bool), &VariableType::grid_sampler_3d_backward>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::gru_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &), &VariableType::gru_cell>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::hann_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (int64_t, const TensorOptions &), &VariableType::hann_window>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::hann_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (int64_t, bool, const TensorOptions &), &VariableType::hann_window>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::hardshrink(Tensor self, Scalar lambd=0.5) -> Tensor")
    .kernel<Tensor (const Tensor &, Scalar), &VariableType::hardshrink>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::ifft(Tensor self, int signal_ndim, bool normalized=False) -> Tensor")
    .kernel<Tensor (const Tensor &, int64_t, bool), &VariableType::ifft>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::index_select.out(Tensor self, int dim, Tensor index, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, int64_t, const Tensor &), &VariableType::index_select_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::index_select.dimname_out(Tensor self, Dimname dim, Tensor index, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, Dimname, const Tensor &), &VariableType::index_select_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::indices(Tensor(a) self) -> Tensor(a)")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &), &VariableType::indices>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::is_complex(Tensor self) -> bool")
    .kernel<bool (const Tensor &), &VariableType::is_complex>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::is_same_size(Tensor self, Tensor other) -> bool")
    .kernel<bool (const Tensor &, const Tensor &), &VariableType::is_same_size>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::l1_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &, int64_t), &VariableType::l1_loss_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, IntArrayRef, const Tensor &, const Tensor &, double, bool), &VariableType::layer_norm>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::leaky_relu_backward(Tensor grad_output, Tensor self, Scalar negative_slope) -> Tensor")
    .kernel<Tensor (const Tensor &, const Tensor &, Scalar), &VariableType::leaky_relu_backward>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::lerp.Scalar(Tensor self, Tensor end, Scalar weight) -> Tensor")
    .kernel<Tensor (const Tensor &, const Tensor &, Scalar), &VariableType::lerp>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::lerp.Tensor(Tensor self, Tensor end, Tensor weight) -> Tensor")
    .kernel<Tensor (const Tensor &, const Tensor &, const Tensor &), &VariableType::lerp>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::lerp_.Scalar(Tensor(a!) self, Tensor end, Scalar weight) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, Scalar), &VariableType::lerp_>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::lerp_.Tensor(Tensor(a!) self, Tensor end, Tensor weight) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &), &VariableType::lerp_>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const Tensor &, const Tensor &), &VariableType::linear>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::log_sigmoid(Tensor self) -> Tensor")
    .kernel<Tensor (const Tensor &), &VariableType::log_sigmoid>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::log_sigmoid_backward.grad_input(Tensor grad_output, Tensor self, Tensor buffer, *, Tensor(a!) grad_input) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &), &VariableType::log_sigmoid_backward_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::logical_xor.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &), &VariableType::logical_xor_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::logspace.out(Scalar start, Scalar end, int steps=100, float base=10.0, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, Scalar, Scalar, int64_t, double), &VariableType::logspace_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::logsumexp.out(Tensor self, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, IntArrayRef, bool), &VariableType::logsumexp_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::logsumexp.names_out(Tensor self, Dimname[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, DimnameList, bool), &VariableType::logsumexp_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::matmul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &), &VariableType::matmul_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::max.dim_max(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values) -> (Tensor(a!) values, Tensor(b!) indices)")
    .impl_unboxedOnlyKernel<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, int64_t, bool), &VariableType::max_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::max.names_dim_max(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values) -> (Tensor(a!) values, Tensor(b!) indices)")
    .impl_unboxedOnlyKernel<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, Dimname, bool), &VariableType::max_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::max.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &), &VariableType::max_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::max_unpool2d(Tensor self, Tensor indices, int[2] output_size) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const Tensor &, IntArrayRef), &VariableType::max_unpool2d>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::max_unpool2d_backward.grad_input(Tensor grad_output, Tensor self, Tensor indices, int[2] output_size, *, Tensor(a!) grad_input) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef), &VariableType::max_unpool2d_backward_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::max_unpool3d.out(Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef), &VariableType::max_unpool3d_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::min_values(Tensor self, int[1] dim, bool keepdim=False) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, IntArrayRef, bool), &VariableType::min_values>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::min_values.names(Tensor self, Dimname[1] dim, bool keepdim=False) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, DimnameList, bool), &VariableType::min_values>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::miopen_convolution_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool[3] output_mask) -> (Tensor, Tensor, Tensor)")
    .impl_unboxedOnlyKernel<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool, std::array<bool,3>), &VariableType::miopen_convolution_backward>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::miopen_convolution_backward_bias(Tensor grad_output) -> Tensor")
    .kernel<Tensor (const Tensor &), &VariableType::miopen_convolution_backward_bias>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::miopen_convolution_backward_input(int[] self_size, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool), &VariableType::miopen_convolution_backward_input>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::miopen_convolution_transpose(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool), &VariableType::miopen_convolution_transpose>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::mkldnn_adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, IntArrayRef), &VariableType::mkldnn_adaptive_avg_pool2d>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::mkldnn_convolution(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t), &VariableType::mkldnn_convolution>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::mkldnn_reorder_conv2d_weight(Tensor self, int[2] padding=0, int[2] stride=1, int[2] dilation=1, int groups=1) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t), &VariableType::mkldnn_reorder_conv2d_weight>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::mode.values(Tensor self, int dim=-1, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)")
    .impl_unboxedOnlyKernel<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, int64_t, bool), &VariableType::mode_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::mode.dimname_out(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)")
    .impl_unboxedOnlyKernel<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, Dimname, bool), &VariableType::mode_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::mse_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor")
    .kernel<Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t), &VariableType::mse_loss_backward>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::multilabel_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, Tensor is_target) -> Tensor")
    .kernel<Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t, const Tensor &), &VariableType::multilabel_margin_loss_backward>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::multilabel_margin_loss_forward.output(Tensor self, Tensor target, int reduction, *, Tensor(a!) output, Tensor(b!) is_target) -> (Tensor(a!), Tensor(b!))")
    .impl_unboxedOnlyKernel<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, const Tensor &, int64_t), &VariableType::multilabel_margin_loss_forward_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::multinomial(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, int64_t, bool, Generator *), &VariableType::multinomial>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::mvlgamma(Tensor self, int p) -> Tensor")
    .kernel<Tensor (const Tensor &, int64_t), &VariableType::mvlgamma>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::mvlgamma_(Tensor(a!) self, int p) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, int64_t), &VariableType::mvlgamma_>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::narrow(Tensor(a) self, int dim, int start, int length) -> Tensor(a)")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, int64_t, int64_t, int64_t), &VariableType::narrow>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)")
    .impl_unboxedOnlyKernel<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double), &VariableType::native_batch_norm>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::ne.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, Scalar), &VariableType::ne_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::ne.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &), &VariableType::ne_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::new_full(Tensor self, int[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, IntArrayRef, Scalar, const TensorOptions &), &VariableType::new_full>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::nll_loss(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t), &VariableType::nll_loss>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::nll_loss2d(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t), &VariableType::nll_loss2d>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::nll_loss2d_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight, *, Tensor(a!) grad_input) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, const Tensor &), &VariableType::nll_loss2d_backward_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::nll_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight, *, Tensor(a!) grad_input) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, const Tensor &), &VariableType::nll_loss_backward_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::nuclear_norm(Tensor self, bool keepdim=False) -> Tensor")
    .kernel<Tensor (const Tensor &, bool), &VariableType::nuclear_norm>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::nuclear_norm.dim(Tensor self, int[2] dim, bool keepdim=False) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, IntArrayRef, bool), &VariableType::nuclear_norm>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::numel(Tensor self) -> int")
    .kernel<int64_t (const Tensor &), &VariableType::numel>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::orgqr(Tensor self, Tensor input2) -> Tensor")
    .kernel<Tensor (const Tensor &, const Tensor &), &VariableType::orgqr>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::ormqr.out(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, bool), &VariableType::ormqr_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, IntArrayRef), &VariableType::permute>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::pixel_shuffle(Tensor self, int upscale_factor) -> Tensor")
    .kernel<Tensor (const Tensor &, int64_t), &VariableType::pixel_shuffle>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::put_(Tensor(a!) self, Tensor index, Tensor source, bool accumulate=False) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &, bool), &VariableType::put_>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::q_zero_point(Tensor self) -> int")
    .kernel<int64_t (const Tensor &), &VariableType::q_zero_point>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::quantize_per_tensor(Tensor self, float scale, int zero_point, ScalarType dtype) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, double, int64_t, ScalarType), &VariableType::quantize_per_tensor>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::quantized_gru.input(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)")
    .impl_unboxedOnlyKernel<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, TensorList, bool, int64_t, double, bool, bool, bool), &VariableType::quantized_gru>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::quantized_gru.data(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)")
    .impl_unboxedOnlyKernel<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, TensorList, bool, int64_t, double, bool, bool), &VariableType::quantized_gru>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::quantized_lstm_cell(Tensor input, Tensor[] hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> (Tensor, Tensor)")
    .impl_unboxedOnlyKernel<std::tuple<Tensor,Tensor> (const Tensor &, TensorList, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, Scalar, Scalar), &VariableType::quantized_lstm_cell>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::rand.out(int[] size, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, IntArrayRef), &VariableType::rand_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::rand.generator_out(int[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, IntArrayRef, Generator *), &VariableType::rand_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::randn.out(int[] size, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, IntArrayRef), &VariableType::randn_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::randn.generator_out(int[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, IntArrayRef, Generator *), &VariableType::randn_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::range.step(Scalar start, Scalar end, Scalar step=1, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (Scalar, Scalar, Scalar, const TensorOptions &), &VariableType::range>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::range(Scalar start, Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (Scalar, Scalar, const TensorOptions &), &VariableType::range>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::reciprocal(Tensor self) -> Tensor")
    .kernel<Tensor (const Tensor &), &VariableType::reciprocal>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::reciprocal_(Tensor(a!) self) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &), &VariableType::reciprocal_>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::refine_names(Tensor(a) self, DimnameList names) -> Tensor(a)")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, DimnameList), &VariableType::refine_names>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::reflection_pad1d(Tensor self, int[2] padding) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, IntArrayRef), &VariableType::reflection_pad1d>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::reflection_pad1d_backward.grad_input(Tensor grad_output, Tensor self, int[2] padding, *, Tensor(a!) grad_input) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef), &VariableType::reflection_pad1d_backward_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::reflection_pad2d.out(Tensor self, int[4] padding, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, IntArrayRef), &VariableType::reflection_pad2d_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::relu(Tensor self) -> Tensor")
    .kernel<Tensor (const Tensor &), &VariableType::relu>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::relu_(Tensor(a!) self) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &), &VariableType::relu_>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::remainder.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, Scalar), &VariableType::remainder_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::remainder.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &), &VariableType::remainder_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::replication_pad1d.out(Tensor self, int[2] padding, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, IntArrayRef), &VariableType::replication_pad1d_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::resize_as_(Tensor(a!) self, Tensor the_template) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &), &VariableType::resize_as_>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::rnn_relu_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &), &VariableType::rnn_relu_cell>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::rrelu_with_noise(Tensor self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const Tensor &, Scalar, Scalar, bool, Generator *), &VariableType::rrelu_with_noise>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::rrelu_with_noise_(Tensor(a!) self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, Scalar, Scalar, bool, Generator *), &VariableType::rrelu_with_noise_>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::rrelu_with_noise_backward.grad_input(Tensor grad_output, Tensor self, Tensor noise, Scalar lower, Scalar upper, bool training, *, Tensor(a!) grad_input) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, bool), &VariableType::rrelu_with_noise_backward_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::scatter_add(Tensor self, int dim, Tensor index, Tensor src) -> Tensor")
    .kernel<Tensor (const Tensor &, int64_t, const Tensor &, const Tensor &), &VariableType::scatter_add>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::scatter_add.dimname(Tensor self, Dimname dim, Tensor index, Tensor src) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, Dimname, const Tensor &, const Tensor &), &VariableType::scatter_add>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::scatter_add_(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, int64_t, const Tensor &, const Tensor &), &VariableType::scatter_add_>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::select.Dimname(Tensor(a) self, Dimname dim, int index) -> Tensor(a)")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, Dimname, int64_t), &VariableType::select>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, int64_t, int64_t), &VariableType::select>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::sin(Tensor self) -> Tensor")
    .kernel<Tensor (const Tensor &), &VariableType::sin>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::sin_(Tensor(a!) self) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &), &VariableType::sin_>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::slow_conv_dilated3d_backward(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)")
    .impl_unboxedOnlyKernel<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, std::array<bool,3>), &VariableType::slow_conv_dilated3d_backward>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::soft_margin_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &, int64_t), &VariableType::soft_margin_loss_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::softmax(Tensor self, int dim, ScalarType? dtype=None) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, int64_t, c10::optional<ScalarType>), &VariableType::softmax>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::softmax(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, Dimname, c10::optional<ScalarType>), &VariableType::softmax>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::softplus.out(Tensor self, Scalar beta=1, Scalar threshold=20, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, Scalar, Scalar), &VariableType::softplus_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::softshrink_backward(Tensor grad_output, Tensor self, Scalar lambd) -> Tensor")
    .kernel<Tensor (const Tensor &, const Tensor &, Scalar), &VariableType::softshrink_backward>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::sort.values(Tensor self, int dim=-1, bool descending=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)")
    .impl_unboxedOnlyKernel<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, int64_t, bool), &VariableType::sort_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::sort.dimname_values(Tensor self, Dimname dim, bool descending=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)")
    .impl_unboxedOnlyKernel<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, Dimname, bool), &VariableType::sort_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::squeeze(Tensor(a) self) -> Tensor(a)")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &), &VariableType::squeeze>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, int64_t), &VariableType::squeeze>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::squeeze.dimname(Tensor(a) self, Dimname dim) -> Tensor(a)")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, Dimname), &VariableType::squeeze>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::squeeze_(Tensor(a!) self) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &), &VariableType::squeeze_>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::squeeze_.dim(Tensor(a!) self, int dim) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, int64_t), &VariableType::squeeze_>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::squeeze_.dimname(Tensor(a!) self, Dimname dim) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, Dimname), &VariableType::squeeze_>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::std.out(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, IntArrayRef, bool, bool), &VariableType::std_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::std.names_out(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, DimnameList, bool, bool), &VariableType::std_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor")
    .kernel<Tensor (const Tensor &, const Tensor &, Scalar), &VariableType::sub>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor")
    .kernel<Tensor (const Tensor &, Scalar, Scalar), &VariableType::sub>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, Scalar), &VariableType::sub_>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::sub_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, Scalar, Scalar), &VariableType::sub_>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::sum.IntList_out(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, IntArrayRef, bool, c10::optional<ScalarType>), &VariableType::sum_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::sum.DimnameList_out(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, DimnameList, bool, c10::optional<ScalarType>), &VariableType::sum_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::take.out(Tensor self, Tensor index, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &), &VariableType::take_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::thnn_conv2d_forward(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding) -> (Tensor output, Tensor finput, Tensor fgrad_input)")
    .impl_unboxedOnlyKernel<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef), &VariableType::thnn_conv2d_forward>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::thnn_conv3d_backward.output_mask(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, Tensor finput, Tensor fgrad_input, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)")
    .impl_unboxedOnlyKernel<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,3>), &VariableType::thnn_conv3d_backward>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::thnn_conv3d_forward.output(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias, int[3] stride, int[3] padding, *, Tensor(a!) output, Tensor(b!) finput, Tensor(c!) fgrad_input) -> (Tensor(a!), Tensor(b!), Tensor(c!))")
    .impl_unboxedOnlyKernel<std::tuple<Tensor &,Tensor &,Tensor &> (Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef), &VariableType::thnn_conv3d_forward_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::thnn_conv_depthwise2d.out(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef), &VariableType::thnn_conv_depthwise2d_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::to_dense_backward(Tensor grad, Tensor input) -> Tensor")
    .kernel<Tensor (const Tensor &, const Tensor &), &VariableType::to_dense_backward>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::topk.values(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True, *, Tensor(a!) values, Tensor(b!) indices) ->(Tensor(a!) values, Tensor(b!) indices)")
    .impl_unboxedOnlyKernel<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, int64_t, int64_t, bool, bool), &VariableType::topk_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::trunc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &), &VariableType::trunc_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::unbind.int(Tensor(a) self, int dim=0) -> Tensor(a)[]")
    .impl_unboxedOnlyKernel<std::vector<Tensor> (const Tensor &, int64_t), &VariableType::unbind>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::unbind.Dimname(Tensor(a) self, Dimname dim) -> Tensor(a)[]")
    .impl_unboxedOnlyKernel<std::vector<Tensor> (const Tensor &, Dimname), &VariableType::unbind>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::unique_consecutive(Tensor self, bool return_inverse=False, bool return_counts=False, int? dim=None) -> (Tensor, Tensor, Tensor)")
    .impl_unboxedOnlyKernel<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, bool, bool, c10::optional<int64_t>), &VariableType::unique_consecutive>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::upsample_bilinear2d_backward(Tensor grad_output, int[2] output_size, int[4] input_size, bool align_corners) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, IntArrayRef, IntArrayRef, bool), &VariableType::upsample_bilinear2d_backward>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::upsample_linear1d(Tensor self, int[1] output_size, bool align_corners) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, IntArrayRef, bool), &VariableType::upsample_linear1d>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::upsample_linear1d_backward.grad_input(Tensor grad_output, int[1] output_size, int[3] input_size, bool align_corners, *, Tensor(a!) grad_input) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef, bool), &VariableType::upsample_linear1d_backward_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::upsample_nearest1d_backward(Tensor grad_output, int[1] output_size, int[3] input_size) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, IntArrayRef, IntArrayRef), &VariableType::upsample_nearest1d_backward>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::upsample_nearest2d(Tensor self, int[2] output_size) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, IntArrayRef), &VariableType::upsample_nearest2d>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::upsample_nearest2d_backward.grad_input(Tensor grad_output, int[2] output_size, int[4] input_size, *, Tensor(a!) grad_input) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef), &VariableType::upsample_nearest2d_backward_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::upsample_nearest3d.out(Tensor self, int[3] output_size, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, IntArrayRef), &VariableType::upsample_nearest3d_out>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::view_as(Tensor self, Tensor other) -> Tensor")
    .kernel<Tensor (const Tensor &, const Tensor &), &VariableType::view_as>(TensorTypeId::VariableTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));

}

}} // namespace torch::autograd
