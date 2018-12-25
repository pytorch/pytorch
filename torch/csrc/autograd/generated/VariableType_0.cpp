#include "torch/csrc/autograd/VariableTypeUtils.h"

// @generated from tools/autograd/templates/VariableType_0.cpp

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

Tensor & VariableType::__ilshift__(Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("__ilshift__", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::__ilshift__(self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::__ilshift__(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("__ilshift__", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::__ilshift__(self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::__ior__(Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("__ior__", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::__ior__(self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::__ior__(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("__ior__", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::__ior__(self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::__ixor__(Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("__ixor__", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::__ixor__(self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::__ixor__(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("__ixor__", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::__ixor__(self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::__lshift__(const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("__lshift__", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::__lshift__(self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::__lshift__(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("__lshift__", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::__lshift__(self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::__or__(const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("__or__", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::__or__(self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::__or__(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("__or__", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::__or__(self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::__xor__(const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("__xor__", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::__xor__(self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::__xor__(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("__xor__", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::__xor__(self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_argmax(const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("_argmax", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_argmax");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::_argmax(self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_baddbmm_mkl_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("_baddbmm_mkl_", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
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
Tensor VariableType::_cast_Double(const Tensor & self, bool non_blocking) const {
  profiler::RecordFunction profiler("_cast_Double", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::_cast_Double(self, non_blocking);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_cast_Short(const Tensor & self, bool non_blocking) const {
  profiler::RecordFunction profiler("_cast_Short", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::_cast_Short(self, non_blocking);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void VariableType::_copy_same_type_(Tensor & self, const Tensor & src) const {
  profiler::RecordFunction profiler("_copy_same_type_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& src_ = unpack(src, "src", 1);
  baseType->_copy_same_type_(self_, src_);
}
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> VariableType::_cudnn_rnn(const Tensor & input, TensorList weight, int64_t weight_stride0, const Tensor & weight_buf, const Tensor & hx, const Tensor & cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntList batch_sizes, const Tensor & dropout_state) const {
  profiler::RecordFunction profiler("_cudnn_rnn", Function::peek_at_next_sequence_nr());
  auto& input_ = unpack(input, "input", 0);
  auto weight_ = unpack(weight, "weight", 1);
  auto weight_buf_ = unpack_opt(weight_buf, "weight_buf", 3);
  auto& hx_ = unpack(hx, "hx", 4);
  auto cx_ = unpack_opt(cx, "cx", 5);
  auto dropout_state_ = unpack_opt(dropout_state, "dropout_state", 14);
  check_no_requires_grad(weight_buf, "weight_buf");
  std::shared_ptr<CudnnRnnBackward> grad_fn;
  if (compute_requires_grad( input, weight, hx, cx )) {
    grad_fn = std::shared_ptr<CudnnRnnBackward>(new CudnnRnnBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1, result2, result3, result4) = as_variable(baseType->_cudnn_rnn(input_, weight_, weight_stride0, weight_buf_, hx_, cx_, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state_));
  set_history(flatten_tensor_args( result0, result1, result2 ), grad_fn);
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
int64_t VariableType::_dimV(const Tensor & self) const {
  profiler::RecordFunction profiler("_dimV", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto result = baseType->_dimV(self_);
  return result;
}
Tensor & VariableType::_dirichlet_grad_out(Tensor & output, const Tensor & x, const Tensor & alpha, const Tensor & total) const {
  profiler::RecordFunction profiler("_dirichlet_grad_out", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_dirichlet_grad");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "total", total);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_dirichlet_grad_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::_dirichlet_grad_out(output, x, alpha, total);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
std::tuple<Tensor,Tensor,Tensor,Tensor> VariableType::_embedding_bag(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse) const {
  profiler::RecordFunction profiler("_embedding_bag", Function::peek_at_next_sequence_nr());
  auto& weight_ = unpack(weight, "weight", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& offsets_ = unpack(offsets, "offsets", 2);
  std::shared_ptr<EmbeddingBagBackward> grad_fn;
  if (compute_requires_grad( weight )) {
    grad_fn = std::shared_ptr<EmbeddingBagBackward>(new EmbeddingBagBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( weight ));
    grad_fn->weight_argsize_0 = weight.size(0);
    grad_fn->indices_ = SavedVariable(indices, false);
    grad_fn->offsets_ = SavedVariable(offsets, false);
    grad_fn->scale_grad_by_freq = scale_grad_by_freq;
    grad_fn->mode = mode;
    grad_fn->sparse = sparse;
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1, result2, result3) = as_variable(baseType->_embedding_bag(weight_, indices_, offsets_, scale_grad_by_freq, mode, sparse));
  set_history(flatten_tensor_args( result0 ), grad_fn);
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
Tensor VariableType::_embedding_bag_sparse_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode) const {
  profiler::RecordFunction profiler("_embedding_bag_sparse_backward", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::_embedding_bag_sparse_backward(grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_indices(const Tensor & self) const {
  profiler::RecordFunction profiler("_indices", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_view(self, baseType->_indices(self_), false);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Scalar VariableType::_local_scalar_dense(const Tensor & self) const {
  profiler::RecordFunction profiler("_local_scalar_dense", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto result = baseType->_local_scalar_dense(self_);
  return result;
}
Tensor VariableType::_pdist_forward(const Tensor & self, double p) const {
  profiler::RecordFunction profiler("_pdist_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<PdistBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<PdistBackward>(new PdistBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_pdist_forward(self_, p));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor VariableType::_softmax(const Tensor & self, int64_t dim, bool half_to_float) const {
  profiler::RecordFunction profiler("_softmax", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SoftmaxBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SoftmaxBackward>(new SoftmaxBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_softmax(self_, dim, half_to_float));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor & VariableType::_sparse_add_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
  profiler::RecordFunction profiler("_sparse_add_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("_sparse_add");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_sparse_add");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_add");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_sparse_add_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_sparse_add_out(result_, self_, other_, alpha);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_sparse_coo_tensor_with_dims(int64_t sparse_dim, int64_t dense_dim, IntList size, const TensorOptions & options) const {
  profiler::RecordFunction profiler("_sparse_coo_tensor_with_dims", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_sparse_coo_tensor_with_dims(sparse_dim, dense_dim, size, options_));
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_sparse_sum(const Tensor & self) const {
  profiler::RecordFunction profiler("_sparse_sum", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_sum");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::_sparse_sum(self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_sparse_sum(const Tensor & self, ScalarType dtype) const {
  profiler::RecordFunction profiler("_sparse_sum", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::_sparse_sum(self, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_sparse_sum(const Tensor & self, IntList dim) const {
  profiler::RecordFunction profiler("_sparse_sum", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SparseSumBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SparseSumBackward>(new SparseSumBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_sparse_sum(self_, dim));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_sparse_sum(const Tensor & self, IntList dim, ScalarType dtype) const {
  profiler::RecordFunction profiler("_sparse_sum", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::_sparse_sum(self, dim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_abs(const Tensor & self) const {
  profiler::RecordFunction profiler("_th_abs", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_abs"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_abs");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_abs(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::s__th_addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
  profiler::RecordFunction profiler("_th_addcmul", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& tensor1_ = unpack(tensor1, "tensor1", 1);
  auto& tensor2_ = unpack(tensor2, "tensor2", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, tensor1, tensor2 )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_addcmul"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, tensor1, tensor2 ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_addcmul");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "tensor1", tensor1);
    jit::tracer::addInputs(node, "tensor2", tensor2);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->s__th_addcmul(self_, tensor1_, tensor2_, value));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::s__th_addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
  profiler::RecordFunction profiler("_th_addcmul_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& tensor1_ = unpack(tensor1, "tensor1", 1);
  auto& tensor2_ = unpack(tensor2, "tensor2", 2);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, tensor1, tensor2 )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_addcmul_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, tensor1, tensor2 ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_addcmul");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_addcmul_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "tensor1", tensor1);
    jit::tracer::addInputs(node, "tensor2", tensor2);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_addcmul_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s__th_addcmul_(self_, tensor1_, tensor2_, value);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::s__th_addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("_th_addmm", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& mat1_ = unpack(mat1, "mat1", 1);
  auto& mat2_ = unpack(mat2, "mat2", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, mat1, mat2 )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_addmm"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, mat1, mat2 ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_addmm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat1", mat1);
    jit::tracer::addInputs(node, "mat2", mat2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->s__th_addmm(self_, mat1_, mat2_, beta, alpha));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("_th_addmm_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& mat1_ = unpack(mat1, "mat1", 1);
  auto& mat2_ = unpack(mat2, "mat2", 2);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, mat1, mat2 )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_addmm_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, mat1, mat2 ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_addmm");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_addmm_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat1", mat1);
    jit::tracer::addInputs(node, "mat2", mat2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_addmm_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_addmm_(self_, mat1_, mat2_, beta, alpha);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::s__th_addr_out(Tensor & result, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("_th_addr_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& vec1_ = unpack(vec1, "vec1", 2);
  auto& vec2_ = unpack(vec2, "vec2", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, vec1, vec2 )) {
    throw_error_out_requires_grad("_th_addr");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_addr");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_addr");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "vec1", vec1);
    jit::tracer::addInputs(node, "vec2", vec2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_addr_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s__th_addr_out(result_, self_, vec1_, vec2_, beta, alpha);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_atan(const Tensor & self) const {
  profiler::RecordFunction profiler("_th_atan", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_atan"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_atan");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_atan(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::s__th_atan2(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("_th_atan2", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_atan2"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_atan2");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->s__th_atan2(self_, other_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::s__th_atan2_(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("_th_atan2_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_atan2_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_atan2");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_atan2_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_atan2_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s__th_atan2_(self_, other_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::s__th_baddbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("_th_baddbmm_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& batch1_ = unpack(batch1, "batch1", 2);
  auto& batch2_ = unpack(batch2, "batch2", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, batch1, batch2 )) {
    throw_error_out_requires_grad("_th_baddbmm");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_baddbmm");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_baddbmm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "batch1", batch1);
    jit::tracer::addInputs(node, "batch2", batch2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_baddbmm_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s__th_baddbmm_out(result_, self_, batch1_, batch2_, beta, alpha);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_bmm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const {
  profiler::RecordFunction profiler("_th_bmm_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& mat2_ = unpack(mat2, "mat2", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, mat2 )) {
    throw_error_out_requires_grad("_th_bmm");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_bmm");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_bmm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat2", mat2);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_bmm_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_bmm_out(result_, self_, mat2_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_btrisolve_out(Tensor & result, const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) const {
  profiler::RecordFunction profiler("_th_btrisolve_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& LU_data_ = unpack(LU_data, "LU_data", 2);
  auto& LU_pivots_ = unpack(LU_pivots, "LU_pivots", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, LU_data, LU_pivots )) {
    throw_error_out_requires_grad("_th_btrisolve");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_btrisolve");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_btrisolve");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "LU_data", LU_data);
    jit::tracer::addInputs(node, "LU_pivots", LU_pivots);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_btrisolve_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_btrisolve_out(result_, self_, LU_data_, LU_pivots_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_clamp(const Tensor & self, Scalar min, Scalar max) const {
  profiler::RecordFunction profiler("_th_clamp", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_clamp"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_clamp");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min", min);
    jit::tracer::addInputs(node, "max", max);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_clamp(self_, min, max));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_clamp_max_out(Tensor & result, const Tensor & self, Scalar max) const {
  profiler::RecordFunction profiler("_th_clamp_max_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_clamp_max");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_clamp_max");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_clamp_max");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "max", max);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_clamp_max_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_clamp_max_out(result_, self_, max);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_copy_ignoring_overlaps_(Tensor & self, const Tensor & src) const {
  profiler::RecordFunction profiler("_th_copy_ignoring_overlaps_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& src_ = unpack(src, "src", 1);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, src )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_copy_ignoring_overlaps_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, src ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_copy_ignoring_overlaps");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_copy_ignoring_overlaps_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "src", src);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_copy_ignoring_overlaps_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_copy_ignoring_overlaps_(self_, src_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::_th_cos(const Tensor & self) const {
  profiler::RecordFunction profiler("_th_cos", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_cos"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_cos");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_cos(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_cumsum_out(Tensor & result, const Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("_th_cumsum_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_cumsum");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_cumsum");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_cumsum");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_cumsum_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_cumsum_out(result_, self_, dim);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_diag(const Tensor & self, int64_t diagonal) const {
  profiler::RecordFunction profiler("_th_diag", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_diag"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_diag");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "diagonal", diagonal);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_diag(self_, diagonal));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_digamma(const Tensor & self) const {
  profiler::RecordFunction profiler("_th_digamma", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_digamma"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_digamma");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_digamma(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_digamma_(Tensor & self) const {
  profiler::RecordFunction profiler("_th_digamma_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_digamma_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_digamma");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_digamma_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_digamma_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_digamma_(self_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::_th_dirichlet_grad_out(Tensor & output, const Tensor & x, const Tensor & alpha, const Tensor & total) const {
  profiler::RecordFunction profiler("_th_dirichlet_grad_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& x_ = unpack(x, "x", 1);
  auto& alpha_ = unpack(alpha, "alpha", 2);
  auto& total_ = unpack(total, "total", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( x, alpha, total )) {
    throw_error_out_requires_grad("_th_dirichlet_grad");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_th_dirichlet_grad");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_dirichlet_grad");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "total", total);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_dirichlet_grad_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_dirichlet_grad_out(output_, x_, alpha_, total_);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor & VariableType::_th_erfc_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("_th_erfc_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_erfc");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_erfc");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_erfc");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_erfc_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_erfc_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_erfinv(const Tensor & self) const {
  profiler::RecordFunction profiler("_th_erfinv", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_erfinv"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_erfinv");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_erfinv(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_erfinv_(Tensor & self) const {
  profiler::RecordFunction profiler("_th_erfinv_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_erfinv_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_erfinv");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_erfinv_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_erfinv_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_erfinv_(self_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::_th_expm1_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("_th_expm1_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_expm1");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_expm1");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_expm1");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_expm1_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_expm1_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_floor_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("_th_floor_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_floor");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_floor");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_floor");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_floor_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_floor_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_gather(const Tensor & self, int64_t dim, const Tensor & index) const {
  profiler::RecordFunction profiler("_th_gather", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_gather"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_gather");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_gather(self_, dim, index_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_getri_single_out(Tensor & output, const Tensor & self) const {
  profiler::RecordFunction profiler("_th_getri_single_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_getri_single");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_th_getri_single");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_getri_single");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_getri_single_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_getri_single_out(output_, self_);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor & VariableType::_th_ilshift_(Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("_th_ilshift_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_ilshift_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_ilshift");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_ilshift_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_ilshift_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_ilshift_(self_, other);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::s__th_ilshift_(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("_th_ilshift_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_ilshift_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_ilshift");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_ilshift_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_ilshift_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s__th_ilshift_(self_, other_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::_th_index_select_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) const {
  profiler::RecordFunction profiler("_th_index_select_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& index_ = unpack(index, "index", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_index_select");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_index_select");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_index_select");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_index_select_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_index_select_out(result_, self_, dim, index_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_ior_(Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("_th_ior_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_ior_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_ior");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_ior_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_ior_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_ior_(self_, other);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::s__th_ior_(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("_th_ior_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_ior_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_ior");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_ior_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_ior_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s__th_ior_(self_, other_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::_th_ixor_(Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("_th_ixor_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_ixor_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_ixor");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_ixor_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_ixor_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_ixor_(self_, other);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::s__th_ixor_(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("_th_ixor_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_ixor_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_ixor");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_ixor_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_ixor_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s__th_ixor_(self_, other_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::s__th_lerp(const Tensor & self, const Tensor & end, Scalar weight) const {
  profiler::RecordFunction profiler("_th_lerp", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& end_ = unpack(end, "end", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, end )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_lerp"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, end ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_lerp");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "weight", weight);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->s__th_lerp(self_, end_, weight));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::s__th_lerp_(Tensor & self, const Tensor & end, Scalar weight) const {
  profiler::RecordFunction profiler("_th_lerp_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& end_ = unpack(end, "end", 1);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, end )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_lerp_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, end ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_lerp");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_lerp_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "weight", weight);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_lerp_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s__th_lerp_(self_, end_, weight);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::_th_logspace_out(Tensor & result, Scalar start, Scalar end, int64_t steps) const {
  profiler::RecordFunction profiler("_th_logspace_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_logspace");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "steps", steps);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_logspace_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_logspace_out(result_, start, end, steps);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_lshift(const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("_th_lshift", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_lshift"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_lshift");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_lshift(self_, other));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::s__th_lshift(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("_th_lshift", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_lshift"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_lshift");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->s__th_lshift(self_, other_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::s__th_max_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("_th_max_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("_th_max");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_max");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_max");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_max_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s__th_max_out(result_, self_, other_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor &,Tensor &> VariableType::_th_max_out(Tensor & max, Tensor & max_indices, const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("_th_max_out", Function::peek_at_next_sequence_nr());
  auto& max_ = unpack(max, "max", 0);
  auto& max_indices_ = unpack(max_indices, "max_indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_max");
  }
  if (compute_requires_grad( max )) {
    throw_error_out_requires_grad("_th_max");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_max");
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_max_out", max);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_max_out(max_, max_indices_, self_, dim, keepdim);
  increment_version(max);
  rebase_history(flatten_tensor_args( max ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, max);
    jit::tracer::addOutput(node, max_indices);
  }
  return std::forward_as_tuple(max, max_indices);
}
std::tuple<Tensor &,Tensor &> VariableType::_th_mode_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("_th_mode_out", Function::peek_at_next_sequence_nr());
  auto& values_ = unpack(values, "values", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_mode");
  }
  if (compute_requires_grad( values )) {
    throw_error_out_requires_grad("_th_mode");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_mode");
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_mode_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_mode_out(values_, indices_, self_, dim, keepdim);
  increment_version(values);
  rebase_history(flatten_tensor_args( values ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(values, indices);
}
Tensor VariableType::_th_multinomial(const Tensor & self, int64_t num_samples, bool replacement, Generator * generator) const {
  profiler::RecordFunction profiler("_th_multinomial", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_multinomial");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "num_samples", num_samples);
    jit::tracer::addInputs(node, "replacement", replacement);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_multinomial(self_, num_samples, replacement, generator));
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_ne_out(Tensor & result, const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("_th_ne_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_ne");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_ne_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_ne_out(result_, self_, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::s__th_ne_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("_th_ne_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_ne");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_ne_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s__th_ne_out(result_, self_, other_);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_or(const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("_th_or", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_or"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_or");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_or(self_, other));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::s__th_or(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("_th_or", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_or"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_or");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->s__th_or(self_, other_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_orgqr(const Tensor & self, const Tensor & input2) const {
  profiler::RecordFunction profiler("_th_orgqr", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& input2_ = unpack(input2, "input2", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, input2 )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_orgqr"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, input2 ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_orgqr");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "input2", input2);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_orgqr(self_, input2_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_ormqr_out(Tensor & result, const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) const {
  profiler::RecordFunction profiler("_th_ormqr_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& input2_ = unpack(input2, "input2", 2);
  auto& input3_ = unpack(input3, "input3", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, input2, input3 )) {
    throw_error_out_requires_grad("_th_ormqr");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_ormqr");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_ormqr");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "input2", input2);
    jit::tracer::addInputs(node, "input3", input3);
    jit::tracer::addInputs(node, "left", left);
    jit::tracer::addInputs(node, "transpose", transpose);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_ormqr_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_ormqr_out(result_, self_, input2_, input3_, left, transpose);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_potrs_single(const Tensor & self, const Tensor & input2, bool upper) const {
  profiler::RecordFunction profiler("_th_potrs_single", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& input2_ = unpack(input2, "input2", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, input2 )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_potrs_single"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, input2 ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_potrs_single");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "input2", input2);
    jit::tracer::addInputs(node, "upper", upper);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_potrs_single(self_, input2_, upper));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_put_(Tensor & self, const Tensor & index, const Tensor & source, bool accumulate) const {
  profiler::RecordFunction profiler("_th_put_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 1);
  auto& source_ = unpack(source, "source", 2);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, source )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_put_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, source ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_put");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_put_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "source", source);
    jit::tracer::addInputs(node, "accumulate", accumulate);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_put_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_put_(self_, index_, source_, accumulate);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::_th_range(Scalar start, Scalar end, Scalar step) const {
  profiler::RecordFunction profiler("_th_range", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_range");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "step", step);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_range(start, end, step));
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_reciprocal(const Tensor & self) const {
  profiler::RecordFunction profiler("_th_reciprocal", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_reciprocal"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_reciprocal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_reciprocal(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_reciprocal_(Tensor & self) const {
  profiler::RecordFunction profiler("_th_reciprocal_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_reciprocal_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_reciprocal");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_reciprocal_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_reciprocal_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_reciprocal_(self_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::_th_remainder_out(Tensor & result, const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("_th_remainder_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_remainder");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_remainder");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_remainder");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_remainder_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_remainder_out(result_, self_, other);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::s__th_remainder_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("_th_remainder_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("_th_remainder");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_remainder");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_remainder");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_remainder_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s__th_remainder_out(result_, self_, other_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_resize_as_(Tensor & self, const Tensor & the_template) const {
  profiler::RecordFunction profiler("_th_resize_as_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& the_template_ = unpack(the_template, "the_template", 1);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, the_template )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_resize_as_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, the_template ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_resize_as");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_resize_as_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "the_template", the_template);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_resize_as_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_resize_as_(self_, the_template_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::_th_rshift_out(Tensor & result, const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("_th_rshift_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_rshift");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_rshift");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_rshift");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_rshift_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_rshift_out(result_, self_, other);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::s__th_rshift_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("_th_rshift_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("_th_rshift");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_rshift");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_rshift");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_rshift_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s__th_rshift_out(result_, self_, other_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_scatter_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
  profiler::RecordFunction profiler("_th_scatter_add_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  auto& src_ = unpack(src, "src", 3);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, src )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_scatter_add_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, src ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_scatter_add");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_scatter_add_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "src", src);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_scatter_add_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_scatter_add_(self_, dim, index_, src_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::_th_sin(const Tensor & self) const {
  profiler::RecordFunction profiler("_th_sin", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_sin"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_sin");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_sin(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor &,Tensor &> VariableType::_th_sort_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool descending) const {
  profiler::RecordFunction profiler("_th_sort_out", Function::peek_at_next_sequence_nr());
  auto& values_ = unpack(values, "values", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_sort");
  }
  if (compute_requires_grad( values )) {
    throw_error_out_requires_grad("_th_sort");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_sort");
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_sort_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_sort_out(values_, indices_, self_, dim, descending);
  increment_version(values);
  rebase_history(flatten_tensor_args( values ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(values, indices);
}
Tensor & VariableType::_th_std_out(Tensor & result, const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
  profiler::RecordFunction profiler("_th_std_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_std");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_std");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_std");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "unbiased", unbiased);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_std_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_std_out(result_, self_, dim, unbiased, keepdim);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_take_out(Tensor & result, const Tensor & self, const Tensor & index) const {
  profiler::RecordFunction profiler("_th_take_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& index_ = unpack(index, "index", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_take");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_take");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_take");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "index", index);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_take_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_take_out(result_, self_, index_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor &,Tensor &> VariableType::_th_topk_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) const {
  profiler::RecordFunction profiler("_th_topk_out", Function::peek_at_next_sequence_nr());
  auto& values_ = unpack(values, "values", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_topk");
  }
  if (compute_requires_grad( values )) {
    throw_error_out_requires_grad("_th_topk");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_topk");
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_topk_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_topk_out(values_, indices_, self_, k, dim, largest, sorted);
  increment_version(values);
  rebase_history(flatten_tensor_args( values ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(values, indices);
}
std::tuple<Tensor,Tensor> VariableType::_th_trtrs(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) const {
  profiler::RecordFunction profiler("_th_trtrs", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& A_ = unpack(A, "A", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, A )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_trtrs"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, A ));
  }
  Tensor res1;
  Tensor res2;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_trtrs");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "A", A);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "transpose", transpose);
    jit::tracer::addInputs(node, "unitriangular", unitriangular);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(res1, res2) = as_variable(baseType->_th_trtrs(self_, A_, upper, transpose, unitriangular));
  set_history(flatten_tensor_args( res1, res2 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, res1);
    jit::tracer::addOutput(node, res2);
  }
  return std::make_tuple(std::move(res1), std::move(res2));
}
Tensor & VariableType::_th_trunc_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("_th_trunc_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_trunc");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_trunc");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_trunc");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_trunc_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_trunc_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_xor(const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("_th_xor", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_xor"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_xor");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_xor(self_, other));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::s__th_xor(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("_th_xor", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_xor"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_xor");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->s__th_xor(self_, other_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_thnn_adaptive_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices) const {
  profiler::RecordFunction profiler("_thnn_adaptive_max_pool2d_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_adaptive_max_pool2d_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_adaptive_max_pool2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto grad_input = as_variable(baseType->_thnn_adaptive_max_pool2d_backward(grad_output_, self_, indices_));
  set_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
std::tuple<Tensor &,Tensor &> VariableType::_thnn_adaptive_max_pool2d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("_thnn_adaptive_max_pool2d_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_thnn_adaptive_max_pool2d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_thnn_adaptive_max_pool2d_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_adaptive_max_pool2d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_adaptive_max_pool2d_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_adaptive_max_pool2d_forward_out(output_, indices_, self_, output_size);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(output, indices);
}
Tensor & VariableType::_thnn_adaptive_max_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices) const {
  profiler::RecordFunction profiler("_thnn_adaptive_max_pool3d_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& indices_ = unpack(indices, "indices", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("_thnn_adaptive_max_pool3d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("_thnn_adaptive_max_pool3d_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_adaptive_max_pool3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_adaptive_max_pool3d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_adaptive_max_pool3d_backward_out(grad_input_, grad_output_, self_, indices_);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_avg_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
  profiler::RecordFunction profiler("_thnn_avg_pool2d_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("_thnn_avg_pool2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("_thnn_avg_pool2d_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_avg_pool2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    jit::tracer::addInputs(node, "count_include_pad", count_include_pad);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_avg_pool2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_avg_pool2d_backward_out(grad_input_, grad_output_, self_, kernel_size, stride, padding, ceil_mode, count_include_pad);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::_thnn_binary_cross_entropy_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) const {
  profiler::RecordFunction profiler("_thnn_binary_cross_entropy_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  auto weight_ = unpack_opt(weight, "weight", 3);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, self, target, weight )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_binary_cross_entropy_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, target, weight ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_binary_cross_entropy_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto grad_input = as_variable(baseType->_thnn_binary_cross_entropy_backward(grad_output_, self_, target_, weight_, reduction));
  set_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_binary_cross_entropy_forward_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) const {
  profiler::RecordFunction profiler("_thnn_binary_cross_entropy_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  auto weight_ = unpack_opt(weight, "weight", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, target, weight )) {
    throw_error_out_requires_grad("_thnn_binary_cross_entropy_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_thnn_binary_cross_entropy_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_binary_cross_entropy_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_binary_cross_entropy_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_binary_cross_entropy_forward_out(output_, self_, target_, weight_, reduction);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::_thnn_conv2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
  profiler::RecordFunction profiler("_thnn_conv2d_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 3);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_conv2d_forward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
  }
  Tensor output;
  Tensor finput;
  Tensor fgrad_input;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_conv2d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(output, finput, fgrad_input) = as_variable(baseType->_thnn_conv2d_forward(self_, weight_, kernel_size, bias_, stride, padding));
  set_history(flatten_tensor_args( output, finput, fgrad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, finput);
    jit::tracer::addOutput(node, fgrad_input);
  }
  return std::make_tuple(std::move(output), std::move(finput), std::move(fgrad_input));
}
std::tuple<Tensor,Tensor,Tensor> VariableType::_thnn_conv3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) const {
  profiler::RecordFunction profiler("_thnn_conv3d_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto& finput_ = unpack(finput, "finput", 6);
  auto& fgrad_input_ = unpack(fgrad_input, "fgrad_input", 7);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, self, weight, finput, fgrad_input )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_conv3d_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, weight, finput, fgrad_input ));
  }
  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_conv3d_backward");
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(grad_input, grad_weight, grad_bias) = as_variable(baseType->_thnn_conv3d_backward(grad_output_, self_, weight_, kernel_size, stride, padding, finput_, fgrad_input_, output_mask));
  set_history(flatten_tensor_args( grad_input, grad_weight, grad_bias ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
    jit::tracer::addOutput(node, grad_weight);
    jit::tracer::addOutput(node, grad_bias);
  }
  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::_thnn_conv3d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
  profiler::RecordFunction profiler("_thnn_conv3d_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& finput_ = unpack(finput, "finput", 1);
  auto& fgrad_input_ = unpack(fgrad_input, "fgrad_input", 2);
  auto& self_ = unpack(self, "self", 3);
  auto& weight_ = unpack(weight, "weight", 4);
  auto bias_ = unpack_opt(bias, "bias", 6);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    throw_error_out_requires_grad("_thnn_conv3d_forward");
  }
  if (compute_requires_grad( output, finput, fgrad_input )) {
    throw_error_out_requires_grad("_thnn_conv3d_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_conv3d_forward");
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_conv3d_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_conv3d_forward_out(output_, finput_, fgrad_input_, self_, weight_, kernel_size, bias_, stride, padding);
  increment_version(output);
  increment_version(finput);
  increment_version(fgrad_input);
  rebase_history(flatten_tensor_args( output, finput, fgrad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, finput);
    jit::tracer::addOutput(node, fgrad_input);
  }
  return std::forward_as_tuple(output, finput, fgrad_input);
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::_thnn_conv_dilated2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones) const {
  profiler::RecordFunction profiler("_thnn_conv_dilated2d_backward_out", Function::peek_at_next_sequence_nr());
  auto grad_input_ = unpack_opt(grad_input, "grad_input", 0);
  auto grad_weight_ = unpack_opt(grad_weight, "grad_weight", 1);
  auto grad_bias_ = unpack_opt(grad_bias, "grad_bias", 2);
  auto& grad_output_ = unpack(grad_output, "grad_output", 3);
  auto& self_ = unpack(self, "self", 4);
  auto& weight_ = unpack(weight, "weight", 5);
  auto& columns_ = unpack(columns, "columns", 10);
  auto& ones_ = unpack(ones, "ones", 11);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, weight, columns, ones )) {
    throw_error_out_requires_grad("_thnn_conv_dilated2d_backward");
  }
  if (compute_requires_grad( grad_input, grad_weight, grad_bias )) {
    throw_error_out_requires_grad("_thnn_conv_dilated2d_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_conv_dilated2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_weight", grad_weight);
    jit::tracer::addInputs(node, "grad_bias", grad_bias);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "columns", columns);
    jit::tracer::addInputs(node, "ones", ones);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_conv_dilated2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_conv_dilated2d_backward_out(grad_input_, grad_weight_, grad_bias_, grad_output_, self_, weight_, kernel_size, stride, padding, dilation, columns_, ones_);
  increment_version(grad_input);
  increment_version(grad_weight);
  increment_version(grad_bias);
  rebase_history(flatten_tensor_args( grad_input, grad_weight, grad_bias ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
    jit::tracer::addOutput(node, grad_weight);
    jit::tracer::addOutput(node, grad_bias);
  }
  return std::forward_as_tuple(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> VariableType::_thnn_conv_transpose2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) const {
  profiler::RecordFunction profiler("_thnn_conv_transpose2d_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto& columns_ = unpack(columns, "columns", 8);
  auto& ones_ = unpack(ones, "ones", 9);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, self, weight, columns, ones )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_conv_transpose2d_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, weight, columns, ones ));
  }
  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_conv_transpose2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "columns", columns);
    jit::tracer::addInputs(node, "ones", ones);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(grad_input, grad_weight, grad_bias) = as_variable(baseType->_thnn_conv_transpose2d_backward(grad_output_, self_, weight_, kernel_size, stride, padding, output_padding, dilation, columns_, ones_, output_mask));
  set_history(flatten_tensor_args( grad_input, grad_weight, grad_bias ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
    jit::tracer::addOutput(node, grad_weight);
    jit::tracer::addOutput(node, grad_bias);
  }
  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::_thnn_conv_transpose2d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
  profiler::RecordFunction profiler("_thnn_conv_transpose2d_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& columns_ = unpack(columns, "columns", 1);
  auto& ones_ = unpack(ones, "ones", 2);
  auto& self_ = unpack(self, "self", 3);
  auto& weight_ = unpack(weight, "weight", 4);
  auto bias_ = unpack_opt(bias, "bias", 6);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    throw_error_out_requires_grad("_thnn_conv_transpose2d_forward");
  }
  if (compute_requires_grad( output, columns, ones )) {
    throw_error_out_requires_grad("_thnn_conv_transpose2d_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_conv_transpose2d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "columns", columns);
    jit::tracer::addInputs(node, "ones", ones);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_conv_transpose2d_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_conv_transpose2d_forward_out(output_, columns_, ones_, self_, weight_, kernel_size, bias_, stride, padding, output_padding, dilation);
  increment_version(output);
  increment_version(columns);
  increment_version(ones);
  rebase_history(flatten_tensor_args( output, columns, ones ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, columns);
    jit::tracer::addOutput(node, ones);
  }
  return std::forward_as_tuple(output, columns, ones);
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::_thnn_conv_transpose3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & finput, const Tensor & fgrad_input) const {
  profiler::RecordFunction profiler("_thnn_conv_transpose3d_backward_out", Function::peek_at_next_sequence_nr());
  auto grad_input_ = unpack_opt(grad_input, "grad_input", 0);
  auto grad_weight_ = unpack_opt(grad_weight, "grad_weight", 1);
  auto grad_bias_ = unpack_opt(grad_bias, "grad_bias", 2);
  auto& grad_output_ = unpack(grad_output, "grad_output", 3);
  auto& self_ = unpack(self, "self", 4);
  auto& weight_ = unpack(weight, "weight", 5);
  auto& finput_ = unpack(finput, "finput", 11);
  auto& fgrad_input_ = unpack(fgrad_input, "fgrad_input", 12);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, weight, finput, fgrad_input )) {
    throw_error_out_requires_grad("_thnn_conv_transpose3d_backward");
  }
  if (compute_requires_grad( grad_input, grad_weight, grad_bias )) {
    throw_error_out_requires_grad("_thnn_conv_transpose3d_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_conv_transpose3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_weight", grad_weight);
    jit::tracer::addInputs(node, "grad_bias", grad_bias);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "finput", finput);
    jit::tracer::addInputs(node, "fgrad_input", fgrad_input);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_conv_transpose3d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_conv_transpose3d_backward_out(grad_input_, grad_weight_, grad_bias_, grad_output_, self_, weight_, kernel_size, stride, padding, output_padding, dilation, finput_, fgrad_input_);
  increment_version(grad_input);
  increment_version(grad_weight);
  increment_version(grad_bias);
  rebase_history(flatten_tensor_args( grad_input, grad_weight, grad_bias ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
    jit::tracer::addOutput(node, grad_weight);
    jit::tracer::addOutput(node, grad_bias);
  }
  return std::forward_as_tuple(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor> VariableType::_thnn_fused_gru_cell(const Tensor & input_gates, const Tensor & hidden_gates, const Tensor & hx, const Tensor & input_bias, const Tensor & hidden_bias) const {
  profiler::RecordFunction profiler("_thnn_fused_gru_cell", Function::peek_at_next_sequence_nr());
  auto& input_gates_ = unpack(input_gates, "input_gates", 0);
  auto& hidden_gates_ = unpack(hidden_gates, "hidden_gates", 1);
  auto& hx_ = unpack(hx, "hx", 2);
  auto input_bias_ = unpack_opt(input_bias, "input_bias", 3);
  auto hidden_bias_ = unpack_opt(hidden_bias, "hidden_bias", 4);
  std::shared_ptr<ThnnFusedGruCellBackward> grad_fn;
  if (compute_requires_grad( input_gates, hidden_gates, hx, input_bias, hidden_bias )) {
    grad_fn = std::shared_ptr<ThnnFusedGruCellBackward>(new ThnnFusedGruCellBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = as_variable(baseType->_thnn_fused_gru_cell(input_gates_, hidden_gates_, hx_, input_bias_, hidden_bias_));
  set_history(flatten_tensor_args( result0 ), grad_fn);
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
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> VariableType::_thnn_fused_lstm_cell_backward(const Tensor & grad_hy, const Tensor & grad_cy, const Tensor & cx, const Tensor & cy, const Tensor & workspace, bool has_bias) const {
  profiler::RecordFunction profiler("_thnn_fused_lstm_cell_backward", Function::peek_at_next_sequence_nr());
  auto grad_hy_ = unpack_opt(grad_hy, "grad_hy", 0);
  auto grad_cy_ = unpack_opt(grad_cy, "grad_cy", 1);
  auto& cx_ = unpack(cx, "cx", 2);
  auto& cy_ = unpack(cy, "cy", 3);
  auto& workspace_ = unpack(workspace, "workspace", 4);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_hy, grad_cy, cx, cy, workspace )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_fused_lstm_cell_backward"), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1, result2, result3, result4) = as_variable(baseType->_thnn_fused_lstm_cell_backward(grad_hy_, grad_cy_, cx_, cy_, workspace_, has_bias));
  set_history(flatten_tensor_args( result0, result1, result2, result3, result4 ), grad_fn);
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
Tensor VariableType::_thnn_glu_forward(const Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("_thnn_glu_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_glu_forward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_glu_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto output = as_variable(baseType->_thnn_glu_forward(self_, dim));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_thnn_leaky_relu_backward(const Tensor & grad_output, const Tensor & self, Scalar negative_slope) const {
  profiler::RecordFunction profiler("_thnn_leaky_relu_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_leaky_relu_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_leaky_relu_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "negative_slope", negative_slope);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto grad_input = as_variable(baseType->_thnn_leaky_relu_backward(grad_output_, self_, negative_slope));
  set_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_leaky_relu_forward_out(Tensor & output, const Tensor & self, Scalar negative_slope) const {
  profiler::RecordFunction profiler("_thnn_leaky_relu_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_thnn_leaky_relu_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_thnn_leaky_relu_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_leaky_relu_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "negative_slope", negative_slope);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_leaky_relu_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_leaky_relu_forward_out(output_, self_, negative_slope);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor & VariableType::_thnn_log_sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & buffer) const {
  profiler::RecordFunction profiler("_thnn_log_sigmoid_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& buffer_ = unpack(buffer, "buffer", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, buffer )) {
    throw_error_out_requires_grad("_thnn_log_sigmoid_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("_thnn_log_sigmoid_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_log_sigmoid_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "buffer", buffer);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_log_sigmoid_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_log_sigmoid_backward_out(grad_input_, grad_output_, self_, buffer_);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
std::tuple<Tensor,Tensor> VariableType::_thnn_max_pool3d_with_indices_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
  profiler::RecordFunction profiler("_thnn_max_pool3d_with_indices_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_max_pool3d_with_indices_forward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor output;
  Tensor indices;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_max_pool3d_with_indices_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(output, indices) = as_variable(baseType->_thnn_max_pool3d_with_indices_forward(self_, kernel_size, stride, padding, dilation, ceil_mode));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, indices);
  }
  return std::make_tuple(std::move(output), std::move(indices));
}
Tensor & VariableType::_thnn_max_unpool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size) const {
  profiler::RecordFunction profiler("_thnn_max_unpool2d_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& indices_ = unpack(indices, "indices", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("_thnn_max_unpool2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("_thnn_max_unpool2d_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_max_unpool2d_backward");
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_max_unpool2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_max_unpool2d_backward_out(grad_input_, grad_output_, self_, indices_, output_size);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::_thnn_mse_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) const {
  profiler::RecordFunction profiler("_thnn_mse_loss_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, self, target )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_mse_loss_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, target ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_mse_loss_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto grad_input = as_variable(baseType->_thnn_mse_loss_backward(grad_output_, self_, target_, reduction));
  set_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_mse_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, int64_t reduction) const {
  profiler::RecordFunction profiler("_thnn_mse_loss_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, target )) {
    throw_error_out_requires_grad("_thnn_mse_loss_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_thnn_mse_loss_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_mse_loss_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_mse_loss_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_mse_loss_forward_out(output_, self_, target_, reduction);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_thnn_multilabel_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, const Tensor & is_target) const {
  profiler::RecordFunction profiler("_thnn_multilabel_margin_loss_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  auto& is_target_ = unpack(is_target, "is_target", 4);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, self, is_target )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_multilabel_margin_loss_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, is_target ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_multilabel_margin_loss_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "is_target", is_target);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto grad_input = as_variable(baseType->_thnn_multilabel_margin_loss_backward(grad_output_, self_, target_, reduction, is_target_));
  set_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
std::tuple<Tensor &,Tensor &> VariableType::_thnn_multilabel_margin_loss_forward_out(Tensor & output, Tensor & is_target, const Tensor & self, const Tensor & target, int64_t reduction) const {
  profiler::RecordFunction profiler("_thnn_multilabel_margin_loss_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& is_target_ = unpack(is_target, "is_target", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& target_ = unpack(target, "target", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_thnn_multilabel_margin_loss_forward");
  }
  if (compute_requires_grad( output, is_target )) {
    throw_error_out_requires_grad("_thnn_multilabel_margin_loss_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_multilabel_margin_loss_forward");
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_multilabel_margin_loss_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_multilabel_margin_loss_forward_out(output_, is_target_, self_, target_, reduction);
  increment_version(output);
  increment_version(is_target);
  rebase_history(flatten_tensor_args( output, is_target ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, is_target);
  }
  return std::forward_as_tuple(output, is_target);
}
Tensor & VariableType::_thnn_nll_loss2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) const {
  profiler::RecordFunction profiler("_thnn_nll_loss2d_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& target_ = unpack(target, "target", 3);
  auto weight_ = unpack_opt(weight, "weight", 4);
  auto& total_weight_ = unpack(total_weight, "total_weight", 7);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, weight, total_weight )) {
    throw_error_out_requires_grad("_thnn_nll_loss2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("_thnn_nll_loss2d_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_nll_loss2d_backward");
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_nll_loss2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_nll_loss2d_backward_out(grad_input_, grad_output_, self_, target_, weight_, reduction, ignore_index, total_weight_);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_nll_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) const {
  profiler::RecordFunction profiler("_thnn_nll_loss_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& target_ = unpack(target, "target", 3);
  auto weight_ = unpack_opt(weight, "weight", 4);
  auto& total_weight_ = unpack(total_weight, "total_weight", 7);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, weight, total_weight )) {
    throw_error_out_requires_grad("_thnn_nll_loss_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("_thnn_nll_loss_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_nll_loss_backward");
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_nll_loss_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_nll_loss_backward_out(grad_input_, grad_output_, self_, target_, weight_, reduction, ignore_index, total_weight_);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_reflection_pad1d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("_thnn_reflection_pad1d_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("_thnn_reflection_pad1d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("_thnn_reflection_pad1d_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_reflection_pad1d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_reflection_pad1d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_reflection_pad1d_backward_out(grad_input_, grad_output_, self_, padding);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::_thnn_replication_pad3d_forward(const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("_thnn_replication_pad3d_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_replication_pad3d_forward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_replication_pad3d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto output = as_variable(baseType->_thnn_replication_pad3d_forward(self_, padding));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor & VariableType::_thnn_rrelu_with_noise_(Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
  profiler::RecordFunction profiler("_thnn_rrelu_with_noise_", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_thnn_rrelu_with_noise");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_thnn_rrelu_with_noise_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "noise", noise);
    jit::tracer::addInputs(node, "lower", lower);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_rrelu_with_noise_", self);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::_thnn_rrelu_with_noise_(self, noise, lower, upper, training, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::_thnn_rrelu_with_noise_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training) const {
  profiler::RecordFunction profiler("_thnn_rrelu_with_noise_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& noise_ = unpack(noise, "noise", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, noise )) {
    throw_error_out_requires_grad("_thnn_rrelu_with_noise_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("_thnn_rrelu_with_noise_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_rrelu_with_noise_backward");
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_rrelu_with_noise_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_rrelu_with_noise_backward_out(grad_input_, grad_output_, self_, noise_, lower, upper, training);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::_thnn_sigmoid_forward(const Tensor & self) const {
  profiler::RecordFunction profiler("_thnn_sigmoid_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_sigmoid_forward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_sigmoid_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto output = as_variable(baseType->_thnn_sigmoid_forward(self_));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_thnn_softshrink_backward(const Tensor & grad_output, const Tensor & self, Scalar lambd) const {
  profiler::RecordFunction profiler("_thnn_softshrink_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_softshrink_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_softshrink_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "lambd", lambd);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto grad_input = as_variable(baseType->_thnn_softshrink_backward(grad_output_, self_, lambd));
  set_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_softshrink_forward_out(Tensor & output, const Tensor & self, Scalar lambd) const {
  profiler::RecordFunction profiler("_thnn_softshrink_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_thnn_softshrink_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_thnn_softshrink_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_softshrink_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "lambd", lambd);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_softshrink_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_softshrink_forward_out(output_, self_, lambd);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_thnn_upsample_bilinear2d_backward(const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
  profiler::RecordFunction profiler("_thnn_upsample_bilinear2d_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_upsample_bilinear2d_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_upsample_bilinear2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto grad_input = as_variable(baseType->_thnn_upsample_bilinear2d_backward(grad_output_, output_size, input_size, align_corners));
  set_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_upsample_bilinear2d_forward_out(Tensor & output, const Tensor & self, IntList output_size, bool align_corners) const {
  profiler::RecordFunction profiler("_thnn_upsample_bilinear2d_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_thnn_upsample_bilinear2d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_thnn_upsample_bilinear2d_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_upsample_bilinear2d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_upsample_bilinear2d_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_upsample_bilinear2d_forward_out(output_, self_, output_size, align_corners);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor & VariableType::_thnn_upsample_linear1d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
  profiler::RecordFunction profiler("_thnn_upsample_linear1d_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output )) {
    throw_error_out_requires_grad("_thnn_upsample_linear1d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("_thnn_upsample_linear1d_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_upsample_linear1d_backward");
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_upsample_linear1d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_upsample_linear1d_backward_out(grad_input_, grad_output_, output_size, input_size, align_corners);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::_thnn_upsample_nearest1d_backward(const Tensor & grad_output, IntList output_size, IntList input_size) const {
  profiler::RecordFunction profiler("_thnn_upsample_nearest1d_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_upsample_nearest1d_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_upsample_nearest1d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto grad_input = as_variable(baseType->_thnn_upsample_nearest1d_backward(grad_output_, output_size, input_size));
  set_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_upsample_nearest1d_forward_out(Tensor & output, const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("_thnn_upsample_nearest1d_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_thnn_upsample_nearest1d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_thnn_upsample_nearest1d_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_upsample_nearest1d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_upsample_nearest1d_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_upsample_nearest1d_forward_out(output_, self_, output_size);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor & VariableType::_thnn_upsample_nearest2d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size) const {
  profiler::RecordFunction profiler("_thnn_upsample_nearest2d_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output )) {
    throw_error_out_requires_grad("_thnn_upsample_nearest2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("_thnn_upsample_nearest2d_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_upsample_nearest2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_upsample_nearest2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_upsample_nearest2d_backward_out(grad_input_, grad_output_, output_size, input_size);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::_trilinear(const Tensor & i1, const Tensor & i2, const Tensor & i3, IntList expand1, IntList expand2, IntList expand3, IntList sumdim, int64_t unroll_dim) const {
  profiler::RecordFunction profiler("_trilinear", Function::peek_at_next_sequence_nr());
  auto& i1_ = unpack(i1, "i1", 0);
  auto& i2_ = unpack(i2, "i2", 1);
  auto& i3_ = unpack(i3, "i3", 2);
  std::shared_ptr<TrilinearBackward> grad_fn;
  if (compute_requires_grad( i1, i2, i3 )) {
    grad_fn = std::shared_ptr<TrilinearBackward>(new TrilinearBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_trilinear(i1_, i2_, i3_, expand1, expand2, expand3, sumdim, unroll_dim));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_unsafe_view(const Tensor & self, IntList size) const {
  profiler::RecordFunction profiler("_unsafe_view", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<UnsafeViewBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<UnsafeViewBackward>(new UnsafeViewBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_unsafe_view(self_, size));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::_weight_norm_cuda_interface(const Tensor & v, const Tensor & g, int64_t dim) const {
  profiler::RecordFunction profiler("_weight_norm_cuda_interface", Function::peek_at_next_sequence_nr());
  auto& v_ = unpack(v, "v", 0);
  auto& g_ = unpack(g, "g", 1);
  std::shared_ptr<WeightNormCudaInterfaceBackward> grad_fn;
  if (compute_requires_grad( v, g )) {
    grad_fn = std::shared_ptr<WeightNormCudaInterfaceBackward>(new WeightNormCudaInterfaceBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = as_variable(baseType->_weight_norm_cuda_interface(v_, g_, dim));
  set_history(flatten_tensor_args( result0 ), grad_fn);
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
Tensor VariableType::abs(const Tensor & self) const {
  profiler::RecordFunction profiler("abs", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AbsBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AbsBackward>(new AbsBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->abs(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::abs_(Tensor & self) const {
  profiler::RecordFunction profiler("abs_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<AbsBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AbsBackward>(new AbsBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("abs_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->abs_(self_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::adaptive_avg_pool1d(const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("adaptive_avg_pool1d", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::adaptive_avg_pool1d(self, output_size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::adaptive_avg_pool2d_out(Tensor & output, const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("adaptive_avg_pool2d_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("adaptive_avg_pool2d");
  }
  if (compute_requires_grad( output )) {
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
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("adaptive_avg_pool2d_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->adaptive_avg_pool2d_out(output_, self_, output_size);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::adaptive_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices) const {
  profiler::RecordFunction profiler("adaptive_max_pool2d_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 2);
  check_no_requires_grad(indices, "indices");
  std::shared_ptr<AdaptiveMaxPool2DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<AdaptiveMaxPool2DBackwardBackward>(new AdaptiveMaxPool2DBackwardBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->adaptive_max_pool2d_backward(grad_output_, self_, indices_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::adaptive_max_pool3d(const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("adaptive_max_pool3d", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AdaptiveMaxPool3DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AdaptiveMaxPool3DBackward>(new AdaptiveMaxPool3DBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  Tensor output;
  Tensor indices;
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(output, indices) = as_variable(baseType->adaptive_max_pool3d(self_, output_size));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, indices);
  }
  if (grad_fn) {
    grad_fn->indices_ = SavedVariable(indices, true);
  }
  return std::make_tuple(std::move(output), std::move(indices));
}
Tensor & VariableType::adaptive_max_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices) const {
  profiler::RecordFunction profiler("adaptive_max_pool3d_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& indices_ = unpack(indices, "indices", 3);
  std::shared_ptr<Function> grad_fn;
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("adaptive_max_pool3d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->adaptive_max_pool3d_backward_out(grad_input_, grad_output_, self_, indices_);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
  profiler::RecordFunction profiler("addcmul", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& tensor1_ = unpack(tensor1, "tensor1", 1);
  auto& tensor2_ = unpack(tensor2, "tensor2", 2);
  std::shared_ptr<AddcmulBackward> grad_fn;
  if (compute_requires_grad( self, tensor1, tensor2 )) {
    grad_fn = std::shared_ptr<AddcmulBackward>(new AddcmulBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, tensor1, tensor2 ));
    grad_fn->tensor2_ = SavedVariable(tensor2, false);
    grad_fn->value = value;
    grad_fn->tensor1_ = SavedVariable(tensor1, false);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->addcmul(self_, tensor1_, tensor2_, value));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
  profiler::RecordFunction profiler("addcmul_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& tensor1_ = unpack(tensor1, "tensor1", 1);
  auto& tensor2_ = unpack(tensor2, "tensor2", 2);
  check_inplace(self);
  std::shared_ptr<AddcmulBackward> grad_fn;
  if (compute_requires_grad( self, tensor1, tensor2 )) {
    grad_fn = std::shared_ptr<AddcmulBackward>(new AddcmulBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, tensor1, tensor2 ));
    grad_fn->tensor2_ = SavedVariable(tensor2, false);
    grad_fn->value = value;
    grad_fn->tensor1_ = SavedVariable(tensor1, false);
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("addcmul_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->addcmul_(self_, tensor1_, tensor2_, value);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("addmm", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& mat1_ = unpack(mat1, "mat1", 1);
  auto& mat2_ = unpack(mat2, "mat2", 2);
  std::shared_ptr<AddmmBackward> grad_fn;
  if (compute_requires_grad( self, mat1, mat2 )) {
    grad_fn = std::shared_ptr<AddmmBackward>(new AddmmBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->addmm(self_, mat1_, mat2_, beta, alpha));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("addmm_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& mat1_ = unpack(mat1, "mat1", 1);
  auto& mat2_ = unpack(mat2, "mat2", 2);
  check_inplace(self);
  std::shared_ptr<AddmmBackward> grad_fn;
  if (compute_requires_grad( self, mat1, mat2 )) {
    grad_fn = std::shared_ptr<AddmmBackward>(new AddmmBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("addmm_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->addmm_(self_, mat1_, mat2_, beta, alpha);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::addr_out(Tensor & result, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("addr_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& vec1_ = unpack(vec1, "vec1", 2);
  auto& vec2_ = unpack(vec2, "vec2", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, vec1, vec2 )) {
    throw_error_out_requires_grad("addr");
  }
  if (compute_requires_grad( result )) {
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
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("addr_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->addr_out(result_, self_, vec1_, vec2_, beta, alpha);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::argmax(const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("argmax", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::argmax(self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::argmax(const Tensor & self) const {
  profiler::RecordFunction profiler("argmax", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::argmax");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::argmax(self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::atan(const Tensor & self) const {
  profiler::RecordFunction profiler("atan", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AtanBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AtanBackward>(new AtanBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->atan(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::atan2(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("atan2", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  std::shared_ptr<Atan2Backward> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<Atan2Backward>(new Atan2Backward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->atan2(self_, other_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::atan2_(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("atan2_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<Atan2Backward> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<Atan2Backward>(new Atan2Backward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("atan2_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->atan2_(self_, other_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::atan_(Tensor & self) const {
  profiler::RecordFunction profiler("atan_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<AtanBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AtanBackward>(new AtanBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("atan_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->atan_(self_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::avg_pool2d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
  profiler::RecordFunction profiler("avg_pool2d", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AvgPool2DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AvgPool2DBackward>(new AvgPool2DBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->ceil_mode = ceil_mode;
    grad_fn->count_include_pad = count_include_pad;
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->avg_pool2d(self_, kernel_size, stride, padding, ceil_mode, count_include_pad));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::avg_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
  profiler::RecordFunction profiler("avg_pool2d_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("avg_pool2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->avg_pool2d_backward_out(grad_input_, grad_output_, self_, kernel_size, stride, padding, ceil_mode, count_include_pad);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::avg_pool3d_out(Tensor & output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
  profiler::RecordFunction profiler("avg_pool3d_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("avg_pool3d");
  }
  if (compute_requires_grad( output )) {
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("avg_pool3d_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->avg_pool3d_out(output_, self_, kernel_size, stride, padding, ceil_mode, count_include_pad);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor & VariableType::baddbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("baddbmm_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& batch1_ = unpack(batch1, "batch1", 2);
  auto& batch2_ = unpack(batch2, "batch2", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, batch1, batch2 )) {
    throw_error_out_requires_grad("baddbmm");
  }
  if (compute_requires_grad( result )) {
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
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("baddbmm_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->baddbmm_out(result_, self_, batch1_, batch2_, beta, alpha);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::binary_cross_entropy_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) const {
  profiler::RecordFunction profiler("binary_cross_entropy_backward", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::binary_cross_entropy_backward(grad_output, self, target, weight, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::bmm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const {
  profiler::RecordFunction profiler("bmm_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& mat2_ = unpack(mat2, "mat2", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, mat2 )) {
    throw_error_out_requires_grad("bmm");
  }
  if (compute_requires_grad( result )) {
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
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bmm_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->bmm_out(result_, self_, mat2_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::btrisolve_out(Tensor & result, const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) const {
  profiler::RecordFunction profiler("btrisolve_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& LU_data_ = unpack(LU_data, "LU_data", 2);
  auto& LU_pivots_ = unpack(LU_pivots, "LU_pivots", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, LU_data, LU_pivots )) {
    throw_error_out_requires_grad("btrisolve");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("btrisolve");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::btrisolve");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "LU_data", LU_data);
    jit::tracer::addInputs(node, "LU_pivots", LU_pivots);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("btrisolve_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->btrisolve_out(result_, self_, LU_data_, LU_pivots_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::celu(const Tensor & self, Scalar alpha) const {
  profiler::RecordFunction profiler("celu", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::celu(self, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::celu_(Tensor & self, Scalar alpha) const {
  profiler::RecordFunction profiler("celu_", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
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
Tensor & VariableType::cholesky_out(Tensor & result, const Tensor & self, bool upper) const {
  profiler::RecordFunction profiler("cholesky_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("cholesky");
  }
  if (compute_requires_grad( result )) {
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
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cholesky_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->cholesky_out(result_, self_, upper);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::clamp(const Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max) const {
  profiler::RecordFunction profiler("clamp", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ClampBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ClampBackward>(new ClampBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->clamp(self_, min, max));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::clamp_(Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max) const {
  profiler::RecordFunction profiler("clamp_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<ClampBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ClampBackward>(new ClampBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("clamp_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->clamp_(self_, min, max);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::clamp_max_out(Tensor & result, const Tensor & self, Scalar max) const {
  profiler::RecordFunction profiler("clamp_max_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("clamp_max");
  }
  if (compute_requires_grad( result )) {
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
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("clamp_max_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->clamp_max_out(result_, self_, max);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::conv_transpose3d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, int64_t groups, IntList dilation) const {
  profiler::RecordFunction profiler("conv_transpose3d", Function::peek_at_next_sequence_nr());
  auto result = TypeDefault::conv_transpose3d(input, weight, bias, stride, padding, output_padding, groups, dilation);
  return result;
}
Tensor VariableType::cos(const Tensor & self) const {
  profiler::RecordFunction profiler("cos", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<CosBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<CosBackward>(new CosBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->cos(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::cos_(Tensor & self) const {
  profiler::RecordFunction profiler("cos_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<CosBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<CosBackward>(new CosBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cos_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->cos_(self_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::cudnn_batch_norm_backward(const Tensor & input, const Tensor & grad_output, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_var, double epsilon) const {
  profiler::RecordFunction profiler("cudnn_batch_norm_backward", Function::peek_at_next_sequence_nr());
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
    grad_fn = std::shared_ptr<CudnnBatchNormBackwardBackward>(new CudnnBatchNormBackwardBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1, result2) = as_variable(baseType->cudnn_batch_norm_backward(input_, grad_output_, weight_, running_mean_, running_var_, save_mean_, save_var_, epsilon));
  set_history(flatten_tensor_args( result0, result1, result2 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
std::tuple<Tensor,Tensor> VariableType::cudnn_grid_sampler_backward(const Tensor & self, const Tensor & grid, const Tensor & grad_output) const {
  profiler::RecordFunction profiler("cudnn_grid_sampler_backward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& grid_ = unpack(grid, "grid", 1);
  auto& grad_output_ = unpack(grad_output, "grad_output", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, grid, grad_output )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("cudnn_grid_sampler_backward"), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(grad_self, grad_grid) = as_variable(baseType->cudnn_grid_sampler_backward(self_, grid_, grad_output_));
  set_history(flatten_tensor_args( grad_self, grad_grid ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_self);
    jit::tracer::addOutput(node, grad_grid);
  }
  return std::make_tuple(std::move(grad_self), std::move(grad_grid));
}
Tensor & VariableType::cumsum_out(Tensor & result, const Tensor & self, int64_t dim, ScalarType dtype) const {
  profiler::RecordFunction profiler("cumsum_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("cumsum");
  }
  if (compute_requires_grad( result )) {
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
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cumsum_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->cumsum_out(result_, self_, dim, dtype);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::cumsum_out(Tensor & result, const Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("cumsum_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("cumsum");
  }
  if (compute_requires_grad( result )) {
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cumsum_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->cumsum_out(result_, self_, dim);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::diag(const Tensor & self, int64_t diagonal) const {
  profiler::RecordFunction profiler("diag", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<DiagBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<DiagBackward>(new DiagBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->diag(self_, diagonal));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::digamma(const Tensor & self) const {
  profiler::RecordFunction profiler("digamma", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<DigammaBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<DigammaBackward>(new DigammaBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->digamma(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::digamma_(Tensor & self) const {
  profiler::RecordFunction profiler("digamma_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<DigammaBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<DigammaBackward>(new DigammaBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("digamma_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->digamma_(self_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::elu_out(Tensor & output, const Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale) const {
  profiler::RecordFunction profiler("elu_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("elu");
  }
  if (compute_requires_grad( output )) {
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
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("elu_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->elu_out(output_, self_, alpha, scale, input_scale);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
std::tuple<Tensor,Tensor,Tensor,Tensor> VariableType::embedding_bag(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse) const {
  profiler::RecordFunction profiler("embedding_bag", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1, result2, result3) = TypeDefault::embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3));
}
Tensor VariableType::embedding_dense_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) const {
  profiler::RecordFunction profiler("embedding_dense_backward", Function::peek_at_next_sequence_nr());
  auto& grad_ = unpack(grad, "grad", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("embedding_dense_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::embedding_dense_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "num_weights", num_weights);
    jit::tracer::addInputs(node, "padding_idx", padding_idx);
    jit::tracer::addInputs(node, "scale_grad_by_freq", scale_grad_by_freq);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->embedding_dense_backward(grad_, indices_, num_weights, padding_idx, scale_grad_by_freq));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::empty_like(const Tensor & self) const {
  profiler::RecordFunction profiler("empty_like", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::empty_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::empty_like(self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::erfc_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("erfc_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("erfc");
  }
  if (compute_requires_grad( result )) {
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
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("erfc_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->erfc_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::erfinv(const Tensor & self) const {
  profiler::RecordFunction profiler("erfinv", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ErfinvBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ErfinvBackward>(new ErfinvBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->erfinv(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::erfinv_(Tensor & self) const {
  profiler::RecordFunction profiler("erfinv_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<ErfinvBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ErfinvBackward>(new ErfinvBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("erfinv_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->erfinv_(self_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::expand(const Tensor & self, IntList size, bool implicit) const {
  profiler::RecordFunction profiler("expand", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ExpandBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ExpandBackward>(new ExpandBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_view(self, baseType->expand(self_, size, implicit), true);
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::expm1_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("expm1_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("expm1");
  }
  if (compute_requires_grad( result )) {
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
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("expm1_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->expm1_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::fft(const Tensor & self, int64_t signal_ndim, bool normalized) const {
  profiler::RecordFunction profiler("fft", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::fft(self, signal_ndim, normalized);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::flatten(const Tensor & self, int64_t start_dim, int64_t end_dim) const {
  profiler::RecordFunction profiler("flatten", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::flatten(self, start_dim, end_dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::floor_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("floor_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("floor");
  }
  if (compute_requires_grad( result )) {
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
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("floor_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->floor_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::gather(const Tensor & self, int64_t dim, const Tensor & index) const {
  profiler::RecordFunction profiler("gather", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  check_no_requires_grad(index, "index");
  std::shared_ptr<GatherBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<GatherBackward>(new GatherBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->dim = dim;
    grad_fn->index_ = SavedVariable(index, false);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->gather(self_, dim, index_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::grid_sampler_3d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode) const {
  profiler::RecordFunction profiler("grid_sampler_3d_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& input_ = unpack(input, "input", 1);
  auto& grid_ = unpack(grid, "grid", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, input, grid )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("grid_sampler_3d_backward"), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = as_variable(baseType->grid_sampler_3d_backward(grad_output_, input_, grid_, interpolation_mode, padding_mode));
  set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor VariableType::gru_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh) const {
  profiler::RecordFunction profiler("gru_cell", Function::peek_at_next_sequence_nr());
  auto result = TypeDefault::gru_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
  return result;
}
Tensor VariableType::hardshrink(const Tensor & self, Scalar lambd) const {
  profiler::RecordFunction profiler("hardshrink", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<HardshrinkBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<HardshrinkBackward>(new HardshrinkBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->hardshrink(self_, lambd));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::ifft(const Tensor & self, int64_t signal_ndim, bool normalized) const {
  profiler::RecordFunction profiler("ifft", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::ifft(self, signal_ndim, normalized);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::index_select_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) const {
  profiler::RecordFunction profiler("index_select_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& index_ = unpack(index, "index", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, index )) {
    throw_error_out_requires_grad("index_select");
  }
  if (compute_requires_grad( result )) {
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
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("index_select_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->index_select_out(result_, self_, dim, index_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::indices(const Tensor & self) const {
  profiler::RecordFunction profiler("indices", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_view(self, baseType->indices(self_), false);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
bool VariableType::is_complex(const Tensor & self) const {
  profiler::RecordFunction profiler("is_complex", Function::peek_at_next_sequence_nr());
  auto result = TypeDefault::is_complex(self);
  return result;
}
bool VariableType::is_same_size(const Tensor & self, const Tensor & other) const {
  auto result = TypeDefault::is_same_size(self, other);
  return result;
}
Tensor & VariableType::l1_loss_out(Tensor & output, const Tensor & self, const Tensor & target, int64_t reduction) const {
  profiler::RecordFunction profiler("l1_loss_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, target )) {
    throw_error_out_requires_grad("l1_loss");
  }
  if (compute_requires_grad( output )) {
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
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("l1_loss_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->l1_loss_out(output_, self_, target_, reduction);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::layer_norm(const Tensor & input, IntList normalized_shape, const Tensor & weight, const Tensor & bias, double eps, bool cudnn_enable) const {
  profiler::RecordFunction profiler("layer_norm", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::layer_norm(input, normalized_shape, weight, bias, eps, cudnn_enable);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::leaky_relu_backward(const Tensor & grad_output, const Tensor & self, Scalar negative_slope) const {
  profiler::RecordFunction profiler("leaky_relu_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<LeakyReluBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<LeakyReluBackwardBackward>(new LeakyReluBackwardBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->leaky_relu_backward(grad_output_, self_, negative_slope));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::lerp(const Tensor & self, const Tensor & end, Scalar weight) const {
  profiler::RecordFunction profiler("lerp", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& end_ = unpack(end, "end", 1);
  std::shared_ptr<LerpBackward> grad_fn;
  if (compute_requires_grad( self, end )) {
    grad_fn = std::shared_ptr<LerpBackward>(new LerpBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->lerp(self_, end_, weight));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::lerp_(Tensor & self, const Tensor & end, Scalar weight) const {
  profiler::RecordFunction profiler("lerp_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& end_ = unpack(end, "end", 1);
  check_inplace(self);
  std::shared_ptr<LerpBackward> grad_fn;
  if (compute_requires_grad( self, end )) {
    grad_fn = std::shared_ptr<LerpBackward>(new LerpBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("lerp_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->lerp_(self_, end_, weight);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::linear(const Tensor & input, const Tensor & weight, const Tensor & bias) const {
  profiler::RecordFunction profiler("linear", Function::peek_at_next_sequence_nr());
  auto result = TypeDefault::linear(input, weight, bias);
  return result;
}
Tensor VariableType::log_sigmoid(const Tensor & self) const {
  profiler::RecordFunction profiler("log_sigmoid", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::log_sigmoid");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::log_sigmoid(self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::log_sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & buffer) const {
  profiler::RecordFunction profiler("log_sigmoid_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& buffer_ = unpack(buffer, "buffer", 3);
  std::shared_ptr<Function> grad_fn;
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("log_sigmoid_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->log_sigmoid_backward_out(grad_input_, grad_output_, self_, buffer_);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::logspace_out(Tensor & result, Scalar start, Scalar end) const {
  profiler::RecordFunction profiler("logspace_out", Function::peek_at_next_sequence_nr());
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
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "result", result.options());
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logspace_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::logspace_out(result, start, end);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::logspace_out(Tensor & result, Scalar start, Scalar end, int64_t steps) const {
  profiler::RecordFunction profiler("logspace_out", Function::peek_at_next_sequence_nr());
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
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "result", result.options());
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logspace_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::logspace_out(result, start, end, steps);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::logsumexp_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("logsumexp_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("logsumexp");
  }
  if (compute_requires_grad( result )) {
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
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logsumexp_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->logsumexp_out(result_, self_, dim, keepdim);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::matmul_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("matmul_out", Function::peek_at_next_sequence_nr());
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
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("matmul_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::matmul_out(result, self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor &,Tensor &> VariableType::max_out(Tensor & max, Tensor & max_values, const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("max_out", Function::peek_at_next_sequence_nr());
  auto& max_ = unpack(max, "max", 0);
  auto& max_values_ = unpack(max_values, "max_values", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("max_out", max);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->max_out(max_, max_values_, self_, dim, keepdim);
  increment_version(max);
  rebase_history(flatten_tensor_args( max ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, max);
    jit::tracer::addOutput(node, max_values);
  }
  return std::forward_as_tuple(max, max_values);
}
Tensor & VariableType::max_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("max_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("max");
  }
  if (compute_requires_grad( result )) {
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
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("max_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->max_out(result_, self_, other_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::max_unpool2d(const Tensor & self, const Tensor & indices, IntList output_size) const {
  profiler::RecordFunction profiler("max_unpool2d", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  check_no_requires_grad(indices, "indices");
  std::shared_ptr<MaxUnpool2DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MaxUnpool2DBackward>(new MaxUnpool2DBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->max_unpool2d(self_, indices_, output_size));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::max_unpool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size) const {
  profiler::RecordFunction profiler("max_unpool2d_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& indices_ = unpack(indices, "indices", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, indices )) {
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("max_unpool2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->max_unpool2d_backward_out(grad_input_, grad_output_, self_, indices_, output_size);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::max_unpool3d_out(Tensor & output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
  profiler::RecordFunction profiler("max_unpool3d_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, indices )) {
    throw_error_out_requires_grad("max_unpool3d");
  }
  if (compute_requires_grad( output )) {
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
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("max_unpool3d_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->max_unpool3d_out(output_, self_, indices_, output_size, stride, padding);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::min_values(const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("min_values", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::min_values(self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::miopen_convolution_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask) const {
  profiler::RecordFunction profiler("miopen_convolution_backward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  std::shared_ptr<MiopenConvolutionBackwardBackward> grad_fn;
  if (compute_requires_grad( self, grad_output, weight )) {
    grad_fn = std::shared_ptr<MiopenConvolutionBackwardBackward>(new MiopenConvolutionBackwardBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1, result2) = as_variable(baseType->miopen_convolution_backward(self_, grad_output_, weight_, padding, stride, dilation, groups, benchmark, deterministic, output_mask));
  set_history(flatten_tensor_args( result0, result1, result2 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor VariableType::miopen_convolution_backward_bias(const Tensor & grad_output) const {
  profiler::RecordFunction profiler("miopen_convolution_backward_bias", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("miopen_convolution_backward_bias"), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->miopen_convolution_backward_bias(grad_output_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::miopen_convolution_backward_input(IntList self_size, const Tensor & grad_output, const Tensor & weight, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic) const {
  profiler::RecordFunction profiler("miopen_convolution_backward_input", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, weight )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("miopen_convolution_backward_input"), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->miopen_convolution_backward_input(self_size, grad_output_, weight_, padding, stride, dilation, groups, benchmark, deterministic));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::miopen_convolution_transpose(const Tensor & self, const Tensor & weight, const Tensor & bias, IntList padding, IntList output_padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic) const {
  profiler::RecordFunction profiler("miopen_convolution_transpose", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 2);
  std::shared_ptr<MiopenConvolutionTransposeBackward> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::shared_ptr<MiopenConvolutionTransposeBackward>(new MiopenConvolutionTransposeBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->miopen_convolution_transpose(self_, weight_, bias_, padding, output_padding, stride, dilation, groups, benchmark, deterministic));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::mkldnn_convolution(const Tensor & self, const Tensor & weight, const Tensor & bias, IntList padding, IntList stride, IntList dilation, int64_t groups) const {
  profiler::RecordFunction profiler("mkldnn_convolution", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 2);
  std::shared_ptr<MkldnnConvolutionBackward> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::shared_ptr<MkldnnConvolutionBackward>(new MkldnnConvolutionBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->mkldnn_convolution(self_, weight_, bias_, padding, stride, dilation, groups));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor &,Tensor &> VariableType::mode_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("mode_out", Function::peek_at_next_sequence_nr());
  auto& values_ = unpack(values, "values", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mode_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->mode_out(values_, indices_, self_, dim, keepdim);
  increment_version(values);
  rebase_history(flatten_tensor_args( values ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(values, indices);
}
Tensor VariableType::mse_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) const {
  profiler::RecordFunction profiler("mse_loss_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  check_no_requires_grad(target, "target");
  std::shared_ptr<MseLossBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<MseLossBackwardBackward>(new MseLossBackwardBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->mse_loss_backward(grad_output_, self_, target_, reduction));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::multilabel_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, const Tensor & is_target) const {
  profiler::RecordFunction profiler("multilabel_margin_loss_backward", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::multilabel_margin_loss_backward(grad_output, self, target, reduction, is_target);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor &,Tensor &> VariableType::multilabel_margin_loss_forward_out(Tensor & output, Tensor & is_target, const Tensor & self, const Tensor & target, int64_t reduction) const {
  profiler::RecordFunction profiler("multilabel_margin_loss_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& is_target_ = unpack(is_target, "is_target", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& target_ = unpack(target, "target", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, target )) {
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("multilabel_margin_loss_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->multilabel_margin_loss_forward_out(output_, is_target_, self_, target_, reduction);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, is_target);
  }
  return std::forward_as_tuple(output, is_target);
}
Tensor VariableType::multinomial(const Tensor & self, int64_t num_samples, bool replacement, Generator * generator) const {
  profiler::RecordFunction profiler("multinomial", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::multinomial(self, num_samples, replacement, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::mvlgamma(const Tensor & self, int64_t p) const {
  profiler::RecordFunction profiler("mvlgamma", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<MvlgammaBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MvlgammaBackward>(new MvlgammaBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->mvlgamma(self_, p));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::mvlgamma_(Tensor & self, int64_t p) const {
  profiler::RecordFunction profiler("mvlgamma_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<MvlgammaBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MvlgammaBackward>(new MvlgammaBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mvlgamma_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->mvlgamma_(self_, p);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::narrow(const Tensor & self, int64_t dim, int64_t start, int64_t length) const {
  profiler::RecordFunction profiler("narrow", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::narrow(self, dim, start, length);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::native_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) const {
  profiler::RecordFunction profiler("native_batch_norm", Function::peek_at_next_sequence_nr());
  auto& input_ = unpack(input, "input", 0);
  auto weight_ = unpack_opt(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 2);
  auto running_mean_ = unpack_opt(running_mean, "running_mean", 3);
  auto running_var_ = unpack_opt(running_var, "running_var", 4);
  check_no_requires_grad(running_mean, "running_mean");
  check_no_requires_grad(running_var, "running_var");
  std::shared_ptr<NativeBatchNormBackward> grad_fn;
  if (compute_requires_grad( input, weight, bias )) {
    grad_fn = std::shared_ptr<NativeBatchNormBackward>(new NativeBatchNormBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1, result2) = as_variable(baseType->native_batch_norm(input_, weight_, bias_, running_mean_, running_var_, training, momentum, eps));
  set_history(flatten_tensor_args( result0 ), grad_fn);
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
Tensor & VariableType::native_zero_(Tensor & self) const {
  profiler::RecordFunction profiler("native_zero_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("native_zero_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::native_zero");
    } else {
      op_name = jit::Symbol::fromQualString("aten::native_zero_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("native_zero_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->native_zero_(self_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::ne_out(Tensor & result, const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("ne_out", Function::peek_at_next_sequence_nr());
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
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ne_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::ne_out(result, self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::ne_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("ne_out", Function::peek_at_next_sequence_nr());
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
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ne_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::ne_out(result, self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::nll_loss(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) const {
  profiler::RecordFunction profiler("nll_loss", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::nll_loss(self, target, weight, reduction, ignore_index);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::nll_loss2d(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) const {
  profiler::RecordFunction profiler("nll_loss2d", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::nll_loss2d(self, target, weight, reduction, ignore_index);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::nll_loss2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) const {
  profiler::RecordFunction profiler("nll_loss2d_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& target_ = unpack(target, "target", 3);
  auto weight_ = unpack_opt(weight, "weight", 4);
  auto& total_weight_ = unpack(total_weight, "total_weight", 7);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, target, weight, total_weight )) {
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("nll_loss2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->nll_loss2d_backward_out(grad_input_, grad_output_, self_, target_, weight_, reduction, ignore_index, total_weight_);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::nll_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) const {
  profiler::RecordFunction profiler("nll_loss_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& target_ = unpack(target, "target", 3);
  auto weight_ = unpack_opt(weight, "weight", 4);
  auto& total_weight_ = unpack(total_weight, "total_weight", 7);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, target, weight, total_weight )) {
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("nll_loss_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->nll_loss_backward_out(grad_input_, grad_output_, self_, target_, weight_, reduction, ignore_index, total_weight_);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::nuclear_norm(const Tensor & self, bool keepdim) const {
  profiler::RecordFunction profiler("nuclear_norm", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::nuclear_norm(self, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
int64_t VariableType::numel(const Tensor & self) const {
  auto result = TypeDefault::numel(self);
  return result;
}
Tensor VariableType::orgqr(const Tensor & self, const Tensor & input2) const {
  profiler::RecordFunction profiler("orgqr", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& input2_ = unpack(input2, "input2", 1);
  std::shared_ptr<OrgqrBackward> grad_fn;
  if (compute_requires_grad( self, input2 )) {
    grad_fn = std::shared_ptr<OrgqrBackward>(new OrgqrBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->orgqr(self_, input2_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::ormqr_out(Tensor & result, const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) const {
  profiler::RecordFunction profiler("ormqr_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& input2_ = unpack(input2, "input2", 2);
  auto& input3_ = unpack(input3, "input3", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, input2, input3 )) {
    throw_error_out_requires_grad("ormqr");
  }
  if (compute_requires_grad( result )) {
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
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ormqr_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->ormqr_out(result_, self_, input2_, input3_, left, transpose);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::permute(const Tensor & self, IntList dims) const {
  profiler::RecordFunction profiler("permute", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<PermuteBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<PermuteBackward>(new PermuteBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_view(self, baseType->permute(self_, dims), true);
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::pixel_shuffle(const Tensor & self, int64_t upscale_factor) const {
  profiler::RecordFunction profiler("pixel_shuffle", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::pixel_shuffle(self, upscale_factor);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::put_(Tensor & self, const Tensor & index, const Tensor & source, bool accumulate) const {
  profiler::RecordFunction profiler("put_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 1);
  auto& source_ = unpack(source, "source", 2);
  check_inplace(self);
  check_no_requires_grad(index, "index");
  std::shared_ptr<PutBackward> grad_fn;
  if (compute_requires_grad( self, source )) {
    grad_fn = std::shared_ptr<PutBackward>(new PutBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("put_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->put_(self_, index_, source_, accumulate);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::rand_out(Tensor & result, IntList size) const {
  profiler::RecordFunction profiler("rand_out", Function::peek_at_next_sequence_nr());
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
      jit::tracer::addInputs(node, "result", result.options());
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("rand_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::rand_out(result, size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::rand_out(Tensor & result, IntList size, Generator * generator) const {
  profiler::RecordFunction profiler("rand_out", Function::peek_at_next_sequence_nr());
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
      jit::tracer::addInputs(node, "result", result.options());
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("rand_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::rand_out(result, size, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::randn_out(Tensor & result, IntList size) const {
  profiler::RecordFunction profiler("randn_out", Function::peek_at_next_sequence_nr());
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
      jit::tracer::addInputs(node, "result", result.options());
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("randn_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::randn_out(result, size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::randn_out(Tensor & result, IntList size, Generator * generator) const {
  profiler::RecordFunction profiler("randn_out", Function::peek_at_next_sequence_nr());
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
      jit::tracer::addInputs(node, "result", result.options());
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("randn_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::randn_out(result, size, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::reciprocal(const Tensor & self) const {
  profiler::RecordFunction profiler("reciprocal", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ReciprocalBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ReciprocalBackward>(new ReciprocalBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->reciprocal(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor & VariableType::reciprocal_(Tensor & self) const {
  profiler::RecordFunction profiler("reciprocal_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<ReciprocalBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ReciprocalBackward>(new ReciprocalBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("reciprocal_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->reciprocal_(self_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(self, true);
  }
  return self;
}
Tensor VariableType::reflection_pad1d(const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("reflection_pad1d", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ReflectionPad1DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ReflectionPad1DBackward>(new ReflectionPad1DBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->reflection_pad1d(self_, padding));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::reflection_pad1d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("reflection_pad1d_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("reflection_pad1d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->reflection_pad1d_backward_out(grad_input_, grad_output_, self_, padding);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::reflection_pad2d_out(Tensor & output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("reflection_pad2d_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("reflection_pad2d");
  }
  if (compute_requires_grad( output )) {
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
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("reflection_pad2d_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->reflection_pad2d_out(output_, self_, padding);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::relu(const Tensor & self) const {
  profiler::RecordFunction profiler("relu", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ReluBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ReluBackward0>(new ReluBackward0(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->relu(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::relu_(Tensor & self) const {
  profiler::RecordFunction profiler("relu_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<ReluBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ReluBackward1>(new ReluBackward1(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("relu_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->relu_(self_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(self, true);
  }
  return self;
}
Tensor & VariableType::remainder_out(Tensor & result, const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("remainder_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("remainder");
  }
  if (compute_requires_grad( result )) {
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
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("remainder_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->remainder_out(result_, self_, other);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::remainder_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("remainder_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("remainder");
  }
  if (compute_requires_grad( result )) {
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
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("remainder_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->remainder_out(result_, self_, other_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::replication_pad1d_out(Tensor & output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("replication_pad1d_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("replication_pad1d");
  }
  if (compute_requires_grad( output )) {
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
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("replication_pad1d_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->replication_pad1d_out(output_, self_, padding);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::rnn_relu_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh) const {
  profiler::RecordFunction profiler("rnn_relu_cell", Function::peek_at_next_sequence_nr());
  auto result = TypeDefault::rnn_relu_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
  return result;
}
Tensor VariableType::rrelu_with_noise(const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
  profiler::RecordFunction profiler("rrelu_with_noise", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& noise_ = unpack(noise, "noise", 1);
  check_no_requires_grad(noise, "noise");
  std::shared_ptr<RreluWithNoiseBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<RreluWithNoiseBackward0>(new RreluWithNoiseBackward0(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->rrelu_with_noise(self_, noise_, lower, upper, training, generator));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::rrelu_with_noise_(Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
  profiler::RecordFunction profiler("rrelu_with_noise_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& noise_ = unpack(noise, "noise", 1);
  check_inplace(self);
  check_no_requires_grad(noise, "noise");
  std::shared_ptr<RreluWithNoiseBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<RreluWithNoiseBackward1>(new RreluWithNoiseBackward1(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("rrelu_with_noise_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->rrelu_with_noise_(self_, noise_, lower, upper, training, generator);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(self, true);
  }
  return self;
}
Tensor & VariableType::rrelu_with_noise_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training) const {
  profiler::RecordFunction profiler("rrelu_with_noise_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& noise_ = unpack(noise, "noise", 3);
  std::shared_ptr<Function> grad_fn;
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("rrelu_with_noise_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->rrelu_with_noise_backward_out(grad_input_, grad_output_, self_, noise_, lower, upper, training);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::scatter_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
  profiler::RecordFunction profiler("scatter_add_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  auto& src_ = unpack(src, "src", 3);
  check_inplace(self);
  check_no_requires_grad(index, "index");
  std::shared_ptr<ScatterAddBackward> grad_fn;
  if (compute_requires_grad( self, src )) {
    grad_fn = std::shared_ptr<ScatterAddBackward>(new ScatterAddBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, src ));
    grad_fn->dim = dim;
    grad_fn->index_ = SavedVariable(index, false);
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("scatter_add_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->scatter_add_(self_, dim, index_, src_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::select(const Tensor & self, int64_t dim, int64_t index) const {
  profiler::RecordFunction profiler("select", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SelectBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SelectBackward>(new SelectBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_view(self, baseType->select(self_, dim, index), true);
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::sin(const Tensor & self) const {
  profiler::RecordFunction profiler("sin", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SinBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SinBackward>(new SinBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->sin(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::sin_(Tensor & self) const {
  profiler::RecordFunction profiler("sin_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<SinBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SinBackward>(new SinBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sin_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->sin_(self_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::soft_margin_loss_out(Tensor & output, const Tensor & self, const Tensor & target, int64_t reduction) const {
  profiler::RecordFunction profiler("soft_margin_loss_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, target )) {
    throw_error_out_requires_grad("soft_margin_loss");
  }
  if (compute_requires_grad( output )) {
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
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("soft_margin_loss_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->soft_margin_loss_out(output_, self_, target_, reduction);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::softmax(const Tensor & self, int64_t dim, ScalarType dtype) const {
  profiler::RecordFunction profiler("softmax", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::softmax(self, dim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::softmax(const Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("softmax", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::softmax(self, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::softplus_out(Tensor & output, const Tensor & self, Scalar beta, Scalar threshold) const {
  profiler::RecordFunction profiler("softplus_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("softplus");
  }
  if (compute_requires_grad( output )) {
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
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("softplus_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->softplus_out(output_, self_, beta, threshold);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::softshrink_backward(const Tensor & grad_output, const Tensor & self, Scalar lambd) const {
  profiler::RecordFunction profiler("softshrink_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<SoftshrinkBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<SoftshrinkBackwardBackward>(new SoftshrinkBackwardBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->softshrink_backward(grad_output_, self_, lambd));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor &,Tensor &> VariableType::sort_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool descending) const {
  profiler::RecordFunction profiler("sort_out", Function::peek_at_next_sequence_nr());
  auto& values_ = unpack(values, "values", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sort_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->sort_out(values_, indices_, self_, dim, descending);
  increment_version(values);
  rebase_history(flatten_tensor_args( values ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(values, indices);
}
Tensor VariableType::squeeze(const Tensor & self) const {
  profiler::RecordFunction profiler("squeeze", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SqueezeBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SqueezeBackward0>(new SqueezeBackward0(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_view(self, baseType->squeeze(self_), true);
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::squeeze(const Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("squeeze", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SqueezeBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SqueezeBackward1>(new SqueezeBackward1(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_view(self, baseType->squeeze(self_, dim), true);
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::squeeze_(Tensor & self) const {
  profiler::RecordFunction profiler("squeeze_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<SqueezeBackward2> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SqueezeBackward2>(new SqueezeBackward2(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("squeeze_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->squeeze_(self_);
  increment_version(self);
  set_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::squeeze_(Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("squeeze_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<SqueezeBackward3> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SqueezeBackward3>(new SqueezeBackward3(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("squeeze_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->squeeze_(self_, dim);
  increment_version(self);
  set_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::std_out(Tensor & result, const Tensor & self, IntList dim, bool unbiased, bool keepdim) const {
  profiler::RecordFunction profiler("std_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("std");
  }
  if (compute_requires_grad( result )) {
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
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("std_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->std_out(result_, self_, dim, unbiased, keepdim);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::sub(const Tensor & self, const Tensor & other, Scalar alpha) const {
  profiler::RecordFunction profiler("sub", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  std::shared_ptr<SubBackward0> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<SubBackward0>(new SubBackward0(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->sub(self_, other_, alpha));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::sub(const Tensor & self, Scalar other, Scalar alpha) const {
  profiler::RecordFunction profiler("sub", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SubBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SubBackward1>(new SubBackward1(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->sub(self_, other, alpha));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::sub_(Tensor & self, const Tensor & other, Scalar alpha) const {
  profiler::RecordFunction profiler("sub_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<SubBackward0> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<SubBackward0>(new SubBackward0(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sub_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->sub_(self_, other_, alpha);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::sub_(Tensor & self, Scalar other, Scalar alpha) const {
  profiler::RecordFunction profiler("sub_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<SubBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SubBackward1>(new SubBackward1(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sub_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->sub_(self_, other, alpha);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::sum_out(Tensor & result, const Tensor & self, IntList dim, bool keepdim, ScalarType dtype) const {
  profiler::RecordFunction profiler("sum_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("sum");
  }
  if (compute_requires_grad( result )) {
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
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sum_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->sum_out(result_, self_, dim, keepdim, dtype);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::sum_out(Tensor & result, const Tensor & self, IntList dim, bool keepdim) const {
  profiler::RecordFunction profiler("sum_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("sum");
  }
  if (compute_requires_grad( result )) {
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sum_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->sum_out(result_, self_, dim, keepdim);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::sum_out(Tensor & result, const Tensor & self, IntList dim, ScalarType dtype) const {
  profiler::RecordFunction profiler("sum_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("sum");
  }
  if (compute_requires_grad( result )) {
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
    jit::tracer::addInputs(node, "dtype", dtype);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sum_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->sum_out(result_, self_, dim, dtype);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::take_out(Tensor & result, const Tensor & self, const Tensor & index) const {
  profiler::RecordFunction profiler("take_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& index_ = unpack(index, "index", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, index )) {
    throw_error_out_requires_grad("take");
  }
  if (compute_requires_grad( result )) {
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
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("take_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->take_out(result_, self_, index_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::thnn_conv2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
  profiler::RecordFunction profiler("thnn_conv2d_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 3);
  std::shared_ptr<ThnnConv2DBackward> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::shared_ptr<ThnnConv2DBackward>(new ThnnConv2DBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(output, finput, fgrad_input) = as_variable(baseType->thnn_conv2d_forward(self_, weight_, kernel_size, bias_, stride, padding));
  set_history(flatten_tensor_args( output ), grad_fn);
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
std::tuple<Tensor,Tensor,Tensor> VariableType::thnn_conv3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) const {
  profiler::RecordFunction profiler("thnn_conv3d_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto& finput_ = unpack(finput, "finput", 6);
  auto& fgrad_input_ = unpack(fgrad_input, "fgrad_input", 7);
  check_no_requires_grad(finput, "finput");
  check_no_requires_grad(fgrad_input, "fgrad_input");
  std::shared_ptr<ThnnConv3DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self, weight )) {
    grad_fn = std::shared_ptr<ThnnConv3DBackwardBackward>(new ThnnConv3DBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, weight ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
  }
  Tensor grad_input;
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(grad_input, grad_weight, grad_bias) = as_variable(baseType->thnn_conv3d_backward(grad_output_, self_, weight_, kernel_size, stride, padding, finput_, fgrad_input_, output_mask));
  set_history(flatten_tensor_args( grad_input, grad_weight, grad_bias ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
    jit::tracer::addOutput(node, grad_weight);
    jit::tracer::addOutput(node, grad_bias);
  }
  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::thnn_conv3d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
  profiler::RecordFunction profiler("thnn_conv3d_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& finput_ = unpack(finput, "finput", 1);
  auto& fgrad_input_ = unpack(fgrad_input, "fgrad_input", 2);
  auto& self_ = unpack(self, "self", 3);
  auto& weight_ = unpack(weight, "weight", 4);
  auto bias_ = unpack_opt(bias, "bias", 6);
  std::shared_ptr<Function> grad_fn;
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("thnn_conv3d_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->thnn_conv3d_forward_out(output_, finput_, fgrad_input_, self_, weight_, kernel_size, bias_, stride, padding);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, finput);
    jit::tracer::addOutput(node, fgrad_input);
  }
  return std::forward_as_tuple(output, finput, fgrad_input);
}
Tensor & VariableType::thnn_conv_depthwise2d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
  profiler::RecordFunction profiler("thnn_conv_depthwise2d_out", Function::peek_at_next_sequence_nr());
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
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("thnn_conv_depthwise2d_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::thnn_conv_depthwise2d_out(output, self, weight, kernel_size, bias, stride, padding, dilation);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::thnn_conv_dilated2d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
  profiler::RecordFunction profiler("thnn_conv_dilated2d", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_conv_dilated2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::thnn_conv_dilated2d(self, weight, kernel_size, bias, stride, padding, dilation);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::thnn_conv_dilated2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones) const {
  profiler::RecordFunction profiler("thnn_conv_dilated2d_backward_out", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_conv_dilated2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_weight", grad_weight);
    jit::tracer::addInputs(node, "grad_bias", grad_bias);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "columns", columns);
    jit::tracer::addInputs(node, "ones", ones);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("thnn_conv_dilated2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::thnn_conv_dilated2d_backward_out(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, dilation, columns, ones);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
    jit::tracer::addOutput(node, grad_weight);
    jit::tracer::addOutput(node, grad_bias);
  }
  return std::forward_as_tuple(grad_input, grad_weight, grad_bias);
}
Tensor & VariableType::thnn_conv_dilated3d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
  profiler::RecordFunction profiler("thnn_conv_dilated3d_out", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_conv_dilated3d");
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
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("thnn_conv_dilated3d_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::thnn_conv_dilated3d_out(output, self, weight, kernel_size, bias, stride, padding, dilation);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::thnn_conv_transpose2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) const {
  profiler::RecordFunction profiler("thnn_conv_transpose2d_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto& columns_ = unpack(columns, "columns", 8);
  auto& ones_ = unpack(ones, "ones", 9);
  check_no_requires_grad(columns, "columns");
  check_no_requires_grad(ones, "ones");
  std::shared_ptr<ThnnConvTranspose2DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self, weight )) {
    grad_fn = std::shared_ptr<ThnnConvTranspose2DBackwardBackward>(new ThnnConvTranspose2DBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, weight ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->output_padding = output_padding.vec();
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
    op_name = jit::Symbol::fromQualString("aten::thnn_conv_transpose2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "columns", columns);
    jit::tracer::addInputs(node, "ones", ones);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(grad_input, grad_weight, grad_bias) = as_variable(baseType->thnn_conv_transpose2d_backward(grad_output_, self_, weight_, kernel_size, stride, padding, output_padding, dilation, columns_, ones_, output_mask));
  set_history(flatten_tensor_args( grad_input, grad_weight, grad_bias ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
    jit::tracer::addOutput(node, grad_weight);
    jit::tracer::addOutput(node, grad_bias);
  }
  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::thnn_conv_transpose2d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
  profiler::RecordFunction profiler("thnn_conv_transpose2d_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& columns_ = unpack(columns, "columns", 1);
  auto& ones_ = unpack(ones, "ones", 2);
  auto& self_ = unpack(self, "self", 3);
  auto& weight_ = unpack(weight, "weight", 4);
  auto bias_ = unpack_opt(bias, "bias", 6);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    throw_error_out_requires_grad("thnn_conv_transpose2d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("thnn_conv_transpose2d_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_conv_transpose2d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "columns", columns);
    jit::tracer::addInputs(node, "ones", ones);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("thnn_conv_transpose2d_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->thnn_conv_transpose2d_forward_out(output_, columns_, ones_, self_, weight_, kernel_size, bias_, stride, padding, output_padding, dilation);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, columns);
    jit::tracer::addOutput(node, ones);
  }
  return std::forward_as_tuple(output, columns, ones);
}
Tensor VariableType::thnn_conv_transpose3d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
  profiler::RecordFunction profiler("thnn_conv_transpose3d", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_conv_transpose3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::thnn_conv_transpose3d(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::thnn_conv_transpose3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & finput, const Tensor & fgrad_input) const {
  profiler::RecordFunction profiler("thnn_conv_transpose3d_backward_out", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_conv_transpose3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_weight", grad_weight);
    jit::tracer::addInputs(node, "grad_bias", grad_bias);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "finput", finput);
    jit::tracer::addInputs(node, "fgrad_input", fgrad_input);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("thnn_conv_transpose3d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::thnn_conv_transpose3d_backward_out(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, finput, fgrad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
    jit::tracer::addOutput(node, grad_weight);
    jit::tracer::addOutput(node, grad_bias);
  }
  return std::forward_as_tuple(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor &,Tensor &> VariableType::topk_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) const {
  profiler::RecordFunction profiler("topk_out", Function::peek_at_next_sequence_nr());
  auto& values_ = unpack(values, "values", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("topk_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->topk_out(values_, indices_, self_, k, dim, largest, sorted);
  increment_version(values);
  rebase_history(flatten_tensor_args( values ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(values, indices);
}
std::tuple<Tensor,Tensor> VariableType::trtrs(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) const {
  profiler::RecordFunction profiler("trtrs", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& A_ = unpack(A, "A", 1);
  std::shared_ptr<TrtrsBackward> grad_fn;
  if (compute_requires_grad( self, A )) {
    grad_fn = std::shared_ptr<TrtrsBackward>(new TrtrsBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, A ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->A_ = SavedVariable(A, false);
    grad_fn->upper = upper;
    grad_fn->transpose = transpose;
    grad_fn->unitriangular = unitriangular;
  }
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::trtrs");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "A", A);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "transpose", transpose);
    jit::tracer::addInputs(node, "unitriangular", unitriangular);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = as_variable(baseType->trtrs(self_, A_, upper, transpose, unitriangular));
  set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  if (grad_fn) {
    grad_fn->result0_ = SavedVariable(result0, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor & VariableType::trunc_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("trunc_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("trunc");
  }
  if (compute_requires_grad( result )) {
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
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("trunc_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->trunc_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::vector<Tensor> VariableType::unbind(const Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("unbind", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<UnbindBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<UnbindBackward>(new UnbindBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_view(self, baseType->unbind(self_, dim), true);
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::upsample_bilinear2d_backward(const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
  profiler::RecordFunction profiler("upsample_bilinear2d_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  std::shared_ptr<UpsampleBilinear2DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output )) {
    grad_fn = std::shared_ptr<UpsampleBilinear2DBackwardBackward>(new UpsampleBilinear2DBackwardBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->upsample_bilinear2d_backward(grad_output_, output_size, input_size, align_corners));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::upsample_linear1d(const Tensor & self, IntList output_size, bool align_corners) const {
  profiler::RecordFunction profiler("upsample_linear1d", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<UpsampleLinear1DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<UpsampleLinear1DBackward>(new UpsampleLinear1DBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->upsample_linear1d(self_, output_size, align_corners));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::upsample_linear1d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
  profiler::RecordFunction profiler("upsample_linear1d_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  std::shared_ptr<Function> grad_fn;
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_linear1d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->upsample_linear1d_backward_out(grad_input_, grad_output_, output_size, input_size, align_corners);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::upsample_nearest1d_backward(const Tensor & grad_output, IntList output_size, IntList input_size) const {
  profiler::RecordFunction profiler("upsample_nearest1d_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  std::shared_ptr<UpsampleNearest1DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output )) {
    grad_fn = std::shared_ptr<UpsampleNearest1DBackwardBackward>(new UpsampleNearest1DBackwardBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->upsample_nearest1d_backward(grad_output_, output_size, input_size));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::upsample_nearest2d(const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("upsample_nearest2d", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<UpsampleNearest2DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<UpsampleNearest2DBackward>(new UpsampleNearest2DBackward(), deleteFunction);
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->upsample_nearest2d(self_, output_size));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::upsample_nearest2d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size) const {
  profiler::RecordFunction profiler("upsample_nearest2d_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  std::shared_ptr<Function> grad_fn;
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
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_nearest2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->upsample_nearest2d_backward_out(grad_input_, grad_output_, output_size, input_size);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::upsample_nearest3d_out(Tensor & output, const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("upsample_nearest3d_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("upsample_nearest3d");
  }
  if (compute_requires_grad( output )) {
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
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_nearest3d_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->upsample_nearest3d_out(output_, self_, output_size);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::view_as(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("view_as", Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::view_as(self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

}} // namespace torch::autograd
