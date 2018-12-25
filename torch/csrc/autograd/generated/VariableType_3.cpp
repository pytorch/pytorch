#include "torch/csrc/autograd/VariableTypeUtils.h"

// @generated from tools/autograd/templates/VariableType_3.cpp

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

std::tuple<Tensor,Tensor> VariableType::RoiPooling2d_forward(const Tensor & input, const Tensor & rois, int64_t pooledHeight, int64_t pooledWidth, double spatialScale) const {
  profiler::RecordFunction profiler("RoiPooling2d_forward", Function::peek_at_next_sequence_nr());
  auto& input_ = unpack(input, "input", 0);
  auto& rois_ = unpack(rois, "rois", 1);
  check_no_requires_grad(rois, "rois");
  std::shared_ptr<Roipooling2DBackward> grad_fn;
  if (compute_requires_grad( input )) {
    grad_fn = std::shared_ptr<Roipooling2DBackward>(new Roipooling2DBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( input ));
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->rois_ = SavedVariable(rois, false);
    grad_fn->pooledHeight = pooledHeight;
    grad_fn->pooledWidth = pooledWidth;
    grad_fn->spatialScale = spatialScale;
  }
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::RoiPooling2d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "rois", rois);
    jit::tracer::addInputs(node, "pooledHeight", pooledHeight);
    jit::tracer::addInputs(node, "pooledWidth", pooledWidth);
    jit::tracer::addInputs(node, "spatialScale", spatialScale);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = as_variable(baseType->RoiPooling2d_forward(input_, rois_, pooledHeight, pooledWidth, spatialScale));
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
Tensor VariableType::_argmin(const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("_argmin", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_argmin");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::_argmin(self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::_ctc_loss(const Tensor & log_probs, const Tensor & targets, IntList input_lengths, IntList target_lengths, int64_t blank) const {
  profiler::RecordFunction profiler("_ctc_loss", Function::peek_at_next_sequence_nr());
  auto& log_probs_ = unpack(log_probs, "log_probs", 0);
  auto& targets_ = unpack(targets, "targets", 1);
  check_no_requires_grad(targets, "targets");
  std::shared_ptr<CtcLossBackward> grad_fn;
  if (compute_requires_grad( log_probs )) {
    grad_fn = std::shared_ptr<CtcLossBackward>(new CtcLossBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( log_probs ));
    grad_fn->log_probs_ = SavedVariable(log_probs, false);
    grad_fn->targets_ = SavedVariable(targets, false);
    grad_fn->input_lengths = input_lengths.vec();
    grad_fn->target_lengths = target_lengths.vec();
    grad_fn->blank = blank;
  }
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_ctc_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "log_probs", log_probs);
    jit::tracer::addInputs(node, "targets", targets);
    jit::tracer::addInputs(node, "input_lengths", input_lengths);
    jit::tracer::addInputs(node, "target_lengths", target_lengths);
    jit::tracer::addInputs(node, "blank", blank);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = as_variable(baseType->_ctc_loss(log_probs_, targets_, input_lengths, target_lengths, blank));
  set_history(flatten_tensor_args( result0 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  if (grad_fn) {
    grad_fn->result0_ = SavedVariable(result0, true);
    grad_fn->result1_ = SavedVariable(result1, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor VariableType::_cudnn_init_dropout_state(double dropout, bool train, int64_t dropout_seed, const TensorOptions & options) const {
  profiler::RecordFunction profiler("_cudnn_init_dropout_state", Function::peek_at_next_sequence_nr());
  auto options_ = TensorOptions(options).is_variable(false);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cudnn_init_dropout_state");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "dropout_seed", dropout_seed);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_cudnn_init_dropout_state(dropout, train, dropout_seed, options_));
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_cudnn_rnn_flatten_weight(TensorList weight_arr, int64_t weight_stride0, int64_t input_size, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, bool bidirectional) const {
  profiler::RecordFunction profiler("_cudnn_rnn_flatten_weight", Function::peek_at_next_sequence_nr());
  auto weight_arr_ = unpack(weight_arr, "weight_arr", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( weight_arr )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_cudnn_rnn_flatten_weight"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( weight_arr ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cudnn_rnn_flatten_weight");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "weight_arr", weight_arr);
    jit::tracer::addInputs(node, "weight_stride0", weight_stride0);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "hidden_size", hidden_size);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_cudnn_rnn_flatten_weight(weight_arr_, weight_stride0, input_size, mode, hidden_size, num_layers, batch_first, bidirectional));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_embedding_bag_dense_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, const Tensor & maximum_indices, int64_t num_weights, bool scale_grad_by_freq, int64_t mode) const {
  profiler::RecordFunction profiler("_embedding_bag_dense_backward", Function::peek_at_next_sequence_nr());
  auto& grad_ = unpack(grad, "grad", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& offsets_ = unpack(offsets, "offsets", 2);
  auto& offset2bag_ = unpack(offset2bag, "offset2bag", 3);
  auto& bag_size_ = unpack(bag_size, "bag_size", 4);
  auto& maximum_indices_ = unpack(maximum_indices, "maximum_indices", 5);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_embedding_bag_dense_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_embedding_bag_dense_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "offsets", offsets);
    jit::tracer::addInputs(node, "offset2bag", offset2bag);
    jit::tracer::addInputs(node, "bag_size", bag_size);
    jit::tracer::addInputs(node, "maximum_indices", maximum_indices);
    jit::tracer::addInputs(node, "num_weights", num_weights);
    jit::tracer::addInputs(node, "scale_grad_by_freq", scale_grad_by_freq);
    jit::tracer::addInputs(node, "mode", mode);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_embedding_bag_dense_backward(grad_, indices_, offsets_, offset2bag_, bag_size_, maximum_indices_, num_weights, scale_grad_by_freq, mode));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_log_softmax_backward_data(const Tensor & grad_output, const Tensor & output, int64_t dim, const Tensor & self) const {
  profiler::RecordFunction profiler("_log_softmax_backward_data", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& output_ = unpack(output, "output", 1);
  auto& self_ = unpack(self, "self", 3);
  std::shared_ptr<LogSoftmaxBackwardDataBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<LogSoftmaxBackwardDataBackward>(new LogSoftmaxBackwardDataBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->output_ = SavedVariable(output, false);
    grad_fn->dim = dim;
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_log_softmax_backward_data");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output", output);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_log_softmax_backward_data(grad_output_, output_, dim, self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_pack_padded_sequence_backward(const Tensor & grad, IntList input_size, const Tensor & batch_sizes, bool batch_first) const {
  profiler::RecordFunction profiler("_pack_padded_sequence_backward", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_pack_padded_sequence_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "batch_sizes", batch_sizes);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::_pack_padded_sequence_backward(grad, input_size, batch_sizes, batch_first);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_potrs_helper(const Tensor & self, const Tensor & A, bool upper) const {
  profiler::RecordFunction profiler("_potrs_helper", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& A_ = unpack(A, "A", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, A )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_potrs_helper"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, A ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_potrs_helper");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "A", A);
    jit::tracer::addInputs(node, "upper", upper);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_potrs_helper(self_, A_, upper));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_shape_as_tensor(const Tensor & self) const {
  profiler::RecordFunction profiler("_shape_as_tensor", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_shape_as_tensor");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::_shape_as_tensor(self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_sparse_div_zerodim_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("_sparse_div_zerodim_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("_sparse_div_zerodim");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_sparse_div_zerodim");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_div_zerodim");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_sparse_div_zerodim_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_sparse_div_zerodim_out(result_, self_, other_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_standard_gamma_grad(const Tensor & self, const Tensor & output) const {
  profiler::RecordFunction profiler("_standard_gamma_grad", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& output_ = unpack(output, "output", 1);
  std::shared_ptr<StandardGammaGradBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<StandardGammaGradBackward>(new StandardGammaGradBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_standard_gamma_grad");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output", output);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_standard_gamma_grad(self_, output_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::s__th_addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("_th_addbmm", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& batch1_ = unpack(batch1, "batch1", 1);
  auto& batch2_ = unpack(batch2, "batch2", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, batch1, batch2 )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_addbmm"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, batch1, batch2 ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_addbmm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "batch1", batch1);
    jit::tracer::addInputs(node, "batch2", batch2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->s__th_addbmm(self_, batch1_, batch2_, beta, alpha));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_addbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("_th_addbmm_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& batch1_ = unpack(batch1, "batch1", 1);
  auto& batch2_ = unpack(batch2, "batch2", 2);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, batch1, batch2 )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_addbmm_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, batch1, batch2 ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_addbmm");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_addbmm_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "batch1", batch1);
    jit::tracer::addInputs(node, "batch2", batch2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_addbmm_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_addbmm_(self_, batch1_, batch2_, beta, alpha);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::s__th_addcdiv_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
  profiler::RecordFunction profiler("_th_addcdiv_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& tensor1_ = unpack(tensor1, "tensor1", 2);
  auto& tensor2_ = unpack(tensor2, "tensor2", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, tensor1, tensor2 )) {
    throw_error_out_requires_grad("_th_addcdiv");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_addcdiv");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_addcdiv");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "tensor1", tensor1);
    jit::tracer::addInputs(node, "tensor2", tensor2);
    jit::tracer::addInputs(node, "value", value);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_addcdiv_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s__th_addcdiv_out(result_, self_, tensor1_, tensor2_, value);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::s__th_addmv_out(Tensor & result, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("_th_addmv_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& mat_ = unpack(mat, "mat", 2);
  auto& vec_ = unpack(vec, "vec", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, mat, vec )) {
    throw_error_out_requires_grad("_th_addmv");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_addmv");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_addmv");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat", mat);
    jit::tracer::addInputs(node, "vec", vec);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_addmv_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s__th_addmv_out(result_, self_, mat_, vec_, beta, alpha);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_all(const Tensor & self) const {
  profiler::RecordFunction profiler("_th_all", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_all"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_all");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_all(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_all(const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("_th_all", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_all"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_all");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_all(self_, dim, keepdim));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_any(const Tensor & self) const {
  profiler::RecordFunction profiler("_th_any", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_any"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_any");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_any(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_any(const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("_th_any", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_any"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_any");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_any(self_, dim, keepdim));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::_th_btrifact_with_info_out(Tensor & result, Tensor & pivots, Tensor & info, const Tensor & self, bool pivot) const {
  profiler::RecordFunction profiler("_th_btrifact_with_info_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& pivots_ = unpack(pivots, "pivots", 1);
  auto& info_ = unpack(info, "info", 2);
  auto& self_ = unpack(self, "self", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_btrifact_with_info");
  }
  if (compute_requires_grad( result, pivots, info )) {
    throw_error_out_requires_grad("_th_btrifact_with_info");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_btrifact_with_info");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "pivots", pivots);
    jit::tracer::addInputs(node, "info", info);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "pivot", pivot);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_btrifact_with_info_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_btrifact_with_info_out(result_, pivots_, info_, self_, pivot);
  increment_version(result);
  increment_version(pivots);
  increment_version(info);
  rebase_history(flatten_tensor_args( result, pivots, info ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
    jit::tracer::addOutput(node, pivots);
    jit::tracer::addOutput(node, info);
  }
  return std::forward_as_tuple(result, pivots, info);
}
Tensor VariableType::_th_ceil(const Tensor & self) const {
  profiler::RecordFunction profiler("_th_ceil", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_ceil"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_ceil");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_ceil(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_clamp_min_out(Tensor & result, const Tensor & self, Scalar min) const {
  profiler::RecordFunction profiler("_th_clamp_min_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_clamp_min");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_clamp_min");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_clamp_min");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min", min);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_clamp_min_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_clamp_min_out(result_, self_, min);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_cosh_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("_th_cosh_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_cosh");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_cosh");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_cosh");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_cosh_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_cosh_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_cross_out(Tensor & result, const Tensor & self, const Tensor & other, int64_t dim) const {
  profiler::RecordFunction profiler("_th_cross_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("_th_cross");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_cross");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_cross");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "dim", dim);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_cross_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_cross_out(result_, self_, other_, dim);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor &,Tensor &> VariableType::_th_eig_out(Tensor & res1, Tensor & res2, const Tensor & self, bool eigenvectors) const {
  profiler::RecordFunction profiler("_th_eig_out", Function::peek_at_next_sequence_nr());
  auto& res1_ = unpack(res1, "res1", 0);
  auto& res2_ = unpack(res2, "res2", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_eig");
  }
  if (compute_requires_grad( res1, res2 )) {
    throw_error_out_requires_grad("_th_eig");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_eig");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "res2", res2);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "eigenvectors", eigenvectors);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "res1", res1);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_eig_out", res1);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_eig_out(res1_, res2_, self_, eigenvectors);
  increment_version(res1);
  increment_version(res2);
  rebase_history(flatten_tensor_args( res1, res2 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, res1);
    jit::tracer::addOutput(node, res2);
  }
  return std::forward_as_tuple(res1, res2);
}
Tensor & VariableType::_th_eq_out(Tensor & result, const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("_th_eq_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_eq");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_eq_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_eq_out(result_, self_, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::s__th_eq_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("_th_eq_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_eq");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_eq_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s__th_eq_out(result_, self_, other_);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_exp(const Tensor & self) const {
  profiler::RecordFunction profiler("_th_exp", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_exp"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_exp");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_exp(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_fill_(Tensor & self, Scalar value) const {
  profiler::RecordFunction profiler("_th_fill_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_fill_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_fill");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_fill_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_fill_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_fill_(self_, value);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::_th_fill_(Tensor & self, const Tensor & value) const {
  profiler::RecordFunction profiler("_th_fill_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& value_ = unpack(value, "value", 1);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, value )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_fill_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, value ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_fill");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_fill_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_fill_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_fill_(self_, value_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::_th_ge_out(Tensor & result, const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("_th_ge_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_ge");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_ge_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_ge_out(result_, self_, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::s__th_ge_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("_th_ge_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_ge");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_ge_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s__th_ge_out(result_, self_, other_);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::_th_geqrf(const Tensor & self) const {
  profiler::RecordFunction profiler("_th_geqrf", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_geqrf"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor res1;
  Tensor res2;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_geqrf");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(res1, res2) = as_variable(baseType->_th_geqrf(self_));
  set_history(flatten_tensor_args( res1, res2 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, res1);
    jit::tracer::addOutput(node, res2);
  }
  return std::make_tuple(std::move(res1), std::move(res2));
}
Tensor VariableType::_th_ger(const Tensor & self, const Tensor & vec2) const {
  profiler::RecordFunction profiler("_th_ger", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& vec2_ = unpack(vec2, "vec2", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, vec2 )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_ger"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, vec2 ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_ger");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "vec2", vec2);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_ger(self_, vec2_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor &,Tensor &> VariableType::_th_gesv_single_out(Tensor & solution, Tensor & lu, const Tensor & self, const Tensor & A) const {
  profiler::RecordFunction profiler("_th_gesv_single_out", Function::peek_at_next_sequence_nr());
  auto& solution_ = unpack(solution, "solution", 0);
  auto& lu_ = unpack(lu, "lu", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& A_ = unpack(A, "A", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, A )) {
    throw_error_out_requires_grad("_th_gesv_single");
  }
  if (compute_requires_grad( solution, lu )) {
    throw_error_out_requires_grad("_th_gesv_single");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_gesv_single");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "lu", lu);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "A", A);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "solution", solution);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_gesv_single_out", solution);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_gesv_single_out(solution_, lu_, self_, A_);
  increment_version(solution);
  increment_version(lu);
  rebase_history(flatten_tensor_args( solution, lu ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, solution);
    jit::tracer::addOutput(node, lu);
  }
  return std::forward_as_tuple(solution, lu);
}
Tensor & VariableType::_th_gt_out(Tensor & result, const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("_th_gt_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_gt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_gt_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_gt_out(result_, self_, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::s__th_gt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("_th_gt_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_gt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_gt_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s__th_gt_out(result_, self_, other_);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_histc_out(Tensor & result, const Tensor & self, int64_t bins, Scalar min, Scalar max) const {
  profiler::RecordFunction profiler("_th_histc_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_histc");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_histc");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_histc");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "bins", bins);
    jit::tracer::addInputs(node, "min", min);
    jit::tracer::addInputs(node, "max", max);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_histc_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_histc_out(result_, self_, bins, min, max);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_index_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
  profiler::RecordFunction profiler("_th_index_add_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  auto& source_ = unpack(source, "source", 3);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, source )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_index_add_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, source ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_index_add");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_index_add_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "source", source);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_index_add_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_index_add_(self_, dim, index_, source_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
std::tuple<Tensor,Tensor> VariableType::_th_kthvalue(const Tensor & self, int64_t k, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("_th_kthvalue", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_kthvalue"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor values;
  Tensor indices;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_kthvalue");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "k", k);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(values, indices) = as_variable(baseType->_th_kthvalue(self_, k, dim, keepdim));
  set_history(flatten_tensor_args( values ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
Tensor & VariableType::_th_le_out(Tensor & result, const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("_th_le_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_le");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_le_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_le_out(result_, self_, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::s__th_le_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("_th_le_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_le");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_le_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s__th_le_out(result_, self_, other_);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_lgamma(const Tensor & self) const {
  profiler::RecordFunction profiler("_th_lgamma", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_lgamma"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_lgamma");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_lgamma(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_lgamma_(Tensor & self) const {
  profiler::RecordFunction profiler("_th_lgamma_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_lgamma_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_lgamma");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_lgamma_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_lgamma_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_lgamma_(self_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::_th_log10_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("_th_log10_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_log10");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_log10");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_log10");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_log10_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_log10_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_log1p(const Tensor & self) const {
  profiler::RecordFunction profiler("_th_log1p", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_log1p"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_log1p");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_log1p(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_lt_out(Tensor & result, const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("_th_lt_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_lt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_lt_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_lt_out(result_, self_, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::s__th_lt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("_th_lt_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_lt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_lt_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s__th_lt_out(result_, self_, other_);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::s__th_masked_select_out(Tensor & result, const Tensor & self, const Tensor & mask) const {
  profiler::RecordFunction profiler("_th_masked_select_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& mask_ = unpack(mask, "mask", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_masked_select");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_masked_select");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_masked_select");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_masked_select_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s__th_masked_select_out(result_, self_, mask_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::s__th_min_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("_th_min_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("_th_min");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_min");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_min");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_min_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s__th_min_out(result_, self_, other_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor &,Tensor &> VariableType::_th_min_out(Tensor & min, Tensor & min_indices, const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("_th_min_out", Function::peek_at_next_sequence_nr());
  auto& min_ = unpack(min, "min", 0);
  auto& min_indices_ = unpack(min_indices, "min_indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_min");
  }
  if (compute_requires_grad( min )) {
    throw_error_out_requires_grad("_th_min");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_min");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "min_indices", min_indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "min", min);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_min_out", min);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_min_out(min_, min_indices_, self_, dim, keepdim);
  increment_version(min);
  rebase_history(flatten_tensor_args( min ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, min);
    jit::tracer::addOutput(node, min_indices);
  }
  return std::forward_as_tuple(min, min_indices);
}
Tensor VariableType::_th_mm(const Tensor & self, const Tensor & mat2) const {
  profiler::RecordFunction profiler("_th_mm", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& mat2_ = unpack(mat2, "mat2", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, mat2 )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_mm"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, mat2 ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_mm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat2", mat2);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_mm(self_, mat2_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_neg_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("_th_neg_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_neg");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_neg");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_neg");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_neg_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_neg_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_nonzero_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("_th_nonzero_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_nonzero");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_nonzero_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_nonzero_out(result_, self_);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_norm_out(Tensor & result, const Tensor & self, Scalar p, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("_th_norm_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_norm");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_norm");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_norm_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_norm_out(result_, self_, p, dim, keepdim);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_normal_out(Tensor & output, const Tensor & mean, double std, Generator * generator) const {
  profiler::RecordFunction profiler("_th_normal_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& mean_ = unpack(mean, "mean", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( mean )) {
    throw_error_out_requires_grad("_th_normal");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_th_normal");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_normal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_normal_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_normal_out(output_, mean_, std, generator);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor & VariableType::_th_normal_out(Tensor & output, double mean, const Tensor & std, Generator * generator) const {
  profiler::RecordFunction profiler("_th_normal_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& std_ = unpack(std, "std", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( std )) {
    throw_error_out_requires_grad("_th_normal");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_th_normal");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_normal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_normal_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_normal_out(output_, mean, std_, generator);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor & VariableType::_th_normal_out(Tensor & output, const Tensor & mean, const Tensor & std, Generator * generator) const {
  profiler::RecordFunction profiler("_th_normal_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& mean_ = unpack(mean, "mean", 1);
  auto& std_ = unpack(std, "std", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( mean, std )) {
    throw_error_out_requires_grad("_th_normal");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_th_normal");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_normal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_normal_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_normal_out(output_, mean_, std_, generator);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_th_potri(const Tensor & self, bool upper) const {
  profiler::RecordFunction profiler("_th_potri", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_potri"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_potri");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "upper", upper);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto output = as_variable(baseType->_th_potri(self_, upper));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
std::tuple<Tensor &,Tensor &> VariableType::_th_pstrf_out(Tensor & res1, Tensor & res2, const Tensor & self, bool upper, Scalar tol) const {
  profiler::RecordFunction profiler("_th_pstrf_out", Function::peek_at_next_sequence_nr());
  auto& res1_ = unpack(res1, "res1", 0);
  auto& res2_ = unpack(res2, "res2", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_pstrf");
  }
  if (compute_requires_grad( res1, res2 )) {
    throw_error_out_requires_grad("_th_pstrf");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_pstrf");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "res2", res2);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "tol", tol);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "res1", res1);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_pstrf_out", res1);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_pstrf_out(res1_, res2_, self_, upper, tol);
  increment_version(res1);
  increment_version(res2);
  rebase_history(flatten_tensor_args( res1, res2 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, res1);
    jit::tracer::addOutput(node, res2);
  }
  return std::forward_as_tuple(res1, res2);
}
Tensor & VariableType::_th_renorm_out(Tensor & result, const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const {
  profiler::RecordFunction profiler("_th_renorm_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_renorm");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_renorm");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_renorm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "maxnorm", maxnorm);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_renorm_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_renorm_out(result_, self_, p, dim, maxnorm);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
  profiler::RecordFunction profiler("_th_scatter_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  auto& src_ = unpack(src, "src", 3);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, src )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_scatter_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, src ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_scatter");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_scatter_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "src", src);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_scatter_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_scatter_(self_, dim, index_, src_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::_th_scatter_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
  profiler::RecordFunction profiler("_th_scatter_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_scatter_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_scatter");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_scatter_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_scatter_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_scatter_(self_, dim, index_, value);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::_th_sigmoid(const Tensor & self) const {
  profiler::RecordFunction profiler("_th_sigmoid", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_sigmoid"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_sigmoid");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_sigmoid(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_sign(const Tensor & self) const {
  profiler::RecordFunction profiler("_th_sign", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_sign"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_sign");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_sign(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_sign_(Tensor & self) const {
  profiler::RecordFunction profiler("_th_sign_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_sign_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_sign");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_sign_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_sign_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_sign_(self_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::_th_sinh_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("_th_sinh_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_sinh");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_sinh");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_sinh");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_sinh_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_sinh_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_sqrt(const Tensor & self) const {
  profiler::RecordFunction profiler("_th_sqrt", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_sqrt"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_sqrt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_sqrt(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::_th_svd(const Tensor & self, bool some, bool compute_uv) const {
  profiler::RecordFunction profiler("_th_svd", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_svd"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor res1;
  Tensor res2;
  Tensor res3;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_svd");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "some", some);
    jit::tracer::addInputs(node, "compute_uv", compute_uv);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(res1, res2, res3) = as_variable(baseType->_th_svd(self_, some, compute_uv));
  set_history(flatten_tensor_args( res1, res2, res3 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, res1);
    jit::tracer::addOutput(node, res2);
    jit::tracer::addOutput(node, res3);
  }
  return std::make_tuple(std::move(res1), std::move(res2), std::move(res3));
}
std::tuple<Tensor &,Tensor &> VariableType::_th_symeig_out(Tensor & res1, Tensor & res2, const Tensor & self, bool eigenvectors, bool upper) const {
  profiler::RecordFunction profiler("_th_symeig_out", Function::peek_at_next_sequence_nr());
  auto& res1_ = unpack(res1, "res1", 0);
  auto& res2_ = unpack(res2, "res2", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_symeig");
  }
  if (compute_requires_grad( res1, res2 )) {
    throw_error_out_requires_grad("_th_symeig");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_symeig");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "res2", res2);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "eigenvectors", eigenvectors);
    jit::tracer::addInputs(node, "upper", upper);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "res1", res1);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_symeig_out", res1);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_symeig_out(res1_, res2_, self_, eigenvectors, upper);
  increment_version(res1);
  increment_version(res2);
  rebase_history(flatten_tensor_args( res1, res2 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, res1);
    jit::tracer::addOutput(node, res2);
  }
  return std::forward_as_tuple(res1, res2);
}
Tensor VariableType::_th_tan(const Tensor & self) const {
  profiler::RecordFunction profiler("_th_tan", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_tan"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_tan");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_tan(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_tril(const Tensor & self, int64_t diagonal) const {
  profiler::RecordFunction profiler("_th_tril", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_tril"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_tril");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "diagonal", diagonal);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_tril(self_, diagonal));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_tril_(Tensor & self, int64_t diagonal) const {
  profiler::RecordFunction profiler("_th_tril_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_tril_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_tril");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_tril_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "diagonal", diagonal);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_tril_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_tril_(self_, diagonal);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::_th_unfold(const Tensor & self, int64_t dimension, int64_t size, int64_t step) const {
  profiler::RecordFunction profiler("_th_unfold", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_unfold"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_unfold");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dimension", dimension);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "step", step);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_unfold(self_, dimension, size, step));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_uniform_(Tensor & self, double from, double to, Generator * generator) const {
  profiler::RecordFunction profiler("_th_uniform_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_uniform_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_uniform");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_uniform_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "from", from);
    jit::tracer::addInputs(node, "to", to);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_uniform_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_uniform_(self_, from, to, generator);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::_th_var_out(Tensor & result, const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
  profiler::RecordFunction profiler("_th_var_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_var");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_var");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_var");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_var_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_var_out(result_, self_, dim, unbiased, keepdim);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_view(const Tensor & self, IntList size) const {
  profiler::RecordFunction profiler("_th_view", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_view"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_view");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_view(self_, size));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_zero_(Tensor & self) const {
  profiler::RecordFunction profiler("_th_zero_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_zero_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_zero");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_zero_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_zero_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_zero_(self_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::_thnn_adaptive_avg_pool3d_backward(const Tensor & grad_output, const Tensor & self) const {
  profiler::RecordFunction profiler("_thnn_adaptive_avg_pool3d_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_adaptive_avg_pool3d_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_adaptive_avg_pool3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto grad_input = as_variable(baseType->_thnn_adaptive_avg_pool3d_backward(grad_output_, self_));
  set_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_adaptive_avg_pool3d_forward_out(Tensor & output, const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("_thnn_adaptive_avg_pool3d_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_thnn_adaptive_avg_pool3d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_thnn_adaptive_avg_pool3d_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_adaptive_avg_pool3d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_adaptive_avg_pool3d_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_adaptive_avg_pool3d_forward_out(output_, self_, output_size);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_thnn_avg_pool3d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
  profiler::RecordFunction profiler("_thnn_avg_pool3d_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_avg_pool3d_forward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_avg_pool3d_forward");
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
  auto output = as_variable(baseType->_thnn_avg_pool3d_forward(self_, kernel_size, stride, padding, ceil_mode, count_include_pad));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_thnn_col2im_backward(const Tensor & grad_output, IntList kernel_size, IntList dilation, IntList padding, IntList stride) const {
  profiler::RecordFunction profiler("_thnn_col2im_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_col2im_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_col2im_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto grad_input = as_variable(baseType->_thnn_col2im_backward(grad_output_, kernel_size, dilation, padding, stride));
  set_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_col2im_forward_out(Tensor & output, const Tensor & self, IntList output_size, IntList kernel_size, IntList dilation, IntList padding, IntList stride) const {
  profiler::RecordFunction profiler("_thnn_col2im_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_thnn_col2im_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_thnn_col2im_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_col2im_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_col2im_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_col2im_forward_out(output_, self_, output_size, kernel_size, dilation, padding, stride);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::_thnn_conv2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input) const {
  profiler::RecordFunction profiler("_thnn_conv2d_backward_out", Function::peek_at_next_sequence_nr());
  auto grad_input_ = unpack_opt(grad_input, "grad_input", 0);
  auto grad_weight_ = unpack_opt(grad_weight, "grad_weight", 1);
  auto grad_bias_ = unpack_opt(grad_bias, "grad_bias", 2);
  auto& grad_output_ = unpack(grad_output, "grad_output", 3);
  auto& self_ = unpack(self, "self", 4);
  auto& weight_ = unpack(weight, "weight", 5);
  auto& finput_ = unpack(finput, "finput", 9);
  auto& fgrad_input_ = unpack(fgrad_input, "fgrad_input", 10);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, weight, finput, fgrad_input )) {
    throw_error_out_requires_grad("_thnn_conv2d_backward");
  }
  if (compute_requires_grad( grad_input, grad_weight, grad_bias )) {
    throw_error_out_requires_grad("_thnn_conv2d_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_conv2d_backward");
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
    jit::tracer::addInputs(node, "finput", finput);
    jit::tracer::addInputs(node, "fgrad_input", fgrad_input);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_conv2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_conv2d_backward_out(grad_input_, grad_weight_, grad_bias_, grad_output_, self_, weight_, kernel_size, stride, padding, finput_, fgrad_input_);
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
Tensor VariableType::_thnn_conv_depthwise2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
  profiler::RecordFunction profiler("_thnn_conv_depthwise2d_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 3);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_conv_depthwise2d_forward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_conv_depthwise2d_forward");
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
  auto output = as_variable(baseType->_thnn_conv_depthwise2d_forward(self_, weight_, kernel_size, bias_, stride, padding, dilation));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::_thnn_conv_dilated3d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
  profiler::RecordFunction profiler("_thnn_conv_dilated3d_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 3);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_conv_dilated3d_forward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
  }
  Tensor output;
  Tensor columns;
  Tensor ones;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_conv_dilated3d_forward");
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
  std::tie(output, columns, ones) = as_variable(baseType->_thnn_conv_dilated3d_forward(self_, weight_, kernel_size, bias_, stride, padding, dilation));
  set_history(flatten_tensor_args( output, columns, ones ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, columns);
    jit::tracer::addOutput(node, ones);
  }
  return std::make_tuple(std::move(output), std::move(columns), std::move(ones));
}
Tensor VariableType::_thnn_elu_forward(const Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale) const {
  profiler::RecordFunction profiler("_thnn_elu_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_elu_forward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_elu_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "input_scale", input_scale);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto output = as_variable(baseType->_thnn_elu_forward(self_, alpha, scale, input_scale));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor & VariableType::_thnn_elu_forward_(Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale) const {
  profiler::RecordFunction profiler("_thnn_elu_forward_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_elu_forward_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_thnn_elu_forward");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_thnn_elu_forward_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "input_scale", input_scale);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_elu_forward_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_elu_forward_(self_, alpha, scale, input_scale);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::_thnn_fractional_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & indices) const {
  profiler::RecordFunction profiler("_thnn_fractional_max_pool2d_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 4);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_fractional_max_pool2d_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_fractional_max_pool2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "indices", indices);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto grad_input = as_variable(baseType->_thnn_fractional_max_pool2d_backward(grad_output_, self_, kernel_size, output_size, indices_));
  set_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
std::tuple<Tensor &,Tensor &> VariableType::_thnn_fractional_max_pool2d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & random_samples) const {
  profiler::RecordFunction profiler("_thnn_fractional_max_pool2d_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& random_samples_ = unpack(random_samples, "random_samples", 5);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, random_samples )) {
    throw_error_out_requires_grad("_thnn_fractional_max_pool2d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_thnn_fractional_max_pool2d_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_fractional_max_pool2d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "random_samples", random_samples);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_fractional_max_pool2d_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_fractional_max_pool2d_forward_out(output_, indices_, self_, kernel_size, output_size, random_samples_);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(output, indices);
}
Tensor & VariableType::_thnn_glu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("_thnn_glu_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("_thnn_glu_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("_thnn_glu_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_glu_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_glu_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_glu_backward_out(grad_input_, grad_output_, self_, dim);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::_thnn_hardtanh_backward(const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val) const {
  profiler::RecordFunction profiler("_thnn_hardtanh_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_hardtanh_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_hardtanh_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min_val", min_val);
    jit::tracer::addInputs(node, "max_val", max_val);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto grad_input = as_variable(baseType->_thnn_hardtanh_backward(grad_output_, self_, min_val, max_val));
  set_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_hardtanh_forward_out(Tensor & output, const Tensor & self, Scalar min_val, Scalar max_val) const {
  profiler::RecordFunction profiler("_thnn_hardtanh_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_thnn_hardtanh_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_thnn_hardtanh_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_hardtanh_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min_val", min_val);
    jit::tracer::addInputs(node, "max_val", max_val);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_hardtanh_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_hardtanh_forward_out(output_, self_, min_val, max_val);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_thnn_im2col_backward(const Tensor & grad_output, IntList input_size, IntList kernel_size, IntList dilation, IntList padding, IntList stride) const {
  profiler::RecordFunction profiler("_thnn_im2col_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_im2col_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_im2col_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto grad_input = as_variable(baseType->_thnn_im2col_backward(grad_output_, input_size, kernel_size, dilation, padding, stride));
  set_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_im2col_forward_out(Tensor & output, const Tensor & self, IntList kernel_size, IntList dilation, IntList padding, IntList stride) const {
  profiler::RecordFunction profiler("_thnn_im2col_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_thnn_im2col_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_thnn_im2col_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_im2col_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_im2col_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_im2col_forward_out(output_, self_, kernel_size, dilation, padding, stride);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_thnn_l1_loss_forward(const Tensor & self, const Tensor & target, int64_t reduction) const {
  profiler::RecordFunction profiler("_thnn_l1_loss_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, target )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_l1_loss_forward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, target ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_l1_loss_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto output = as_variable(baseType->_thnn_l1_loss_forward(self_, target_, reduction));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_thnn_max_pool2d_with_indices_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const {
  profiler::RecordFunction profiler("_thnn_max_pool2d_with_indices_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 7);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_max_pool2d_with_indices_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_max_pool2d_with_indices_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    jit::tracer::addInputs(node, "indices", indices);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto grad_input = as_variable(baseType->_thnn_max_pool2d_with_indices_backward(grad_output_, self_, kernel_size, stride, padding, dilation, ceil_mode, indices_));
  set_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
std::tuple<Tensor &,Tensor &> VariableType::_thnn_max_pool2d_with_indices_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
  profiler::RecordFunction profiler("_thnn_max_pool2d_with_indices_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_thnn_max_pool2d_with_indices_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_thnn_max_pool2d_with_indices_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_max_pool2d_with_indices_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_max_pool2d_with_indices_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_max_pool2d_with_indices_forward_out(output_, indices_, self_, kernel_size, stride, padding, dilation, ceil_mode);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(output, indices);
}
Tensor & VariableType::_thnn_max_pool3d_with_indices_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const {
  profiler::RecordFunction profiler("_thnn_max_pool3d_with_indices_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& indices_ = unpack(indices, "indices", 8);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("_thnn_max_pool3d_with_indices_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("_thnn_max_pool3d_with_indices_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_max_pool3d_with_indices_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    jit::tracer::addInputs(node, "indices", indices);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_max_pool3d_with_indices_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_max_pool3d_with_indices_backward_out(grad_input_, grad_output_, self_, kernel_size, stride, padding, dilation, ceil_mode, indices_);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::_thnn_max_unpool3d_forward(const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
  profiler::RecordFunction profiler("_thnn_max_unpool3d_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_max_unpool3d_forward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_max_unpool3d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto output = as_variable(baseType->_thnn_max_unpool3d_forward(self_, indices_, output_size, stride, padding));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_thnn_multi_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) const {
  profiler::RecordFunction profiler("_thnn_multi_margin_loss_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  auto weight_ = unpack_opt(weight, "weight", 5);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, self, weight )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_multi_margin_loss_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, weight ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_multi_margin_loss_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "margin", margin);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto grad_input = as_variable(baseType->_thnn_multi_margin_loss_backward(grad_output_, self_, target_, p, margin, weight_, reduction));
  set_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_multi_margin_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) const {
  profiler::RecordFunction profiler("_thnn_multi_margin_loss_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  auto weight_ = unpack_opt(weight, "weight", 5);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, weight )) {
    throw_error_out_requires_grad("_thnn_multi_margin_loss_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_thnn_multi_margin_loss_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_multi_margin_loss_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "margin", margin);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_multi_margin_loss_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_multi_margin_loss_forward_out(output_, self_, target_, p, margin, weight_, reduction);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_thnn_reflection_pad2d_forward(const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("_thnn_reflection_pad2d_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_reflection_pad2d_forward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_reflection_pad2d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto output = as_variable(baseType->_thnn_reflection_pad2d_forward(self_, padding));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_thnn_replication_pad1d_forward(const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("_thnn_replication_pad1d_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_replication_pad1d_forward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_replication_pad1d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto output = as_variable(baseType->_thnn_replication_pad1d_forward(self_, padding));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_thnn_replication_pad2d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("_thnn_replication_pad2d_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_replication_pad2d_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_replication_pad2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto grad_input = as_variable(baseType->_thnn_replication_pad2d_backward(grad_output_, self_, padding));
  set_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_replication_pad2d_forward_out(Tensor & output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("_thnn_replication_pad2d_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_thnn_replication_pad2d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_thnn_replication_pad2d_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_replication_pad2d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_replication_pad2d_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_replication_pad2d_forward_out(output_, self_, padding);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor & VariableType::_thnn_replication_pad3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("_thnn_replication_pad3d_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("_thnn_replication_pad3d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("_thnn_replication_pad3d_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_replication_pad3d_backward");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_replication_pad3d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_replication_pad3d_backward_out(grad_input_, grad_output_, self_, padding);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output) const {
  profiler::RecordFunction profiler("_thnn_sigmoid_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& output_ = unpack(output, "output", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, output )) {
    throw_error_out_requires_grad("_thnn_sigmoid_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("_thnn_sigmoid_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_sigmoid_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output", output);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_sigmoid_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_sigmoid_backward_out(grad_input_, grad_output_, output_);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::_thnn_smooth_l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) const {
  profiler::RecordFunction profiler("_thnn_smooth_l1_loss_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, self, target )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_smooth_l1_loss_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, target ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_smooth_l1_loss_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto grad_input = as_variable(baseType->_thnn_smooth_l1_loss_backward(grad_output_, self_, target_, reduction));
  set_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_smooth_l1_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, int64_t reduction) const {
  profiler::RecordFunction profiler("_thnn_smooth_l1_loss_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, target )) {
    throw_error_out_requires_grad("_thnn_smooth_l1_loss_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_thnn_smooth_l1_loss_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_smooth_l1_loss_forward");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_smooth_l1_loss_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_smooth_l1_loss_forward_out(output_, self_, target_, reduction);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_thnn_soft_margin_loss_forward(const Tensor & self, const Tensor & target, int64_t reduction) const {
  profiler::RecordFunction profiler("_thnn_soft_margin_loss_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, target )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_soft_margin_loss_forward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, target ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_soft_margin_loss_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto output = as_variable(baseType->_thnn_soft_margin_loss_forward(self_, target_, reduction));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_thnn_softplus_forward(const Tensor & self, Scalar beta, Scalar threshold) const {
  profiler::RecordFunction profiler("_thnn_softplus_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_softplus_forward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_softplus_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "threshold", threshold);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto output = as_variable(baseType->_thnn_softplus_forward(self_, beta, threshold));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_thnn_tanh_backward(const Tensor & grad_output, const Tensor & output) const {
  profiler::RecordFunction profiler("_thnn_tanh_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& output_ = unpack(output, "output", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, output )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_tanh_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, output ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_tanh_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output", output);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto grad_input = as_variable(baseType->_thnn_tanh_backward(grad_output_, output_));
  set_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_tanh_forward_out(Tensor & output, const Tensor & self) const {
  profiler::RecordFunction profiler("_thnn_tanh_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_thnn_tanh_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_thnn_tanh_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_tanh_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_tanh_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_tanh_forward_out(output_, self_);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_thnn_upsample_bicubic2d_backward(const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
  profiler::RecordFunction profiler("_thnn_upsample_bicubic2d_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_upsample_bicubic2d_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_upsample_bicubic2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto grad_input = as_variable(baseType->_thnn_upsample_bicubic2d_backward(grad_output_, output_size, input_size, align_corners));
  set_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_upsample_bicubic2d_forward_out(Tensor & output, const Tensor & self, IntList output_size, bool align_corners) const {
  profiler::RecordFunction profiler("_thnn_upsample_bicubic2d_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_thnn_upsample_bicubic2d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_thnn_upsample_bicubic2d_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_upsample_bicubic2d_forward");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_upsample_bicubic2d_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_upsample_bicubic2d_forward_out(output_, self_, output_size, align_corners);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_thnn_upsample_nearest3d_forward(const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("_thnn_upsample_nearest3d_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_upsample_nearest3d_forward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_upsample_nearest3d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto output = as_variable(baseType->_thnn_upsample_nearest3d_forward(self_, output_size));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_thnn_upsample_trilinear3d_backward(const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
  profiler::RecordFunction profiler("_thnn_upsample_trilinear3d_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_upsample_trilinear3d_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_upsample_trilinear3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto grad_input = as_variable(baseType->_thnn_upsample_trilinear3d_backward(grad_output_, output_size, input_size, align_corners));
  set_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_upsample_trilinear3d_forward_out(Tensor & output, const Tensor & self, IntList output_size, bool align_corners) const {
  profiler::RecordFunction profiler("_thnn_upsample_trilinear3d_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_thnn_upsample_trilinear3d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_thnn_upsample_trilinear3d_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_upsample_trilinear3d_forward");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_upsample_trilinear3d_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_upsample_trilinear3d_forward_out(output_, self_, output_size, align_corners);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
std::tuple<Tensor,Tensor> VariableType::_unique(const Tensor & self, bool sorted, bool return_inverse) const {
  profiler::RecordFunction profiler("_unique", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<UniqueBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<UniqueBackward>(new UniqueBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_unique");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "sorted", sorted);
    jit::tracer::addInputs(node, "return_inverse", return_inverse);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = as_variable(baseType->_unique(self_, sorted, return_inverse));
  set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
std::tuple<Tensor,Tensor> VariableType::_weight_norm_differentiable_backward(const Tensor & grad_w, const Tensor & saved_v, const Tensor & saved_g, const Tensor & saved_norms, int64_t dim) const {
  profiler::RecordFunction profiler("_weight_norm_differentiable_backward", Function::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_weight_norm_differentiable_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_w", grad_w);
    jit::tracer::addInputs(node, "saved_v", saved_v);
    jit::tracer::addInputs(node, "saved_g", saved_g);
    jit::tracer::addInputs(node, "saved_norms", saved_norms);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = TypeDefault::_weight_norm_differentiable_backward(grad_w, saved_v, saved_g, saved_norms, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor VariableType::adaptive_avg_pool3d_backward(const Tensor & grad_output, const Tensor & self) const {
  profiler::RecordFunction profiler("adaptive_avg_pool3d_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<AdaptiveAvgPool3DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<AdaptiveAvgPool3DBackwardBackward>(new AdaptiveAvgPool3DBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_info = self;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::adaptive_avg_pool3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->adaptive_avg_pool3d_backward(grad_output_, self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::adaptive_max_pool1d(const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("adaptive_max_pool1d", Function::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::adaptive_max_pool1d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = TypeDefault::adaptive_max_pool1d(self, output_size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
std::tuple<Tensor &,Tensor &> VariableType::adaptive_max_pool2d_out(Tensor & output, Tensor & indices, const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("adaptive_max_pool2d_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("adaptive_max_pool2d");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("adaptive_max_pool2d");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::adaptive_max_pool2d");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("adaptive_max_pool2d_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->adaptive_max_pool2d_out(output_, indices_, self_, output_size);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(output, indices);
}
Tensor VariableType::addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("addbmm", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& batch1_ = unpack(batch1, "batch1", 1);
  auto& batch2_ = unpack(batch2, "batch2", 2);
  std::shared_ptr<AddbmmBackward> grad_fn;
  if (compute_requires_grad( self, batch1, batch2 )) {
    grad_fn = std::shared_ptr<AddbmmBackward>(new AddbmmBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, batch1, batch2 ));
    grad_fn->batch1_argsize_0 = batch1.size(0);
    grad_fn->batch1_argsize_1 = batch1.size(1);
    grad_fn->batch2_argsize_2 = batch2.size(2);
    grad_fn->batch2_ = SavedVariable(batch2, false);
    grad_fn->alpha = alpha;
    grad_fn->batch1_ = SavedVariable(batch1, false);
    grad_fn->beta = beta;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::addbmm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "batch1", batch1);
    jit::tracer::addInputs(node, "batch2", batch2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->addbmm(self_, batch1_, batch2_, beta, alpha));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::addbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("addbmm_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& batch1_ = unpack(batch1, "batch1", 1);
  auto& batch2_ = unpack(batch2, "batch2", 2);
  check_inplace(self);
  std::shared_ptr<AddbmmBackward> grad_fn;
  if (compute_requires_grad( self, batch1, batch2 )) {
    grad_fn = std::shared_ptr<AddbmmBackward>(new AddbmmBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, batch1, batch2 ));
    grad_fn->batch1_argsize_0 = batch1.size(0);
    grad_fn->batch1_argsize_1 = batch1.size(1);
    grad_fn->batch2_argsize_2 = batch2.size(2);
    grad_fn->batch2_ = SavedVariable(batch2, false);
    grad_fn->alpha = alpha;
    grad_fn->batch1_ = SavedVariable(batch1, false);
    grad_fn->beta = beta;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::addbmm");
    } else {
      op_name = jit::Symbol::fromQualString("aten::addbmm_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "batch1", batch1);
    jit::tracer::addInputs(node, "batch2", batch2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("addbmm_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->addbmm_(self_, batch1_, batch2_, beta, alpha);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::addcdiv_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
  profiler::RecordFunction profiler("addcdiv_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& tensor1_ = unpack(tensor1, "tensor1", 2);
  auto& tensor2_ = unpack(tensor2, "tensor2", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, tensor1, tensor2 )) {
    throw_error_out_requires_grad("addcdiv");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("addcdiv");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::addcdiv");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "tensor1", tensor1);
    jit::tracer::addInputs(node, "tensor2", tensor2);
    jit::tracer::addInputs(node, "value", value);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("addcdiv_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->addcdiv_out(result_, self_, tensor1_, tensor2_, value);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::addmv_out(Tensor & result, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("addmv_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& mat_ = unpack(mat, "mat", 2);
  auto& vec_ = unpack(vec, "vec", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, mat, vec )) {
    throw_error_out_requires_grad("addmv");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("addmv");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::addmv");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat", mat);
    jit::tracer::addInputs(node, "vec", vec);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("addmv_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->addmv_out(result_, self_, mat_, vec_, beta, alpha);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::all(const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("all", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::all");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::all(self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::all(const Tensor & self) const {
  profiler::RecordFunction profiler("all", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::all");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::all(self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::any(const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("any", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::any");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::any(self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::any(const Tensor & self) const {
  profiler::RecordFunction profiler("any", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::any");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::any(self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::argmin(const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("argmin", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::argmin");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::argmin(self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::argmin(const Tensor & self) const {
  profiler::RecordFunction profiler("argmin", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::argmin");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::argmin(self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::as_strided(const Tensor & self, IntList size, IntList stride) const {
  profiler::RecordFunction profiler("as_strided", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::as_strided");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "stride", stride);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::as_strided(self, size, stride);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::as_strided(const Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
  profiler::RecordFunction profiler("as_strided", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AsStridedBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AsStridedBackward>(new AsStridedBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_geometry = TensorGeometry(self);
    grad_fn->size = size.vec();
    grad_fn->stride = stride.vec();
    grad_fn->storage_offset = storage_offset;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::as_strided");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "storage_offset", storage_offset);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_view(self, baseType->as_strided(self_, size, stride, storage_offset), true);
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::as_strided_(Tensor & self, IntList size, IntList stride) const {
  profiler::RecordFunction profiler("as_strided_", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::as_strided");
    } else {
      op_name = jit::Symbol::fromQualString("aten::as_strided_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "stride", stride);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("as_strided_", self);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::as_strided_(self, size, stride);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::as_strided_(Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
  profiler::RecordFunction profiler("as_strided_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<AsStridedBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AsStridedBackward>(new AsStridedBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_geometry = TensorGeometry(self);
    grad_fn->size = size.vec();
    grad_fn->stride = stride.vec();
    grad_fn->storage_offset = storage_offset;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::as_strided");
    } else {
      op_name = jit::Symbol::fromQualString("aten::as_strided_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "storage_offset", storage_offset);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("as_strided_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->as_strided_(self_, size, stride, storage_offset);
  increment_version(self);
  set_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps, bool cudnn_enabled) const {
  profiler::RecordFunction profiler("batch_norm", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::batch_norm");
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
    jit::tracer::addInputs(node, "cudnn_enabled", cudnn_enabled);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::bilinear(const Tensor & input1, const Tensor & input2, const Tensor & weight, const Tensor & bias) const {
  profiler::RecordFunction profiler("bilinear", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bilinear");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input1", input1);
    jit::tracer::addInputs(node, "input2", input2);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::bilinear(input1, input2, weight, bias);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::binary_cross_entropy_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) const {
  profiler::RecordFunction profiler("binary_cross_entropy_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  auto weight_ = unpack_opt(weight, "weight", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, target, weight )) {
    throw_error_out_requires_grad("binary_cross_entropy");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("binary_cross_entropy");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::binary_cross_entropy");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("binary_cross_entropy_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->binary_cross_entropy_out(output_, self_, target_, weight_, reduction);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::btrifact_with_info_out(Tensor & A_LU, Tensor & pivots, Tensor & info, const Tensor & self, bool pivot) const {
  profiler::RecordFunction profiler("btrifact_with_info_out", Function::peek_at_next_sequence_nr());
  auto& A_LU_ = unpack(A_LU, "A_LU", 0);
  auto& pivots_ = unpack(pivots, "pivots", 1);
  auto& info_ = unpack(info, "info", 2);
  auto& self_ = unpack(self, "self", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("btrifact_with_info");
  }
  if (compute_requires_grad( A_LU, pivots, info )) {
    throw_error_out_requires_grad("btrifact_with_info");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::btrifact_with_info");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "pivots", pivots);
    jit::tracer::addInputs(node, "info", info);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "pivot", pivot);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "A_LU", A_LU);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("btrifact_with_info_out", A_LU);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->btrifact_with_info_out(A_LU_, pivots_, info_, self_, pivot);
  increment_version(A_LU);
  increment_version(pivots);
  increment_version(info);
  rebase_history(flatten_tensor_args( A_LU, pivots, info ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, A_LU);
    jit::tracer::addOutput(node, pivots);
    jit::tracer::addOutput(node, info);
  }
  return std::forward_as_tuple(A_LU, pivots, info);
}
Tensor VariableType::ceil(const Tensor & self) const {
  profiler::RecordFunction profiler("ceil", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<CeilBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<CeilBackward>(new CeilBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ceil");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->ceil(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::ceil_(Tensor & self) const {
  profiler::RecordFunction profiler("ceil_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<CeilBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<CeilBackward>(new CeilBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::ceil");
    } else {
      op_name = jit::Symbol::fromQualString("aten::ceil_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ceil_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->ceil_(self_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::clamp_min_out(Tensor & result, const Tensor & self, Scalar min) const {
  profiler::RecordFunction profiler("clamp_min_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("clamp_min");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("clamp_min");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::clamp_min");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min", min);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("clamp_min_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->clamp_min_out(result_, self_, min);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::constant_pad_nd(const Tensor & self, IntList pad, Scalar value) const {
  profiler::RecordFunction profiler("constant_pad_nd", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ConstantPadNdBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ConstantPadNdBackward>(new ConstantPadNdBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->pad = pad.vec();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::constant_pad_nd");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "pad", pad);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->constant_pad_nd(self_, pad, value));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::conv2d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, int64_t groups) const {
  profiler::RecordFunction profiler("conv2d", Function::peek_at_next_sequence_nr());
  auto result = TypeDefault::conv2d(input, weight, bias, stride, padding, dilation, groups);
  return result;
}
Tensor VariableType::conv_transpose1d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, int64_t groups, IntList dilation) const {
  profiler::RecordFunction profiler("conv_transpose1d", Function::peek_at_next_sequence_nr());
  auto result = TypeDefault::conv_transpose1d(input, weight, bias, stride, padding, output_padding, groups, dilation);
  return result;
}
Tensor & VariableType::copy_sparse_to_sparse_(Tensor & self, const Tensor & src, bool non_blocking) const {
  profiler::RecordFunction profiler("copy_sparse_to_sparse_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& src_ = unpack(src, "src", 1);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, src )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("copy_sparse_to_sparse_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, src ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::copy_sparse_to_sparse");
    } else {
      op_name = jit::Symbol::fromQualString("aten::copy_sparse_to_sparse_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "src", src);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("copy_sparse_to_sparse_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->copy_sparse_to_sparse_(self_, src_, non_blocking);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::cosh_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("cosh_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("cosh");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("cosh");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cosh");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cosh_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->cosh_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::cross_out(Tensor & result, const Tensor & self, const Tensor & other, int64_t dim) const {
  profiler::RecordFunction profiler("cross_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("cross");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("cross");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cross");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "dim", dim);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cross_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->cross_out(result_, self_, other_, dim);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::ctc_loss(const Tensor & log_probs, const Tensor & targets, IntList input_lengths, IntList target_lengths, int64_t blank, int64_t reduction) const {
  profiler::RecordFunction profiler("ctc_loss", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ctc_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "log_probs", log_probs);
    jit::tracer::addInputs(node, "targets", targets);
    jit::tracer::addInputs(node, "input_lengths", input_lengths);
    jit::tracer::addInputs(node, "target_lengths", target_lengths);
    jit::tracer::addInputs(node, "blank", blank);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::ctc_loss(const Tensor & log_probs, const Tensor & targets, const Tensor & input_lengths, const Tensor & target_lengths, int64_t blank, int64_t reduction) const {
  profiler::RecordFunction profiler("ctc_loss", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ctc_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "log_probs", log_probs);
    jit::tracer::addInputs(node, "targets", targets);
    jit::tracer::addInputs(node, "input_lengths", input_lengths);
    jit::tracer::addInputs(node, "target_lengths", target_lengths);
    jit::tracer::addInputs(node, "blank", blank);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::cudnn_affine_grid_generator_backward(const Tensor & grad, int64_t N, int64_t C, int64_t H, int64_t W) const {
  profiler::RecordFunction profiler("cudnn_affine_grid_generator_backward", Function::peek_at_next_sequence_nr());
  auto& grad_ = unpack(grad, "grad", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("cudnn_affine_grid_generator_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cudnn_affine_grid_generator_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "N", N);
    jit::tracer::addInputs(node, "C", C);
    jit::tracer::addInputs(node, "H", H);
    jit::tracer::addInputs(node, "W", W);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto grad_theta = as_variable(baseType->cudnn_affine_grid_generator_backward(grad_, N, C, H, W));
  set_history(flatten_tensor_args( grad_theta ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_theta);
  }
  return grad_theta;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::cudnn_convolution_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask) const {
  profiler::RecordFunction profiler("cudnn_convolution_backward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  std::shared_ptr<CudnnConvolutionBackwardBackward> grad_fn;
  if (compute_requires_grad( self, grad_output, weight )) {
    grad_fn = std::shared_ptr<CudnnConvolutionBackwardBackward>(new CudnnConvolutionBackwardBackward(), deleteFunction);
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
    op_name = jit::Symbol::fromQualString("aten::cudnn_convolution_backward");
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
  std::tie(result0, result1, result2) = as_variable(baseType->cudnn_convolution_backward(self_, grad_output_, weight_, padding, stride, dilation, groups, benchmark, deterministic, output_mask));
  set_history(flatten_tensor_args( result0, result1, result2 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor VariableType::cudnn_convolution_backward_bias(const Tensor & grad_output) const {
  profiler::RecordFunction profiler("cudnn_convolution_backward_bias", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("cudnn_convolution_backward_bias"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cudnn_convolution_backward_bias");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->cudnn_convolution_backward_bias(grad_output_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::cudnn_convolution_backward_input(IntList self_size, const Tensor & grad_output, const Tensor & weight, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic) const {
  profiler::RecordFunction profiler("cudnn_convolution_backward_input", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, weight )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("cudnn_convolution_backward_input"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, weight ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cudnn_convolution_backward_input");
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
  auto result = as_variable(baseType->cudnn_convolution_backward_input(self_size, grad_output_, weight_, padding, stride, dilation, groups, benchmark, deterministic));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::cudnn_convolution_transpose(const Tensor & self, const Tensor & weight, const Tensor & bias, IntList padding, IntList output_padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic) const {
  profiler::RecordFunction profiler("cudnn_convolution_transpose", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 2);
  std::shared_ptr<CudnnConvolutionTransposeBackward> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::shared_ptr<CudnnConvolutionTransposeBackward>(new CudnnConvolutionTransposeBackward(), deleteFunction);
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
    op_name = jit::Symbol::fromQualString("aten::cudnn_convolution_transpose");
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
  auto result = as_variable(baseType->cudnn_convolution_transpose(self_, weight_, bias_, padding, output_padding, stride, dilation, groups, benchmark, deterministic));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::diagflat(const Tensor & self, int64_t offset) const {
  profiler::RecordFunction profiler("diagflat", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::diagflat");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "offset", offset);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::diagflat(self, offset);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::div(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("div", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  std::shared_ptr<DivBackward0> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<DivBackward0>(new DivBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->other_ = SavedVariable(other, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::div");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->div(self_, other_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::div(const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("div", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<DivBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<DivBackward1>(new DivBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->other = other;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::div");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->div(self_, other));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::div_(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("div_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<DivBackward0> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<DivBackward0>(new DivBackward0(), deleteFunction);
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
      op_name = jit::Symbol::fromQualString("aten::div");
    } else {
      op_name = jit::Symbol::fromQualString("aten::div_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("div_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->div_(self_, other_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::div_(Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("div_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<DivBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<DivBackward1>(new DivBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->other = other;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::div");
    } else {
      op_name = jit::Symbol::fromQualString("aten::div_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("div_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->div_(self_, other);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
std::tuple<Tensor &,Tensor &> VariableType::eig_out(Tensor & e, Tensor & v, const Tensor & self, bool eigenvectors) const {
  profiler::RecordFunction profiler("eig_out", Function::peek_at_next_sequence_nr());
  auto& e_ = unpack(e, "e", 0);
  auto& v_ = unpack(v, "v", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("eig");
  }
  if (compute_requires_grad( e, v )) {
    throw_error_out_requires_grad("eig");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::eig");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "v", v);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "eigenvectors", eigenvectors);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "e", e);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("eig_out", e);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->eig_out(e_, v_, self_, eigenvectors);
  increment_version(e);
  increment_version(v);
  rebase_history(flatten_tensor_args( e, v ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, e);
    jit::tracer::addOutput(node, v);
  }
  return std::forward_as_tuple(e, v);
}
Tensor VariableType::embedding_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) const {
  profiler::RecordFunction profiler("embedding_backward", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::embedding_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "num_weights", num_weights);
    jit::tracer::addInputs(node, "padding_idx", padding_idx);
    jit::tracer::addInputs(node, "scale_grad_by_freq", scale_grad_by_freq);
    jit::tracer::addInputs(node, "sparse", sparse);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::embedding_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::empty_out(Tensor & result, IntList size) const {
  profiler::RecordFunction profiler("empty_out", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::empty");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("empty_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::empty_out(result, size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::eq_out(Tensor & result, const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("eq_out", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::eq");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("eq_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::eq_out(result, self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::eq_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("eq_out", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::eq");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("eq_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::eq_out(result, self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::exp(const Tensor & self) const {
  profiler::RecordFunction profiler("exp", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ExpBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ExpBackward>(new ExpBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::exp");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->exp(self_));
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
Tensor & VariableType::exp_(Tensor & self) const {
  profiler::RecordFunction profiler("exp_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<ExpBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ExpBackward>(new ExpBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::exp");
    } else {
      op_name = jit::Symbol::fromQualString("aten::exp_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("exp_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->exp_(self_);
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
Tensor & VariableType::fill_(Tensor & self, Scalar value) const {
  profiler::RecordFunction profiler("fill_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<FillBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<FillBackward0>(new FillBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::full_like");
    } else {
      op_name = jit::Symbol::fromQualString("aten::fill_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fill_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->fill_(self_, value);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::fill_(Tensor & self, const Tensor & value) const {
  profiler::RecordFunction profiler("fill_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& value_ = unpack(value, "value", 1);
  check_inplace(self);
  std::shared_ptr<FillBackward1> grad_fn;
  if (compute_requires_grad( self, value )) {
    grad_fn = std::shared_ptr<FillBackward1>(new FillBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, value ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::full_like");
    } else {
      op_name = jit::Symbol::fromQualString("aten::fill_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fill_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->fill_(self_, value_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::fractional_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & indices) const {
  profiler::RecordFunction profiler("fractional_max_pool2d_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 4);
  check_no_requires_grad(indices, "indices");
  std::shared_ptr<FractionalMaxPool2DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<FractionalMaxPool2DBackwardBackward>(new FractionalMaxPool2DBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->indices_ = SavedVariable(indices, false);
    grad_fn->self_info = self;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fractional_max_pool2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "indices", indices);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->fractional_max_pool2d_backward(grad_output_, self_, kernel_size, output_size, indices_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::ge_out(Tensor & result, const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("ge_out", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ge");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ge_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::ge_out(result, self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::ge_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("ge_out", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ge");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ge_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::ge_out(result, self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::geqrf(const Tensor & self) const {
  profiler::RecordFunction profiler("geqrf", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<GeqrfBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<GeqrfBackward>(new GeqrfBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::geqrf");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = as_variable(baseType->geqrf(self_));
  set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor VariableType::ger(const Tensor & self, const Tensor & vec2) const {
  profiler::RecordFunction profiler("ger", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& vec2_ = unpack(vec2, "vec2", 1);
  std::shared_ptr<GerBackward> grad_fn;
  if (compute_requires_grad( self, vec2 )) {
    grad_fn = std::shared_ptr<GerBackward>(new GerBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, vec2 ));
    grad_fn->vec2_ = SavedVariable(vec2, false);
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ger");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "vec2", vec2);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->ger(self_, vec2_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::glu(const Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("glu", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<GluBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<GluBackward>(new GluBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::glu");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->glu(self_, dim));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::glu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("glu_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("glu_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("glu_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::glu_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("glu_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->glu_backward_out(grad_input_, grad_output_, self_, dim);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::grid_sampler(const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode) const {
  profiler::RecordFunction profiler("grid_sampler", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::grid_sampler");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "grid", grid);
    jit::tracer::addInputs(node, "interpolation_mode", interpolation_mode);
    jit::tracer::addInputs(node, "padding_mode", padding_mode);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::grid_sampler(input, grid, interpolation_mode, padding_mode);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::grid_sampler_2d(const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode) const {
  profiler::RecordFunction profiler("grid_sampler_2d", Function::peek_at_next_sequence_nr());
  auto& input_ = unpack(input, "input", 0);
  auto& grid_ = unpack(grid, "grid", 1);
  std::shared_ptr<GridSampler2DBackward> grad_fn;
  if (compute_requires_grad( input, grid )) {
    grad_fn = std::shared_ptr<GridSampler2DBackward>(new GridSampler2DBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( input, grid ));
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->grid_ = SavedVariable(grid, false);
    grad_fn->interpolation_mode = interpolation_mode;
    grad_fn->padding_mode = padding_mode;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::grid_sampler_2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "grid", grid);
    jit::tracer::addInputs(node, "interpolation_mode", interpolation_mode);
    jit::tracer::addInputs(node, "padding_mode", padding_mode);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->grid_sampler_2d(input_, grid_, interpolation_mode, padding_mode));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::gt_out(Tensor & result, const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("gt_out", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::gt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("gt_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::gt_out(result, self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::gt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("gt_out", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::gt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("gt_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::gt_out(result, self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::hardtanh_backward(const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val) const {
  profiler::RecordFunction profiler("hardtanh_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<HardtanhBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<HardtanhBackwardBackward>(new HardtanhBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->min_val = min_val;
    grad_fn->max_val = max_val;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hardtanh_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min_val", min_val);
    jit::tracer::addInputs(node, "max_val", max_val);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->hardtanh_backward(grad_output_, self_, min_val, max_val));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::histc_out(Tensor & result, const Tensor & self, int64_t bins, Scalar min, Scalar max) const {
  profiler::RecordFunction profiler("histc_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("histc");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("histc");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::histc");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "bins", bins);
    jit::tracer::addInputs(node, "min", min);
    jit::tracer::addInputs(node, "max", max);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("histc_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->histc_out(result_, self_, bins, min, max);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::hspmm_out(Tensor & result, const Tensor & mat1, const Tensor & mat2) const {
  profiler::RecordFunction profiler("hspmm_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& mat1_ = unpack(mat1, "mat1", 1);
  auto& mat2_ = unpack(mat2, "mat2", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( mat1, mat2 )) {
    throw_error_out_requires_grad("hspmm");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("hspmm");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hspmm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mat1", mat1);
    jit::tracer::addInputs(node, "mat2", mat2);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("hspmm_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->hspmm_out(result_, mat1_, mat2_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::index_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
  profiler::RecordFunction profiler("index_add_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  auto& source_ = unpack(source, "source", 3);
  check_inplace(self);
  check_no_requires_grad(index, "index");
  std::shared_ptr<IndexAddBackward> grad_fn;
  if (compute_requires_grad( self, source )) {
    grad_fn = std::shared_ptr<IndexAddBackward>(new IndexAddBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, source ));
    grad_fn->dim = dim;
    grad_fn->index_ = SavedVariable(index, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::index_add");
    } else {
      op_name = jit::Symbol::fromQualString("aten::index_add_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "source", source);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("index_add_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->index_add_(self_, dim, index_, source_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::inverse_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("inverse_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("inverse");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("inverse");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::inverse");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("inverse_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->inverse_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::kl_div(const Tensor & self, const Tensor & target, int64_t reduction) const {
  profiler::RecordFunction profiler("kl_div", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  std::shared_ptr<KlDivBackward> grad_fn;
  if (compute_requires_grad( self, target )) {
    grad_fn = std::shared_ptr<KlDivBackward>(new KlDivBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, target ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->reduction = reduction;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::kl_div");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->kl_div(self_, target_, reduction));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::kthvalue(const Tensor & self, int64_t k, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("kthvalue", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<KthvalueBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<KthvalueBackward>(new KthvalueBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->dim = dim;
    grad_fn->keepdim = keepdim;
  }
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::kthvalue");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "k", k);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = as_variable(baseType->kthvalue(self_, k, dim, keepdim));
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
Tensor & VariableType::le_out(Tensor & result, const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("le_out", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::le");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("le_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::le_out(result, self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::le_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("le_out", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::le");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("le_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::le_out(result, self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::leaky_relu_out(Tensor & output, const Tensor & self, Scalar negative_slope) const {
  profiler::RecordFunction profiler("leaky_relu_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("leaky_relu");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("leaky_relu");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::leaky_relu");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "negative_slope", negative_slope);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("leaky_relu_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->leaky_relu_out(output_, self_, negative_slope);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::lgamma(const Tensor & self) const {
  profiler::RecordFunction profiler("lgamma", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<LgammaBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<LgammaBackward>(new LgammaBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::lgamma");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->lgamma(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::lgamma_(Tensor & self) const {
  profiler::RecordFunction profiler("lgamma_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<LgammaBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<LgammaBackward>(new LgammaBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::lgamma");
    } else {
      op_name = jit::Symbol::fromQualString("aten::lgamma_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("lgamma_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->lgamma_(self_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::log10_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("log10_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("log10");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("log10");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::log10");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("log10_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->log10_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::log1p(const Tensor & self) const {
  profiler::RecordFunction profiler("log1p", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<Log1PBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<Log1PBackward>(new Log1PBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::log1p");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->log1p(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::log1p_(Tensor & self) const {
  profiler::RecordFunction profiler("log1p_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<Log1PBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<Log1PBackward>(new Log1PBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::log1p");
    } else {
      op_name = jit::Symbol::fromQualString("aten::log1p_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("log1p_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->log1p_(self_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::lstm(const Tensor & input, TensorList hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) const {
  profiler::RecordFunction profiler("lstm", Function::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  Tensor result2;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::lstm");
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1, result2) = TypeDefault::lstm(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
std::tuple<Tensor,Tensor,Tensor> VariableType::lstm(const Tensor & data, const Tensor & batch_sizes, TensorList hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) const {
  profiler::RecordFunction profiler("lstm", Function::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  Tensor result2;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::lstm");
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1, result2) = TypeDefault::lstm(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor & VariableType::lt_out(Tensor & result, const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("lt_out", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::lt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("lt_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::lt_out(result, self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::lt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("lt_out", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::lt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("lt_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::lt_out(result, self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::margin_ranking_loss(const Tensor & input1, const Tensor & input2, const Tensor & target, double margin, int64_t reduction) const {
  profiler::RecordFunction profiler("margin_ranking_loss", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::margin_ranking_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input1", input1);
    jit::tracer::addInputs(node, "input2", input2);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "margin", margin);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::margin_ranking_loss(input1, input2, target, margin, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::masked_select_out(Tensor & result, const Tensor & self, const Tensor & mask) const {
  profiler::RecordFunction profiler("masked_select_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& mask_ = unpack(mask, "mask", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, mask )) {
    throw_error_out_requires_grad("masked_select");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("masked_select");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::masked_select");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("masked_select_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->masked_select_out(result_, self_, mask_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::matrix_power(const Tensor & self, int64_t n) const {
  profiler::RecordFunction profiler("matrix_power", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::matrix_power");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::matrix_power(self, n);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::max_pool2d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
  profiler::RecordFunction profiler("max_pool2d", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max_pool2d");
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
  auto result = TypeDefault::max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::max_pool2d_with_indices_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const {
  profiler::RecordFunction profiler("max_pool2d_with_indices_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 7);
  check_no_requires_grad(indices, "indices");
  std::shared_ptr<MaxPool2DWithIndicesBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<MaxPool2DWithIndicesBackwardBackward>(new MaxPool2DWithIndicesBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->indices_ = SavedVariable(indices, false);
    grad_fn->self_info = self;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max_pool2d_with_indices_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    jit::tracer::addInputs(node, "indices", indices);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->max_pool2d_with_indices_backward(grad_output_, self_, kernel_size, stride, padding, dilation, ceil_mode, indices_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::max_pool3d_with_indices(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
  profiler::RecordFunction profiler("max_pool3d_with_indices", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<MaxPool3DWithIndicesBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MaxPool3DWithIndicesBackward>(new MaxPool3DWithIndicesBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->dilation = dilation.vec();
    grad_fn->ceil_mode = ceil_mode;
  }
  Tensor output;
  Tensor indices;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max_pool3d_with_indices");
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
  std::tie(output, indices) = as_variable(baseType->max_pool3d_with_indices(self_, kernel_size, stride, padding, dilation, ceil_mode));
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
Tensor & VariableType::max_pool3d_with_indices_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const {
  profiler::RecordFunction profiler("max_pool3d_with_indices_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& indices_ = unpack(indices, "indices", 8);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, indices )) {
    throw_error_out_requires_grad("max_pool3d_with_indices_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("max_pool3d_with_indices_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max_pool3d_with_indices_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    jit::tracer::addInputs(node, "indices", indices);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("max_pool3d_with_indices_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->max_pool3d_with_indices_backward_out(grad_input_, grad_output_, self_, kernel_size, stride, padding, dilation, ceil_mode, indices_);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
std::tuple<Tensor &,Tensor &> VariableType::min_out(Tensor & min, Tensor & min_indices, const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("min_out", Function::peek_at_next_sequence_nr());
  auto& min_ = unpack(min, "min", 0);
  auto& min_indices_ = unpack(min_indices, "min_indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("min");
  }
  if (compute_requires_grad( min )) {
    throw_error_out_requires_grad("min");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::min");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "min_indices", min_indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "min", min);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("min_out", min);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->min_out(min_, min_indices_, self_, dim, keepdim);
  increment_version(min);
  rebase_history(flatten_tensor_args( min ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, min);
    jit::tracer::addOutput(node, min_indices);
  }
  return std::forward_as_tuple(min, min_indices);
}
Tensor & VariableType::min_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("min_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("min");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("min");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::min");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("min_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->min_out(result_, self_, other_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::miopen_convolution_backward_weight(IntList weight_size, const Tensor & grad_output, const Tensor & self, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic) const {
  profiler::RecordFunction profiler("miopen_convolution_backward_weight", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("miopen_convolution_backward_weight"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::miopen_convolution_backward_weight");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "weight_size", weight_size);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "benchmark", benchmark);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->miopen_convolution_backward_weight(weight_size, grad_output_, self_, padding, stride, dilation, groups, benchmark, deterministic));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::mm(const Tensor & self, const Tensor & mat2) const {
  profiler::RecordFunction profiler("mm", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& mat2_ = unpack(mat2, "mat2", 1);
  std::shared_ptr<MmBackward> grad_fn;
  if (compute_requires_grad( self, mat2 )) {
    grad_fn = std::shared_ptr<MmBackward>(new MmBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, mat2 ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->mat2_sizes = mat2.sizes().vec();
    grad_fn->mat2_ = SavedVariable(mat2, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat2", mat2);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->mm(self_, mat2_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::mse_loss_out(Tensor & output, const Tensor & self, const Tensor & target, int64_t reduction) const {
  profiler::RecordFunction profiler("mse_loss_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, target )) {
    throw_error_out_requires_grad("mse_loss");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("mse_loss");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mse_loss");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("mse_loss_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->mse_loss_out(output_, self_, target_, reduction);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor & VariableType::mul_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("mul_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("mul");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("mul");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mul");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mul_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->mul_out(result_, self_, other_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::multi_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) const {
  profiler::RecordFunction profiler("multi_margin_loss_backward", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::multi_margin_loss_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "margin", margin);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::multi_margin_loss_backward(grad_output, self, target, p, margin, weight, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::multilabel_margin_loss_out(Tensor & output, const Tensor & self, const Tensor & target, int64_t reduction) const {
  profiler::RecordFunction profiler("multilabel_margin_loss_out", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::multilabel_margin_loss");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("multilabel_margin_loss_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::multilabel_margin_loss_out(output, self, target, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::narrow_copy(const Tensor & self, int64_t dim, int64_t start, int64_t length) const {
  profiler::RecordFunction profiler("narrow_copy", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("narrow_copy"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::narrow_copy");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "length", length);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->narrow_copy(self_, dim, start, length));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::native_pow_out(Tensor & result, const Tensor & self, Scalar exponent) const {
  profiler::RecordFunction profiler("native_pow_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("native_pow");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("native_pow");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::native_pow");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "exponent", exponent);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("native_pow_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->native_pow_out(result_, self_, exponent);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::neg_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("neg_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("neg");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("neg");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::neg");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("neg_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->neg_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::nonzero_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("nonzero_out", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::nonzero");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("nonzero_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::nonzero_out(result, self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::norm_out(Tensor & result, const Tensor & self, Scalar p, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("norm_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("norm");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("norm");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("norm_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->norm_out(result_, self_, p, dim, keepdim);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::normal_out(Tensor & output, const Tensor & mean, double std, Generator * generator) const {
  profiler::RecordFunction profiler("normal_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& mean_ = unpack(mean, "mean", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( mean )) {
    throw_error_out_requires_grad("normal");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("normal");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::normal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("normal_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->normal_out(output_, mean_, std, generator);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor & VariableType::normal_out(Tensor & output, double mean, const Tensor & std, Generator * generator) const {
  profiler::RecordFunction profiler("normal_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& std_ = unpack(std, "std", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( std )) {
    throw_error_out_requires_grad("normal");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("normal");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::normal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("normal_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->normal_out(output_, mean, std_, generator);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor & VariableType::normal_out(Tensor & output, const Tensor & mean, const Tensor & std, Generator * generator) const {
  profiler::RecordFunction profiler("normal_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& mean_ = unpack(mean, "mean", 1);
  auto& std_ = unpack(std, "std", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( mean, std )) {
    throw_error_out_requires_grad("normal");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("normal");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::normal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("normal_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->normal_out(output_, mean_, std_, generator);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::ones_like(const Tensor & self) const {
  profiler::RecordFunction profiler("ones_like", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ones_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::ones_like(self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::pdist(const Tensor & self, double p) const {
  profiler::RecordFunction profiler("pdist", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::pdist");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::pdist(self, p);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::potri(const Tensor & self, bool upper) const {
  profiler::RecordFunction profiler("potri", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<PotriBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<PotriBackward>(new PotriBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::potri");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "upper", upper);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->potri(self_, upper));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::potrs(const Tensor & self, const Tensor & input2, bool upper) const {
  profiler::RecordFunction profiler("potrs", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& input2_ = unpack(input2, "input2", 1);
  std::shared_ptr<PotrsBackward> grad_fn;
  if (compute_requires_grad( self, input2 )) {
    grad_fn = std::shared_ptr<PotrsBackward>(new PotrsBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, input2 ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::potrs");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "input2", input2);
    jit::tracer::addInputs(node, "upper", upper);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->potrs(self_, input2_, upper));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::prelu_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight) const {
  profiler::RecordFunction profiler("prelu_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  std::shared_ptr<PreluBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self, weight )) {
    grad_fn = std::shared_ptr<PreluBackwardBackward>(new PreluBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, weight ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
  }
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::prelu_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = as_variable(baseType->prelu_backward(grad_output_, self_, weight_));
  set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
std::tuple<Tensor &,Tensor &> VariableType::pstrf_out(Tensor & u, Tensor & piv, const Tensor & self, bool upper, Scalar tol) const {
  profiler::RecordFunction profiler("pstrf_out", Function::peek_at_next_sequence_nr());
  auto& u_ = unpack(u, "u", 0);
  auto& piv_ = unpack(piv, "piv", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("pstrf");
  }
  if (compute_requires_grad( u, piv )) {
    throw_error_out_requires_grad("pstrf");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::pstrf");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "piv", piv);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "tol", tol);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "u", u);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("pstrf_out", u);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->pstrf_out(u_, piv_, self_, upper, tol);
  increment_version(u);
  increment_version(piv);
  rebase_history(flatten_tensor_args( u, piv ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, u);
    jit::tracer::addOutput(node, piv);
  }
  return std::forward_as_tuple(u, piv);
}
Tensor VariableType::randint_like(const Tensor & self, int64_t high) const {
  profiler::RecordFunction profiler("randint_like", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randint_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "high", high);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::randint_like(self, high);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::randint_like(const Tensor & self, int64_t low, int64_t high) const {
  profiler::RecordFunction profiler("randint_like", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randint_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "low", low);
    jit::tracer::addInputs(node, "high", high);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::randint_like(self, low, high);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::renorm_out(Tensor & result, const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const {
  profiler::RecordFunction profiler("renorm_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("renorm");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("renorm");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::renorm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "maxnorm", maxnorm);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("renorm_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->renorm_out(result_, self_, p, dim, maxnorm);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::replication_pad2d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("replication_pad2d_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<ReplicationPad2DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<ReplicationPad2DBackwardBackward>(new ReplicationPad2DBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->padding = padding.vec();
    grad_fn->self_info = self;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::replication_pad2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->replication_pad2d_backward(grad_output_, self_, padding));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::replication_pad3d(const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("replication_pad3d", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ReplicationPad3DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ReplicationPad3DBackward>(new ReplicationPad3DBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->padding = padding.vec();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::replication_pad3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->replication_pad3d(self_, padding));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::replication_pad3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("replication_pad3d_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("replication_pad3d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("replication_pad3d_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::replication_pad3d_backward");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("replication_pad3d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->replication_pad3d_backward_out(grad_input_, grad_output_, self_, padding);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
  profiler::RecordFunction profiler("scatter_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  auto& src_ = unpack(src, "src", 3);
  check_inplace(self);
  check_no_requires_grad(index, "index");
  std::shared_ptr<ScatterBackward0> grad_fn;
  if (compute_requires_grad( self, src )) {
    grad_fn = std::shared_ptr<ScatterBackward0>(new ScatterBackward0(), deleteFunction);
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
      op_name = jit::Symbol::fromQualString("aten::scatter");
    } else {
      op_name = jit::Symbol::fromQualString("aten::scatter_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "src", src);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("scatter_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->scatter_(self_, dim, index_, src_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::scatter_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
  profiler::RecordFunction profiler("scatter_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  check_inplace(self);
  check_no_requires_grad(index, "index");
  std::shared_ptr<ScatterBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ScatterBackward1>(new ScatterBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->dim = dim;
    grad_fn->index_ = SavedVariable(index, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::scatter");
    } else {
      op_name = jit::Symbol::fromQualString("aten::scatter_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("scatter_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->scatter_(self_, dim, index_, value);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::sigmoid(const Tensor & self) const {
  profiler::RecordFunction profiler("sigmoid", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SigmoidBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SigmoidBackward>(new SigmoidBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sigmoid");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->sigmoid(self_));
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
Tensor & VariableType::sigmoid_(Tensor & self) const {
  profiler::RecordFunction profiler("sigmoid_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<SigmoidBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SigmoidBackward>(new SigmoidBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::sigmoid");
    } else {
      op_name = jit::Symbol::fromQualString("aten::sigmoid_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sigmoid_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->sigmoid_(self_);
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
Tensor & VariableType::sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output) const {
  profiler::RecordFunction profiler("sigmoid_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& output_ = unpack(output, "output", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, output )) {
    throw_error_out_requires_grad("sigmoid_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("sigmoid_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sigmoid_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output", output);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sigmoid_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->sigmoid_backward_out(grad_input_, grad_output_, output_);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::sign(const Tensor & self) const {
  profiler::RecordFunction profiler("sign", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SignBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SignBackward>(new SignBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sign");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->sign(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::sign_(Tensor & self) const {
  profiler::RecordFunction profiler("sign_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<SignBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SignBackward>(new SignBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::sign");
    } else {
      op_name = jit::Symbol::fromQualString("aten::sign_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sign_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->sign_(self_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::sinh_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("sinh_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("sinh");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("sinh");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sinh");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sinh_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->sinh_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
int64_t VariableType::size(const Tensor & self, int64_t dim) const {
  auto result = TypeDefault::size(self, dim);
  return result;
}
Tensor VariableType::slice(const Tensor & self, int64_t dim, int64_t start, int64_t end, int64_t step) const {
  profiler::RecordFunction profiler("slice", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SliceBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SliceBackward>(new SliceBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->dim = dim;
    grad_fn->start = start;
    grad_fn->end = end;
    grad_fn->step = step;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::slice");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "step", step);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_view(self, baseType->slice(self_, dim, start, end, step), true);
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::smm(const Tensor & self, const Tensor & mat2) const {
  profiler::RecordFunction profiler("smm", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::smm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat2", mat2);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::smm(self, mat2);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::smooth_l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) const {
  profiler::RecordFunction profiler("smooth_l1_loss_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  check_no_requires_grad(target, "target");
  std::shared_ptr<SmoothL1LossBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<SmoothL1LossBackwardBackward>(new SmoothL1LossBackwardBackward(), deleteFunction);
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
    op_name = jit::Symbol::fromQualString("aten::smooth_l1_loss_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->smooth_l1_loss_backward(grad_output_, self_, target_, reduction));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::softshrink_out(Tensor & output, const Tensor & self, Scalar lambd) const {
  profiler::RecordFunction profiler("softshrink_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("softshrink");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("softshrink");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::softshrink");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "lambd", lambd);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("softshrink_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->softshrink_out(output_, self_, lambd);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
int64_t VariableType::sparse_dim(const Tensor & self) const {
  profiler::RecordFunction profiler("sparse_dim", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto result = baseType->sparse_dim(self_);
  return result;
}
Tensor & VariableType::sparse_resize_and_clear_(Tensor & self, IntList size, int64_t sparse_dim, int64_t dense_dim) const {
  profiler::RecordFunction profiler("sparse_resize_and_clear_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("sparse_resize_and_clear_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::sparse_resize_and_clear");
    } else {
      op_name = jit::Symbol::fromQualString("aten::sparse_resize_and_clear_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "sparse_dim", sparse_dim);
    jit::tracer::addInputs(node, "dense_dim", dense_dim);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sparse_resize_and_clear_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->sparse_resize_and_clear_(self_, size, sparse_dim, dense_dim);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
std::vector<Tensor> VariableType::split_with_sizes(const Tensor & self, IntList split_sizes, int64_t dim) const {
  profiler::RecordFunction profiler("split_with_sizes", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SplitWithSizesBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SplitWithSizesBackward>(new SplitWithSizesBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->split_sizes = split_sizes.vec();
    grad_fn->dim = dim;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::split_with_sizes");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "split_sizes", split_sizes);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->split_with_sizes(self_, split_sizes, dim));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::sqrt(const Tensor & self) const {
  profiler::RecordFunction profiler("sqrt", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SqrtBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SqrtBackward>(new SqrtBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sqrt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->sqrt(self_));
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
Tensor & VariableType::sqrt_(Tensor & self) const {
  profiler::RecordFunction profiler("sqrt_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<SqrtBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SqrtBackward>(new SqrtBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::sqrt");
    } else {
      op_name = jit::Symbol::fromQualString("aten::sqrt_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sqrt_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->sqrt_(self_);
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
Tensor & VariableType::stack_out(Tensor & result, TensorList tensors, int64_t dim) const {
  profiler::RecordFunction profiler("stack_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto tensors_ = unpack(tensors, "tensors", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( tensors )) {
    throw_error_out_requires_grad("stack");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("stack");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::stack");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    jit::tracer::addInputs(node, "dim", dim);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("stack_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->stack_out(result_, tensors_, dim);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::svd(const Tensor & self, bool some, bool compute_uv) const {
  profiler::RecordFunction profiler("svd", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SvdBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SvdBackward>(new SvdBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->some = some;
    grad_fn->compute_uv = compute_uv;
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::svd");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "some", some);
    jit::tracer::addInputs(node, "compute_uv", compute_uv);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1, result2) = as_variable(baseType->svd(self_, some, compute_uv));
  set_history(flatten_tensor_args( result0, result1, result2 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  if (grad_fn) {
    grad_fn->result0_ = SavedVariable(result0, true);
    grad_fn->result1_ = SavedVariable(result1, true);
    grad_fn->result2_ = SavedVariable(result2, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
std::tuple<Tensor &,Tensor &> VariableType::symeig_out(Tensor & e, Tensor & V, const Tensor & self, bool eigenvectors, bool upper) const {
  profiler::RecordFunction profiler("symeig_out", Function::peek_at_next_sequence_nr());
  auto& e_ = unpack(e, "e", 0);
  auto& V_ = unpack(V, "V", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("symeig");
  }
  if (compute_requires_grad( e, V )) {
    throw_error_out_requires_grad("symeig");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::symeig");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "V", V);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "eigenvectors", eigenvectors);
    jit::tracer::addInputs(node, "upper", upper);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "e", e);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("symeig_out", e);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->symeig_out(e_, V_, self_, eigenvectors, upper);
  increment_version(e);
  increment_version(V);
  rebase_history(flatten_tensor_args( e, V ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, e);
    jit::tracer::addOutput(node, V);
  }
  return std::forward_as_tuple(e, V);
}
Tensor VariableType::tan(const Tensor & self) const {
  profiler::RecordFunction profiler("tan", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<TanBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<TanBackward>(new TanBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::tan");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->tan(self_));
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
Tensor & VariableType::tan_(Tensor & self) const {
  profiler::RecordFunction profiler("tan_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<TanBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<TanBackward>(new TanBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::tan");
    } else {
      op_name = jit::Symbol::fromQualString("aten::tan_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("tan_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->tan_(self_);
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
Tensor VariableType::tanh_backward(const Tensor & grad_output, const Tensor & output) const {
  profiler::RecordFunction profiler("tanh_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& output_ = unpack(output, "output", 1);
  std::shared_ptr<TanhBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, output )) {
    grad_fn = std::shared_ptr<TanhBackwardBackward>(new TanhBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, output ));
    grad_fn->output_ = SavedVariable(output, false);
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::tanh_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output", output);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->tanh_backward(grad_output_, output_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::thnn_col2im_backward(const Tensor & grad_output, IntList kernel_size, IntList dilation, IntList padding, IntList stride) const {
  profiler::RecordFunction profiler("thnn_col2im_backward", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_col2im_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::thnn_col2im_backward(grad_output, kernel_size, dilation, padding, stride);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::thnn_conv2d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
  profiler::RecordFunction profiler("thnn_conv2d", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_conv2d");
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
  auto result = TypeDefault::thnn_conv2d(self, weight, kernel_size, bias, stride, padding);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::thnn_conv2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input) const {
  profiler::RecordFunction profiler("thnn_conv2d_backward_out", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_conv2d_backward");
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
    jit::tracer::addInputs(node, "finput", finput);
    jit::tracer::addInputs(node, "fgrad_input", fgrad_input);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("thnn_conv2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::thnn_conv2d_backward_out(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
    jit::tracer::addOutput(node, grad_weight);
    jit::tracer::addOutput(node, grad_bias);
  }
  return std::forward_as_tuple(grad_input, grad_weight, grad_bias);
}
Tensor & VariableType::thnn_conv3d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
  profiler::RecordFunction profiler("thnn_conv3d_out", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_conv3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
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
    jit::tracer::ensureUniqueIfOutOfPlaced("thnn_conv3d_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::thnn_conv3d_out(output, self, weight, kernel_size, bias, stride, padding);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::thnn_conv_depthwise2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
  profiler::RecordFunction profiler("thnn_conv_depthwise2d_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 3);
  std::shared_ptr<ThnnConvDepthwise2DBackward> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::shared_ptr<ThnnConvDepthwise2DBackward>(new ThnnConvDepthwise2DBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->dilation = dilation.vec();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_conv_depthwise2d_forward");
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
  auto result = as_variable(baseType->thnn_conv_depthwise2d_forward(self_, weight_, kernel_size, bias_, stride, padding, dilation));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::thnn_conv_dilated3d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
  profiler::RecordFunction profiler("thnn_conv_dilated3d_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 3);
  std::shared_ptr<ThnnConvDilated3DBackward> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::shared_ptr<ThnnConvDilated3DBackward>(new ThnnConvDilated3DBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->dilation = dilation.vec();
  }
  Tensor output;
  Tensor columns;
  Tensor ones;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_conv_dilated3d_forward");
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
  std::tie(output, columns, ones) = as_variable(baseType->thnn_conv_dilated3d_forward(self_, weight_, kernel_size, bias_, stride, padding, dilation));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, columns);
    jit::tracer::addOutput(node, ones);
  }
  if (grad_fn) {
    grad_fn->columns_ = SavedVariable(columns, true);
    grad_fn->ones_ = SavedVariable(ones, true);
  }
  return std::make_tuple(std::move(output), std::move(columns), std::move(ones));
}
Tensor & VariableType::thnn_conv_transpose2d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
  profiler::RecordFunction profiler("thnn_conv_transpose2d_out", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_conv_transpose2d");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("thnn_conv_transpose2d_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::thnn_conv_transpose2d_out(output, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::thnn_im2col_backward(const Tensor & grad_output, IntList input_size, IntList kernel_size, IntList dilation, IntList padding, IntList stride) const {
  profiler::RecordFunction profiler("thnn_im2col_backward", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_im2col_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::thnn_im2col_backward(grad_output, input_size, kernel_size, dilation, padding, stride);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::threshold(const Tensor & self, Scalar threshold, Scalar value) const {
  profiler::RecordFunction profiler("threshold", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ThresholdBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ThresholdBackward0>(new ThresholdBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->threshold = threshold;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::threshold");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "threshold", threshold);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->threshold(self_, threshold, value));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::threshold_(Tensor & self, Scalar threshold, Scalar value) const {
  profiler::RecordFunction profiler("threshold_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<ThresholdBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ThresholdBackward1>(new ThresholdBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->threshold = threshold;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::threshold");
    } else {
      op_name = jit::Symbol::fromQualString("aten::threshold_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "threshold", threshold);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("threshold_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->threshold_(self_, threshold, value);
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
Tensor VariableType::tril(const Tensor & self, int64_t diagonal) const {
  profiler::RecordFunction profiler("tril", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<TrilBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<TrilBackward>(new TrilBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->diagonal = diagonal;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::tril");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "diagonal", diagonal);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->tril(self_, diagonal));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::tril_(Tensor & self, int64_t diagonal) const {
  profiler::RecordFunction profiler("tril_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<TrilBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<TrilBackward>(new TrilBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->diagonal = diagonal;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::tril");
    } else {
      op_name = jit::Symbol::fromQualString("aten::tril_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "diagonal", diagonal);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("tril_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->tril_(self_, diagonal);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::unfold(const Tensor & self, int64_t dimension, int64_t size, int64_t step) const {
  profiler::RecordFunction profiler("unfold", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<UnfoldBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<UnfoldBackward>(new UnfoldBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->dimension = dimension;
    grad_fn->size = size;
    grad_fn->step = step;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::unfold");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dimension", dimension);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "step", step);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_view(self, baseType->unfold(self_, dimension, size, step), true);
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::uniform_(Tensor & self, double from, double to, Generator * generator) const {
  profiler::RecordFunction profiler("uniform_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<UniformBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<UniformBackward>(new UniformBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::uniform");
    } else {
      op_name = jit::Symbol::fromQualString("aten::uniform_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "from", from);
    jit::tracer::addInputs(node, "to", to);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("uniform_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->uniform_(self_, from, to, generator);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::upsample_bicubic2d_backward(const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
  profiler::RecordFunction profiler("upsample_bicubic2d_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  std::shared_ptr<UpsampleBicubic2DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output )) {
    grad_fn = std::shared_ptr<UpsampleBicubic2DBackwardBackward>(new UpsampleBicubic2DBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
    grad_fn->output_size = output_size.vec();
    grad_fn->align_corners = align_corners;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_bicubic2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->upsample_bicubic2d_backward(grad_output_, output_size, input_size, align_corners));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::upsample_bilinear2d_out(Tensor & output, const Tensor & self, IntList output_size, bool align_corners) const {
  profiler::RecordFunction profiler("upsample_bilinear2d_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("upsample_bilinear2d");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("upsample_bilinear2d");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_bilinear2d");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_bilinear2d_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->upsample_bilinear2d_out(output_, self_, output_size, align_corners);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor & VariableType::upsample_nearest1d_out(Tensor & output, const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("upsample_nearest1d_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("upsample_nearest1d");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("upsample_nearest1d");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_nearest1d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_nearest1d_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->upsample_nearest1d_out(output_, self_, output_size);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::upsample_trilinear3d_backward(const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
  profiler::RecordFunction profiler("upsample_trilinear3d_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  std::shared_ptr<UpsampleTrilinear3DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output )) {
    grad_fn = std::shared_ptr<UpsampleTrilinear3DBackwardBackward>(new UpsampleTrilinear3DBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
    grad_fn->output_size = output_size.vec();
    grad_fn->align_corners = align_corners;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_trilinear3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->upsample_trilinear3d_backward(grad_output_, output_size, input_size, align_corners));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::var_out(Tensor & result, const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
  profiler::RecordFunction profiler("var_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("var");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("var");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::var");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("var_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->var_out(result_, self_, dim, unbiased, keepdim);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::view(const Tensor & self, IntList size) const {
  profiler::RecordFunction profiler("view", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ViewBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ViewBackward>(new ViewBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::view");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_view(self, baseType->view(self_, size), true);
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::zero_(Tensor & self) const {
  profiler::RecordFunction profiler("zero_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<ZeroBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ZeroBackward>(new ZeroBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::zeros_like");
    } else {
      op_name = jit::Symbol::fromQualString("aten::zero_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("zero_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->zero_(self_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}

}} // namespace torch::autograd
