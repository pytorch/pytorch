#include "torch/csrc/autograd/VariableTypeUtils.h"

// @generated from tools/autograd/templates/VariableType_2.cpp

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

Tensor VariableType::RoiPooling2d_backward(const Tensor & input, const Tensor & rois, int64_t pooledHeight, int64_t pooledWidth, double spatialScale, const Tensor & gradOutput, const Tensor & argmaxes) const {
  profiler::RecordFunction profiler("RoiPooling2d_backward", Function::peek_at_next_sequence_nr());
  auto& input_ = unpack(input, "input", 0);
  auto& rois_ = unpack(rois, "rois", 1);
  auto& gradOutput_ = unpack(gradOutput, "gradOutput", 5);
  auto& argmaxes_ = unpack(argmaxes, "argmaxes", 6);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( input, rois, gradOutput, argmaxes )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("RoiPooling2d_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( input, rois, gradOutput, argmaxes ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::RoiPooling2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "rois", rois);
    jit::tracer::addInputs(node, "pooledHeight", pooledHeight);
    jit::tracer::addInputs(node, "pooledWidth", pooledWidth);
    jit::tracer::addInputs(node, "spatialScale", spatialScale);
    jit::tracer::addInputs(node, "gradOutput", gradOutput);
    jit::tracer::addInputs(node, "argmaxes", argmaxes);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->RoiPooling2d_backward(input_, rois_, pooledHeight, pooledWidth, spatialScale, gradOutput_, argmaxes_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::__and__(const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("__and__", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__and__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::__and__(self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::__and__(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("__and__", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__and__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::__and__(self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::__iand__(Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("__iand__", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__iand__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::__iand__(self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::__iand__(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("__iand__", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__iand__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::__iand__(self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::_cast_Long(const Tensor & self, bool non_blocking) const {
  profiler::RecordFunction profiler("_cast_Long", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cast_Long");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::_cast_Long(self, non_blocking);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::_convolution_double_backward(const Tensor & ggI, const Tensor & ggW, const Tensor & ggb, const Tensor & gO, const Tensor & weight, const Tensor & self, IntList stride, IntList padding, IntList dilation, bool transposed, IntList output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled, std::array<bool,3> output_mask) const {
  profiler::RecordFunction profiler("_convolution_double_backward", Function::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  Tensor result2;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_convolution_double_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "ggI", ggI);
    jit::tracer::addInputs(node, "ggW", ggW);
    jit::tracer::addInputs(node, "ggb", ggb);
    jit::tracer::addInputs(node, "gO", gO);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "transposed", transposed);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "benchmark", benchmark);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    jit::tracer::addInputs(node, "cudnn_enabled", cudnn_enabled);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1, result2) = TypeDefault::_convolution_double_backward(ggI, ggW, ggb, gO, weight, self, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, output_mask);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
void VariableType::_cufft_clear_plan_cache() const {
  profiler::RecordFunction profiler("_cufft_clear_plan_cache", Function::peek_at_next_sequence_nr());
  TypeDefault::_cufft_clear_plan_cache();
}
int64_t VariableType::_cufft_get_plan_cache_max_size() const {
  profiler::RecordFunction profiler("_cufft_get_plan_cache_max_size", Function::peek_at_next_sequence_nr());
  auto result = TypeDefault::_cufft_get_plan_cache_max_size();
  return result;
}
int64_t VariableType::_dimI(const Tensor & self) const {
  profiler::RecordFunction profiler("_dimI", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto result = baseType->_dimI(self_);
  return result;
}
Tensor VariableType::_fft_with_size(const Tensor & self, int64_t signal_ndim, bool complex_input, bool complex_output, bool inverse, IntList checked_signal_sizes, bool normalized, bool onesided, IntList output_sizes) const {
  profiler::RecordFunction profiler("_fft_with_size", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<FftWithSizeBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<FftWithSizeBackward>(new FftWithSizeBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->signal_ndim = signal_ndim;
    grad_fn->complex_input = complex_input;
    grad_fn->complex_output = complex_output;
    grad_fn->inverse = inverse;
    grad_fn->checked_signal_sizes = checked_signal_sizes.vec();
    grad_fn->normalized = normalized;
    grad_fn->onesided = onesided;
    grad_fn->output_sizes = output_sizes.vec();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_fft_with_size");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "signal_ndim", signal_ndim);
    jit::tracer::addInputs(node, "complex_input", complex_input);
    jit::tracer::addInputs(node, "complex_output", complex_output);
    jit::tracer::addInputs(node, "inverse", inverse);
    jit::tracer::addInputs(node, "checked_signal_sizes", checked_signal_sizes);
    jit::tracer::addInputs(node, "normalized", normalized);
    jit::tracer::addInputs(node, "onesided", onesided);
    jit::tracer::addInputs(node, "output_sizes", output_sizes);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_fft_with_size(self_, signal_ndim, complex_input, complex_output, inverse, checked_signal_sizes, normalized, onesided, output_sizes));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::_gesv_helper(const Tensor & self, const Tensor & A) const {
  profiler::RecordFunction profiler("_gesv_helper", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& A_ = unpack(A, "A", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, A )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_gesv_helper"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, A ));
  }
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_gesv_helper");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "A", A);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = as_variable(baseType->_gesv_helper(self_, A_));
  set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor VariableType::_log_softmax(const Tensor & self, int64_t dim, bool half_to_float) const {
  profiler::RecordFunction profiler("_log_softmax", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<LogSoftmaxBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<LogSoftmaxBackward>(new LogSoftmaxBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_log_softmax");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "half_to_float", half_to_float);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_log_softmax(self_, dim, half_to_float));
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
int64_t VariableType::_nnz(const Tensor & self) const {
  profiler::RecordFunction profiler("_nnz", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto result = baseType->_nnz(self_);
  return result;
}
std::tuple<Tensor,Tensor> VariableType::_pack_padded_sequence(const Tensor & input, const Tensor & lengths, bool batch_first) const {
  profiler::RecordFunction profiler("_pack_padded_sequence", Function::peek_at_next_sequence_nr());
  auto& input_ = unpack(input, "input", 0);
  auto& lengths_ = unpack(lengths, "lengths", 1);
  check_no_requires_grad(lengths, "lengths");
  std::shared_ptr<PackPaddedSequenceBackward> grad_fn;
  if (compute_requires_grad( input )) {
    grad_fn = std::shared_ptr<PackPaddedSequenceBackward>(new PackPaddedSequenceBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( input ));
    grad_fn->input_sizes = input.sizes().vec();
    grad_fn->batch_first = batch_first;
  }
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_pack_padded_sequence");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "lengths", lengths);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = as_variable(baseType->_pack_padded_sequence(input_, lengths_, batch_first));
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
std::tuple<Tensor,Tensor> VariableType::_pad_packed_sequence(const Tensor & data, const Tensor & batch_sizes, bool batch_first, Scalar padding_value, int64_t total_length) const {
  profiler::RecordFunction profiler("_pad_packed_sequence", Function::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_pad_packed_sequence");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "data", data);
    jit::tracer::addInputs(node, "batch_sizes", batch_sizes);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    jit::tracer::addInputs(node, "padding_value", padding_value);
    jit::tracer::addInputs(node, "total_length", total_length);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = TypeDefault::_pad_packed_sequence(data, batch_sizes, batch_first, padding_value, total_length);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor VariableType::_reshape_from_tensor(const Tensor & self, const Tensor & shape) const {
  profiler::RecordFunction profiler("_reshape_from_tensor", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_reshape_from_tensor");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "shape", shape);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::_reshape_from_tensor(self, shape);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_sparse_dense_add_out(Tensor & result, const Tensor & self, SparseTensorRef other, Scalar alpha) const {
  profiler::RecordFunction profiler("_sparse_dense_add_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto other_ = unpack(other, "other", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, other.tref )) {
    throw_error_out_requires_grad("_sparse_dense_add");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_sparse_dense_add");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_dense_add");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("_sparse_dense_add_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_sparse_dense_add_out(result_, self_, other_, alpha);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_sparse_mm(const Tensor & sparse, const Tensor & dense) const {
  profiler::RecordFunction profiler("_sparse_mm", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_mm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "sparse", sparse);
    jit::tracer::addInputs(node, "dense", dense);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::_sparse_mm(sparse, dense);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_sparse_mul_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("_sparse_mul_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("_sparse_mul");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_sparse_mul");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_mul");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_sparse_mul_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_sparse_mul_out(result_, self_, other_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_sparse_mul_scalar_out(Tensor & result, const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("_sparse_mul_scalar_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_sparse_mul_scalar");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_sparse_mul_scalar");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_mul_scalar");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_sparse_mul_scalar_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_sparse_mul_scalar_out(result_, self_, other);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_acos(const Tensor & self) const {
  profiler::RecordFunction profiler("_th_acos", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_acos"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_acos");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_acos(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::s__th_addbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("_th_addbmm_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& batch1_ = unpack(batch1, "batch1", 2);
  auto& batch2_ = unpack(batch2, "batch2", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, batch1, batch2 )) {
    throw_error_out_requires_grad("_th_addbmm");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_addbmm");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_addbmm_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s__th_addbmm_out(result_, self_, batch1_, batch2_, beta, alpha);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_alias(const Tensor & self) const {
  profiler::RecordFunction profiler("_th_alias", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_alias"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_alias");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_alias(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_all_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("_th_all_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_all");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_all");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_all_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_all_out(result_, self_, dim, keepdim);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_and(const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("_th_and", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_and"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_and");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_and(self_, other));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::s__th_and(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("_th_and", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_and"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_and");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->s__th_and(self_, other_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_any_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("_th_any_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_any");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_any");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_any_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_any_out(result_, self_, dim, keepdim);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_arange(Scalar start, Scalar end, Scalar step) const {
  profiler::RecordFunction profiler("_th_arange", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_arange");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "step", step);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_arange(start, end, step));
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_arange(Scalar end) const {
  profiler::RecordFunction profiler("_th_arange", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_arange");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "end", end);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_arange(end));
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_asin(const Tensor & self) const {
  profiler::RecordFunction profiler("_th_asin", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_asin"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_asin");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_asin(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::_th_btrifact(const Tensor & self, bool pivot) const {
  profiler::RecordFunction profiler("_th_btrifact", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_btrifact"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor result;
  Tensor pivots;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_btrifact");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "pivot", pivot);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result, pivots) = as_variable(baseType->_th_btrifact(self_, pivot));
  set_history(flatten_tensor_args( result, pivots ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
    jit::tracer::addOutput(node, pivots);
  }
  return std::make_tuple(std::move(result), std::move(pivots));
}
Tensor VariableType::_th_cat(TensorList tensors, int64_t dim) const {
  profiler::RecordFunction profiler("_th_cat", Function::peek_at_next_sequence_nr());
  auto tensors_ = unpack(tensors, "tensors", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( tensors )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_cat"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( tensors ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_cat");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto self = as_variable(baseType->_th_cat(tensors_, dim));
  set_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::_th_cauchy_(Tensor & self, double median, double sigma, Generator * generator) const {
  profiler::RecordFunction profiler("_th_cauchy_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_cauchy_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_cauchy");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_cauchy_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "median", median);
    jit::tracer::addInputs(node, "sigma", sigma);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_cauchy_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_cauchy_(self_, median, sigma, generator);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::_th_ceil_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("_th_ceil_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_ceil");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_ceil");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_ceil_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_ceil_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_cumprod(const Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("_th_cumprod", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_cumprod"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_cumprod");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_cumprod(self_, dim));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_dot(const Tensor & self, const Tensor & tensor) const {
  profiler::RecordFunction profiler("_th_dot", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& tensor_ = unpack(tensor, "tensor", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, tensor )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_dot"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, tensor ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_dot");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "tensor", tensor);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_dot(self_, tensor_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_erf(const Tensor & self) const {
  profiler::RecordFunction profiler("_th_erf", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_erf"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_erf");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_erf(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_exp_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("_th_exp_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_exp");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_exp");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_exp_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_exp_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_fmod(const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("_th_fmod", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_fmod"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_fmod");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_fmod(self_, other));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::s__th_fmod(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("_th_fmod", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_fmod"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_fmod");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->s__th_fmod(self_, other_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_fmod_(Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("_th_fmod_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_fmod_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_fmod");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_fmod_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_fmod_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_fmod_(self_, other);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::s__th_fmod_(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("_th_fmod_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_fmod_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_fmod");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_fmod_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_fmod_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s__th_fmod_(self_, other_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::_th_frac(const Tensor & self) const {
  profiler::RecordFunction profiler("_th_frac", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_frac"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_frac");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_frac(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_frac_(Tensor & self) const {
  profiler::RecordFunction profiler("_th_frac_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_frac_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_frac");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_frac_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_frac_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_frac_(self_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
std::tuple<Tensor,Tensor> VariableType::_th_gels(const Tensor & self, const Tensor & A) const {
  profiler::RecordFunction profiler("_th_gels", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& A_ = unpack(A, "A", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, A )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_gels"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, A ));
  }
  Tensor res1;
  Tensor res2;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_gels");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "A", A);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(res1, res2) = as_variable(baseType->_th_gels(self_, A_));
  set_history(flatten_tensor_args( res1, res2 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, res1);
    jit::tracer::addOutput(node, res2);
  }
  return std::make_tuple(std::move(res1), std::move(res2));
}
std::tuple<Tensor &,Tensor &> VariableType::_th_geqrf_out(Tensor & res1, Tensor & res2, const Tensor & self) const {
  profiler::RecordFunction profiler("_th_geqrf_out", Function::peek_at_next_sequence_nr());
  auto& res1_ = unpack(res1, "res1", 0);
  auto& res2_ = unpack(res2, "res2", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_geqrf");
  }
  if (compute_requires_grad( res1, res2 )) {
    throw_error_out_requires_grad("_th_geqrf");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_geqrf");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "res2", res2);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "res1", res1);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_geqrf_out", res1);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_geqrf_out(res1_, res2_, self_);
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
Tensor & VariableType::_th_ger_out(Tensor & result, const Tensor & self, const Tensor & vec2) const {
  profiler::RecordFunction profiler("_th_ger_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& vec2_ = unpack(vec2, "vec2", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, vec2 )) {
    throw_error_out_requires_grad("_th_ger");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_ger");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_ger_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_ger_out(result_, self_, vec2_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_iand_(Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("_th_iand_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_iand_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_iand");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_iand_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_iand_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_iand_(self_, other);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::s__th_iand_(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("_th_iand_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_iand_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_iand");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_iand_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_iand_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s__th_iand_(self_, other_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
std::tuple<Tensor &,Tensor &> VariableType::_th_kthvalue_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("_th_kthvalue_out", Function::peek_at_next_sequence_nr());
  auto& values_ = unpack(values, "values", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_kthvalue");
  }
  if (compute_requires_grad( values )) {
    throw_error_out_requires_grad("_th_kthvalue");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_kthvalue");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "k", k);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "values", values);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_kthvalue_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_kthvalue_out(values_, indices_, self_, k, dim, keepdim);
  increment_version(values);
  rebase_history(flatten_tensor_args( values ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(values, indices);
}
Tensor & VariableType::_th_lgamma_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("_th_lgamma_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_lgamma");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_lgamma");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_lgamma_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_lgamma_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_log(const Tensor & self) const {
  profiler::RecordFunction profiler("_th_log", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_log"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_log");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_log(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_log1p_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("_th_log1p_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_log1p");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_log1p");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_log1p_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_log1p_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_log2(const Tensor & self) const {
  profiler::RecordFunction profiler("_th_log2", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_log2"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_log2");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_log2(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::s__th_masked_fill_(Tensor & self, const Tensor & mask, Scalar value) const {
  profiler::RecordFunction profiler("_th_masked_fill_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& mask_ = unpack(mask, "mask", 1);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_masked_fill_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_masked_fill");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_masked_fill_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_masked_fill_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s__th_masked_fill_(self_, mask_, value);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::s__th_masked_fill_(Tensor & self, const Tensor & mask, const Tensor & value) const {
  profiler::RecordFunction profiler("_th_masked_fill_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& mask_ = unpack(mask, "mask", 1);
  auto& value_ = unpack(value, "value", 2);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, value )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_masked_fill_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, value ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_masked_fill");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_masked_fill_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_masked_fill_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s__th_masked_fill_(self_, mask_, value_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::s__th_masked_scatter_(Tensor & self, const Tensor & mask, const Tensor & source) const {
  profiler::RecordFunction profiler("_th_masked_scatter_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& mask_ = unpack(mask, "mask", 1);
  auto& source_ = unpack(source, "source", 2);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, source )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_masked_scatter_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, source ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_masked_scatter");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_masked_scatter_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "source", source);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_masked_scatter_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s__th_masked_scatter_(self_, mask_, source_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::_th_median(const Tensor & self) const {
  profiler::RecordFunction profiler("_th_median", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_median"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_median");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_median(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::_th_median(const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("_th_median", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_median"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor values;
  Tensor indices;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_median");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(values, indices) = as_variable(baseType->_th_median(self_, dim, keepdim));
  set_history(flatten_tensor_args( values ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
Tensor & VariableType::_th_mm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const {
  profiler::RecordFunction profiler("_th_mm_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& mat2_ = unpack(mat2, "mat2", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, mat2 )) {
    throw_error_out_requires_grad("_th_mm");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_mm");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_mm_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_mm_out(result_, self_, mat2_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_mv(const Tensor & self, const Tensor & vec) const {
  profiler::RecordFunction profiler("_th_mv", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& vec_ = unpack(vec, "vec", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, vec )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_mv"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, vec ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_mv");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "vec", vec);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_mv(self_, vec_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_polygamma(int64_t n, const Tensor & self) const {
  profiler::RecordFunction profiler("_th_polygamma", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_polygamma"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_polygamma");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_polygamma(n, self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_polygamma_(Tensor & self, int64_t n) const {
  profiler::RecordFunction profiler("_th_polygamma_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_polygamma_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_polygamma");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_polygamma_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_polygamma_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_polygamma_(self_, n);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::_th_potrf_single(const Tensor & self, bool upper) const {
  profiler::RecordFunction profiler("_th_potrf_single", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_potrf_single"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_potrf_single");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "upper", upper);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto output = as_variable(baseType->_th_potrf_single(self_, upper));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor & VariableType::_th_potri_out(Tensor & output, const Tensor & self, bool upper) const {
  profiler::RecordFunction profiler("_th_potri_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_potri");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_th_potri");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_potri_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_potri_out(output_, self_, upper);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_th_pow(const Tensor & self, Scalar exponent) const {
  profiler::RecordFunction profiler("_th_pow", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_pow"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_pow");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "exponent", exponent);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_pow(self_, exponent));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::s__th_pow(const Tensor & self, const Tensor & exponent) const {
  profiler::RecordFunction profiler("_th_pow", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& exponent_ = unpack(exponent, "exponent", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, exponent )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_pow"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, exponent ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_pow");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "exponent", exponent);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->s__th_pow(self_, exponent_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_pow(Scalar self, const Tensor & exponent) const {
  profiler::RecordFunction profiler("_th_pow", Function::peek_at_next_sequence_nr());
  auto& exponent_ = unpack(exponent, "exponent", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( exponent )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_pow"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( exponent ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_pow");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "exponent", exponent);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_pow(self, exponent_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_pow_(Tensor & self, Scalar exponent) const {
  profiler::RecordFunction profiler("_th_pow_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_pow_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_pow");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_pow_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "exponent", exponent);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_pow_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_pow_(self_, exponent);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::s__th_pow_(Tensor & self, const Tensor & exponent) const {
  profiler::RecordFunction profiler("_th_pow_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& exponent_ = unpack(exponent, "exponent", 1);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, exponent )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_pow_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, exponent ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_pow");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_pow_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "exponent", exponent);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_pow_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s__th_pow_(self_, exponent_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
std::tuple<Tensor,Tensor> VariableType::_th_qr(const Tensor & self) const {
  profiler::RecordFunction profiler("_th_qr", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_qr"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor res1;
  Tensor res2;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_qr");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(res1, res2) = as_variable(baseType->_th_qr(self_));
  set_history(flatten_tensor_args( res1, res2 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, res1);
    jit::tracer::addOutput(node, res2);
  }
  return std::make_tuple(std::move(res1), std::move(res2));
}
Tensor VariableType::_th_round(const Tensor & self) const {
  profiler::RecordFunction profiler("_th_round", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_round"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_round");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_round(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_rsqrt(const Tensor & self) const {
  profiler::RecordFunction profiler("_th_rsqrt", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_rsqrt"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_rsqrt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_rsqrt(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_set_(Tensor & self, Storage source) const {
  profiler::RecordFunction profiler("_th_set_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_set_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  baseType->_th_set_(self_, source);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  return self;
}
Tensor & VariableType::_th_set_(Tensor & self, Storage source, int64_t storage_offset, IntList size, IntList stride) const {
  profiler::RecordFunction profiler("_th_set_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_set_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  baseType->_th_set_(self_, source, storage_offset, size, stride);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  return self;
}
Tensor & VariableType::_th_set_(Tensor & self, const Tensor & source) const {
  profiler::RecordFunction profiler("_th_set_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& source_ = unpack(source, "source", 1);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, source )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_set_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, source ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_set");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_set_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "source", source);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_set_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_set_(self_, source_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::_th_set_(Tensor & self) const {
  profiler::RecordFunction profiler("_th_set_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_set_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_set");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_set_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_set_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_set_(self_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::_th_sigmoid_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("_th_sigmoid_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_sigmoid");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_sigmoid");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_sigmoid_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_sigmoid_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_sign_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("_th_sign_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_sign");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_sign");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_sign_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_sign_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_sqrt_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("_th_sqrt_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_sqrt");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_sqrt");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_sqrt_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_sqrt_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::_th_svd_out(Tensor & res1, Tensor & res2, Tensor & res3, const Tensor & self, bool some, bool compute_uv) const {
  profiler::RecordFunction profiler("_th_svd_out", Function::peek_at_next_sequence_nr());
  auto& res1_ = unpack(res1, "res1", 0);
  auto& res2_ = unpack(res2, "res2", 1);
  auto& res3_ = unpack(res3, "res3", 2);
  auto& self_ = unpack(self, "self", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_svd");
  }
  if (compute_requires_grad( res1, res2, res3 )) {
    throw_error_out_requires_grad("_th_svd");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_svd");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "res2", res2);
    jit::tracer::addInputs(node, "res3", res3);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "some", some);
    jit::tracer::addInputs(node, "compute_uv", compute_uv);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "res1", res1);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_svd_out", res1);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_svd_out(res1_, res2_, res3_, self_, some, compute_uv);
  increment_version(res1);
  increment_version(res2);
  increment_version(res3);
  rebase_history(flatten_tensor_args( res1, res2, res3 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, res1);
    jit::tracer::addOutput(node, res2);
    jit::tracer::addOutput(node, res3);
  }
  return std::forward_as_tuple(res1, res2, res3);
}
Tensor & VariableType::_th_tan_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("_th_tan_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_tan");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_tan");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_tan_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_tan_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_tanh(const Tensor & self) const {
  profiler::RecordFunction profiler("_th_tanh", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_tanh"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_tanh");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_tanh(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_tensor(Storage storage, int64_t storageOffset, IntList size, IntList stride) const {
  profiler::RecordFunction profiler("_th_tensor", Function::peek_at_next_sequence_nr());
  auto result = as_variable(baseType->_th_tensor(storage, storageOffset, size, stride));
  return result;
}
Tensor VariableType::_th_tensor(IntList size, IntList stride) const {
  profiler::RecordFunction profiler("_th_tensor", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_tensor");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "stride", stride);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_tensor(size, stride));
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_trace(const Tensor & self) const {
  profiler::RecordFunction profiler("_th_trace", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_trace"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_trace");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_trace(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_tril_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
  profiler::RecordFunction profiler("_th_tril_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_tril");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_tril");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_tril_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_tril_out(result_, self_, diagonal);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_triu(const Tensor & self, int64_t diagonal) const {
  profiler::RecordFunction profiler("_th_triu", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_triu"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_triu");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "diagonal", diagonal);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_triu(self_, diagonal));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_triu_(Tensor & self, int64_t diagonal) const {
  profiler::RecordFunction profiler("_th_triu_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_triu_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_triu");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_triu_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "diagonal", diagonal);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_triu_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_triu_(self_, diagonal);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::_th_unfold_out(Tensor & result, const Tensor & self, int64_t dimension, int64_t size, int64_t step) const {
  profiler::RecordFunction profiler("_th_unfold_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_unfold");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_unfold");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_unfold_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_unfold_out(result_, self_, dimension, size, step);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_thnn_adaptive_avg_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self) const {
  profiler::RecordFunction profiler("_thnn_adaptive_avg_pool3d_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("_thnn_adaptive_avg_pool3d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("_thnn_adaptive_avg_pool3d_backward");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_adaptive_avg_pool3d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_adaptive_avg_pool3d_backward_out(grad_input_, grad_output_, self_);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
std::tuple<Tensor,Tensor> VariableType::_thnn_adaptive_max_pool3d_forward(const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("_thnn_adaptive_max_pool3d_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_adaptive_max_pool3d_forward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor output;
  Tensor indices;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_adaptive_max_pool3d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(output, indices) = as_variable(baseType->_thnn_adaptive_max_pool3d_forward(self_, output_size));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, indices);
  }
  return std::make_tuple(std::move(output), std::move(indices));
}
Tensor VariableType::_thnn_avg_pool2d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
  profiler::RecordFunction profiler("_thnn_avg_pool2d_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_avg_pool2d_forward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_avg_pool2d_forward");
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
  auto output = as_variable(baseType->_thnn_avg_pool2d_forward(self_, kernel_size, stride, padding, ceil_mode, count_include_pad));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_thnn_avg_pool3d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
  profiler::RecordFunction profiler("_thnn_avg_pool3d_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_avg_pool3d_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_avg_pool3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    jit::tracer::addInputs(node, "count_include_pad", count_include_pad);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto grad_input = as_variable(baseType->_thnn_avg_pool3d_backward(grad_output_, self_, kernel_size, stride, padding, ceil_mode, count_include_pad));
  set_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_avg_pool3d_forward_out(Tensor & output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
  profiler::RecordFunction profiler("_thnn_avg_pool3d_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_thnn_avg_pool3d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_thnn_avg_pool3d_forward");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_avg_pool3d_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_avg_pool3d_forward_out(output_, self_, kernel_size, stride, padding, ceil_mode, count_include_pad);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor & VariableType::_thnn_col2im_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList kernel_size, IntList dilation, IntList padding, IntList stride) const {
  profiler::RecordFunction profiler("_thnn_col2im_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output )) {
    throw_error_out_requires_grad("_thnn_col2im_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("_thnn_col2im_backward");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_col2im_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_col2im_backward_out(grad_input_, grad_output_, kernel_size, dilation, padding, stride);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
std::tuple<Tensor,Tensor> VariableType::_thnn_conv_depthwise2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, std::array<bool,2> output_mask) const {
  profiler::RecordFunction profiler("_thnn_conv_depthwise2d_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, self, weight )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_conv_depthwise2d_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, weight ));
  }
  Tensor grad_input;
  Tensor grad_weight;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_conv_depthwise2d_backward");
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(grad_input, grad_weight) = as_variable(baseType->_thnn_conv_depthwise2d_backward(grad_output_, self_, weight_, kernel_size, stride, padding, dilation, output_mask));
  set_history(flatten_tensor_args( grad_input, grad_weight ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
    jit::tracer::addOutput(node, grad_weight);
  }
  return std::make_tuple(std::move(grad_input), std::move(grad_weight));
}
Tensor & VariableType::_thnn_conv_depthwise2d_forward_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
  profiler::RecordFunction profiler("_thnn_conv_depthwise2d_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto bias_ = unpack_opt(bias, "bias", 4);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    throw_error_out_requires_grad("_thnn_conv_depthwise2d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_thnn_conv_depthwise2d_forward");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_conv_depthwise2d_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_conv_depthwise2d_forward_out(output_, self_, weight_, kernel_size, bias_, stride, padding, dilation);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::_thnn_conv_dilated2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
  profiler::RecordFunction profiler("_thnn_conv_dilated2d_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 3);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_conv_dilated2d_forward"), deleteFunction);
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
    op_name = jit::Symbol::fromQualString("aten::_thnn_conv_dilated2d_forward");
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
  std::tie(output, columns, ones) = as_variable(baseType->_thnn_conv_dilated2d_forward(self_, weight_, kernel_size, bias_, stride, padding, dilation));
  set_history(flatten_tensor_args( output, columns, ones ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, columns);
    jit::tracer::addOutput(node, ones);
  }
  return std::make_tuple(std::move(output), std::move(columns), std::move(ones));
}
std::tuple<Tensor,Tensor,Tensor> VariableType::_thnn_conv_dilated3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) const {
  profiler::RecordFunction profiler("_thnn_conv_dilated3d_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto& columns_ = unpack(columns, "columns", 7);
  auto& ones_ = unpack(ones, "ones", 8);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, self, weight, columns, ones )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_conv_dilated3d_backward"), deleteFunction);
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
    op_name = jit::Symbol::fromQualString("aten::_thnn_conv_dilated3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "columns", columns);
    jit::tracer::addInputs(node, "ones", ones);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(grad_input, grad_weight, grad_bias) = as_variable(baseType->_thnn_conv_dilated3d_backward(grad_output_, self_, weight_, kernel_size, stride, padding, dilation, columns_, ones_, output_mask));
  set_history(flatten_tensor_args( grad_input, grad_weight, grad_bias ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
    jit::tracer::addOutput(node, grad_weight);
    jit::tracer::addOutput(node, grad_bias);
  }
  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::_thnn_conv_dilated3d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
  profiler::RecordFunction profiler("_thnn_conv_dilated3d_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& columns_ = unpack(columns, "columns", 1);
  auto& ones_ = unpack(ones, "ones", 2);
  auto& self_ = unpack(self, "self", 3);
  auto& weight_ = unpack(weight, "weight", 4);
  auto bias_ = unpack_opt(bias, "bias", 6);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    throw_error_out_requires_grad("_thnn_conv_dilated3d_forward");
  }
  if (compute_requires_grad( output, columns, ones )) {
    throw_error_out_requires_grad("_thnn_conv_dilated3d_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_conv_dilated3d_forward");
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
    jit::tracer::addInputs(node, "dilation", dilation);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_conv_dilated3d_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_conv_dilated3d_forward_out(output_, columns_, ones_, self_, weight_, kernel_size, bias_, stride, padding, dilation);
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
std::tuple<Tensor,Tensor,Tensor> VariableType::_thnn_conv_transpose3d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
  profiler::RecordFunction profiler("_thnn_conv_transpose3d_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 3);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_conv_transpose3d_forward"), deleteFunction);
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
    op_name = jit::Symbol::fromQualString("aten::_thnn_conv_transpose3d_forward");
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
  std::tie(output, finput, fgrad_input) = as_variable(baseType->_thnn_conv_transpose3d_forward(self_, weight_, kernel_size, bias_, stride, padding, output_padding, dilation));
  set_history(flatten_tensor_args( output, finput, fgrad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, finput);
    jit::tracer::addOutput(node, fgrad_input);
  }
  return std::make_tuple(std::move(output), std::move(finput), std::move(fgrad_input));
}
Tensor VariableType::_thnn_elu_backward(const Tensor & grad_output, Scalar alpha, Scalar scale, Scalar input_scale, const Tensor & output) const {
  profiler::RecordFunction profiler("_thnn_elu_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& output_ = unpack(output, "output", 4);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, output )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_elu_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, output ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_elu_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "input_scale", input_scale);
    jit::tracer::addInputs(node, "output", output);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto grad_input = as_variable(baseType->_thnn_elu_backward(grad_output_, alpha, scale, input_scale, output_));
  set_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_elu_forward_out(Tensor & output, const Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale) const {
  profiler::RecordFunction profiler("_thnn_elu_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_thnn_elu_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_thnn_elu_forward");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_elu_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_elu_forward_out(output_, self_, alpha, scale, input_scale);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor & VariableType::_thnn_fractional_max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & indices) const {
  profiler::RecordFunction profiler("_thnn_fractional_max_pool2d_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& indices_ = unpack(indices, "indices", 5);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("_thnn_fractional_max_pool2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("_thnn_fractional_max_pool2d_backward");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_fractional_max_pool2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_fractional_max_pool2d_backward_out(grad_input_, grad_output_, self_, kernel_size, output_size, indices_);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_hardtanh_(Tensor & self, Scalar min_val, Scalar max_val) const {
  profiler::RecordFunction profiler("_thnn_hardtanh_", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_thnn_hardtanh");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_thnn_hardtanh_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min_val", min_val);
    jit::tracer::addInputs(node, "max_val", max_val);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_hardtanh_", self);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::_thnn_hardtanh_(self, min_val, max_val);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::_thnn_hardtanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val) const {
  profiler::RecordFunction profiler("_thnn_hardtanh_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("_thnn_hardtanh_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("_thnn_hardtanh_backward");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_hardtanh_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_hardtanh_backward_out(grad_input_, grad_output_, self_, min_val, max_val);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_im2col_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList input_size, IntList kernel_size, IntList dilation, IntList padding, IntList stride) const {
  profiler::RecordFunction profiler("_thnn_im2col_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output )) {
    throw_error_out_requires_grad("_thnn_im2col_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("_thnn_im2col_backward");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_im2col_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_im2col_backward_out(grad_input_, grad_output_, input_size, kernel_size, dilation, padding, stride);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::_thnn_l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) const {
  profiler::RecordFunction profiler("_thnn_l1_loss_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, self, target )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_l1_loss_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, target ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_l1_loss_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto grad_input = as_variable(baseType->_thnn_l1_loss_backward(grad_output_, self_, target_, reduction));
  set_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_l1_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, int64_t reduction) const {
  profiler::RecordFunction profiler("_thnn_l1_loss_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, target )) {
    throw_error_out_requires_grad("_thnn_l1_loss_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_thnn_l1_loss_forward");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_l1_loss_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_l1_loss_forward_out(output_, self_, target_, reduction);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
std::tuple<Tensor,Tensor> VariableType::_thnn_log_sigmoid_forward(const Tensor & self) const {
  profiler::RecordFunction profiler("_thnn_log_sigmoid_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_log_sigmoid_forward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor output;
  Tensor buffer;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_log_sigmoid_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(output, buffer) = as_variable(baseType->_thnn_log_sigmoid_forward(self_));
  set_history(flatten_tensor_args( output, buffer ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, buffer);
  }
  return std::make_tuple(std::move(output), std::move(buffer));
}
Tensor & VariableType::_thnn_max_pool2d_with_indices_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const {
  profiler::RecordFunction profiler("_thnn_max_pool2d_with_indices_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& indices_ = unpack(indices, "indices", 8);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("_thnn_max_pool2d_with_indices_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("_thnn_max_pool2d_with_indices_backward");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_max_pool2d_with_indices_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_max_pool2d_with_indices_backward_out(grad_input_, grad_output_, self_, kernel_size, stride, padding, dilation, ceil_mode, indices_);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::_thnn_max_unpool2d_forward(const Tensor & self, const Tensor & indices, IntList output_size) const {
  profiler::RecordFunction profiler("_thnn_max_unpool2d_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_max_unpool2d_forward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_max_unpool2d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto output = as_variable(baseType->_thnn_max_unpool2d_forward(self_, indices_, output_size));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_thnn_max_unpool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
  profiler::RecordFunction profiler("_thnn_max_unpool3d_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_max_unpool3d_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_max_unpool3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto grad_input = as_variable(baseType->_thnn_max_unpool3d_backward(grad_output_, self_, indices_, output_size, stride, padding));
  set_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_max_unpool3d_forward_out(Tensor & output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
  profiler::RecordFunction profiler("_thnn_max_unpool3d_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_thnn_max_unpool3d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_thnn_max_unpool3d_forward");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_max_unpool3d_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_max_unpool3d_forward_out(output_, self_, indices_, output_size, stride, padding);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor & VariableType::_thnn_multi_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) const {
  profiler::RecordFunction profiler("_thnn_multi_margin_loss_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& target_ = unpack(target, "target", 3);
  auto weight_ = unpack_opt(weight, "weight", 6);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, weight )) {
    throw_error_out_requires_grad("_thnn_multi_margin_loss_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("_thnn_multi_margin_loss_backward");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_multi_margin_loss_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_multi_margin_loss_backward_out(grad_input_, grad_output_, self_, target_, p, margin, weight_, reduction);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
std::tuple<Tensor,Tensor> VariableType::_thnn_nll_loss2d_forward(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) const {
  profiler::RecordFunction profiler("_thnn_nll_loss2d_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  auto weight_ = unpack_opt(weight, "weight", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, weight )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_nll_loss2d_forward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, weight ));
  }
  Tensor output;
  Tensor total_weight;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_nll_loss2d_forward");
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
  std::tie(output, total_weight) = as_variable(baseType->_thnn_nll_loss2d_forward(self_, target_, weight_, reduction, ignore_index));
  set_history(flatten_tensor_args( output, total_weight ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, total_weight);
  }
  return std::make_tuple(std::move(output), std::move(total_weight));
}
std::tuple<Tensor,Tensor> VariableType::_thnn_nll_loss_forward(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) const {
  profiler::RecordFunction profiler("_thnn_nll_loss_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  auto weight_ = unpack_opt(weight, "weight", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, weight )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_nll_loss_forward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, weight ));
  }
  Tensor output;
  Tensor total_weight;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_nll_loss_forward");
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
  std::tie(output, total_weight) = as_variable(baseType->_thnn_nll_loss_forward(self_, target_, weight_, reduction, ignore_index));
  set_history(flatten_tensor_args( output, total_weight ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, total_weight);
  }
  return std::make_tuple(std::move(output), std::move(total_weight));
}
Tensor VariableType::_thnn_reflection_pad1d_forward(const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("_thnn_reflection_pad1d_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_reflection_pad1d_forward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_reflection_pad1d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto output = as_variable(baseType->_thnn_reflection_pad1d_forward(self_, padding));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_thnn_reflection_pad2d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("_thnn_reflection_pad2d_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_reflection_pad2d_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_reflection_pad2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto grad_input = as_variable(baseType->_thnn_reflection_pad2d_backward(grad_output_, self_, padding));
  set_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_reflection_pad2d_forward_out(Tensor & output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("_thnn_reflection_pad2d_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_thnn_reflection_pad2d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_thnn_reflection_pad2d_forward");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_reflection_pad2d_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_reflection_pad2d_forward_out(output_, self_, padding);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_thnn_replication_pad1d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("_thnn_replication_pad1d_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_replication_pad1d_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_replication_pad1d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto grad_input = as_variable(baseType->_thnn_replication_pad1d_backward(grad_output_, self_, padding));
  set_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_replication_pad1d_forward_out(Tensor & output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("_thnn_replication_pad1d_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_thnn_replication_pad1d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_thnn_replication_pad1d_forward");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_replication_pad1d_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_replication_pad1d_forward_out(output_, self_, padding);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor & VariableType::_thnn_replication_pad2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("_thnn_replication_pad2d_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("_thnn_replication_pad2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("_thnn_replication_pad2d_backward");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_replication_pad2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_replication_pad2d_backward_out(grad_input_, grad_output_, self_, padding);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::_thnn_rrelu_with_noise_forward(const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
  profiler::RecordFunction profiler("_thnn_rrelu_with_noise_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& noise_ = unpack(noise, "noise", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, noise )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_rrelu_with_noise_forward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, noise ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_rrelu_with_noise_forward");
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
  auto output = as_variable(baseType->_thnn_rrelu_with_noise_forward(self_, noise_, lower, upper, training, generator));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor & VariableType::_thnn_rrelu_with_noise_forward_(Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
  profiler::RecordFunction profiler("_thnn_rrelu_with_noise_forward_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& noise_ = unpack(noise, "noise", 1);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, noise )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_rrelu_with_noise_forward_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, noise ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_thnn_rrelu_with_noise_forward");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_thnn_rrelu_with_noise_forward_");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_rrelu_with_noise_forward_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_rrelu_with_noise_forward_(self_, noise_, lower, upper, training, generator);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::_thnn_smooth_l1_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) const {
  profiler::RecordFunction profiler("_thnn_smooth_l1_loss_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& target_ = unpack(target, "target", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, target )) {
    throw_error_out_requires_grad("_thnn_smooth_l1_loss_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("_thnn_smooth_l1_loss_backward");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_smooth_l1_loss_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_smooth_l1_loss_backward_out(grad_input_, grad_output_, self_, target_, reduction);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::_thnn_soft_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) const {
  profiler::RecordFunction profiler("_thnn_soft_margin_loss_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, self, target )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_soft_margin_loss_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, target ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_soft_margin_loss_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto grad_input = as_variable(baseType->_thnn_soft_margin_loss_backward(grad_output_, self_, target_, reduction));
  set_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_soft_margin_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, int64_t reduction) const {
  profiler::RecordFunction profiler("_thnn_soft_margin_loss_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, target )) {
    throw_error_out_requires_grad("_thnn_soft_margin_loss_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_thnn_soft_margin_loss_forward");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_soft_margin_loss_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_soft_margin_loss_forward_out(output_, self_, target_, reduction);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_thnn_softplus_backward(const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output) const {
  profiler::RecordFunction profiler("_thnn_softplus_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& output_ = unpack(output, "output", 4);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, self, output )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_softplus_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, output ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_softplus_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "threshold", threshold);
    jit::tracer::addInputs(node, "output", output);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto grad_input = as_variable(baseType->_thnn_softplus_backward(grad_output_, self_, beta, threshold, output_));
  set_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_softplus_forward_out(Tensor & output, const Tensor & self, Scalar beta, Scalar threshold) const {
  profiler::RecordFunction profiler("_thnn_softplus_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_thnn_softplus_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_thnn_softplus_forward");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_softplus_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_softplus_forward_out(output_, self_, beta, threshold);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor & VariableType::_thnn_tanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output) const {
  profiler::RecordFunction profiler("_thnn_tanh_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& output_ = unpack(output, "output", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, output )) {
    throw_error_out_requires_grad("_thnn_tanh_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("_thnn_tanh_backward");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_tanh_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_tanh_backward_out(grad_input_, grad_output_, output_);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_upsample_bicubic2d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
  profiler::RecordFunction profiler("_thnn_upsample_bicubic2d_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output )) {
    throw_error_out_requires_grad("_thnn_upsample_bicubic2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("_thnn_upsample_bicubic2d_backward");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_upsample_bicubic2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_upsample_bicubic2d_backward_out(grad_input_, grad_output_, output_size, input_size, align_corners);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::_thnn_upsample_linear1d_forward(const Tensor & self, IntList output_size, bool align_corners) const {
  profiler::RecordFunction profiler("_thnn_upsample_linear1d_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_upsample_linear1d_forward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_upsample_linear1d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto output = as_variable(baseType->_thnn_upsample_linear1d_forward(self_, output_size, align_corners));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_thnn_upsample_nearest2d_forward(const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("_thnn_upsample_nearest2d_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_upsample_nearest2d_forward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_upsample_nearest2d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto output = as_variable(baseType->_thnn_upsample_nearest2d_forward(self_, output_size));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_thnn_upsample_nearest3d_backward(const Tensor & grad_output, IntList output_size, IntList input_size) const {
  profiler::RecordFunction profiler("_thnn_upsample_nearest3d_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_upsample_nearest3d_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_upsample_nearest3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto grad_input = as_variable(baseType->_thnn_upsample_nearest3d_backward(grad_output_, output_size, input_size));
  set_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_upsample_nearest3d_forward_out(Tensor & output, const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("_thnn_upsample_nearest3d_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_thnn_upsample_nearest3d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_thnn_upsample_nearest3d_forward");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_upsample_nearest3d_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_upsample_nearest3d_forward_out(output_, self_, output_size);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor & VariableType::_thnn_upsample_trilinear3d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
  profiler::RecordFunction profiler("_thnn_upsample_trilinear3d_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output )) {
    throw_error_out_requires_grad("_thnn_upsample_trilinear3d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("_thnn_upsample_trilinear3d_backward");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_upsample_trilinear3d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_thnn_upsample_trilinear3d_backward_out(grad_input_, grad_output_, output_size, input_size, align_corners);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
std::tuple<Tensor,Tensor> VariableType::_unique_dim(const Tensor & self, int64_t dim, bool sorted, bool return_inverse) const {
  profiler::RecordFunction profiler("_unique_dim", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_unique_dim"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_unique_dim");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "sorted", sorted);
    jit::tracer::addInputs(node, "return_inverse", return_inverse);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = as_variable(baseType->_unique_dim(self_, dim, sorted, return_inverse));
  set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor VariableType::_weight_norm(const Tensor & v, const Tensor & g, int64_t dim) const {
  profiler::RecordFunction profiler("_weight_norm", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_weight_norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "v", v);
    jit::tracer::addInputs(node, "g", g);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::_weight_norm(v, g, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::acos(const Tensor & self) const {
  profiler::RecordFunction profiler("acos", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AcosBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AcosBackward>(new AcosBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::acos");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->acos(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::acos_(Tensor & self) const {
  profiler::RecordFunction profiler("acos_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<AcosBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AcosBackward>(new AcosBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::acos");
    } else {
      op_name = jit::Symbol::fromQualString("aten::acos_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("acos_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->acos_(self_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::adaptive_avg_pool2d_backward(const Tensor & grad_output, const Tensor & self) const {
  profiler::RecordFunction profiler("adaptive_avg_pool2d_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<AdaptiveAvgPool2DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<AdaptiveAvgPool2DBackwardBackward>(new AdaptiveAvgPool2DBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_info = self;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::adaptive_avg_pool2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->adaptive_avg_pool2d_backward(grad_output_, self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::adaptive_avg_pool3d(const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("adaptive_avg_pool3d", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AdaptiveAvgPool3DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AdaptiveAvgPool3DBackward>(new AdaptiveAvgPool3DBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::adaptive_avg_pool3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->adaptive_avg_pool3d(self_, output_size));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::adaptive_avg_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self) const {
  profiler::RecordFunction profiler("adaptive_avg_pool3d_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("adaptive_avg_pool3d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("adaptive_avg_pool3d_backward");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("adaptive_avg_pool3d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->adaptive_avg_pool3d_backward_out(grad_input_, grad_output_, self_);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::add(const Tensor & self, const Tensor & other, Scalar alpha) const {
  profiler::RecordFunction profiler("add", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  std::shared_ptr<AddBackward0> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<AddBackward0>(new AddBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->alpha = alpha;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::add");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->add(self_, other_, alpha));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::add(const Tensor & self, Scalar other, Scalar alpha) const {
  profiler::RecordFunction profiler("add", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AddBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AddBackward1>(new AddBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::add");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->add(self_, other, alpha));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::add_(Tensor & self, const Tensor & other, Scalar alpha) const {
  profiler::RecordFunction profiler("add_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<AddBackward0> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<AddBackward0>(new AddBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->alpha = alpha;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::add");
    } else {
      op_name = jit::Symbol::fromQualString("aten::add_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("add_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->add_(self_, other_, alpha);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::add_(Tensor & self, Scalar other, Scalar alpha) const {
  profiler::RecordFunction profiler("add_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<AddBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AddBackward1>(new AddBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::add");
    } else {
      op_name = jit::Symbol::fromQualString("aten::add_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("add_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->add_(self_, other, alpha);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::addbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("addbmm_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& batch1_ = unpack(batch1, "batch1", 2);
  auto& batch2_ = unpack(batch2, "batch2", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, batch1, batch2 )) {
    throw_error_out_requires_grad("addbmm");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("addbmm");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("addbmm_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->addbmm_out(result_, self_, batch1_, batch2_, beta, alpha);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::affine_grid_generator_backward(const Tensor & grad, IntList size) const {
  profiler::RecordFunction profiler("affine_grid_generator_backward", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::affine_grid_generator_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "size", size);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::affine_grid_generator_backward(grad, size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::alias(const Tensor & self) const {
  profiler::RecordFunction profiler("alias", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AliasBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AliasBackward>(new AliasBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::alias");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_view(self, baseType->alias(self_), true);
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::all_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("all_out", Function::peek_at_next_sequence_nr());
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("all_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::all_out(result, self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
bool VariableType::allclose(const Tensor & self, const Tensor & other, double rtol, double atol, bool equal_nan) const {
  profiler::RecordFunction profiler("allclose", Function::peek_at_next_sequence_nr());
  auto result = TypeDefault::allclose(self, other, rtol, atol, equal_nan);
  return result;
}
Tensor & VariableType::any_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("any_out", Function::peek_at_next_sequence_nr());
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("any_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::any_out(result, self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::asin(const Tensor & self) const {
  profiler::RecordFunction profiler("asin", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AsinBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AsinBackward>(new AsinBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::asin");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->asin(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::asin_(Tensor & self) const {
  profiler::RecordFunction profiler("asin_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<AsinBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AsinBackward>(new AsinBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::asin");
    } else {
      op_name = jit::Symbol::fromQualString("aten::asin_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("asin_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->asin_(self_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::avg_pool3d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
  profiler::RecordFunction profiler("avg_pool3d_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<AvgPool3DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<AvgPool3DBackwardBackward>(new AvgPool3DBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->ceil_mode = ceil_mode;
    grad_fn->count_include_pad = count_include_pad;
    grad_fn->self_info = self;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::avg_pool3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    jit::tracer::addInputs(node, "count_include_pad", count_include_pad);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->avg_pool3d_backward(grad_output_, self_, kernel_size, stride, padding, ceil_mode, count_include_pad));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::bernoulli(const Tensor & self, Generator * generator) const {
  profiler::RecordFunction profiler("bernoulli", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<BernoulliBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<BernoulliBackward0>(new BernoulliBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bernoulli");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->bernoulli(self_, generator));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::bernoulli(const Tensor & self, double p, Generator * generator) const {
  profiler::RecordFunction profiler("bernoulli", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bernoulli");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::bernoulli(self, p, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::bernoulli_(Tensor & self, const Tensor & p, Generator * generator) const {
  profiler::RecordFunction profiler("bernoulli_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& p_ = unpack(p, "p", 1);
  check_inplace(self);
  std::shared_ptr<BernoulliBackward1> grad_fn;
  if (compute_requires_grad( self, p )) {
    grad_fn = std::shared_ptr<BernoulliBackward1>(new BernoulliBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, p ));
    grad_fn->p_info = p;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::bernoulli");
    } else {
      op_name = jit::Symbol::fromQualString("aten::bernoulli_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bernoulli_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->bernoulli_(self_, p_, generator);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::bernoulli_(Tensor & self, double p, Generator * generator) const {
  profiler::RecordFunction profiler("bernoulli_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<BernoulliBackward2> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<BernoulliBackward2>(new BernoulliBackward2(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::bernoulli");
    } else {
      op_name = jit::Symbol::fromQualString("aten::bernoulli_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bernoulli_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->bernoulli_(self_, p, generator);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::binary_cross_entropy_with_logits_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, const Tensor & pos_weight, int64_t reduction) const {
  profiler::RecordFunction profiler("binary_cross_entropy_with_logits_backward", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::binary_cross_entropy_with_logits_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "pos_weight", pos_weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::binary_cross_entropy_with_logits_backward(grad_output, self, target, weight, pos_weight, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::btrifact(const Tensor & self, bool pivot) const {
  profiler::RecordFunction profiler("btrifact", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<BtrifactBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<BtrifactBackward>(new BtrifactBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::btrifact");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "pivot", pivot);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = as_variable(baseType->btrifact(self_, pivot));
  set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor VariableType::cat(TensorList tensors, int64_t dim) const {
  profiler::RecordFunction profiler("cat", Function::peek_at_next_sequence_nr());
  auto tensors_ = unpack(tensors, "tensors", 0);
  std::shared_ptr<CatBackward> grad_fn;
  if (compute_requires_grad( tensors )) {
    grad_fn = std::shared_ptr<CatBackward>(new CatBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( tensors ));
    grad_fn->tensors_args_sizes = to_args_sizes(tensors);
    grad_fn->dim = dim;
    grad_fn->tensors_size_ = tensors.size();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cat");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->cat(tensors_, dim));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::cauchy_(Tensor & self, double median, double sigma, Generator * generator) const {
  profiler::RecordFunction profiler("cauchy_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<CauchyBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<CauchyBackward>(new CauchyBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::cauchy");
    } else {
      op_name = jit::Symbol::fromQualString("aten::cauchy_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "median", median);
    jit::tracer::addInputs(node, "sigma", sigma);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cauchy_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->cauchy_(self_, median, sigma, generator);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::ceil_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("ceil_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("ceil");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("ceil");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ceil_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->ceil_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::vector<Tensor> VariableType::chunk(const Tensor & self, int64_t chunks, int64_t dim) const {
  profiler::RecordFunction profiler("chunk", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::chunk");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "chunks", chunks);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::chunk(self, chunks, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::conv1d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, int64_t groups) const {
  profiler::RecordFunction profiler("conv1d", Function::peek_at_next_sequence_nr());
  auto result = TypeDefault::conv1d(input, weight, bias, stride, padding, dilation, groups);
  return result;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::conv_tbc_backward(const Tensor & self, const Tensor & input, const Tensor & weight, const Tensor & bias, int64_t pad) const {
  profiler::RecordFunction profiler("conv_tbc_backward", Function::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  Tensor result2;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::conv_tbc_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "pad", pad);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1, result2) = TypeDefault::conv_tbc_backward(self, input, weight, bias, pad);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor VariableType::cosine_embedding_loss(const Tensor & input1, const Tensor & input2, const Tensor & target, double margin, int64_t reduction) const {
  profiler::RecordFunction profiler("cosine_embedding_loss", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cosine_embedding_loss");
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
  auto result = TypeDefault::cosine_embedding_loss(input1, input2, target, margin, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::cudnn_affine_grid_generator(const Tensor & theta, int64_t N, int64_t C, int64_t H, int64_t W) const {
  profiler::RecordFunction profiler("cudnn_affine_grid_generator", Function::peek_at_next_sequence_nr());
  auto& theta_ = unpack(theta, "theta", 0);
  std::shared_ptr<CudnnAffineGridGeneratorBackward> grad_fn;
  if (compute_requires_grad( theta )) {
    grad_fn = std::shared_ptr<CudnnAffineGridGeneratorBackward>(new CudnnAffineGridGeneratorBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( theta ));
    grad_fn->N = N;
    grad_fn->C = C;
    grad_fn->H = H;
    grad_fn->W = W;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cudnn_affine_grid_generator");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "theta", theta);
    jit::tracer::addInputs(node, "N", N);
    jit::tracer::addInputs(node, "C", C);
    jit::tracer::addInputs(node, "H", H);
    jit::tracer::addInputs(node, "W", W);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto grid = as_variable(baseType->cudnn_affine_grid_generator(theta_, N, C, H, W));
  set_history(flatten_tensor_args( grid ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grid);
  }
  return grid;
}
Tensor VariableType::cudnn_convolution(const Tensor & self, const Tensor & weight, const Tensor & bias, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic) const {
  profiler::RecordFunction profiler("cudnn_convolution", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 2);
  std::shared_ptr<CudnnConvolutionBackward> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::shared_ptr<CudnnConvolutionBackward>(new CudnnConvolutionBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->padding = padding.vec();
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
    op_name = jit::Symbol::fromQualString("aten::cudnn_convolution");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "benchmark", benchmark);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->cudnn_convolution(self_, weight_, bias_, padding, stride, dilation, groups, benchmark, deterministic));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::cudnn_convolution_transpose_backward_weight(IntList weight_size, const Tensor & grad_output, const Tensor & self, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic) const {
  profiler::RecordFunction profiler("cudnn_convolution_transpose_backward_weight", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("cudnn_convolution_transpose_backward_weight"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cudnn_convolution_transpose_backward_weight");
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
  auto result = as_variable(baseType->cudnn_convolution_transpose_backward_weight(weight_size, grad_output_, self_, padding, stride, dilation, groups, benchmark, deterministic));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::cumprod(const Tensor & self, int64_t dim, ScalarType dtype) const {
  profiler::RecordFunction profiler("cumprod", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<CumprodBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<CumprodBackward1>(new CumprodBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim;
    grad_fn->dtype = dtype;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cumprod");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->cumprod(self_, dim, dtype));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::cumprod(const Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("cumprod", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<CumprodBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<CumprodBackward0>(new CumprodBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cumprod");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->cumprod(self_, dim));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void* VariableType::data_ptr(const Tensor & self) const {
  auto result = TypeDefault::data_ptr(self);
  return result;
}
Tensor VariableType::det(const Tensor & self) const {
  profiler::RecordFunction profiler("det", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<DetBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<DetBackward>(new DetBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::det");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->det(self_));
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
Tensor & VariableType::div_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("div_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("div");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("div");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("div_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->div_out(result_, self_, other_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::dot(const Tensor & self, const Tensor & tensor) const {
  profiler::RecordFunction profiler("dot", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& tensor_ = unpack(tensor, "tensor", 1);
  std::shared_ptr<DotBackward> grad_fn;
  if (compute_requires_grad( self, tensor )) {
    grad_fn = std::shared_ptr<DotBackward>(new DotBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, tensor ));
    grad_fn->tensor_ = SavedVariable(tensor, false);
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::dot");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "tensor", tensor);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->dot(self_, tensor_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::einsum(std::string equation, TensorList tensors) const {
  profiler::RecordFunction profiler("einsum", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::einsum");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "equation", equation);
    jit::tracer::addInputs(node, "tensors", tensors);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::einsum(equation, tensors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::elu_backward(const Tensor & grad_output, Scalar alpha, Scalar scale, Scalar input_scale, const Tensor & output) const {
  profiler::RecordFunction profiler("elu_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& output_ = unpack(output, "output", 4);
  std::shared_ptr<EluBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, output )) {
    grad_fn = std::shared_ptr<EluBackwardBackward>(new EluBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, output ));
    grad_fn->alpha = alpha;
    grad_fn->scale = scale;
    grad_fn->input_scale = input_scale;
    grad_fn->output_ = SavedVariable(output, false);
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::elu_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "input_scale", input_scale);
    jit::tracer::addInputs(node, "output", output);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->elu_backward(grad_output_, alpha, scale, input_scale, output_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::embedding(const Tensor & weight, const Tensor & indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) const {
  profiler::RecordFunction profiler("embedding", Function::peek_at_next_sequence_nr());
  auto& weight_ = unpack(weight, "weight", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  std::shared_ptr<EmbeddingBackward> grad_fn;
  if (compute_requires_grad( weight )) {
    grad_fn = std::shared_ptr<EmbeddingBackward>(new EmbeddingBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( weight ));
    grad_fn->weight_argsize_0 = weight.size(0);
    grad_fn->indices_ = SavedVariable(indices, false);
    grad_fn->padding_idx = padding_idx;
    grad_fn->scale_grad_by_freq = scale_grad_by_freq;
    grad_fn->sparse = sparse;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::embedding");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "padding_idx", padding_idx);
    jit::tracer::addInputs(node, "scale_grad_by_freq", scale_grad_by_freq);
    jit::tracer::addInputs(node, "sparse", sparse);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->embedding(weight_, indices_, padding_idx, scale_grad_by_freq, sparse));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::embedding_sparse_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) const {
  profiler::RecordFunction profiler("embedding_sparse_backward", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::embedding_sparse_backward");
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
  auto result = TypeDefault::embedding_sparse_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::erf(const Tensor & self) const {
  profiler::RecordFunction profiler("erf", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ErfBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ErfBackward>(new ErfBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::erf");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->erf(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::erf_(Tensor & self) const {
  profiler::RecordFunction profiler("erf_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<ErfBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ErfBackward>(new ErfBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::erf");
    } else {
      op_name = jit::Symbol::fromQualString("aten::erf_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("erf_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->erf_(self_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::exp_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("exp_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("exp");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("exp");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("exp_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->exp_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::expand_as(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("expand_as", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::expand_as");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::expand_as(self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::eye_out(Tensor & result, int64_t n) const {
  profiler::RecordFunction profiler("eye_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::eye");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "n", n);
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "result", result.options());
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("eye_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->eye_out(result_, n);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::eye_out(Tensor & result, int64_t n, int64_t m) const {
  profiler::RecordFunction profiler("eye_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::eye");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "m", m);
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "result", result.options());
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("eye_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->eye_out(result_, n, m);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::feature_alpha_dropout(const Tensor & input, double p, bool train) const {
  profiler::RecordFunction profiler("feature_alpha_dropout", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::feature_alpha_dropout");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "train", train);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::feature_alpha_dropout(input, p, train);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::feature_alpha_dropout_(Tensor & self, double p, bool train) const {
  profiler::RecordFunction profiler("feature_alpha_dropout_", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::feature_alpha_dropout");
    } else {
      op_name = jit::Symbol::fromQualString("aten::feature_alpha_dropout_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "train", train);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("feature_alpha_dropout_", self);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::feature_alpha_dropout_(self, p, train);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::flip(const Tensor & self, IntList dims) const {
  profiler::RecordFunction profiler("flip", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<FlipBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<FlipBackward>(new FlipBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->dims = dims.vec();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::flip");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dims", dims);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->flip(self_, dims));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::fmod(const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("fmod", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<FmodBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<FmodBackward0>(new FmodBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fmod");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->fmod(self_, other));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::fmod(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("fmod", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  std::shared_ptr<FmodBackward1> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<FmodBackward1>(new FmodBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->other_ = SavedVariable(other, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fmod");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->fmod(self_, other_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::fmod_(Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("fmod_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<FmodBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<FmodBackward0>(new FmodBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::fmod");
    } else {
      op_name = jit::Symbol::fromQualString("aten::fmod_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fmod_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->fmod_(self_, other);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::fmod_(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("fmod_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<FmodBackward1> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<FmodBackward1>(new FmodBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->other_ = SavedVariable(other, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::fmod");
    } else {
      op_name = jit::Symbol::fromQualString("aten::fmod_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fmod_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->fmod_(self_, other_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::frac(const Tensor & self) const {
  profiler::RecordFunction profiler("frac", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<FracBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<FracBackward>(new FracBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::frac");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->frac(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::frac_(Tensor & self) const {
  profiler::RecordFunction profiler("frac_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<FracBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<FracBackward>(new FracBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::frac");
    } else {
      op_name = jit::Symbol::fromQualString("aten::frac_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("frac_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->frac_(self_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
std::tuple<Tensor,Tensor> VariableType::fractional_max_pool2d(const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & random_samples) const {
  profiler::RecordFunction profiler("fractional_max_pool2d", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& random_samples_ = unpack(random_samples, "random_samples", 3);
  check_no_requires_grad(random_samples, "random_samples");
  std::shared_ptr<FractionalMaxPool2DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<FractionalMaxPool2DBackward>(new FractionalMaxPool2DBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->output_size = output_size.vec();
  }
  Tensor output;
  Tensor indices;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fractional_max_pool2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "random_samples", random_samples);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(output, indices) = as_variable(baseType->fractional_max_pool2d(self_, kernel_size, output_size, random_samples_));
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
Tensor & VariableType::fractional_max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & indices) const {
  profiler::RecordFunction profiler("fractional_max_pool2d_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& indices_ = unpack(indices, "indices", 5);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, indices )) {
    throw_error_out_requires_grad("fractional_max_pool2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("fractional_max_pool2d_backward");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fractional_max_pool2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->fractional_max_pool2d_backward_out(grad_input_, grad_output_, self_, kernel_size, output_size, indices_);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::frobenius_norm(const Tensor & self) const {
  profiler::RecordFunction profiler("frobenius_norm", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::frobenius_norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::frobenius_norm(self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::frobenius_norm(const Tensor & self, IntList dim, bool keepdim) const {
  profiler::RecordFunction profiler("frobenius_norm", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::frobenius_norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::frobenius_norm(self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::gels(const Tensor & self, const Tensor & A) const {
  profiler::RecordFunction profiler("gels", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& A_ = unpack(A, "A", 1);
  std::shared_ptr<GelsBackward> grad_fn;
  if (compute_requires_grad( self, A )) {
    grad_fn = std::shared_ptr<GelsBackward>(new GelsBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, A ));
  }
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::gels");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "A", A);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = as_variable(baseType->gels(self_, A_));
  set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
std::tuple<Tensor &,Tensor &> VariableType::geqrf_out(Tensor & result0, Tensor & result1, const Tensor & self) const {
  profiler::RecordFunction profiler("geqrf_out", Function::peek_at_next_sequence_nr());
  auto& result0_ = unpack(result0, "result0", 0);
  auto& result1_ = unpack(result1, "result1", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("geqrf");
  }
  if (compute_requires_grad( result0, result1 )) {
    throw_error_out_requires_grad("geqrf");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::geqrf");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "result1", result1);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result0", result0);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("geqrf_out", result0);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->geqrf_out(result0_, result1_, self_);
  increment_version(result0);
  increment_version(result1);
  rebase_history(flatten_tensor_args( result0, result1 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::forward_as_tuple(result0, result1);
}
Tensor & VariableType::ger_out(Tensor & result, const Tensor & self, const Tensor & vec2) const {
  profiler::RecordFunction profiler("ger_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& vec2_ = unpack(vec2, "vec2", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, vec2 )) {
    throw_error_out_requires_grad("ger");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("ger");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ger_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->ger_out(result_, self_, vec2_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::gesv(const Tensor & self, const Tensor & A) const {
  profiler::RecordFunction profiler("gesv", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& A_ = unpack(A, "A", 1);
  std::shared_ptr<GesvBackward> grad_fn;
  if (compute_requires_grad( self, A )) {
    grad_fn = std::shared_ptr<GesvBackward>(new GesvBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, A ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->A_ = SavedVariable(A, false);
  }
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::gesv");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "A", A);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = as_variable(baseType->gesv(self_, A_));
  set_history(flatten_tensor_args( result0 ), grad_fn);
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
Tensor & VariableType::glu_out(Tensor & output, const Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("glu_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("glu");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("glu");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("glu_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->glu_out(output_, self_, dim);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::hardtanh(const Tensor & self, Scalar min_val, Scalar max_val) const {
  profiler::RecordFunction profiler("hardtanh", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<HardtanhBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<HardtanhBackward0>(new HardtanhBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->min_val = min_val;
    grad_fn->max_val = max_val;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hardtanh");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min_val", min_val);
    jit::tracer::addInputs(node, "max_val", max_val);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->hardtanh(self_, min_val, max_val));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::hardtanh_(Tensor & self, Scalar min_val, Scalar max_val) const {
  profiler::RecordFunction profiler("hardtanh_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<HardtanhBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<HardtanhBackward1>(new HardtanhBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->min_val = min_val;
    grad_fn->max_val = max_val;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::hardtanh");
    } else {
      op_name = jit::Symbol::fromQualString("aten::hardtanh_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min_val", min_val);
    jit::tracer::addInputs(node, "max_val", max_val);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("hardtanh_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->hardtanh_(self_, min_val, max_val);
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
Tensor & VariableType::hardtanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val) const {
  profiler::RecordFunction profiler("hardtanh_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("hardtanh_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("hardtanh_backward");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("hardtanh_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->hardtanh_backward_out(grad_input_, grad_output_, self_, min_val, max_val);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::instance_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool use_input_stats, double momentum, double eps, bool cudnn_enabled) const {
  profiler::RecordFunction profiler("instance_norm", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::instance_norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "use_input_stats", use_input_stats);
    jit::tracer::addInputs(node, "momentum", momentum);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "cudnn_enabled", cudnn_enabled);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::instance_norm(input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
bool VariableType::is_distributed(const Tensor & self) const {
  auto result = TypeDefault::is_distributed(self);
  return result;
}
std::tuple<Tensor &,Tensor &> VariableType::kthvalue_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("kthvalue_out", Function::peek_at_next_sequence_nr());
  auto& values_ = unpack(values, "values", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("kthvalue");
  }
  if (compute_requires_grad( values )) {
    throw_error_out_requires_grad("kthvalue");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::kthvalue");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "k", k);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "values", values);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("kthvalue_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->kthvalue_out(values_, indices_, self_, k, dim, keepdim);
  increment_version(values);
  rebase_history(flatten_tensor_args( values ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(values, indices);
}
Tensor VariableType::l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) const {
  profiler::RecordFunction profiler("l1_loss_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  check_no_requires_grad(target, "target");
  std::shared_ptr<L1LossBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<L1LossBackwardBackward>(new L1LossBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->reduction = reduction;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::l1_loss_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->l1_loss_backward(grad_output_, self_, target_, reduction));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::lgamma_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("lgamma_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("lgamma");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("lgamma");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("lgamma_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->lgamma_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::log(const Tensor & self) const {
  profiler::RecordFunction profiler("log", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<LogBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<LogBackward>(new LogBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::log");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->log(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::log1p_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("log1p_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("log1p");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("log1p");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("log1p_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->log1p_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::log2(const Tensor & self) const {
  profiler::RecordFunction profiler("log2", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<Log2Backward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<Log2Backward>(new Log2Backward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::log2");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->log2(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::log2_(Tensor & self) const {
  profiler::RecordFunction profiler("log2_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<Log2Backward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<Log2Backward>(new Log2Backward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::log2");
    } else {
      op_name = jit::Symbol::fromQualString("aten::log2_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("log2_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->log2_(self_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::log_(Tensor & self) const {
  profiler::RecordFunction profiler("log_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<LogBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<LogBackward>(new LogBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::log");
    } else {
      op_name = jit::Symbol::fromQualString("aten::log_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("log_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->log_(self_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
std::tuple<Tensor,Tensor> VariableType::log_sigmoid_forward(const Tensor & self) const {
  profiler::RecordFunction profiler("log_sigmoid_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<LogSigmoidBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<LogSigmoidBackward>(new LogSigmoidBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  Tensor output;
  Tensor buffer;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::log_sigmoid_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(output, buffer) = as_variable(baseType->log_sigmoid_forward(self_));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, buffer);
  }
  if (grad_fn) {
    grad_fn->buffer_ = SavedVariable(buffer, true);
  }
  return std::make_tuple(std::move(output), std::move(buffer));
}
Tensor VariableType::log_softmax(const Tensor & self, int64_t dim, ScalarType dtype) const {
  profiler::RecordFunction profiler("log_softmax", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::log_softmax");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::log_softmax(self, dim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::log_softmax(const Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("log_softmax", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::log_softmax");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::log_softmax(self, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::masked_fill_(Tensor & self, const Tensor & mask, Scalar value) const {
  profiler::RecordFunction profiler("masked_fill_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& mask_ = unpack(mask, "mask", 1);
  check_inplace(self);
  check_no_requires_grad(mask, "mask");
  std::shared_ptr<MaskedFillBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MaskedFillBackward0>(new MaskedFillBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->mask_ = SavedVariable(mask, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::masked_fill");
    } else {
      op_name = jit::Symbol::fromQualString("aten::masked_fill_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("masked_fill_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->masked_fill_(self_, mask_, value);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::masked_fill_(Tensor & self, const Tensor & mask, const Tensor & value) const {
  profiler::RecordFunction profiler("masked_fill_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& mask_ = unpack(mask, "mask", 1);
  auto& value_ = unpack(value, "value", 2);
  check_inplace(self);
  check_no_requires_grad(mask, "mask");
  std::shared_ptr<MaskedFillBackward1> grad_fn;
  if (compute_requires_grad( self, value )) {
    grad_fn = std::shared_ptr<MaskedFillBackward1>(new MaskedFillBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, value ));
    grad_fn->mask_ = SavedVariable(mask, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::masked_fill");
    } else {
      op_name = jit::Symbol::fromQualString("aten::masked_fill_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("masked_fill_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->masked_fill_(self_, mask_, value_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::masked_scatter_(Tensor & self, const Tensor & mask, const Tensor & source) const {
  profiler::RecordFunction profiler("masked_scatter_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& mask_ = unpack(mask, "mask", 1);
  auto& source_ = unpack(source, "source", 2);
  check_inplace(self);
  check_no_requires_grad(mask, "mask");
  std::shared_ptr<MaskedScatterBackward> grad_fn;
  if (compute_requires_grad( self, source )) {
    grad_fn = std::shared_ptr<MaskedScatterBackward>(new MaskedScatterBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, source ));
    grad_fn->mask_ = SavedVariable(mask, false);
    grad_fn->source_sizes = source.sizes().vec();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::masked_scatter");
    } else {
      op_name = jit::Symbol::fromQualString("aten::masked_scatter_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "source", source);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("masked_scatter_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->masked_scatter_(self_, mask_, source_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::max_pool1d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
  profiler::RecordFunction profiler("max_pool1d", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max_pool1d");
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
  auto result = TypeDefault::max_pool1d(self, kernel_size, stride, padding, dilation, ceil_mode);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::max_pool2d_with_indices(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
  profiler::RecordFunction profiler("max_pool2d_with_indices", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<MaxPool2DWithIndicesBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MaxPool2DWithIndicesBackward>(new MaxPool2DWithIndicesBackward(), deleteFunction);
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
    op_name = jit::Symbol::fromQualString("aten::max_pool2d_with_indices");
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
  std::tie(output, indices) = as_variable(baseType->max_pool2d_with_indices(self_, kernel_size, stride, padding, dilation, ceil_mode));
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
Tensor & VariableType::max_pool2d_with_indices_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const {
  profiler::RecordFunction profiler("max_pool2d_with_indices_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& indices_ = unpack(indices, "indices", 8);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, indices )) {
    throw_error_out_requires_grad("max_pool2d_with_indices_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("max_pool2d_with_indices_backward");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("max_pool2d_with_indices_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->max_pool2d_with_indices_backward_out(grad_input_, grad_output_, self_, kernel_size, stride, padding, dilation, ceil_mode, indices_);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
std::tuple<Tensor &,Tensor &> VariableType::max_pool3d_with_indices_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
  profiler::RecordFunction profiler("max_pool3d_with_indices_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("max_pool3d_with_indices");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("max_pool3d_with_indices");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max_pool3d_with_indices");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("max_pool3d_with_indices_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->max_pool3d_with_indices_out(output_, indices_, self_, kernel_size, stride, padding, dilation, ceil_mode);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(output, indices);
}
Tensor VariableType::max_unpool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
  profiler::RecordFunction profiler("max_unpool3d_backward", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max_unpool3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::max_unpool3d_backward(grad_output, self, indices, output_size, stride, padding);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::max_values(const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("max_values", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max_values");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::max_values(self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::mean(const Tensor & self, ScalarType dtype) const {
  profiler::RecordFunction profiler("mean", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<MeanBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MeanBackward1>(new MeanBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->self_numel = self.numel();
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mean");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->mean(self_, dtype));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::mean(const Tensor & self) const {
  profiler::RecordFunction profiler("mean", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<MeanBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MeanBackward0>(new MeanBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->self_numel = self.numel();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mean");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->mean(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::mean(const Tensor & self, IntList dim, bool keepdim, ScalarType dtype) const {
  profiler::RecordFunction profiler("mean", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<MeanBackward4> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MeanBackward4>(new MeanBackward4(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim.vec();
    grad_fn->keepdim = keepdim;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mean");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->mean(self_, dim, keepdim, dtype));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::mean(const Tensor & self, IntList dim, bool keepdim) const {
  profiler::RecordFunction profiler("mean", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<MeanBackward2> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MeanBackward2>(new MeanBackward2(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->dim = dim.vec();
    grad_fn->keepdim = keepdim;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mean");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->mean(self_, dim, keepdim));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::mean(const Tensor & self, IntList dim, ScalarType dtype) const {
  profiler::RecordFunction profiler("mean", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<MeanBackward3> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MeanBackward3>(new MeanBackward3(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim.vec();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mean");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->mean(self_, dim, dtype));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::median(const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("median", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<MedianBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MedianBackward1>(new MedianBackward1(), deleteFunction);
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
    op_name = jit::Symbol::fromQualString("aten::median");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = as_variable(baseType->median(self_, dim, keepdim));
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
Tensor VariableType::median(const Tensor & self) const {
  profiler::RecordFunction profiler("median", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<MedianBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MedianBackward0>(new MedianBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::median");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->median(self_));
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
std::tuple<Tensor,Tensor,Tensor> VariableType::miopen_batch_norm_backward(const Tensor & input, const Tensor & grad_output, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_var, double epsilon) const {
  profiler::RecordFunction profiler("miopen_batch_norm_backward", Function::peek_at_next_sequence_nr());
  auto& input_ = unpack(input, "input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto running_mean_ = unpack_opt(running_mean, "running_mean", 3);
  auto running_var_ = unpack_opt(running_var, "running_var", 4);
  auto save_mean_ = unpack_opt(save_mean, "save_mean", 5);
  auto save_var_ = unpack_opt(save_var, "save_var", 6);
  check_no_requires_grad(running_mean, "running_mean");
  check_no_requires_grad(running_var, "running_var");
  std::shared_ptr<MiopenBatchNormBackwardBackward> grad_fn;
  if (compute_requires_grad( input, grad_output, weight, save_mean, save_var )) {
    grad_fn = std::shared_ptr<MiopenBatchNormBackwardBackward>(new MiopenBatchNormBackwardBackward(), deleteFunction);
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
    op_name = jit::Symbol::fromQualString("aten::miopen_batch_norm_backward");
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
  std::tie(result0, result1, result2) = as_variable(baseType->miopen_batch_norm_backward(input_, grad_output_, weight_, running_mean_, running_var_, save_mean_, save_var_, epsilon));
  set_history(flatten_tensor_args( result0, result1, result2 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor & VariableType::mm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const {
  profiler::RecordFunction profiler("mm_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& mat2_ = unpack(mat2, "mat2", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, mat2 )) {
    throw_error_out_requires_grad("mm");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("mm");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mm_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->mm_out(result_, self_, mat2_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::multi_margin_loss(const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) const {
  profiler::RecordFunction profiler("multi_margin_loss", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  auto weight_ = unpack_opt(weight, "weight", 4);
  check_no_requires_grad(target, "target");
  check_no_requires_grad(weight, "weight");
  std::shared_ptr<MultiMarginLossBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MultiMarginLossBackward>(new MultiMarginLossBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->p = p;
    grad_fn->margin = margin;
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->reduction = reduction;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::multi_margin_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "margin", margin);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->multi_margin_loss(self_, target_, p, margin, weight_, reduction));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::multi_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) const {
  profiler::RecordFunction profiler("multi_margin_loss_backward_out", Function::peek_at_next_sequence_nr());
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("multi_margin_loss_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::multi_margin_loss_backward_out(grad_input, grad_output, self, target, p, margin, weight, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::mv(const Tensor & self, const Tensor & vec) const {
  profiler::RecordFunction profiler("mv", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& vec_ = unpack(vec, "vec", 1);
  std::shared_ptr<MvBackward> grad_fn;
  if (compute_requires_grad( self, vec )) {
    grad_fn = std::shared_ptr<MvBackward>(new MvBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, vec ));
    grad_fn->vec_ = SavedVariable(vec, false);
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mv");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "vec", vec);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->mv(self_, vec_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::native_resize_as_(Tensor & self, const Tensor & the_template) const {
  profiler::RecordFunction profiler("native_resize_as_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& the_template_ = unpack(the_template, "the_template", 1);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, the_template )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("native_resize_as_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, the_template ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::native_resize_as");
    } else {
      op_name = jit::Symbol::fromQualString("aten::native_resize_as_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "the_template", the_template);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("native_resize_as_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->native_resize_as_(self_, the_template_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
std::tuple<Tensor,Tensor> VariableType::nll_loss2d_forward(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) const {
  profiler::RecordFunction profiler("nll_loss2d_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  auto weight_ = unpack_opt(weight, "weight", 2);
  check_no_requires_grad(target, "target");
  check_no_requires_grad(weight, "weight");
  std::shared_ptr<NllLoss2DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NllLoss2DBackward>(new NllLoss2DBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->reduction = reduction;
    grad_fn->ignore_index = ignore_index;
  }
  Tensor output;
  Tensor total_weight;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::nll_loss2d_forward");
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
  std::tie(output, total_weight) = as_variable(baseType->nll_loss2d_forward(self_, target_, weight_, reduction, ignore_index));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, total_weight);
  }
  if (grad_fn) {
    grad_fn->total_weight_ = SavedVariable(total_weight, true);
  }
  return std::make_tuple(std::move(output), std::move(total_weight));
}
std::tuple<Tensor,Tensor> VariableType::nll_loss_forward(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) const {
  profiler::RecordFunction profiler("nll_loss_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  auto weight_ = unpack_opt(weight, "weight", 2);
  check_no_requires_grad(target, "target");
  check_no_requires_grad(weight, "weight");
  std::shared_ptr<NllLossBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NllLossBackward>(new NllLossBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->reduction = reduction;
    grad_fn->ignore_index = ignore_index;
  }
  Tensor output;
  Tensor total_weight;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::nll_loss_forward");
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
  std::tie(output, total_weight) = as_variable(baseType->nll_loss_forward(self_, target_, weight_, reduction, ignore_index));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, total_weight);
  }
  if (grad_fn) {
    grad_fn->total_weight_ = SavedVariable(total_weight, true);
  }
  return std::make_tuple(std::move(output), std::move(total_weight));
}
Tensor VariableType::norm_except_dim(const Tensor & v, int64_t pow, int64_t dim) const {
  profiler::RecordFunction profiler("norm_except_dim", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::norm_except_dim");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "v", v);
    jit::tracer::addInputs(node, "pow", pow);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::norm_except_dim(v, pow, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::pin_memory(const Tensor & self) const {
  profiler::RecordFunction profiler("pin_memory", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::pin_memory");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::pin_memory(self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::polygamma(int64_t n, const Tensor & self) const {
  profiler::RecordFunction profiler("polygamma", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<PolygammaBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<PolygammaBackward>(new PolygammaBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->n = n;
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::polygamma");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->polygamma(n, self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::polygamma_(Tensor & self, int64_t n) const {
  profiler::RecordFunction profiler("polygamma_", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::polygamma");
    } else {
      op_name = jit::Symbol::fromQualString("aten::polygamma_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("polygamma_", self);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::polygamma_(self, n);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::potri_out(Tensor & result, const Tensor & self, bool upper) const {
  profiler::RecordFunction profiler("potri_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("potri");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("potri");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("potri_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->potri_out(result_, self_, upper);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::potrs_out(Tensor & result, const Tensor & self, const Tensor & input2, bool upper) const {
  profiler::RecordFunction profiler("potrs_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& input2_ = unpack(input2, "input2", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, input2 )) {
    throw_error_out_requires_grad("potrs");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("potrs");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("potrs_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->potrs_out(result_, self_, input2_, upper);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::pow(const Tensor & self, Scalar exponent) const {
  profiler::RecordFunction profiler("pow", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<PowBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<PowBackward0>(new PowBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->exponent = exponent;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::pow");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "exponent", exponent);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->pow(self_, exponent));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::pow(const Tensor & self, const Tensor & exponent) const {
  profiler::RecordFunction profiler("pow", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& exponent_ = unpack(exponent, "exponent", 1);
  std::shared_ptr<PowBackward1> grad_fn;
  if (compute_requires_grad( self, exponent )) {
    grad_fn = std::shared_ptr<PowBackward1>(new PowBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, exponent ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->exponent_ = SavedVariable(exponent, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::pow");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "exponent", exponent);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->pow(self_, exponent_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::pow(Scalar self, const Tensor & exponent) const {
  profiler::RecordFunction profiler("pow", Function::peek_at_next_sequence_nr());
  auto& exponent_ = unpack(exponent, "exponent", 1);
  std::shared_ptr<PowBackward2> grad_fn;
  if (compute_requires_grad( exponent )) {
    grad_fn = std::shared_ptr<PowBackward2>(new PowBackward2(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( exponent ));
    grad_fn->self = self;
    grad_fn->exponent_ = SavedVariable(exponent, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::pow");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "exponent", exponent);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->pow(self, exponent_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::pow_(Tensor & self, Scalar exponent) const {
  profiler::RecordFunction profiler("pow_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<PowBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<PowBackward0>(new PowBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
    grad_fn->exponent = exponent;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::pow");
    } else {
      op_name = jit::Symbol::fromQualString("aten::pow_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "exponent", exponent);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("pow_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->pow_(self_, exponent);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::pow_(Tensor & self, const Tensor & exponent) const {
  profiler::RecordFunction profiler("pow_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& exponent_ = unpack(exponent, "exponent", 1);
  check_inplace(self);
  std::shared_ptr<PowBackward1> grad_fn;
  if (compute_requires_grad( self, exponent )) {
    grad_fn = std::shared_ptr<PowBackward1>(new PowBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, exponent ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
    grad_fn->exponent_ = SavedVariable(exponent, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::pow");
    } else {
      op_name = jit::Symbol::fromQualString("aten::pow_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "exponent", exponent);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("pow_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->pow_(self_, exponent_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::prelu(const Tensor & self, const Tensor & weight) const {
  profiler::RecordFunction profiler("prelu", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  std::shared_ptr<PreluBackward> grad_fn;
  if (compute_requires_grad( self, weight )) {
    grad_fn = std::shared_ptr<PreluBackward>(new PreluBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, weight ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::prelu");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->prelu(self_, weight_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::prod(const Tensor & self, ScalarType dtype) const {
  profiler::RecordFunction profiler("prod", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ProdBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ProdBackward1>(new ProdBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::prod");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->prod(self_, dtype));
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
Tensor VariableType::prod(const Tensor & self) const {
  profiler::RecordFunction profiler("prod", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ProdBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ProdBackward0>(new ProdBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::prod");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->prod(self_));
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
Tensor VariableType::prod(const Tensor & self, int64_t dim, bool keepdim, ScalarType dtype) const {
  profiler::RecordFunction profiler("prod", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ProdBackward4> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ProdBackward4>(new ProdBackward4(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim;
    grad_fn->keepdim = keepdim;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::prod");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->prod(self_, dim, keepdim, dtype));
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
Tensor VariableType::prod(const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("prod", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ProdBackward2> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ProdBackward2>(new ProdBackward2(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim;
    grad_fn->keepdim = keepdim;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::prod");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->prod(self_, dim, keepdim));
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
Tensor VariableType::prod(const Tensor & self, int64_t dim, ScalarType dtype) const {
  profiler::RecordFunction profiler("prod", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ProdBackward3> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ProdBackward3>(new ProdBackward3(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::prod");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->prod(self_, dim, dtype));
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
std::tuple<Tensor,Tensor> VariableType::qr(const Tensor & self) const {
  profiler::RecordFunction profiler("qr", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<QrBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<QrBackward>(new QrBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::qr");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = as_variable(baseType->qr(self_));
  set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor VariableType::rand_like(const Tensor & self) const {
  profiler::RecordFunction profiler("rand_like", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rand_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::rand_like(self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::randn_like(const Tensor & self) const {
  profiler::RecordFunction profiler("randn_like", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randn_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::randn_like(self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::reflection_pad2d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("reflection_pad2d_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<ReflectionPad2DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<ReflectionPad2DBackwardBackward>(new ReflectionPad2DBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->padding = padding.vec();
    grad_fn->self_info = self;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::reflection_pad2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->reflection_pad2d_backward(grad_output_, self_, padding));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::replication_pad1d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("replication_pad1d_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<ReplicationPad1DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<ReplicationPad1DBackwardBackward>(new ReplicationPad1DBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->padding = padding.vec();
    grad_fn->self_info = self;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::replication_pad1d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->replication_pad1d_backward(grad_output_, self_, padding));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::replication_pad2d(const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("replication_pad2d", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ReplicationPad2DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ReplicationPad2DBackward>(new ReplicationPad2DBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->padding = padding.vec();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::replication_pad2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->replication_pad2d(self_, padding));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::replication_pad2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("replication_pad2d_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("replication_pad2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("replication_pad2d_backward");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("replication_pad2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->replication_pad2d_backward_out(grad_input_, grad_output_, self_, padding);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::replication_pad3d_out(Tensor & output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("replication_pad3d_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("replication_pad3d");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("replication_pad3d");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("replication_pad3d_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->replication_pad3d_out(output_, self_, padding);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::rnn_tanh_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh) const {
  profiler::RecordFunction profiler("rnn_tanh_cell", Function::peek_at_next_sequence_nr());
  auto result = TypeDefault::rnn_tanh_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
  return result;
}
Tensor VariableType::round(const Tensor & self) const {
  profiler::RecordFunction profiler("round", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<RoundBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<RoundBackward>(new RoundBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::round");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->round(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::round_(Tensor & self) const {
  profiler::RecordFunction profiler("round_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<RoundBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<RoundBackward>(new RoundBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::round");
    } else {
      op_name = jit::Symbol::fromQualString("aten::round_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("round_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->round_(self_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::rsqrt(const Tensor & self) const {
  profiler::RecordFunction profiler("rsqrt", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<RsqrtBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<RsqrtBackward>(new RsqrtBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rsqrt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->rsqrt(self_));
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
Tensor & VariableType::rsqrt_(Tensor & self) const {
  profiler::RecordFunction profiler("rsqrt_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<RsqrtBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<RsqrtBackward>(new RsqrtBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::rsqrt");
    } else {
      op_name = jit::Symbol::fromQualString("aten::rsqrt_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("rsqrt_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->rsqrt_(self_);
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
Tensor VariableType::s_native_addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("s_native_addmm", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& mat1_ = unpack(mat1, "mat1", 1);
  auto& mat2_ = unpack(mat2, "mat2", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, mat1, mat2 )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("s_native_addmm"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, mat1, mat2 ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::s_native_addmm");
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
  auto result = as_variable(baseType->s_native_addmm(self_, mat1_, mat2_, beta, alpha));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::s_native_addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("s_native_addmm_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& mat1_ = unpack(mat1, "mat1", 1);
  auto& mat2_ = unpack(mat2, "mat2", 2);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, mat1, mat2 )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("s_native_addmm_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, mat1, mat2 ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::s_native_addmm");
    } else {
      op_name = jit::Symbol::fromQualString("aten::s_native_addmm_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat1", mat1);
    jit::tracer::addInputs(node, "mat2", mat2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("s_native_addmm_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s_native_addmm_(self_, mat1_, mat2_, beta, alpha);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::set_(Tensor & self, Storage source) const {
  profiler::RecordFunction profiler("set_", Function::peek_at_next_sequence_nr());
  TypeDefault::set_(self, source);
  return self;
}
Tensor & VariableType::set_(Tensor & self, Storage source, int64_t storage_offset, IntList size, IntList stride) const {
  profiler::RecordFunction profiler("set_", Function::peek_at_next_sequence_nr());
  TypeDefault::set_(self, source, storage_offset, size, stride);
  return self;
}
Tensor & VariableType::set_(Tensor & self, const Tensor & source) const {
  profiler::RecordFunction profiler("set_", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::set");
    } else {
      op_name = jit::Symbol::fromQualString("aten::set_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "source", source);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("set_", self);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::set_(self, source);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::set_(Tensor & self) const {
  profiler::RecordFunction profiler("set_", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::set");
    } else {
      op_name = jit::Symbol::fromQualString("aten::set_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("set_", self);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::set_(self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::sigmoid_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("sigmoid_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("sigmoid");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("sigmoid");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sigmoid_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->sigmoid_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::sign_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("sign_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("sign");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("sign");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sign_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->sign_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::smooth_l1_loss(const Tensor & self, const Tensor & target, int64_t reduction) const {
  profiler::RecordFunction profiler("smooth_l1_loss", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  check_no_requires_grad(target, "target");
  std::shared_ptr<SmoothL1LossBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SmoothL1LossBackward>(new SmoothL1LossBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->reduction = reduction;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::smooth_l1_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->smooth_l1_loss(self_, target_, reduction));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::smooth_l1_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) const {
  profiler::RecordFunction profiler("smooth_l1_loss_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& target_ = unpack(target, "target", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, target )) {
    throw_error_out_requires_grad("smooth_l1_loss_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("smooth_l1_loss_backward");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("smooth_l1_loss_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->smooth_l1_loss_backward_out(grad_input_, grad_output_, self_, target_, reduction);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::soft_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) const {
  profiler::RecordFunction profiler("soft_margin_loss_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  check_no_requires_grad(target, "target");
  std::shared_ptr<SoftMarginLossBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<SoftMarginLossBackwardBackward>(new SoftMarginLossBackwardBackward(), deleteFunction);
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
    op_name = jit::Symbol::fromQualString("aten::soft_margin_loss_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->soft_margin_loss_backward(grad_output_, self_, target_, reduction));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::softplus_backward(const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output) const {
  profiler::RecordFunction profiler("softplus_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& output_ = unpack(output, "output", 4);
  std::shared_ptr<SoftplusBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<SoftplusBackwardBackward>(new SoftplusBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->beta = beta;
    grad_fn->threshold = threshold;
    grad_fn->output_ = SavedVariable(output, false);
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::softplus_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "threshold", threshold);
    jit::tracer::addInputs(node, "output", output);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->softplus_backward(grad_output_, self_, beta, threshold, output_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::sparse_mask(const Tensor & self, SparseTensorRef mask) const {
  profiler::RecordFunction profiler("sparse_mask", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto mask_ = unpack(mask, "mask", 1);
  std::shared_ptr<SparseMaskBackward> grad_fn;
  if (compute_requires_grad( self, mask.tref )) {
    grad_fn = std::shared_ptr<SparseMaskBackward>(new SparseMaskBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, mask.tref ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sparse_mask");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->sparse_mask(self_, mask_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::sparse_resize_(Tensor & self, IntList size, int64_t sparse_dim, int64_t dense_dim) const {
  profiler::RecordFunction profiler("sparse_resize_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("sparse_resize_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::sparse_resize");
    } else {
      op_name = jit::Symbol::fromQualString("aten::sparse_resize_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "sparse_dim", sparse_dim);
    jit::tracer::addInputs(node, "dense_dim", dense_dim);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sparse_resize_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->sparse_resize_(self_, size, sparse_dim, dense_dim);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::sqrt_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("sqrt_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("sqrt");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("sqrt");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sqrt_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->sqrt_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::sspaddmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("sspaddmm", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sspaddmm");
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
  auto result = TypeDefault::sspaddmm(self, mat1, mat2, beta, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::svd_out(Tensor & U, Tensor & S, Tensor & V, const Tensor & self, bool some, bool compute_uv) const {
  profiler::RecordFunction profiler("svd_out", Function::peek_at_next_sequence_nr());
  auto& U_ = unpack(U, "U", 0);
  auto& S_ = unpack(S, "S", 1);
  auto& V_ = unpack(V, "V", 2);
  auto& self_ = unpack(self, "self", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("svd");
  }
  if (compute_requires_grad( U, S, V )) {
    throw_error_out_requires_grad("svd");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::svd");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "S", S);
    jit::tracer::addInputs(node, "V", V);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "some", some);
    jit::tracer::addInputs(node, "compute_uv", compute_uv);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "U", U);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("svd_out", U);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->svd_out(U_, S_, V_, self_, some, compute_uv);
  increment_version(U);
  increment_version(S);
  increment_version(V);
  rebase_history(flatten_tensor_args( U, S, V ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, U);
    jit::tracer::addOutput(node, S);
    jit::tracer::addOutput(node, V);
  }
  return std::forward_as_tuple(U, S, V);
}
Tensor & VariableType::tan_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("tan_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("tan");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("tan");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("tan_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->tan_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::tanh(const Tensor & self) const {
  profiler::RecordFunction profiler("tanh", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<TanhBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<TanhBackward>(new TanhBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::tanh");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->tanh(self_));
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
Tensor & VariableType::tanh_(Tensor & self) const {
  profiler::RecordFunction profiler("tanh_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<TanhBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<TanhBackward>(new TanhBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::tanh");
    } else {
      op_name = jit::Symbol::fromQualString("aten::tanh_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("tanh_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->tanh_(self_);
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
Tensor & VariableType::tanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output) const {
  profiler::RecordFunction profiler("tanh_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& output_ = unpack(output, "output", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, output )) {
    throw_error_out_requires_grad("tanh_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("tanh_backward");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("tanh_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->tanh_backward_out(grad_input_, grad_output_, output_);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::thnn_col2im(const Tensor & self, IntList output_size, IntList kernel_size, IntList dilation, IntList padding, IntList stride) const {
  profiler::RecordFunction profiler("thnn_col2im", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ThnnCol2ImBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ThnnCol2ImBackward>(new ThnnCol2ImBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->dilation = dilation.vec();
    grad_fn->padding = padding.vec();
    grad_fn->stride = stride.vec();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_col2im");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->thnn_col2im(self_, output_size, kernel_size, dilation, padding, stride));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::thnn_conv2d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
  profiler::RecordFunction profiler("thnn_conv2d_out", Function::peek_at_next_sequence_nr());
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("thnn_conv2d_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::thnn_conv2d_out(output, self, weight, kernel_size, bias, stride, padding);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
std::tuple<Tensor,Tensor> VariableType::thnn_conv_depthwise2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, std::array<bool,2> output_mask) const {
  profiler::RecordFunction profiler("thnn_conv_depthwise2d_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  std::shared_ptr<ThnnConvDepthwise2DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self, weight )) {
    grad_fn = std::shared_ptr<ThnnConvDepthwise2DBackwardBackward>(new ThnnConvDepthwise2DBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, weight ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_argsize_1 = self.size(1);
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->dilation = dilation.vec();
  }
  Tensor grad_input;
  Tensor grad_weight;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_conv_depthwise2d_backward");
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
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(grad_input, grad_weight) = as_variable(baseType->thnn_conv_depthwise2d_backward(grad_output_, self_, weight_, kernel_size, stride, padding, dilation, output_mask));
  set_history(flatten_tensor_args( grad_input, grad_weight ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
    jit::tracer::addOutput(node, grad_weight);
  }
  return std::make_tuple(std::move(grad_input), std::move(grad_weight));
}
Tensor & VariableType::thnn_conv_depthwise2d_forward_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
  profiler::RecordFunction profiler("thnn_conv_depthwise2d_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto bias_ = unpack_opt(bias, "bias", 4);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    throw_error_out_requires_grad("thnn_conv_depthwise2d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("thnn_conv_depthwise2d_forward");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("thnn_conv_depthwise2d_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->thnn_conv_depthwise2d_forward_out(output_, self_, weight_, kernel_size, bias_, stride, padding, dilation);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::thnn_conv_dilated2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
  profiler::RecordFunction profiler("thnn_conv_dilated2d_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 3);
  std::shared_ptr<ThnnConvDilated2DBackward> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::shared_ptr<ThnnConvDilated2DBackward>(new ThnnConvDilated2DBackward(), deleteFunction);
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
    op_name = jit::Symbol::fromQualString("aten::thnn_conv_dilated2d_forward");
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
  std::tie(output, columns, ones) = as_variable(baseType->thnn_conv_dilated2d_forward(self_, weight_, kernel_size, bias_, stride, padding, dilation));
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
std::tuple<Tensor,Tensor,Tensor> VariableType::thnn_conv_dilated3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) const {
  profiler::RecordFunction profiler("thnn_conv_dilated3d_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto& columns_ = unpack(columns, "columns", 7);
  auto& ones_ = unpack(ones, "ones", 8);
  check_no_requires_grad(columns, "columns");
  check_no_requires_grad(ones, "ones");
  std::shared_ptr<ThnnConvDilated3DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self, weight )) {
    grad_fn = std::shared_ptr<ThnnConvDilated3DBackwardBackward>(new ThnnConvDilated3DBackwardBackward(), deleteFunction);
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
    op_name = jit::Symbol::fromQualString("aten::thnn_conv_dilated3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "columns", columns);
    jit::tracer::addInputs(node, "ones", ones);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(grad_input, grad_weight, grad_bias) = as_variable(baseType->thnn_conv_dilated3d_backward(grad_output_, self_, weight_, kernel_size, stride, padding, dilation, columns_, ones_, output_mask));
  set_history(flatten_tensor_args( grad_input, grad_weight, grad_bias ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
    jit::tracer::addOutput(node, grad_weight);
    jit::tracer::addOutput(node, grad_bias);
  }
  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::thnn_conv_dilated3d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
  profiler::RecordFunction profiler("thnn_conv_dilated3d_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& columns_ = unpack(columns, "columns", 1);
  auto& ones_ = unpack(ones, "ones", 2);
  auto& self_ = unpack(self, "self", 3);
  auto& weight_ = unpack(weight, "weight", 4);
  auto bias_ = unpack_opt(bias, "bias", 6);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    throw_error_out_requires_grad("thnn_conv_dilated3d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("thnn_conv_dilated3d_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_conv_dilated3d_forward");
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
    jit::tracer::addInputs(node, "dilation", dilation);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("thnn_conv_dilated3d_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->thnn_conv_dilated3d_forward_out(output_, columns_, ones_, self_, weight_, kernel_size, bias_, stride, padding, dilation);
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
std::tuple<Tensor,Tensor,Tensor> VariableType::thnn_conv_transpose3d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
  profiler::RecordFunction profiler("thnn_conv_transpose3d_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 3);
  std::shared_ptr<ThnnConvTranspose3DBackward> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::shared_ptr<ThnnConvTranspose3DBackward>(new ThnnConvTranspose3DBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->output_padding = output_padding.vec();
    grad_fn->dilation = dilation.vec();
  }
  Tensor output;
  Tensor finput;
  Tensor fgrad_input;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_conv_transpose3d_forward");
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
  std::tie(output, finput, fgrad_input) = as_variable(baseType->thnn_conv_transpose3d_forward(self_, weight_, kernel_size, bias_, stride, padding, output_padding, dilation));
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
Tensor VariableType::thnn_im2col(const Tensor & self, IntList kernel_size, IntList dilation, IntList padding, IntList stride) const {
  profiler::RecordFunction profiler("thnn_im2col", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ThnnIm2ColBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ThnnIm2ColBackward>(new ThnnIm2ColBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_argsize_2 = self.size(2);
    grad_fn->self_argsize_3 = self.size(3);
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->dilation = dilation.vec();
    grad_fn->padding = padding.vec();
    grad_fn->stride = stride.vec();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_im2col");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->thnn_im2col(self_, kernel_size, dilation, padding, stride));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::threshold_out(Tensor & result, const Tensor & self, Scalar threshold, Scalar value) const {
  profiler::RecordFunction profiler("threshold_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("threshold");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("threshold");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("threshold_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->threshold_out(result_, self_, threshold, value);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::to(const Tensor & self, const TensorOptions & options, bool non_blocking, bool copy) const {
  profiler::RecordFunction profiler("to", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::to");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "options", options);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    jit::tracer::addInputs(node, "copy", copy);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::to(self, options, non_blocking, copy);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::to(const Tensor & self, Device device, ScalarType dtype, bool non_blocking, bool copy) const {
  profiler::RecordFunction profiler("to", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::to");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    jit::tracer::addInputs(node, "copy", copy);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::to(self, device, dtype, non_blocking, copy);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::to(const Tensor & self, ScalarType dtype, bool non_blocking, bool copy) const {
  profiler::RecordFunction profiler("to", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::to");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    jit::tracer::addInputs(node, "copy", copy);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::to(self, dtype, non_blocking, copy);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::to(const Tensor & self, const Tensor & other, bool non_blocking, bool copy) const {
  profiler::RecordFunction profiler("to", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::to");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    jit::tracer::addInputs(node, "copy", copy);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::to(self, other, non_blocking, copy);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::trace(const Tensor & self) const {
  profiler::RecordFunction profiler("trace", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<TraceBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<TraceBackward>(new TraceBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::trace");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->trace(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::tril_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
  profiler::RecordFunction profiler("tril_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("tril");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("tril");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("tril_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->tril_out(result_, self_, diagonal);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::triu(const Tensor & self, int64_t diagonal) const {
  profiler::RecordFunction profiler("triu", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<TriuBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<TriuBackward>(new TriuBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->diagonal = diagonal;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::triu");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "diagonal", diagonal);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->triu(self_, diagonal));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::triu_(Tensor & self, int64_t diagonal) const {
  profiler::RecordFunction profiler("triu_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<TriuBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<TriuBackward>(new TriuBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->diagonal = diagonal;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::triu");
    } else {
      op_name = jit::Symbol::fromQualString("aten::triu_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "diagonal", diagonal);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("triu_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->triu_(self_, diagonal);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::type_as(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("type_as", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::type_as");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::type_as(self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::unsqueeze(const Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("unsqueeze", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<UnsqueezeBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<UnsqueezeBackward0>(new UnsqueezeBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->dim = dim;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::unsqueeze");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_view(self, baseType->unsqueeze(self_, dim), true);
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::unsqueeze_(Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("unsqueeze_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<UnsqueezeBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<UnsqueezeBackward1>(new UnsqueezeBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->dim = dim;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::unsqueeze");
    } else {
      op_name = jit::Symbol::fromQualString("aten::unsqueeze_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("unsqueeze_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->unsqueeze_(self_, dim);
  increment_version(self);
  set_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::upsample_bicubic2d(const Tensor & self, IntList output_size, bool align_corners) const {
  profiler::RecordFunction profiler("upsample_bicubic2d", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<UpsampleBicubic2DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<UpsampleBicubic2DBackward>(new UpsampleBicubic2DBackward(), deleteFunction);
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
    op_name = jit::Symbol::fromQualString("aten::upsample_bicubic2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->upsample_bicubic2d(self_, output_size, align_corners));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::upsample_bicubic2d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
  profiler::RecordFunction profiler("upsample_bicubic2d_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output )) {
    throw_error_out_requires_grad("upsample_bicubic2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("upsample_bicubic2d_backward");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_bicubic2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->upsample_bicubic2d_backward_out(grad_input_, grad_output_, output_size, input_size, align_corners);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::upsample_nearest3d_backward(const Tensor & grad_output, IntList output_size, IntList input_size) const {
  profiler::RecordFunction profiler("upsample_nearest3d_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  std::shared_ptr<UpsampleNearest3DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output )) {
    grad_fn = std::shared_ptr<UpsampleNearest3DBackwardBackward>(new UpsampleNearest3DBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
    grad_fn->output_size = output_size.vec();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_nearest3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->upsample_nearest3d_backward(grad_output_, output_size, input_size));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::upsample_trilinear3d(const Tensor & self, IntList output_size, bool align_corners) const {
  profiler::RecordFunction profiler("upsample_trilinear3d", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<UpsampleTrilinear3DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<UpsampleTrilinear3DBackward>(new UpsampleTrilinear3DBackward(), deleteFunction);
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
    op_name = jit::Symbol::fromQualString("aten::upsample_trilinear3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->upsample_trilinear3d(self_, output_size, align_corners));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::upsample_trilinear3d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
  profiler::RecordFunction profiler("upsample_trilinear3d_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output )) {
    throw_error_out_requires_grad("upsample_trilinear3d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("upsample_trilinear3d_backward");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_trilinear3d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->upsample_trilinear3d_backward_out(grad_input_, grad_output_, output_size, input_size, align_corners);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::zeros_out(Tensor & result, IntList size) const {
  profiler::RecordFunction profiler("zeros_out", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::zeros");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "result", result.options());
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("zeros_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::zeros_out(result, size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

}} // namespace torch::autograd
