#include <ATen/TracerMode.h>
#include <ATen/core/op_registration/op_registration.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/utils/memory.h>
#include <torch/library.h>

using namespace at;

namespace torch { namespace TraceType {

namespace {

Tensor & copy_(Tensor & self, const Tensor & src, bool non_blocking) {
  jit::Value* output = nullptr;
  if(torch::jit::tracer::isTracing()) {
    const jit::tracer::TracingState& state = *jit::tracer::getTracingState();
    auto& graph = state.graph;
    if (state.force_outplace && self.storage().use_count() <= 1) {
      // if you have no views of self, then an in place copy is equivalent to
      // making sure we expand src to the same size as self
      jit::Node* node = graph->create(jit::aten::expand_as, /*num_outputs=*/1);
      jit::tracer::addInputs(node, "src", src);
      jit::tracer::addInputs(node, "self", self);
      graph->insertNode(node);
      output = node->output();
    } else {
      output = graph->insert(
          jit::aten::copy_,
          {jit::tracer::getValueTrace(self), jit::tracer::getValueTrace(src)});
      jit::tracer::recordSourceLocation(output->node());
    }
    jit::tracer::ensureUniqueIfOutOfPlaced("copy_ (possibly due to an assignment)", self);
  }

  {
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    self.copy_(src, non_blocking);
  }

  if(torch::jit::tracer::isTracing()) {
    jit::tracer::setOutput(output, self);
  }
  return self;
}

Tensor& resize_(
    Tensor& self,
    IntArrayRef size,
    c10::optional<MemoryFormat> optional_memory_format) {
  if (torch::jit::tracer::isTracing()) {
    jit::tracer::ArgumentStash::popIntArrayRef("size");
    jit::tracer::warn("resize_", jit::tracer::WARN_RESIZE);
    jit::tracer::delValueTrace(self);
  }

  {
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    self.resize_(size, std::move(optional_memory_format));
  }
  return self;
}

Tensor& resize_as_(
    Tensor& self,
    const Tensor& the_template,
    c10::optional<MemoryFormat> optional_memory_format) {
  if (torch::jit::tracer::isTracing()) {
    jit::tracer::warn("resize_as_", jit::tracer::WARN_RESIZE);
    jit::tracer::delValueTrace(self);
  }

  {
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    self.resize_as_(the_template, std::move(optional_memory_format));
  }
  return self;
}

Tensor detach(const Tensor & self) {
  torch::jit::Node* node = nullptr;
  if (jit::tracer::isTracing()) {
    auto& graph = jit::tracer::getTracingState()->graph;
    node = graph->create(jit::aten::detach, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    graph->insertNode(node);
  }

  auto result = [&]() {
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    return self.detach();
  }();

  if (jit::tracer::isTracing()) {
    jit::tracer::addOutput(node, result);
  }
  return result;
}

Tensor & detach_(Tensor & self) {
  torch::jit::Node* node = nullptr;
  if (jit::tracer::isTracing()) {
    auto& graph = jit::tracer::getTracingState()->graph;
    node = graph->create(jit::aten::detach, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("detach_", self);
  }

  {
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    self.detach_();
  }

  if (jit::tracer::isTracing()) {
    jit::tracer::addOutput(node, self);
  }
  return self;
}

// Invariant:
// - Ops registered to DispatchKey::Tracer below must be included in `MANUAL_TRACER` in tools/autograd/gen_variable_type.py
TORCH_LIBRARY_IMPL(aten, Tracer, m) {
  m.impl("resize_", resize_);
  m.impl("resize_as_", resize_as_);
  m.impl("detach", TORCH_FN(detach));
  m.impl("detach_", detach_);
  m.impl("copy_", copy_);

  // Skip tracing for the following ops by registering fallthrough kernel explicitly.
  m.impl("_backward", CppFunction::makeFallthrough());
  m.impl("set_data", CppFunction::makeFallthrough());
  m.impl("data", CppFunction::makeFallthrough());
  m.impl("is_leaf", CppFunction::makeFallthrough());
  m.impl("output_nr", CppFunction::makeFallthrough());
  m.impl("_version", CppFunction::makeFallthrough());
  m.impl("requires_grad_", CppFunction::makeFallthrough());
  m.impl("retain_grad", CppFunction::makeFallthrough());
  m.impl("_fw_primal", CppFunction::makeFallthrough());
}

}  // namespace

}} // namespace torch::TraceType
