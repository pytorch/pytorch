#include "torch/csrc/jit/assertions.h"
#include "torch/csrc/jit/script/module.h"
#include "torch/csrc/jit/script/compiler.h"
#include "torch/csrc/jit/script/error_report.h"
#include "torch/csrc/jit/export.h"
#include "torch/csrc/jit/operator.h"

namespace torch { namespace jit { namespace script {


struct RecursiveMethodCallError : public std::exception {};
void placeholderCreator(Method&) {
  throw RecursiveMethodCallError();
}

c10::optional<std::vector<Value*>> try_emit_call_to(
    Graph& graph,
    SourceRange loc,
    Method& callee,
    c10::optional<NamedValue> self,
    ArrayRef<NamedValue> args,
    ArrayRef<NamedValue> kwargs,
    std::stringstream& failure_messages,
    Method* caller,
    bool conv_tensors_to_nums) {
  try {
    callee.ensure_defined();
  } catch (RecursiveMethodCallError&) {
    throw ErrorReport(loc) << " method '" << callee.name()
        << "' is called recursively involving this call site. Recursive calls are not supported";
  }
  auto fn = callee.graph();

  auto matched_schema = tryMatchSchema(
    callee.getSchema(),
    loc, graph, self, args, kwargs, failure_messages, conv_tensors_to_nums);
  if(!matched_schema)
    return c10::nullopt;

  // parameters to callee method (which become parameters to _this_ method
  // if they were not already)
  for(at::Tensor* member : callee.params()) {
    if(!caller) {
      throw ErrorReport(loc) << " attempting to call a method with parameters from a raw graph. File a bug report";
    }
    matched_schema->inputs.push_back(caller->get_or_add_parameter(member));
  }
  return inlineCallTo(graph, *callee.graph(), matched_schema->inputs);
}

std::vector<Value*> Method::emit_call_to(SourceRange loc, Method & callee, ArrayRef<NamedValue> args, ArrayRef<NamedValue> kwargs) {
  JIT_ASSERT(!executor);
  std::stringstream failure_messages;
  if (auto result = try_emit_call_to(
          *graph(),
          loc,
          callee,
          c10::nullopt,
          args,
          kwargs,
          failure_messages,
          this,
          /*conv_tensors_to_nums=*/true)) {
    return *result;
  }
  throw ErrorReport(loc) << failure_messages.str();
}

void Method::ensure_defined() {
  if(method_creator) {
    auto creator = method_creator;
    method_creator = placeholderCreator;
    creator(*this);
    method_creator = nullptr;
  }
}

void Module::to(at::Device device, at::ScalarType dtype, bool non_blocking) {
  to_impl(device, dtype, non_blocking);
}

void Module::to(at::ScalarType dtype, bool non_blocking) {
  to_impl(/*device=*/c10::nullopt, dtype, non_blocking);
}

void Module::to(at::Device device, bool non_blocking) {
  to_impl(device, /*dtype=*/c10::nullopt, non_blocking);
}

void Module::save(std::ostream& out) {
  ExportModule(*this, out);
}

void Module::save(const std::string& filename) {
  ExportModule(*this, filename);
}

void Module::to_impl(
    c10::optional<at::Device> device,
    c10::optional<at::ScalarType> dtype,
    bool non_blocking) {
  // First call `to()` on every child module.
  for (auto& child : modules) {
    child->module->to_impl(device, dtype, non_blocking);
  }
  // Then convert every of our parameters.
  for (auto& parameter : parameters) {
    // Need to access the `at::Tensor` as a `Variable` here.
    autograd::Variable variable = *parameter->slot();
    at::Tensor data = variable.data();
    // Use the data's original device or dtype if not supplied here.
    auto new_data = data.to(
        device.value_or(data.device()),
        dtype.value_or(data.scalar_type()),
        non_blocking);
    variable.set_data(new_data);
  }
}

}}}
