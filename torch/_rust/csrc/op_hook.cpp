#include <torch/_rust/csrc/op_hook.h>
#include <torch/_rust/csrc/tensor_ffi.h>

#include <ATen/core/PythonFallbackKernel.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include "torch_rust/src/bindings/ffi.rs.h"

#include <sstream>
#include <string>

namespace torch::rust {

namespace {

struct HookState {
  int depth = 0;
  bool was_included = false;
  at::impl::PreDispatchHook prev = nullptr;
};

thread_local HookState state;

void format_value(std::ostringstream& os, const c10::IValue& v) {
  if (v.isTensor()) {
    const at::Tensor& t = v.toTensor();
    if (!t.defined()) {
      os << "None";
      return;
    }
    os << dtype_short_name(t.scalar_type()) << t.sym_sizes();
    if (!t.device().is_cpu()) {
      os << "@" << t.device();
    }
  } else if (v.isList()) {
    os << "[";
    const char* sep = "";
    for (const c10::IValue& e : v.toListRef()) {
      os << sep;
      format_value(os, e);
      sep = ", ";
    }
    os << "]";
  } else {
    os << v;
  }
}

void hook(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet keys,
    torch::jit::Stack* stack) {
  const c10::FunctionSchema& schema = op.schema();
  const auto& args = schema.arguments();
  std::ostringstream os;
  const char* sep = "";
  auto first = stack->end() - static_cast<std::ptrdiff_t>(args.size());
  for (size_t i = 0; i < args.size(); i++) {
    os << sep;
    if (args[i].kwarg_only()) {
      os << args[i].name() << "=";
    }
    format_value(os, first[static_cast<std::ptrdiff_t>(i)]);
    sep = ", ";
  }
  std::string name = schema.name();
  if (!schema.overload_name().empty()) {
    name += "." + schema.overload_name();
  }
  // Ops run by the callback or inside this op aren't reported.
  c10::impl::ExcludeDispatchKeyGuard nested(c10::DispatchKey::PreDispatch);
  on_torch_op(name, os.str());
  op.redispatchBoxed(keys, stack);
}

} // namespace

void op_hook_enable() {
  if (state.depth++ == 0) {
    state.was_included =
        c10::impl::tls_is_dispatch_key_included(c10::DispatchKey::PreDispatch);
    c10::impl::tls_set_dispatch_key_included(
        c10::DispatchKey::PreDispatch, true);
    state.prev = at::impl::set_pre_dispatch_hook(&hook);
  }
}

void op_hook_pause(bool paused) {
  if (state.depth > 0) {
    c10::impl::tls_set_dispatch_key_included(
        c10::DispatchKey::PreDispatch, paused ? state.was_included : true);
  }
}

void op_hook_disable() {
  if (--state.depth == 0) {
    at::impl::set_pre_dispatch_hook(state.prev);
    c10::impl::tls_set_dispatch_key_included(
        c10::DispatchKey::PreDispatch, state.was_included);
  }
}

} // namespace torch::rust
