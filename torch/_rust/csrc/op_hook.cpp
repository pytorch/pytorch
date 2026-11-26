#include <torch/_rust/csrc/op_hook.h>
#include <torch/_rust/csrc/tensor_ffi.h>

#include <ATen/core/PythonFallbackKernel.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include "torch_rust/src/bindings/ffi.rs.h"

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace torch::rust {

namespace {

struct HookState {
  // Enables of the text path, and whether it wants ops now (not paused).
  int depth = 0;
  bool text = false;
  // Whether the witness wants ops now.
  bool witness = false;
  // Reported ops in flight on this thread.
  std::uint32_t below = 0;
  bool installed = false;
  bool was_included = false;
  at::impl::PreDispatchHook prev = nullptr;
};

thread_local HookState state;

void hook(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet keys,
    torch::jit::Stack* stack);

// What on_witness_op is told of an op; src/exec/calls/witness.rs has the same
// values.
constexpr std::uint32_t kUnreadable = 1; // no schema, or an argument or a written value it does not account for
constexpr std::uint32_t kDraws = 2; // tagged nondeterministic_seeded
constexpr std::uint32_t kReadsValue = 4; // tagged data_dependent_output
constexpr std::uint32_t kValueOut = 8; // a result that is no tensor, list of tensors or optional tensor

void identify(const at::Tensor& t, TensorIdent& out) {
  out.tensor = 0;
  out.storage = 0;
  if (t.defined()) {
    const c10::TensorImpl* impl = t.unsafeGetTensorImpl();
    out.tensor = reinterpret_cast<std::uintptr_t>(impl);
    out.storage = reinterpret_cast<std::uintptr_t>(
        impl->unsafe_storage().unsafeGetStorageImpl());
  }
}

bool tensors_only(const c10::Type& type) {
  switch (type.kind()) {
    case c10::TypeKind::TensorType:
      return true;
    case c10::TypeKind::ListType:
    case c10::TypeKind::OptionalType:
      return tensors_only(*type.containedType(0));
    default:
      return false;
  }
}

void report(
    const c10::OperatorHandle& op,
    const torch::jit::Stack& stack,
    std::uint32_t depth) {
  if (!op.hasSchema()) {
    const c10::OperatorName& name = op.operator_name();
    on_witness_op(name.name, name.overload_name, depth, kUnreadable, {});
    return;
  }
  const c10::FunctionSchema& schema = op.schema();
  const auto& args = schema.arguments();
  std::uint32_t flags = 0;
  std::vector<WitnessWrite> wrote;
  auto write = [&](size_t arg, const at::Tensor& t) {
    WitnessWrite w;
    w.arg = static_cast<std::uint32_t>(arg);
    identify(t, w.id);
    wrote.push_back(w);
  };
  if (schema.is_vararg() || stack.size() < args.size()) {
    flags |= kUnreadable;
  } else {
    auto first = stack.end() - static_cast<std::ptrdiff_t>(args.size());
    for (size_t i = 0; i < args.size(); i++) {
      const c10::AliasInfo* alias = args[i].alias_info();
      if (!alias || !alias->isWrite()) {
        continue;
      }
      const c10::IValue& v = first[static_cast<std::ptrdiff_t>(i)];
      if (v.isTensor()) {
        write(i, v.toTensor());
      } else if (v.isList()) {
        for (const c10::IValue& e : v.toListRef()) {
          if (e.isTensor()) {
            write(i, e.toTensor());
          } else if (!e.isNone()) {
            flags |= kUnreadable;
          }
        }
      } else if (!v.isNone()) {
        // None: an optional argument that was not given is not written.
        flags |= kUnreadable;
      }
    }
  }
  if (op.hasTag(at::Tag::nondeterministic_seeded)) {
    flags |= kDraws;
  }
  if (op.hasTag(at::Tag::data_dependent_output)) {
    flags |= kReadsValue;
  }
  for (const c10::Argument& r : schema.returns()) {
    if (!tensors_only(*r.type())) {
      flags |= kValueOut;
    }
  }
  on_witness_op(
      schema.name(),
      schema.overload_name(),
      depth,
      flags,
      ::rust::Slice<const WitnessWrite>(wrote.data(), wrote.size()));
}

// Installs the hook while anybody is enabled and includes the key while
// anybody wants ops.
void sync() {
  const bool on = state.depth > 0 || state.witness;
  if (on && !state.installed) {
    state.was_included =
        c10::impl::tls_is_dispatch_key_included(c10::DispatchKey::PreDispatch);
    state.prev = at::impl::set_pre_dispatch_hook(&hook);
  }
  if (on || state.installed) {
    c10::impl::tls_set_dispatch_key_included(
        c10::DispatchKey::PreDispatch,
        (on && (state.text || state.witness)) || state.was_included);
  }
  if (!on && state.installed) {
    at::impl::set_pre_dispatch_hook(state.prev);
  }
  state.installed = on;
}

void format_value(std::ostringstream& os, const c10::IValue& v) {
  if (v.isTensor()) {
    const at::Tensor& t = v.toTensor();
    if (!t.defined()) {
      os << "None";
      return;
    }
    os << dtype_short_name(t.scalar_type());
    // A nested tensor has no sizes: asking for them raises.
    if (t.is_nested()) {
      os << "[nested]";
    } else {
      os << t.sym_sizes();
    }
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
  const std::uint32_t depth = state.below;
  if (state.witness) {
    report(op, *stack, depth);
  }
  if (depth == 0 && state.text) {
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
    // Ops run by the callback aren't reported.
    c10::impl::ExcludeDispatchKeyGuard callback(c10::DispatchKey::PreDispatch);
    on_torch_op(name, os.str());
  }
  struct Below {
    Below() {
      state.below++;
    }
    ~Below() {
      state.below--;
    }
  } below;
  if (state.witness) {
    // The key stays included: an op inside this one comes back here, one
    // level down, and continues at the same kernel (`keys` holds nothing at
    // or above PreDispatch).
    op.redispatchBoxed(keys, stack);
  } else {
    // Ops inside this op aren't reported.
    c10::impl::ExcludeDispatchKeyGuard nested(c10::DispatchKey::PreDispatch);
    op.redispatchBoxed(keys, stack);
  }
}

} // namespace

void op_hook_enable() {
  if (state.depth++ == 0) {
    state.text = true;
  }
  sync();
}

void op_hook_pause(bool paused) {
  if (state.depth > 0) {
    state.text = !paused;
    sync();
  }
}

void op_hook_disable() {
  if (--state.depth == 0) {
    state.text = false;
  }
  sync();
}

bool op_hook_witness(bool wanted) {
  state.witness = wanted;
  sync();
  return op_hook_seeing();
}

bool op_hook_seeing() {
  return state.installed &&
      c10::impl::tls_is_dispatch_key_included(c10::DispatchKey::PreDispatch) &&
      !c10::impl::tls_is_dispatch_key_excluded(c10::DispatchKey::PreDispatch);
}

std::uint32_t op_hook_depth() {
  return state.below;
}

void tensor_ident(std::uintptr_t py_obj, TensorIdent& out) noexcept {
  out.tensor = 0;
  out.storage = 0;
  if (const Tensor* t = tensor_by_type(py_obj)) {
    identify(*t, out);
  }
}

} // namespace torch::rust
