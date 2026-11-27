#include <torch/_rust/csrc/op_hook.h>
#include <torch/_rust/csrc/tensor_ffi.h>

#include <ATen/core/PythonFallbackKernel.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/util/SmallVector.h>
#include "torch_rust/src/bindings/ffi.rs.h"

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace torch::rust {

namespace {

// A storage that an op made, or that an op's schema says it wrote, while a
// witness was open.
struct Known {
  Known(const c10::Storage& storage, std::uint64_t made, std::uint64_t wrote)
      : hold(storage.getWeakStorageImpl()), made(made), wrote(wrote) {}
  // Keeps the address from being given to another storage while it is in the
  // table; the memory itself goes when the program lets go of it.
  c10::weak_intrusive_ptr<c10::StorageImpl> hold;
  // The serial of the op that returned it as a new tensor (0: none has), and
  // of the first op to write it before that.
  std::uint64_t made;
  std::uint64_t wrote;
};

constexpr std::size_t kKnownLimit = 8;

struct HookState {
  // Enables of the text path, and whether it wants ops now (not paused).
  int depth = 0;
  bool text = false;
  // Whether the witness wants ops now, and whether one is open (paused or not).
  bool witness = false;
  bool open = false;
  // Reported ops in flight on this thread, the count of ops reported so far
  // (an op's serial), and the serial of the innermost one in flight (0: none).
  std::uint32_t below = 0;
  std::uint64_t serial = 0;
  std::uint64_t innermost = 0;
  // When the table of known storages drops its dead ones.
  std::size_t known_limit = kKnownLimit;
  bool installed = false;
  bool was_included = false;
  at::impl::PreDispatchHook prev = nullptr;
};

thread_local HookState state;
// By the storage's address. Apart from `state`, which stays a plain struct:
// every op reads that.
thread_local std::unordered_map<std::uintptr_t, Known> known_storages;

void hook(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet keys,
    torch::jit::Stack* stack);

constexpr std::uint32_t bit(WitnessFlag flag) {
  return static_cast<std::uint32_t>(flag);
}

using Storages = c10::SmallVector<std::uintptr_t, 8>;

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

// Adds the storages a value of an op's stack holds. False: a kind of value it
// does not know, or a tensor that names no storage (sparse, mkldnn, a
// wrapper), so memory may be reached through it that `out` does not have.
bool storages(const c10::IValue& v, Storages& out) {
  if (v.isTensor()) {
    const at::Tensor& t = v.toTensor();
    if (!t.defined()) {
      return true;
    }
    TensorIdent id;
    identify(t, id);
    out.push_back(id.storage);
    return id.storage != 0;
  }
  if (v.isStorage()) {
    out.push_back(reinterpret_cast<std::uintptr_t>(
        v.toStorage().unsafeGetStorageImpl()));
    return true;
  }
  if (v.isList()) {
    for (const c10::IValue& e : v.toListRef()) {
      if (!storages(e, out)) {
        return false;
      }
    }
    return true;
  }
  return v.isNone() || v.isInt() || v.isDouble() || v.isBool() ||
      v.isComplexDouble() || v.isString() || v.isDevice() || v.isGenerator() ||
      v.isSymInt() || v.isSymFloat() || v.isSymBool();
}

// The table's entry for a storage, new with (made, wrote) if it has none.
Known& known(
    const c10::Storage& storage,
    std::uint64_t made,
    std::uint64_t wrote) {
  auto& table = known_storages;
  if (table.size() >= state.known_limit) {
    std::erase_if(table, [](const auto& entry) {
      return entry.second.hold.expired();
    });
    state.known_limit = std::max(kKnownLimit, 2 * table.size());
  }
  const auto addr =
      reinterpret_cast<std::uintptr_t>(storage.unsafeGetStorageImpl());
  return table.try_emplace(addr, storage, made, wrote).first->second;
}

// Tells the witness of an op that starts. True when it could name every
// storage its arguments hold (`held`), so that a result on none of them is
// one the op made.
bool report(
    const c10::OperatorHandle& op,
    const torch::jit::Stack& stack,
    std::uint32_t depth,
    std::uint64_t serial,
    Storages& held) {
  if (!op.hasSchema()) {
    const c10::OperatorName& name = op.operator_name();
    on_witness_op(
        name.name,
        name.overload_name,
        depth,
        serial,
        bit(WitnessFlag::Unreadable),
        {});
    return false;
  }
  const c10::FunctionSchema& schema = op.schema();
  const auto& args = schema.arguments();
  std::uint32_t flags = 0;
  std::vector<WitnessWrite> wrote;
  auto write = [&](size_t arg, const at::Tensor& t) {
    // An undefined tensor has no memory to write.
    if (!t.defined()) {
      return;
    }
    WitnessWrite w;
    w.arg = static_cast<std::uint32_t>(arg);
    identify(t, w.id);
    w.made = w.id.storage == 0
        ? 0
        : known(t.unsafeGetTensorImpl()->unsafe_storage(), 0, serial).made;
    wrote.push_back(w);
  };
  // from_file maps a file: memory that outlives the call.
  bool judged = schema.name() != "aten::from_file";
  if (schema.is_vararg() || stack.size() < args.size()) {
    flags |= bit(WitnessFlag::Unreadable);
    judged = false;
  } else {
    auto first = stack.end() - static_cast<std::ptrdiff_t>(args.size());
    for (size_t i = 0; i < args.size(); i++) {
      const c10::IValue& v = first[static_cast<std::ptrdiff_t>(i)];
      judged = judged && storages(v, held);
      const c10::AliasInfo* alias = args[i].alias_info();
      if (!alias || !alias->isWrite()) {
        continue;
      }
      if (v.isTensor()) {
        write(i, v.toTensor());
      } else if (v.isList()) {
        for (const c10::IValue& e : v.toListRef()) {
          if (e.isTensor()) {
            write(i, e.toTensor());
          } else if (!e.isNone()) {
            flags |= bit(WitnessFlag::Unreadable);
          }
        }
      } else if (!v.isNone()) {
        // None: an optional argument that was not given is not written.
        flags |= bit(WitnessFlag::Unreadable);
      }
    }
  }
  if (op.hasTag(at::Tag::nondeterministic_seeded)) {
    flags |= bit(WitnessFlag::Draws);
  }
  if (op.hasTag(at::Tag::data_dependent_output)) {
    flags |= bit(WitnessFlag::ReadsValue);
  }
  if (op.hasTag(at::Tag::dynamic_output_shape)) {
    flags |= bit(WitnessFlag::DataSized);
  }
  for (const c10::Argument& r : schema.returns()) {
    if (!tensors_only(*r.type())) {
      flags |= bit(WitnessFlag::ValueOut);
    }
  }
  on_witness_op(
      schema.name(),
      schema.overload_name(),
      depth,
      serial,
      flags,
      ::rust::Slice<const WitnessWrite>(wrote.data(), wrote.size()));
  return judged;
}

// Behind an op whose arguments' storages are `held`: a result that is a new
// tensor (nobody else holds it: not one kept in an argument's state, as a
// tangent or a quantizer's scales are) on a storage none of the arguments
// holds was made by the op. A kernel may have written it before through ops
// of its own (a reduction fills its result): those writes, all inside this
// op, were to what it made.
void note_made(
    const c10::IValue& v,
    const Storages& held,
    std::uint64_t serial) {
  if (v.isList()) {
    for (const c10::IValue& e : v.toListRef()) {
      note_made(e, held, serial);
    }
    return;
  }
  if (!v.isTensor()) {
    return;
  }
  const at::Tensor& t = v.toTensor();
  if (!t.defined() || t.use_count() != 1) {
    return;
  }
  const c10::Storage& storage = t.unsafeGetTensorImpl()->unsafe_storage();
  const auto addr =
      reinterpret_cast<std::uintptr_t>(storage.unsafeGetStorageImpl());
  if (addr == 0 || std::find(held.begin(), held.end(), addr) != held.end()) {
    return;
  }
  Known& entry = known(storage, serial, 0);
  if (entry.made == 0 && entry.wrote > serial) {
    entry.made = serial;
    on_witness_made(addr, entry.wrote, serial);
  }
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
  const std::uint64_t serial = ++state.serial;
  Storages held;
  const bool judged = state.witness && report(op, *stack, depth, serial, held);
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
    explicit Below(std::uint64_t serial) : outer(state.innermost) {
      state.below++;
      state.innermost = serial;
    }
    ~Below() {
      state.below--;
      state.innermost = outer;
    }
    std::uint64_t outer;
  } below(serial);
  if (state.witness) {
    // The key stays included: an op inside this one comes back here, one
    // level down, and continues at the same kernel (`keys` holds nothing at
    // or above PreDispatch).
    op.redispatchBoxed(keys, stack);
    const size_t results = judged ? op.schema().returns().size() : 0;
    if (state.open && results <= stack->size()) {
      for (auto v = stack->end() - static_cast<std::ptrdiff_t>(results);
           v != stack->end();
           v++) {
        note_made(*v, held, serial);
      }
    }
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

bool op_hook_witness(bool wanted, bool open) {
  state.witness = wanted;
  state.open = open;
  if (!open) {
    known_storages.clear();
    state.known_limit = kKnownLimit;
  }
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

std::uint64_t op_hook_serial() {
  return state.serial;
}

std::uint64_t op_hook_innermost() {
  return state.innermost;
}

void tensor_ident(std::uintptr_t py_obj, TensorIdent& out) noexcept {
  out.tensor = 0;
  out.storage = 0;
  if (const Tensor* t = tensor_by_type(py_obj)) {
    identify(*t, out);
  }
}

} // namespace torch::rust
