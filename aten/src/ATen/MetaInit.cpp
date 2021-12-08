// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/MetaInit.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>

#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/flat_hash_map.h>
#include <torch/library.h>

namespace at {
namespace {

// A weak intrusive_target_ptr holder to be used as a key within unordered
// containers (i.e. `unordered_map`).
class WeakRef {
 public:
  explicit WeakRef(intrusive_ptr_target* ptr) noexcept
      : ptr_{ptr} {
    raw::weak_intrusive_ptr::incref(ptr_);
  }

  WeakRef(const WeakRef&) = delete;

  WeakRef& operator=(const WeakRef&) = delete;

  WeakRef(WeakRef&& other) noexcept
      : ptr_{other.ptr_} {
    other.ptr_ = nullptr;
  }

  WeakRef& operator=(WeakRef&& other) noexcept {
    ptr_ = std::exchange(other.ptr_, nullptr);

    return *this;
  }

  ~WeakRef() {
    if (ptr_ != nullptr) {
      raw::weak_intrusive_ptr::decref(ptr_);
    }
  }

  auto unsafe_ptr() const noexcept {
    return ptr_;
  }

 private:
  intrusive_ptr_target* ptr_;
};

} // namespace
} // namespace at

namespace std {

template <>
struct hash<at::WeakRef> {
  size_t operator()(const at::WeakRef& r) {
    return std::hash<at::intrusive_ptr_target*>{}(r.unsafe_ptr());
  }
};

template <>
struct equal_to<at::WeakRef> {
  bool operator()(const at::WeakRef& lhs, const at::WeakRef& rhs) {
    return lhs.unsafe_ptr() < rhs.unsafe_ptr();
  }
};

} // namespace std

namespace at {
namespace {

// NOTE [Foreign and Surrogate Tensors]
//
// A foreign tensor can be two things:
//  - It is either a tensor that was constructed outside of the meta-init
//    context and then later used inside the context.
//  - Or it is a tensor that was constructed from external data via a function
//    such as `torch.tensor()` that cannot be intercepted. Although they are
//    technically constructed inside the meta-init context, we consider such
//    tensors foreign as well.
//
// Since a foreign tensor can have any device type we cannot use it within the
// meta-init context where we expect all tensors to be allocated on the meta
// backend. Therefore, whenever we see an operator argument that was not
// constructed inside the meta-init context, we replace it with a surrogate
// tensor allocated on the meta device and maintain a mapping between them.

class MetaInitState {
 public:
  Tensor getOrCreateSurrogateTensor(const Tensor& tensor);

 private:
  // See the note [Foreign and Surrogate Tensors].
  ska::flat_hash_map<TensorImpl*, Tensor> surrogates_{};

  // We keep a weak reference to all foreign tensors to ensure that their memory
  // addresses do not get recycled even if they get deallocated.
  ska::flat_hash_set<WeakRef> weak_refs_{};
};

thread_local MetaInitState meta_init_state{};

Tensor MetaInitState::getOrCreateSurrogateTensor(const Tensor& tensor) {
  TensorImpl* ptr = tensor.unsafeGetTensorImpl();

  auto pos = surrogates_.find(ptr);

  // If we have seen `tensor` as an argument to a previous operation before,
  // return its cached surrogate since we should preserve its identity.
  if (pos != surrogates_.end()) {
    return pos->second;
  }

  weak_refs_.emplace(ptr);

  // Note that we do not maintain the view relationship of foreign tensors. This
  // means if two foreign tensors share the same storage, we do not reflect this
  // in their surrogates. Since we only deal with meta tensors this is fine.
  Tensor surrogate = empty_like(tensor, device(kMeta));

  surrogates_.emplace(ptr, surrogate);

  return surrogate;
}

using TensorReplacer = std::function<Tensor(const Tensor&)>;

// Replaces the tensors contained in `value` by calling `fn`.
IValue replaceTensors(const IValue& value, const TensorReplacer& fn) {
  bool replaced = false;

  IValue::HashAliasedIValueMap memo{};

  auto visitor = [&replaced, &memo, &fn](const IValue& v) {
    // Deep-copy all compound objects.
    if (v.isTuple() || v.isList() || v.isGenericDict()) {
      return false;
    }

    if (v.isTensor()) {
      const Tensor& inp = v.toTensor();

      Tensor out = fn(inp);
      // If not replaced, we need to put the original value into `memo` so that
      // it does not get cloned by `deepcopy()`.
      if (out.is_same(inp)) {
        memo[v] = v;
      } else {
        memo[v] = out;

        replaced = true;
      }
    } else {
      // Do not clone other non-compound objects.
      memo[v] = v;
    }

    return false;
  };

  value.visit(visitor);

  if (replaced) {
    return value.deepcopy(memo);
  } else {
    return value;
  }
}

void replaceTensorArguments(const OperatorHandle& op, Stack& s, const TensorReplacer& fn) {
  std::size_t num_arguments = op.schema().arguments().size();

  // Perform an in-place update of the tensor arguments in the stack.
  for (std::size_t i = 0; i < num_arguments; i++) {
    IValue& arg = torch::jit::peek(s, i, num_arguments);

    arg = replaceTensors(arg, fn);
  }
}

// Replaces all foreign tensor arguments with surrogate tensors.
void replaceForeignTensorArguments(const OperatorHandle& op, Stack& s) {
  auto replace_foreign_tensor = [](const Tensor& tensor) -> Tensor {
    // If the tensor was not output by an operation that we recorded, we should
    // replace it with a surrogate tensor.
    if (tensor.is_meta()) {
      return tensor;
    } else {
      return meta_init_state.getOrCreateSurrogateTensor(tensor);
    }
  };

  replaceTensorArguments(op, s, replace_foreign_tensor);
}

// Checks whether a `TensorOptions` argument can be parsed from the individual
// arguments of `schema`.
bool hasTensorOptions(const FunctionSchema& schema) noexcept {
  std::array<c10::string_view, 4> tensor_opts = {{
      "dtype", "layout", "device", "pin_memory"
  }};

  const std::vector<Argument>& args = schema.arguments();
  if (args.size() < tensor_opts.size()) {
    return false;
  }

  // Checks if the arguments starting at `arg_pos` represent a `TensorOptions`.
  auto are_tensor_opts = [&tensor_opts](auto arg_pos) noexcept {
    for (const auto& tensor_opt : tensor_opts) {
      if (tensor_opt != arg_pos->name()) {
        return false;
      }
      ++arg_pos;
    }
    return true;
  };

  for (auto pos = args.begin(); pos <= args.end() - tensor_opts.size(); ++pos) {
    if (are_tensor_opts(pos)) {
      return true;
    }
  }
  return false;
}

// TODO: The following function relies on some common heuristics to determine
// whether an operator is a tensor factory. Although in practice it is mostly
// reliable, in long term we need a cleaner solution such as operator tagging.
bool isTensorFactory(const OperatorHandle& op) noexcept {
  const FunctionSchema& schema = op.schema();

  // For an operator to be considered a factory it must return a tensor.
  bool has_tensor_output = std::any_of(
      schema.returns().begin(), schema.returns().end(),
      [](const Argument& return_arg) {
        return return_arg.type() == TensorType::get();
      });

  if (!has_tensor_output) {
    return false;
  }

  // We use a simple heuristic that is similar to Autograd's function generator
  // tool. We check whether the operator has a `TensorOptions` parameter, which
  // means that it is (very likely) a factory.
  return hasTensorOptions(schema);
}

// Replaces the `device` argument in `s` with the meta device if `op` is a
// tensor factory.
void replaceDeviceArgument(const OperatorHandle& op, Stack& s) {
  if (!isTensorFactory(op)) {
    return;
  }

  // It is safe to access the index right away skipping the optional check since
  // we know for sure that a factory has a `device` argument.
  int device_idx = op.schema().argumentIndexWithName("device").value();

  IValue& device = torch::jit::peek(s, device_idx, op.schema().arguments().size());

  // This condition should always be true, but let's be pedantic.
  if (device.isDevice()) {
    device = Device{kMeta};
  }
}

constexpr DispatchKeySet after_meta_init_keyset =
    DispatchKeySet{DispatchKeySet::FULL_AFTER, DispatchKey::MetaInit};

void metaInitFallback(const OperatorHandle& op, DispatchKeySet ks, Stack* s) {
  DisableMetaInitGuard guard{};

  if (s == nullptr) {
    op.redispatchBoxed(ks & after_meta_init_keyset, s);

    return;
  }

  replaceForeignTensorArguments(op, *s);

  replaceDeviceArgument(op, *s);

  // Since we force the use of the meta device, we have to ensure that the Meta
  // dispatch key is in the set even if none of the original dispatch arguments
  // were meta tensors.
  ks = ks.add(DispatchKey::Meta);

  op.redispatchBoxed(ks & after_meta_init_keyset, s);
}

// Used to support nested calls.
thread_local std::size_t meta_init_level = 0;

} // namespace

void enableMetaInit(bool value) {
  if (value) {
    meta_init_level++;

    if (meta_init_level == 1) {
      c10::impl::tls_set_dispatch_key_included(DispatchKey::MetaInit, true);

      clearMetaInitCache();
    }
  } else if (meta_init_level > 0) {
    meta_init_level--;

    if (meta_init_level == 0) {
      c10::impl::tls_set_dispatch_key_included(DispatchKey::MetaInit, false);

      meta_init_state = {};
    }
  }
}

bool isMetaInitEnabled() noexcept {
  if (meta_init_level == 0) {
    return false;
  }
  return !c10::impl::tls_is_dispatch_key_excluded(DispatchKey::MetaInit);
}

void materializeTensor(Tensor& tensor) {
  TORCH_WARN("The meta-init backend is not fully implemented yet.");

  // TODO: Implement!
}

void clearMetaInitCache() {
  TORCH_WARN("The meta-init backend is not fully implemented yet.");

  // TODO: Implement!
}

} // namespace at

TORCH_LIBRARY_IMPL(_, MetaInit, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&at::metaInitFallback>());
}
