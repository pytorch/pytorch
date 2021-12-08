// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/MetaInit.h>

#include <cstdint>

#include <ATen/Tensor.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/DispatchKeySet.h>
#include <torch/library.h>

namespace at {
namespace {

constexpr DispatchKeySet after_meta_init_keyset =
    DispatchKeySet{DispatchKeySet::FULL_AFTER, DispatchKey::MetaInit};

void metaInitFallback(const OperatorHandle& op, DispatchKeySet ks, torch::jit::Stack* s) {
  DisableMetaInitGuard guard{};

  // Just redispatch, effectively noop.
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
  TORCH_WARN("The meta-init backend is not implemented yet.");

  // TODO: Implement!
}

void clearMetaInitCache() {
  TORCH_WARN("The meta-init backend is not implemented yet.");

  // TODO: Implement!
}

} // namespace at

TORCH_LIBRARY_IMPL(_, MetaInit, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&at::metaInitFallback>());
}
