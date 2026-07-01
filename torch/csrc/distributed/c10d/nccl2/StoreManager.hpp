// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <torch/csrc/distributed/c10d/Store.hpp>
#include <string>

namespace c10d::nccl2 {

// Create a PrefixStore wrapping a root TCPStore, using `prefix` to
// namespace all keys.  This avoids key collisions when multiple
// communicators share the same underlying store (e.g. the torchrun
// agent store).
c10::intrusive_ptr<c10d::Store> createPrefixStore(
    const std::string& prefix,
    std::chrono::milliseconds timeout);

// Create an independent TCPStore on an OS-assigned port, using
// `bootstrapStore` only to exchange the chosen port.  The result
// is wrapped in a PrefixStore with `prefix`.  The caller must
// keep `bootstrapStore` alive until all ranks have returned from
// this call.
c10::intrusive_ptr<c10d::Store> dupPrefixStore(
    const std::string& prefix,
    const c10::intrusive_ptr<c10d::Store>& bootstrapStore,
    std::chrono::milliseconds timeout);

} // namespace c10d::nccl2
