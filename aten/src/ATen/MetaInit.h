// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <c10/core/DispatchKey.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/macros/Macros.h>

namespace at {

class Tensor;

/// Enables or disables the meta-init backend.
///
/// When enabled the meta-init backend forces all tensors to use the meta device
/// regardless of their real device and starts recording the ATen operations for
/// materializing those tensors at a later time.
TORCH_API void enableMetaInit(bool value);

/// Indicates whether the meta-init backend is enabled.
TORCH_API bool isMetaInitEnabled() noexcept;

/// Materializes `tensor`.
///
/// Note that the only tensors that can be materialized are the ones that were
/// constructed while the meta-init backend was enabled.
TORCH_API void materializeTensor(Tensor& tensor);

/// Clears the meta-init cache used for materialization.
TORCH_API void clearMetaInitCache();

/// A utility class to temporarily disable the meta-init backend.
class TORCH_API DisableMetaInitGuard {
  c10::impl::ExcludeDispatchKeyGuard guard_{DispatchKey::MetaInit};
};

} // namespace at
