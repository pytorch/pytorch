// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <c10/core/DispatchKey.h>

// This file contains aliases for dispatch keys related to functorch.
// The names were long so we aliased them.

namespace at {
namespace functorch {

constexpr auto kBatchedKey = c10::DispatchKey::FuncTorchBatched;
constexpr auto kVmapModeKey = c10::DispatchKey::FuncTorchVmapMode;
constexpr auto kGradWrapperKey = c10::DispatchKey::FuncTorchGradWrapper;
constexpr auto kDynamicLayerFrontModeKey = c10::DispatchKey::FuncTorchDynamicLayerFrontMode;
constexpr auto kDynamicLayerBackModeKey = c10::DispatchKey::FuncTorchDynamicLayerBackMode;

// Some helper macros
#define SINGLE_ARG(...) __VA_ARGS__

}} // namespace at::functorch
