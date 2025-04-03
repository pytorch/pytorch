// Copyright © 2024 Apple Inc.
// This file is to support the feature for MPS and CPU to share the same
// underlying memory through Copy-on-write context.

#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>
#include <c10/core/Layout.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/intrusive_ptr.h>

#include <optional>

namespace at::mps::cow {

bool to_will_cow(
    const Tensor& self,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    bool copy,
    std::optional<c10::MemoryFormat> optional_memory_format);

c10::intrusive_ptr<c10::TensorImpl> lazy_cloned_tensor_for_unified_memory(
    Tensor const& self,
    std::optional<c10::Device> device,
    c10::intrusive_ptr<c10::StorageImpl> lazy_cloned_storage);

} // namespace at::mps::cow
