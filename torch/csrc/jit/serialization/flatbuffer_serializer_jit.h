#pragma once

#include <torch/csrc/jit/serialization/flatbuffer_serializer.h>

namespace torch {
namespace jit {

TORCH_API void save_jit_module(
    const Module& module,
    const std::string& filename,
    const ExtraFilesMap& extra_files = ExtraFilesMap());

TORCH_API DetachedBuffer::UniqueDetachedBuffer save_jit_module_to_bytes(
    const Module& module,
    const ExtraFilesMap& extra_files = ExtraFilesMap());

TORCH_API Module parse_and_initialize_jit_module(
    std::shared_ptr<char> data,
    size_t size,
    ExtraFilesMap& extra_files,
    c10::optional<at::Device> device = c10::nullopt);

TORCH_API Module load_jit_module_from_file(
    const std::string& filename,
    ExtraFilesMap& extra_files,
    c10::optional<at::Device> device = c10::nullopt);

TORCH_API Module load_jit_module_from_stream(
    std::istream& in,
    ExtraFilesMap& extra_files,
    c10::optional<at::Device> device = c10::nullopt);

// This function will make the capabilities to load and safe
// Module as a flatbuffer file available for use by _load_for_mobile
// and friends. This is NOT needed if using the other functions
// in this file directly.
TORCH_API bool register_flatbuffer_all();

} // namespace jit
} // namespace torch
