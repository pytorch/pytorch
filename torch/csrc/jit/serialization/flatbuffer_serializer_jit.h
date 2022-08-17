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

TORCH_API void save_jit_module_to_write_func(
    const Module& module,
    const ExtraFilesMap& extra_files,
    bool save_mobile_debug_info,
    const std::function<size_t(const void*, size_t)>& writer_func);

TORCH_API Module parse_and_initialize_jit_module(
    std::shared_ptr<char> data,
    size_t size,
    ExtraFilesMap& extra_files,
    c10::optional<at::Device> device);

} // namespace jit
} // namespace torch
