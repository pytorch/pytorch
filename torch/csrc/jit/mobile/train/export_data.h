#pragma once

#include <torch/csrc/jit/mobile/module.h>

namespace torch {
namespace jit {

/**
 * Serializes the provided tensor map to the provided stream.
 *
 * @param[in] map The tensors to serialize.
 * @param[in] out The stream to write the serialized data to.
 * @param[in] use_flatbuffer If true, use Flatbuffers to serialize the data.
 *     If false, use Pickle.
 */
TORCH_API void _save_parameters(
    const std::map<std::string, at::Tensor>& map,
    std::ostream& out,
    bool use_flatbuffer = false);

/**
 * Serializes the provided tensor map to a file.
 *
 * @param[in] map The tensors to serialize.
 * @param[in] filename The stem of the file name to write to. If
 *     @p use_flatbuffer is false, the extension ".pkl" will be appended. If
 *     @p use_flatbuffer is true, the extension ".ff" will be appended.
 * @param[in] use_flatbuffer If true, use Flatbuffers to serialize the data.
 *     If false, use Pickle.
 */
TORCH_API void _save_parameters(
    const std::map<std::string, at::Tensor>& map,
    const std::string& filename,
    bool use_flatbuffer = false);

namespace mobile {

// NOTE: Please prefer using _save_parameters directly over using the 2
// functions below.
TORCH_API mobile::Module tensor_dict_to_mobile(
    const c10::Dict<std::string, at::Tensor>& dict);

c10::Dict<std::string, at::Tensor> tensor_map_to_dict(
    const std::map<std::string, at::Tensor>& map);

} // namespace mobile

extern void (*_save_mobile_module_to)(
    const mobile::Module& module,
    const std::function<size_t(const void*, size_t)>& writer_func);

} // namespace jit
} // namespace torch
