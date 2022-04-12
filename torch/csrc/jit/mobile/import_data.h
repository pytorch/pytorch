#pragma once

#include <ATen/core/TensorBase.h>
#include <c10/core/Device.h>
#include <c10/util/Optional.h>

#include <istream>
#include <map>
#include <string>

namespace torch {
namespace jit {

/**
 * Loads named parameters from the serialized data in @p in.
 *
 * Calls #TORCH_CHECK() if the data format is not recognized.
 */
TORCH_API std::map<std::string, at::Tensor> _load_parameters(
    std::istream& in,
    c10::optional<at::Device> device = c10::nullopt);

/**
 * Loads named parameters from the serialized data in @p filename.
 *
 * Calls #TORCH_CHECK() if the data format is not recognized.
 */
TORCH_API std::map<std::string, at::Tensor> _load_parameters(
    const std::string& filename,
    c10::optional<at::Device> device = c10::nullopt);

} // namespace jit
} // namespace torch
