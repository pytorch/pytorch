#pragma once

#include <ATen/core/TensorBase.h>
#include <c10/core/Device.h>
#include <c10/util/Optional.h>
#include <torch/csrc/jit/mobile/module.h>

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

// NOTE: Please prefer using _load_parameters over using the function below.
TORCH_API std::map<std::string, at::Tensor> mobile_module_to_parameter_map(
    const mobile::Module& module);

} // namespace jit
} // namespace torch
