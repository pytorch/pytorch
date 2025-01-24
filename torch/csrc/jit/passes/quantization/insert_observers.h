#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/passes/quantization/quantization_type.h>

namespace std {

template <>
struct hash<torch::jit::Module> {
  inline size_t operator()(const torch::jit::Module& arg) const {
    return std::hash<c10::intrusive_ptr<c10::ivalue::Object>>()(arg._ivalue());
  }
};

} // namespace std

namespace torch::jit {

using QConfig = std::tuple<Module, Module>;
using QConfigDict = std::unordered_map<std::string, std::optional<QConfig>>;

/** \brief Insert observer module and observer function call for
 *  the Tensors that needs to be observed.
 *
 * For each Tensor that needs to be observed in the method, insert observer
 * module to the input module and add forward calls of observer to the specified
 * method.
 *
 * \param module the input module
 * \param method_name the method we want to insert observers for
 * \param qconfig_dict the qconfig dictionary that specifies how
 * each module is going to be quantized
 * \param inplace whether we want to do inplace modification to the input module
 * or clone the module
 * \param is_dynamic whether the dynamic quantization script is being used.
 */
TORCH_API Module InsertObservers(
    Module& module,
    const std::string& method_name,
    const QConfigDict& qconfig_dict,
    bool inplace,
    QuantType quant_type = QuantType::STATIC);

/** \brief Insert observer module and observer method for
 *  the Tensors that needs to be observed.
 *
 * For each Tensor that needs to be observed in the method, insert observer
 * module to the input module and observe_<method-name> methods to the module.
 * This method is clone of mehtod_name with forward calls of observer added.
 *
 * \param module the input module
 * \param method_name the method we want to insert observers for
 * \param qconfig_dict the qconfig dictionary that specifies how
 * each module is going to be quantized
 * \param inplace whether we want to do inplace modification to the input module
 * or clone the module
 * \param is_dynamic whether the dynamic quantization script is being used.
 */
TORCH_API Module InsertObserversForOnDevicePTQ(
    Module& module,
    const std::string& method_name,
    const QConfigDict& qconfig_dict,
    bool inplace,
    QuantType quant_type = QuantType::STATIC);

} // namespace torch::jit
