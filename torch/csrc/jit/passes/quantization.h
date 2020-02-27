/** \brief This file defines passes used for quantization.
 *
 * The passes have python-bindings and can be invoked directly or as a part of
 * general optimization pipeline (details TBD).
 */
#pragma once

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/script/module.h>

namespace std {

template <>
struct hash<torch::jit::script::Module> {
  inline size_t operator()(const torch::jit::script::Module& arg) const {
    return std::hash<c10::intrusive_ptr<c10::ivalue::Object>>()(arg._ivalue());
  }
};

}

namespace torch {
namespace jit {

using QConfig = std::tuple<script::Module, script::Module>;
using QConfigDict = std::unordered_map<std::string, QConfig>;
using ModuleQConfigMap =
    std::unordered_map<script::ModulePtr, c10::optional<QConfig>>;

struct OptionalQConfigHash {
  inline size_t operator()(const c10::optional<QConfig>& qconfig_opt) const {
    if (qconfig_opt.has_value()) {
      const auto& m1 = std::get<0>(*qconfig_opt);
      const auto& m2 = std::get<1>(*qconfig_opt);
      return std::hash<script::Module>()(m1) + 7 * std::hash<script::Module>()(m2);
    }
    return 0;
  }
};

using QConfigTypePtrMap = std::unordered_map<c10::optional<QConfig>, TypePtr, OptionalQConfigHash>;

/** \brief Quantize model's inputs and outputs.
 *
 * This pass folds quant/dequant ops into the input/output tensors, essentially
 * quantizing these tensors. It's done to reduce model's memory footprint.
 */
TORCH_API void FoldQuantNodesIntoInputsOutputs(std::shared_ptr<Graph>& graph);

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
 */
TORCH_API script::Module InsertObservers(
    script::Module& module,
    const std::string& method_name,
    const std::unordered_map<
        std::string,
        std::tuple<script::Module, script::Module>>& qconfig_dict,
    bool inplace = false);

/** \brief Insert quantize - int_repr - dequantize calls to the Tensors
 *  that are observed in insert_observers pass
 *
 * For each Tensor that is observed, get the observer module and call
 * calculate_qparam on the observer module to get quantization parameters
 * and add quantize - int_repr - dequantize function calls using these
 * parameters we also have special handling for quantizing "bias" right now.
 *
 * \param module the input module
 * \param method_name the method we want to insert quantization calls for
 */
TORCH_API script::Module InsertQuantDeQuant(
    script::Module& module,
    const std::string& method_name,
    bool inplace = false);

/** Replicate dequantize node for each use, so that we can match
 *  quantization patterns
 */
TORCH_API void ReplicateDeQuant(std::shared_ptr<Graph>& graph);

TORCH_API void SwapDeQuant(std::shared_ptr<Graph>& graph);

/** \brief Backend specific pass to fuse dequantize - op - quantize calls
 * as quantized_op calls.
 *
 * Right now this is a fusion for fbgemm backend and only works for quantized
 * conv op, we'll extend to more ops and more backends in the future.
 *
 * Currently supported fusion:
 * q(conv2d(dq(a), dq(w), dq(b))) --> to_nchw(fbgemm_conv2d(prepack(to_nhwc(a)),
 *                                                          prepack(to_nhwc(w)),
 *                                                          prepack(to_nhwc(b))))
 *
 * q(linear(dq(a), dq(w), dq(b))) --> to_nchw(fbgemm_linear(prepack(to_nhwc(a)),
 *                                                          prepack(to_nhwc(w)),
 *                                                          prepack(to_nhwc(b))))
 *
 * \param graph the graph we want to apply fusion
 */
TORCH_API void QuantFusion(std::shared_ptr<Graph>& graph);

/** \brief Fold Conv2d-BatchNorm2d into Conv2d in forward method of this module
 * and all its submodules.
 *
 * The weight and bias of the Conv2d are correspondingly updated. Should only be
 * used on modules in eval mode.
 */
TORCH_API script::Module FoldConvBatchNorm2d(const script::Module& module);

/** \brief Fold quantize function call into module
 *
 *  For the graph of the specified method of module, if we find a
 * quantize_per_tensor call on an attribute("weight") of the module, we'll
 * quantize the attribute directly and register a new buffer "_quantized_weight"
 * on the module and remove the quantize_per_tensor call and replace the use of
 * the quantized weight with
 *  "_quantized_weight".
 */
TORCH_API void FoldQuantizeCallIntoBuffer(
    script::Module& module,
    const std::string& method_name);

/** \brief Insert prepack and unpack function in graph
 *  We want add pack/unpack functions for quantized weight because later we want
 * to fold the packed weight as an attribute of the module, in order to reduce
 * the cost of packing the weight on the fly in quantized models.
 *
 *  Each quantized op has it's corresponding prepack/unpack function,
 *  right now, we only need to do prepack/unpack for quantized::linear
 * and quantized::conv2d.
 */
TORCH_API void InsertPrepackUnpack(std::shared_ptr<Graph>& graph);

/** \brief Insert pack and unpack function in all graphs
 *   of module
 *
 *   Go through graphs of all the methods of all child modules
 *   and call InsertPrepackUnpack on the graph.
 */
TORCH_API void InsertPrepackUnpack(script::Module& module);

/** \brief Fold prepack function call into module
 *
 *  For the graph of the specified method, if we find a
 * `quantized::linear_prepack` call, we'll clone the wrapper module and set the
 * weight and bias of the module and add the wrapper module as a child to the
 * input module. Folding is recursively applied to all methods of all child
 * modules of the input module
 *
 *  Wrapper module is used to overwrite serialization for packed
 *  weight and bias since they are not recognized by JIT, this
 *  is a workaround, a long term solution would be to support serialization of
 *  packed weight and bias using custom types
 *
 */
TORCH_API void FoldPrepackedWeightIntoModule(
    script::Module& module,
    const script::Module& linear_params_module,
    const script::Module& conv_params_module);

/** Recursively deduplicate multiple uses of the same module by
 *  creating an instance clone for each use of the module, which means
 *  the type will be the same as before and all the attributes will be
 *  copied, then we'll change the use of the original module to the use
 *  of cloned module in the Graph.
 */
TORCH_API void DedupModuleUses(script::Module& module);

} // namespace jit
} // namespace torch
