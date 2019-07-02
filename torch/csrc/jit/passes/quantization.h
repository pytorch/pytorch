/** \brief This file defines passes used for quantization.
 *
 * The passes have python-bindings and can be invoked directly or as a part of
 * general optimization pipeline (details TBD).
 */
#pragma once

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/script/module.h>

namespace torch {
namespace jit {

/** \brief Propagates QParams through nodes that are not supposed to change it.
 *
 * An example of such node is `Split`: even though the observed distribution
 * might be different for input and output tensors, we would like to use input's
 * qparams for output as well.
 */
TORCH_API void PropagateQuantInfo(std::shared_ptr<Graph>& graph);

/** \brief Inserts observer nodes for collecting distribution of values taken by
 * a tensor.
 *
 * The distribution can then be used for computing qparams for quantization.
 * \param moduleObj is the module object whose containing methods are modified.
 * \param methodName is module method whose containing graph is instrumented.
 * \param observer_node is a Node representing a call to observer function. It
 * will be cloned into all the places where we need to add instrumentation.
 */
TORCH_API void InsertObserverNodes(
    const script::Module& moduleObj,
    const std::string& methodName,
    Node* observer_node);

/** \brief Inserts observer nodes for collecting distribution of values taken by
 * a tensor. This is overloaded InsertObserverNodes which takes in different
 * arguments and operates on pure functions not associated with module.
 *
 * The distribution can then be used for computing qparams for quantization.
 * \param function_var is a pure script function whose graph is instrumented
 * \param observer_node is a Node representing a call to observer function. It
 * will be cloned into all the places where we need to add instrumentation.
 */
TORCH_API void InsertObserverNodes(
    std::shared_ptr<Function>& function_var,
    Node* observer_node);

/** \brief Inserts quant-dequant nodes.
 *
 * This actually changes the numerical semantics of the original model and thus
 * we only run it when user explicitly wants that. This pass essentially
 * performs quantization of the model by inserting quant-dequant node pairs for
 * quantizatable tensors - later passes only cleanup the IR and
 * make sure the model runs faster/consumes less memory.
 * \moduleObj is the module object whose containing methods are modified.
 * \param method_name whose graph is instrumented for quant-dequant nodes.
 * \param qparam_dict dictionary of tensor unique names to qparams.
 *
 */
TORCH_API void InsertQuantDequantNodes(
    const script::Module& moduleObj,
    const std::string& methodName,
    const std::unordered_map<std::string, std::tuple<std::string, float, int>>&
        qparam_dict);

/** \brief Check that all expected optimizations after quant-dequant nodes
 * insertion actually happened.
 *
 * Even though semantically it would be correct to just execute the initial
 * quant-dequant nodes as is, what we really wanted when we inserted them is to
 * fuse them into adjacent non-quantized ops resulting in quantized ops. Thus,
 * if after all the cleanups, optimizations (particularly, fusion) we find
 * quant-dequant pair in the graph, it indicates that quantization didn't go as
 * planned.
 */
TORCH_API void QuantLinting(std::shared_ptr<Graph>& graph);

/** \brief Quantize model's inputs and outputs.
 *
 * This pass folds quant/dequant ops into the input/output tensors, essentially
 * quantizing these tensors. It's done to reduce model's memory footprint.
 */
TORCH_API void FoldQuantNodesIntoInputsOutputs(std::shared_ptr<Graph>& graph);

/** \brief Inserts quant-dequant nodes for attributes.
 *
 * This is similar to Quant-Dequant pass but it inserts quant-dequant nodes
 * for module parameters. It changes the numerical semantics of the original
 * model and thus we only run it when user explicitly wants that. Later passes
 * only cleanup the IR and make sure the model runs faster/consumes less memory
 * \moduleObj is the module object whose containing methods are modified.
 * \param method_name whose graph is instrumented for quant-dequant nodes.
 * \param param_name parameter for which the nodes are inserted.
 * \param getQParamFunc function to compute qparams.
 * \at::ScalarType t Datatype for param
 */
template <typename Fn>
TORCH_API void InsertQuantDequantNodesForParam(
    const script::Module& moduleObj,
    const std::string& method_name,
    const std::string& param_name,
    const Fn& getQParamFunc,
    at::ScalarType t);
} // namespace jit
} // namespace torch
