#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/quantization/quantization_type.h>

namespace torch {
namespace jit {

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
TORCH_API void QuantFusion(
    std::shared_ptr<Graph>& graph,
    QuantType quant_type = QuantType::STATIC);

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
TORCH_API void InsertPrepackUnpack(Module& module);

TORCH_API script::Module Finalize(
    script::Module& module,
    QuantType quant_type = QuantType::STATIC,
    const std::vector<std::string>& preserved_attrs =
        std::vector<std::string>());

TORCH_API void FoldQuantizedPrepackingOps(Module& module);

} // namespace jit
} // namespace torch
