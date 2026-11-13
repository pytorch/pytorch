#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit::tensorexpr {

// Optimize aten::cat ops in the given subgraph.
//
// Moving users of cat to its inputs.
//    Cat ops get lowered into multiple loops, one per input. When the result
//    of cat is used by some other op, it results in a situation where inlining
//    of cat does not happen. This in turn results in intermediate buffers
//    being created for the result of cat, since it is not inlined.
//
//    For example, consider the following graph:
//       graph(%x : Float(10, strides=[1], device=cpu),
//             %y : Float(20, strides=[1], device=cpu)):
//         %dim : int = prim::Constant[value=0]()
//         %xy_list : Tensor[] = prim::ListConstruct(%x, %y)
//         %cat : Float(60, strides=[1], device=cpu) = aten::cat(%xy_list, %dim)
//         %5 : Float(60, strides=[1], device=cpu) = aten::log(%cat)
//         return (%5))IR";
//
//     This will get lowered into:
//         Allocate(aten_cat);
//         for (...)
//           aten_cat[...] = x[...]
//         for (...)
//           aten_cat[...] = y[...]
//         for (...)
//           aten_log[...] = log(aten_cat[...])
//         Free(aten_cat);
//     Note that aten_cat is not inlined into aten_log and it results in
//     an intermediate buffer allocation as well.
//
//     Optimization:
//        We move the ops that use the result of `cat` into its inputs whenever
//     possible.
//
//     The graph above will be transformed to:
//        graph(%x : Float(10, strides=[1], device=cpu),
//              %y : Float(20, strides=[1], device=cpu)):
//          %3 : int = prim::Constant[value=0]()
//          %7 : Float(10, strides=[1], device=cpu) = aten::log(%x)
//          %8 : Float(20, strides=[1], device=cpu) = aten::log(%y)
//          %9 : Tensor[] = prim::ListConstruct(%7, %8)
//          %10 : Float(60, strides=[1], device=cpu) = aten::cat(%9, %3)
//          return (%10)
//
//     This will get lowered into:
//         for (...)
//           aten_cat[...] = log(x[...])
//         for (...)
//           aten_cat[...] = log(y[...])
//     aten_cat is the output buffer here.

bool OptimizeCat(const std::shared_ptr<Graph>& graph);

TORCH_API void annotateInputShapes(
    const std::shared_ptr<Graph>& graph,
    const std::vector<std::optional<at::Tensor>>& example_inputs);
TORCH_API std::shared_ptr<Graph> removeUnusedSelfArgument(
    const std::shared_ptr<Graph>& graph);
TORCH_API std::shared_ptr<Graph> removeGraphOutput(
    const std::shared_ptr<Graph>& graph,
    size_t idx);
TORCH_API std::shared_ptr<Graph> replaceListOutputWithTuple(
    const std::shared_ptr<Graph>& graph);

// Perform \p ITERS rounds of "trimming" for the given \p GRAPH.
//
// Trimming means that we try to remove a small portion of the graph while
// keeping it valid. This is useful for debugging when we try to find a minimal
// example reproducing the issue at hand. When ITERS is 0, the graph remains
// unchanged, when ITERS is a big number, the graph usually becomes empty.
TORCH_API std::shared_ptr<Graph> trimGraph(
    const std::shared_ptr<Graph>& graph,
    int64_t iters);

// Scan all values in the given graph and replace each dimension with a size Xi
// present in \p SIZES with a symbolic shape Yi. Return a vector of symbol
// values [Y0, Y1, .., Yn].
//
// For example:
// Input:
// graph(%x : Float(10, 20, 30, 40)):
//   %y : Float(10, 20, 30, 40) = aten::relu(%x)
//   return %y
//
// If we run makeShapesSymbolic(graph, {20, 40}), then we'll get:
//
// graph(%x : Float(10, SS(-3), 30, SS(-5))):
//   %y : Float(10, SS(-3), 30, SS(-5)) = aten::relu(%x)
//   return %y
//
// and get {-3, -5} as the return value.
TORCH_API std::vector<int64_t> makeShapesSymbolic(
    std::shared_ptr<Graph>& graph,
    const std::vector<int64_t>& sizes);

// Inspect the graph and report whether it can be converted to TE IR.
// TODO: add error reporting for graphs that can't be converted.
TORCH_API bool isGraphCompilable(const std::shared_ptr<Graph>& graph);

// Examine the graph and (hackily) fill in missing tensor type info, such as
// scalar type, device, and strides. Ideally, this should be done by a proper
// dtype/device/shape propagation passes, but until they are ready we can use
// this, not always correct, workaround pass.
TORCH_API void fixupMissingShapeInfo(const std::shared_ptr<Graph>& graph);

} // namespace torch::jit::tensorexpr
