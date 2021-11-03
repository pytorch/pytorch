#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {
namespace tensorexpr {

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
    const std::vector<c10::optional<at::Tensor>>& example_inputs);
TORCH_API std::shared_ptr<Graph> removeUnusedSelfArgument(
    const std::shared_ptr<Graph>& graph);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
