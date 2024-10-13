#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {

TORCH_API void FuseInferenceOpsForSparseNN(
    std::shared_ptr<torch::jit::Graph>& graph);

TORCH_API void EliminateTrivialEquallySplit(
    std::shared_ptr<torch::jit::Graph>& graph);

TORCH_API void FuseListUnpack(std::shared_ptr<torch::jit::Graph>& graph);

// If outputs_are_immutable is set to false, don't replace the view ops that
// produce aliases of graph outputs with the copy version.
TORCH_API void ReplaceWithCopy(
    std::shared_ptr<torch::jit::Graph>& graph,
    bool outputs_are_immutable = true);

TORCH_API void ReplacePermuteWithCopy(
    std::shared_ptr<torch::jit::Graph>& graph,
    bool outputs_are_immutable = true);

TORCH_API void ReplaceWithMaybeCopy(
    std::shared_ptr<torch::jit::Graph>& graph,
    bool outputs_are_immutable = true);

TORCH_API void RemoveImmutableInputDictLookups(
    std::shared_ptr<torch::jit::Graph>& graph);

TORCH_API bool graphHasOp(std::shared_ptr<Graph>& graph, const char* op_name);

TORCH_API bool forwardHasOp(const Module& module, const char* op_name);

TORCH_API void FuseSignLog1P(std::shared_ptr<Graph>& graph);

TORCH_API void UseVariadicTupleUnpack(const std::shared_ptr<Graph>& graph);

// c10::Symbol::fromQualString is a bit long to type everywhere, and
// we can't use a `using` statement since it's a static class function.
inline c10::Symbol fromQualString(const std::string& qual_string) {
  return c10::Symbol::fromQualString(qual_string);
}

// [Create owned refs for special values]
// StaticRuntimeBlockRunner moves its outputs to the return value at the end of
// run_impl. However, there's a corner case where this can cause problems. If
// we return a constant, then the only reference in the constants_ array can
// be destroyed by this move.
// We could add special logic to handle this in run_impl. But since this is a
// relatively rare corner case, it's simpler to just add an op that does nothing
// but create an owned reference to its input. This owned reference can be
// safely moved out of StaticRuntimeBlockRunner. Note that for scalars,
// this actually does a copy.
// Note that we have to do the same thing if we are returning a value from an
// outer scope in a sub-block.
TORCH_API void CreateOwnedRefsForSpecialValues(Graph& graph);

// [Force non-empty outputs]
// It is technically possible for sub-blocks to not return anything. This is
// problematic for StaticRuntimeBlockRunner because it assumes that at least one
// output is being returned. Rather than slowing down SR with special logic for
// this corner case, we simply force blocks that return nothing to return None.
TORCH_API void ForceNonEmptyOutputs(Graph& graph);

TORCH_API void UseVariadicGroupedAccessor(const std::shared_ptr<Graph>& graph);

TORCH_API void EliminateExtraPermuteOps(std::shared_ptr<Graph>& graph);

TORCH_API void EliminateNoOpSlice(std::shared_ptr<Graph>& graph);

TORCH_API void UseSplitAndSqueeze(std::shared_ptr<Graph>& graph);

// [Remove unnecessary outputs]]
// Removes outputs to reduce compute when it is not used later in the graph.
// Currently used to remove the max_indices output of embedding_bag, which
// isn't necessary to compute the main output.
TORCH_API void RemoveUnnecessaryOutputs(std::shared_ptr<Graph>& graph);

TORCH_API void RemoveUnnecessaryEmbeddingBagOutputs(
    std::shared_ptr<Graph>& graph);

TORCH_API void FuseClampNaNToNum(std::shared_ptr<Graph>& graph);

TORCH_API void UseInPlaceGetRealInputsFromOptionalInputsV2(
    std::shared_ptr<Graph>& graph);

TORCH_API void PrepackWeights(std::shared_ptr<Graph>& graph);

} // namespace torch::jit
