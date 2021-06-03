#include <torch/csrc/jit/passes/annotate_warns.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/inline_forked_closures.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/lift_closures.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/normalize_ops.h>

namespace torch {
namespace jit {

void runCleanupPasses(std::shared_ptr<Graph>& graph) {
  liftClosures(graph);
  inlineForkedClosures(graph);
  if (getInlineEverythingMode()) {
    Inline(*graph);
  }

  // remove any uses of tuples that we inserted that are not needed
  lowerSimpleTuples(graph);

  // full constant propagation runs ops with mutable inputs if it can
  // prove that the inputs are not mutated anywhere in the graph.
  // if a mutating node is removed in the graph (e.g. constant prop inlined a
  // a constant if) then the next time constant prop is run it might be able
  // to run nodes it was not able to previously, and the graph may change
  // (jitter) So we run only constant prop w immutable types here bc
  // successive runs of immutable constant prop does not change the graph
  constantPropagationImmutableTypes(graph);

  // Constant Pooling pass must be after ConstantPropogation, which can create
  // new constants that needs to be pooled.
  constantPooling(graph);

  // For jitter
  canonicalizeOutputs(graph);

  // Annotate aten::warns so that each has its unique ID. This enables us to
  // mimic Python behavior of only emitting each warning only once.
  annotateWarns(graph);
}

} // namespace jit
} // namespace torch
