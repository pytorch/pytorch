#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/reduction_heuristic.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class ExpressionEvaluator;
class SchedulerRuntimeInfo;

namespace scheduler_utils {

// Merge all reduction to the right side and returns total number of***
// reduction axes
size_t mergeReduction(TensorView* tv);

// merge all non-reduction axes to the left side and returns total number of
// iteration axes
size_t mergeNonReduction(TensorView* tv);

// Makes rfactor generic with reduction ops and Welford
TensorView* rfactorHelper(TensorView* red_tv, const std::vector<int>& axes);

// Return immediate producers of tv
std::vector<TensorView*> producerTvsOf(TensorView* tv);

// Return immediate consumers of tv
std::vector<TensorView*> consumerTvsOf(TensorView* tv);

// Return immediate producers of tvs (can return tvs input)
std::vector<TensorView*> producerTvsOf(const std::vector<TensorView*>& tvs);

// Return immediate consumers of tvs (can return tvs input)
std::vector<TensorView*> consumerTvsOf(const std::vector<TensorView*>& tvs);

// Returns producers of tv that are inputs of fusion
std::vector<TensorView*> inputTvsOf(TensorView* tv);

// Returns consumers of tv that are outputs of fusion
std::vector<TensorView*> outputTvsOf(TensorView* tv);

// Returns producers of tvs that are inputs of fusion
std::vector<TensorView*> inputTvsOf(std::vector<TensorView*> tvs);

// Returns consumers of tvs that are outputs of fusion
std::vector<TensorView*> outputTvsOf(std::vector<TensorView*> tvs);

TORCH_CUDA_CU_API std::vector<TensorView*> allTvs(Fusion* fusion);

TORCH_CUDA_CU_API void parallelizeAllLike(
    TensorView* reference_tv,
    const std::vector<TensorView*>& all_tvs);

void computeAtInputs(
    TensorView* consumer,
    int pos,
    ComputeAtMode mode = ComputeAtMode::Standard);

void computeWithOutputs(
    TensorView* producer,
    int pos,
    ComputeAtMode mode = ComputeAtMode::Standard);

// returns all tensor views in fusion that are used between outputs and inputs.
// Order is non-deterministic and non-repeating.
// TODO: This would be good to have determinsitic and to put outside scheduling
// as it's generally useful
std::vector<TensorView*> allTvs(Fusion* fusion);

struct PersistentBufferInfo {
  std::vector<TensorView*> buffers;
  std::unordered_set<IterDomain*> unmappable_dims;
};

// Buffers whos roots can't map to all producer roots based on compute at. These
// are the buffers we would make persistent in a persistent kerenl or would have
// to recompute if we can't make a persistent kernel.
PersistentBufferInfo persistentBuffers(Fusion* fusion);

struct TvProperties {
  // How many elements in tensor view are there to reduce
  int64_t reduction_numel = 1;
  // How many reductions do we need to perform, i.e. how many iter dimension
  // elements are there
  int64_t iteration_numel = 1;
  // Do we reduce the fastest dimension, if no reduction mark true
  bool fastest_dim_reduction = true;
  // What's the iter numel to the left of the reduction (if there is one)
  int64_t iter_outside_red = 1;
  // What's the iter numel to the right of the reduction (if this is or isn't
  // one)
  int64_t iter_inside_red = 1;
};

// Fill TvProperties structure about tv
TvProperties getProperties(
    Fusion* fusion,
    ExpressionEvaluator& evaluator,
    TensorView* tv);
// Will call computeAt once on each producer, with the first consumer found that
// is a consumer of the individual producer
void computeAtBetween(
    const std::vector<TensorView*>& producers,
    const std::vector<TensorView*>& consumers,
    int pos,
    ComputeAtMode mode);

bool registerPersistentBufferCheck(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info);

} // namespace scheduler_utils
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
