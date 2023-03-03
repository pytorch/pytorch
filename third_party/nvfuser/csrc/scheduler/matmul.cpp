#include <scheduler/matmul.h>
#include <scheduler/mma_utils.h>
#include <scheduler/utils.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {
// Move the broadcast axes to the left on the specified number of inner
// dimensions e.g.  (when number_of_inner_pos == 3):
//      [... I0, B, I1] -> [... B, I0, I1]
//  should probably be only used to order innermost mnk axes.
void moveInnerBroadcastLeft(TensorView* tv, int number_of_inner_pos = 3) {
  TORCH_INTERNAL_ASSERT(int(tv->nDims()) >= number_of_inner_pos);
  std::vector<int> broadcast_pos;
  std::vector<int> nonbroadcast_pos;

  for (auto i : c10::irange(number_of_inner_pos)) {
    auto axis_idx = i - number_of_inner_pos;
    auto id = tv->axis(axis_idx);
    if (id->isBroadcast()) {
      broadcast_pos.push_back(axis_idx);
    } else {
      nonbroadcast_pos.push_back(axis_idx);
    }
  }

  auto combined_pos_vec = broadcast_pos;
  combined_pos_vec.insert(
      combined_pos_vec.end(), nonbroadcast_pos.begin(), nonbroadcast_pos.end());

  std::unordered_map<int, int> order_map;
  for (auto i : c10::irange(number_of_inner_pos)) {
    order_map[combined_pos_vec.at(i)] = i - number_of_inner_pos;
  }

  // Apply ordering.
  tv->reorder(order_map);
}

} // namespace

void scheduleMatmul(
    TensorView* c,
    TensorView* a,
    TensorView* b,
    MatmulParam& params) {
  // Unpack from params.
  auto& mma_builder = params.mma_builder;
  auto& gemm_tile = params.tile_sizes;

  // Including current tensor naming convention for reference,
  //  this is very temporary and will change over time and
  //  in fact the whole body of this function will
  //  eventually be a set of utility functions for different
  //  sections of matmul(fusion) kernels, with
  //  each having its own build out to do.
  //
  // Current naming convention:
  //
  //  operands assumed in global memory : a, b
  //
  //  registers staging global load : ar, br (short for a/b read)
  //
  //  shared mem cache of operands : acw_smem, bcw_smem (short for a/b
  //  cache_write smem)
  //
  //  registers at shared memory load output : acr, bcr (short for a/b cache
  //  read)
  //
  //  register tensor input to the actual mma op: ab, bb (short for a/b
  //  broadcasted)
  //
  //  accumulator register: cc (short for c cache)
  //
  //  result in global memory: c

  // Currently only support a, b, c as fusion inputs/outputs
  //  aka. no prolog and epilog fusion yet.
  TORCH_CHECK(
      c->isFusionOutput() && a->isFusionInput() && b->isFusionInput(),
      "not supporting matmul fusion yet");
  TORCH_CHECK(c->definition() && c->definition()->isA<MmaOp>());

  mma_builder.configureMma(c);

  // TODO:
  // Beyond this point, mma_builder really just becomes a populated
  //  list of parameters to describes the mma swizzles that should
  //  be annotated on the tensor domain. Conceptually the mma builder
  //  object should be separated to 2 parts, one as scheduler utility
  //  and the other as matmul heuristic parameters, which we are
  //  starting to build out.

  // Setup register and shared memory stages:
  //   TODO: this section goes to a separate matmul util,
  //   and needs more configurability.

  // Setup accumulator register.
  auto cc = c->cacheBefore();

  // Get the input to the mma op.
  auto mma = dynamic_cast<MmaOp*>(cc->definition());
  TORCH_INTERNAL_ASSERT(mma != nullptr);
  auto ab = mma->inA()->as<TensorView>();
  auto bb = mma->inB()->as<TensorView>();

  // Get exact configurations from mma builder.
  mma_builder.accumulatorTv(cc);
  auto mma_options = mma_builder.build();

  // Staging register for global memory load
  TensorView *ar = a, *br = b;

  if (!params.async_gmem_load_operands) {
    ar = a->cacheAfter();
    br = b->cacheAfter();
  }

  // TODO:
  //  Significant build out needed here
  //   for more flexibility and data type support.
  // Shared memory
  TensorView* acw_smem = nullptr;
  TensorView* bcw_smem = nullptr;
  // Shared memory read
  TensorView* acr = nullptr;
  TensorView* bcr = nullptr;

  // Different paths because Volta swizzle needs to
  //  involve the broadcast dimensions that are concretized
  //  at mma, while Ampere ones should be done before
  //  the broadcast op to be able to use cp.async.
  // TODO:
  // Also a few additional parameters should be introduced
  // to control this stage of scheduling.
  if (isVolta(mma_options.macro)) {
    acw_smem = ab->cacheAfter();
    bcw_smem = bb->cacheAfter();
    // Cache again to be able to vectorize.
    acw_smem = acw_smem->cacheAfter();
    bcw_smem = bcw_smem->cacheAfter();

    acr = acw_smem->cacheAfter();
    bcr = bcw_smem->cacheAfter();
    if (params.double_buffer_options.double_buffer_smem_read) {
      // Provide another copy op between the double buffered
      //  smem load register and the actual mma ops to avoid
      //  complication in double buffered fragment iteration.
      ab = acr->cacheAfter();
      bb = bcr->cacheAfter();
    } else {
      ab = acr;
      bb = bcr;
    }

  } else {
    // Use cp.async as requested in scheduler params.
    c10::optional<LoadStoreOpType> load_op = c10::nullopt;
    if (params.async_gmem_load_operands) {
      load_op = LoadStoreOpType::CpAsync;
    }

    acw_smem = ar->cacheAfter(load_op);
    bcw_smem = br->cacheAfter(load_op);
    acr = acw_smem->cacheAfter(
        mma_builder.operand(MmaOptions::Operand::A).ldMatrix());
    bcr = bcw_smem->cacheAfter(
        mma_builder.operand(MmaOptions::Operand::B).ldMatrix());
  }

  // Make a CTA tile
  // ------------------------------------------------------------------
  scheduler_utils::matmul_utils::canonicalizeMmaTvOrdering(cc);
  // [... M,N,K]
  scheduler_utils::matmul_utils::makeTile(cc, gemm_tile.cta_tile.toVector());

  // [Mo, No, Ko, Mi, Ni, Ki]
  // Propagate tiling globally
  scheduler_utils::transformPropagateToAllFrom(cc, -1);

  // Schedule warp tile
  scheduler_utils::matmul_utils::scheduleWarpTileWithReduction(cc, gemm_tile);

  // Propagate warp tile to main loop and epilog/output tvs
  scheduler_utils::BoundedDirectionalTransformPropagator::bothWays(
      cc, -1, {acw_smem, bcw_smem}, {c});

  // Schedule prolog:
  //   TODO: this section goes to a separate matmul util,
  //   and needs more configurability.
  // ------------------------------------------------------------------
  scheduler_utils::matmul_utils::orderTiledConcreteIdAsRoot(acw_smem);
  // [... M, K]
  acw_smem->merge(-2);
  scheduler_utils::matmul_utils::scheduleContiguousVectorLoad(
      acw_smem, gemm_tile, 8, false);

  // [... N, K]
  scheduler_utils::matmul_utils::orderTiledConcreteIdAsRoot(bcw_smem);
  bcw_smem->merge(-2);
  scheduler_utils::matmul_utils::scheduleContiguousVectorLoad(
      bcw_smem, gemm_tile, 8, false);

  // Propagate prolog tensors
  //  propagate up the DAG, and propagate parallel type.
  scheduler_utils::BoundedDirectionalTransformPropagator::backward(
      acw_smem,
      -1,
      {a},
      scheduler_utils::BoundedDirectionalTransformPropagator::Options()
          .propagateParallelType());
  scheduler_utils::BoundedDirectionalTransformPropagator::backward(
      bcw_smem,
      -1,
      {b},
      scheduler_utils::BoundedDirectionalTransformPropagator::Options()
          .propagateParallelType());

  // Set computeAt, setup the loop nesting structure on the kernel.
  //   TODO: this section goes to a separate matmul util,
  //   and needs more configurability.
  // ------------------------------------------------------------------
  // CTA tile:

  // Swizzle block tiles:
  c->swizzle(Swizzle2DType::ZShape, 0, 1, SwizzleMode::Loop);

  a->computeAt(c, 2);
  b->computeAt(c, 2);

  // Prolog:
  a->computeAt(cc, 3);
  b->computeAt(cc, 3);

  // Main Loop:
  acr->computeAt(cc, -6);
  bcr->computeAt(cc, -6);

  // Add mma swizzle:
  //   TODO: this section goes to a separate matmul util,
  //   and needs more configurability.
  // ------------------------------------------------------------------
  if (isTuring(mma_options.macro) || isAmpere(mma_options.macro)) {
    moveInnerBroadcastLeft(ab);
    moveInnerBroadcastLeft(bb);
  }

  ab->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::A).build());
  bb->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::B).build());

  // Propagate mma input swizzle up the DAG
  //  to all the tensors before mma op and after shared mem read.
  scheduler_utils::BoundedDirectionalTransformPropagator::backward(
      ab,
      -1,
      {acw_smem},
      scheduler_utils::BoundedDirectionalTransformPropagator::Options()
          .propagateParallelType());
  scheduler_utils::BoundedDirectionalTransformPropagator::backward(
      bb,
      -1,
      {bcw_smem},
      scheduler_utils::BoundedDirectionalTransformPropagator::Options()
          .propagateParallelType());

  cc->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::Accumulator).build());

  // Set memory type:
  acw_smem->setMemoryType(MemoryType::Shared);
  bcw_smem->setMemoryType(MemoryType::Shared);

  // Set parallelization:
  //   TODO: this section goes to a separate matmul util,
  //   and needs more configurability.
  // ------------------------------------------------------------------

  // Vectorize smem stores/loads:
  acw_smem->axis(-1)->parallelize(ParallelType::Vectorize);
  bcw_smem->axis(-1)->parallelize(ParallelType::Vectorize);

  acr->axis(-1)->parallelize(ParallelType::Vectorize);
  bcr->axis(-1)->parallelize(ParallelType::Vectorize);

  //  0   1  2  3    4   5  6  7  8  9  10
  // [Mo No Ko Mwo  Nwo Kw Mw Nw (Mi Ni Ki)]
  cc->axis(0)->parallelize(ParallelType::BIDx);
  cc->axis(1)->parallelize(ParallelType::BIDy);
  cc->axis(3)->parallelize(ParallelType::TIDz);
  cc->axis(4)->parallelize(ParallelType::TIDy);

  // Propagate mma output swizzle and parallelization down the DAG
  if (params.double_buffer_options.double_buffer_smem_write) {
    TORCH_CHECK(
        params.double_buffer_options.smem_double_buffer_stage > 1,
        "Invalid buffer stage config")
    if (params.double_buffer_options.smem_double_buffer_stage > 2) {
      TORCH_CHECK(
          params.async_gmem_load_operands,
          "Circular buffer only supports async load");
    }

    acw_smem->circularBuffer(
        params.double_buffer_options.smem_double_buffer_stage);
    bcw_smem->circularBuffer(
        params.double_buffer_options.smem_double_buffer_stage);
  }

  if (params.double_buffer_options.double_buffer_smem_read) {
    acr->doubleBuffer();
    bcr->doubleBuffer();
  }

  scheduler_utils::BoundedDirectionalTransformPropagator::forward(
      cc,
      -1,
      {c},
      scheduler_utils::BoundedDirectionalTransformPropagator::Options()
          .propagateParallelType()
          .propagateToBoundary());
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
