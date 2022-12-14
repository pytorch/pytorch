#pragma once

#include <ATen/core/ivalue.h>

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/pointwise_heuristic.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

/*
 * The 2D pointwise scheduling logic is a bit interesting. We'll start by giving
 * motivation for what the scheduling is attempting to do. What we're going to
 * do with the scheduling is attempt to make it two dimensional in a way that
 * minimizes the refetching of broadcasted dimensions. If we think of the
 * trivial case:
 * T0[i0, b1]
 * T1[b0, i1]
 * T2[i0, i1] = T0 + T1
 * If we scheduled T2 as 1-dimensional we would do something along the lines of
 * merging i0 and i1 then splitting out a block and thread dimension. If i1 is
 * greater than the thread dimension, then all threads would pull the same value
 * from T0. However, they would all be pulling different values from T1. In this
 * case we have perfect reuse of the broadcast dimension T0 but potentially no
 * reuse of the broadcast dimension of T1. "Potentially" because if i1 isn't too
 * big it should be efficiently cached in L2. If i1 is big, then by the time we
 * increment the i0 dimension the i1 dimension will be pushed out of cache.
 *
 * Instead what we do is we map this to a two dimensional problem. Instead of
 * having the schedule that merges the two dimensions, we'll actually leave the
 * dimensions separate and we'll take i0, split it to BIDy, TIDy, and take i1
 * and split it to BIDx and TIDx. Therefore we'll have a parallelization on T2
 * like [BIDy, TIDy | BIDx, TIDx], where | denotes the separation of the
 * original i0 and i1. This helps because all threads in the TIDx dimension will
 * reuse the same value in the i0 dimension (holding BIDy and TIDy constant),
 * all the threads in the TIDy dimension (holding BIDx, and TIDx constant) will
 * reuse the same value in the i1 dimension. This reuse of values reduces the
 * number of redundant values pulled from T0 and T1. The same thing can be said
 * for when incrementing BIDy, but since BIDy is strided on BIDx there's no
 * effective increment of BIDy without incrementing BIDx. Since all threads are
 * executed within a block we can effectively consider the block incrementing
 * TIDx BDIMx times while holding TIDy constant and incrementing TIDy BDIMy
 * times while holding TIDx constant. Since multiple BIDx's are running at the
 * same time on the device we can consider a wave on the GPU of incrementing
 * BIDx (wave number of times), while holding TIDy constant BDIMy * wave number
 * of times.
 *
 * If instead we have a situation like:
 * T0[i0, i1, b2]
 * T1[i0, b1, i2]
 * T2[i0, i1, i2] = T0 + T1
 * It makes sense that the break point would be in position 2, between i1 and
 * i2. This is because when we map [i0, i1 | i2] to [BIDy, TIDy| BIDx, TIDx]
 * BIDx, and TIDx will access the same elements of T0 on b2, and TIDy will
 * likely access the same elements of T1 (as long as i1 > BDIMy). Even if i1 on
 * the order of BDIMy we'll only access ~two unique elements per increment of
 * BIDx or TIDx. This means we'll still reuse many of the same values and limit
 * the amount we need to read duplicate values in T0 and T1.
 *
 * If instead we have:
 * T0[i0, b1, i2]
 * T1[b0, i1, i2]
 * T2[i0, i1, i2] = T0 + T1
 * The analysis gets a bit more complicated. First if i2 is very large and i0
 * and i1 are relatively small it would make sense to have [i0, i1 | i2]. If b0
 * is very small it's unlikely beneficial to have [i0 | i1, i2] as there would
 * be small reuse on b0, and potentially no reuse on b1. If i2 is very small it
 * may be worthwhile to have [i0 | i1, i2]. If i1 and i2 are not small, and
 * their product is relatively large (i.e. you can't fit T2[i, :, :] in L2) then
 * it's unlikely we'll get any significant reuse across i0.
 *
 * What we should (but don't due to complexity) assume then, is that we will get
 * strong reuse across TIDx and TIDy for dimensions that are on the inner
 * portion of the 2D tile.
 *
 * For example if we have:
 * T0[i0, b1, i2]
 * T1[b0, b1, i2]
 * T2[b0, i1, i2]
 * T3[i0, i1, i2] = T0 + T1 + T2
 * We may want to break point at position 1 or position 2 (i.e. [i0 | i1, i2] or
 * [i0, i1 | i2]). We can't immediately tell from the structure.
 *
 * If we choose [i0, i1 | i2] then we'll get:
 * Strong reuse of T0 on TIDy (b1 dim)
 * Perfect reuse across T1 on TIDy (b0 and b1)
 * If BIDx is bound to the LHS of the tile we'll get:
 * Maybe strong reuse of T0 on BIDx (b1 dim if it's large)
 * Perfect reuse across T1 on BIDx
 * Potentially no reuse on T2 if i1 is very large
 *
 * If we pick [i0 | i1, i2], then we'll get:
 * We'll perfect reuse across TIDy on T1 and T2 on b0
 * Some reuse on T0 and T1 on b1 across BIDx if i2 is relatively small and BIDx
 * is bound to the RHS of the 2D schedule Perfect reuse on T1 and T2 on b0
 * across BIDx if BIDx is bound to the LHS of the 2D schedule
 *
 * Materializing these benefits is dependent on the decisions the scheduler
 * makes when parallelizing the problem. The heuristics logic at the moment is
 * fairly simplistic where it assumes that there's only reuse across the break
 * points for tensors that have no iteration domain on the entire side of the
 * breakpoint. This is not optimal but for the time being it seems sufficient.
 * We would ideally take into consideration the parallelization scheme and
 * partial broadcasting on the lhs or rhs.
 *
 * An example of how this analysis is done is given the DAG:
 * T0[i0, i1, b2] float
 * T1[i0, b1, i2] half
 * T2[i0, b1, i2] = cast(T1, float)
 * T4[i0, i1, i2] float = T0 + T2
 * With values of 10, 100, 1000 as [i0, i1, i2]
 * Our break point analysis for positions 0, 1, 2, 3 will be:
 *
 * 0: 10*10 * 100*10 * 1000*10 = 1e9
 * 1: 10*10 * 100*10 * 1000*10 = 1e9
 * 2: 10*10 * 100*10 * 1000*6  = 6e8
 * 3: 10*10 * 100*10 * 1000*10 = 1e9
 *
 * Where for each computation the LHS of the * pairs is the number of elements
 * in that dimension on the reference and the RHS of the * pairs is the
 * broadcast multiple where any tensor that has all broadcasts on the rhs or lhs
 * of the break point doesn't contribute to the broadcast multiple of the rhs or
 * lhs.
 *
 * So we'll pick position 2 since we're confident we can get broadcast reuse on
 * the rhs of tensor 0. As already mentioned this is a pretty big
 * simplification/assumption and in reality it may be harder/easier to take
 * advantage of broadcast on the inner or outer dimension. This is a reasonable
 * way to make relative decisions on break points, however, this computation is
 * ont doing an effective estimate of actual DRAM transfers which it should be
 * modified to do so.
 *
 * For view schedules there can be some incoherent break points for example:
 * T1[i0, i1*i2] = view(T0[i0, i1, i2])
 * would make the position 2 "incoherent". In otherwords we cannot replay
 * through the view a schedule that tries to merge i0 and i1, without i2. So for
 * positions that are incoherent we won't consider break point positions there.
 *
 * See FusionBroadcastViewMultiples_CUDA for what we expect with view handling.
 * Shortly any dimensions that are inputs or outputs of view transformations are
 * considered together, since it's hard to account for partial dimensions that
 * are being broadcasted. So for view it's primarily an all or nothing situation
 * when it comes to the 2D pointwise scheduler.
 */

class SchedulerRuntimeInfo;
class HeuristicSummary;

TORCH_CUDA_CU_API std::shared_ptr<PointwiseParams> getPointwiseHeuristics(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& runtime_inputs,
    HeuristicSummary* data_cache = nullptr);

TORCH_CUDA_CU_API std::shared_ptr<PointwiseParams> getPointwiseHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache = nullptr);

TORCH_CUDA_CU_API void schedulePointwise(
    Fusion* fusion,
    const PointwiseParams& params);

TORCH_CUDA_CU_API LaunchParams schedulePointwise(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& runtime_inputs);

//! Utility for canSchedule interface to check if this fusion has
//!  a fully broadcasted reference tensor, which is necessary for
//!  the pointwise scheduler.
bool hasReferenceTensorView(Fusion* fusion);

// Return reference tensor view.
TensorView* getReferenceTensorView(Fusion* fusion);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
