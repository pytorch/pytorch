namespace fused_reduction {
namespace impl {

// Grouped block welford optimized for outer reductions with
// TIDx and TIDy mapped to non-reduction and reduction domains,
// respectively with unused TIDz.
//
// The main motivation of this optimized version is the same as the
// grouped grid reduction, i.e, by doing multiple reductions together,
// it is possible to reduce the number of synchronizations. However,
// unlike the grouped grid reduction, the cost of grouping can be
// prohitively high, i.e., the size of the work buffer must be
// expanded by a factor of grouping. In the case of grid
// reductions, the buffer is on global memory, so the space requirement
// is not a concern, but that isn't the case with block reductions,
// since the buffer is on shared memory, which has a limited
// capacity.
//
// This implementation tries to benefit from aggregated block
// synchronizations while minimizing the cost of the expanded buffer
// size by first partially reducing the input within each warp. It
// would save the required buffer size by a factor of WARP_SIZE /
// blockDim.x as the reduction is done along threadIdx.y. So to be
// effective, blockDim.x needs to be smaller than WARP_SIZE, and in the
// case of grouped grid welford, it should be typically 8 or 16.
//
// The algorithm is an adaptation of scattered butterfly reduction,
// aka recursive halving, commonly used for implementing
// MPI_Reduce_scatter. For a visual illustration of the data
// organization, see, for example, page 22 of Solomonik,
// Design of Parallel and High-Performance Computing:
// Distributed-Memory Models and Algorithms, 2015
// (https://solomonik.cs.illinois.edu/talks/dphpc-dec-2015.pdf)
//
// Assumptions:
// - blockDim.x and blockDim.y are statically known values so that all
// loops can be completely unrolled
// - blockDim.x is smaller than WARP_SIZE
// - blockDim.x evenly divides WARP_SIZE
// - There are multiple warps per block
// - The gouping factor, NumVals, is at least as large as the warp
// dimY and is divisible by the warp dimY.
//
// This is meant to be used as part of the grouped grid welford
// reduction but should be usable as a standalone block welford routine as
// long as the above assumptions hold.
//
// Note: Having an output reference parameter resulted in using more
// registers than just returing the output. Results would vary
// depending on compiler versions, but it seems safer to return outputs
// as a new value.
template <int NumVals, typename DataType, int BDIMX, int BDIMY>
__inline__ __device__ WelfordTriplet<DataType> blockWelfordOuter(
    DataType* inp_avg,
    DataType* inp_var,
    nvfuser_index_t inp_N,
    DataType* smem) {
  constexpr int num_warps = BDIMX * BDIMY / 32;
  static_assert(num_warps >= 1, "There must be at least a single warp");
  static_assert(32 % BDIMX == 0, "blockDimx.x must be able to divide 32");

  const int tid = threadIdx.x + threadIdx.y * BDIMX;
  const int wid = tid / 32;

  // Dimension of the Y axis within each warp
  constexpr int wdimy = 32 / BDIMX;
  static_assert(NumVals >= wdimy, "NumVals must be >= 32 / blockDim.x");
  static_assert(
      NumVals % wdimy == 0, "NumVals must be divisible by 32 / blockDim.x");
  // There must be at least a single warp

  // Y index within each warp
  const int warp_tidy = threadIdx.y % wdimy;

  // Thread index in each warp
  const int lane_id = threadIdx.x + warp_tidy * BDIMX;

  constexpr int smem_var_offset = num_warps * BDIMX * NumVals;
  constexpr int smem_N_offset = num_warps * BDIMX * NumVals * 2;

  // We define a chunk as a value in a group and a chunk size as the
  // number of group values per thread. Initially, the chunk size is
  // NumVals. After the initial warp reduction, the chunk size is
  // reduced to NumVals/wdimy. For example, suppose NumVals=8,
  // blockDim.x=8, blockDim.y=32, then wdimy=4, so after the initial
  // warp reduction, the chunk size is 2. This is the number of
  // elements each thread stores to shared memory.

  int chunk_size = NumVals;

  // Butterfly reduction, a.k.a. recursive halving as each iteration
  // halves the number of values
#pragma unroll
  for (int lane_mask = 16; lane_mask >= BDIMX; lane_mask /= 2) {
    chunk_size /= 2;

    const auto peer_N = __shfl_xor_sync(0xffffffff, inp_N, lane_mask);
    const auto updated_N = inp_N + peer_N;
    const DataType b_N_div_ab_N =
        updated_N != 0 ? ((DataType)peer_N) / ((DataType)updated_N) : 0;

#pragma unroll
    for (int index_in_chunk = 0; index_in_chunk < chunk_size;
         ++index_in_chunk) {
      DataType pushed_avg = 0;
      DataType pushed_var = 0;
      DataType self_avg = 0;
      DataType self_var = 0;
      // Divergent branch. Not a big deal with independent scheduling?
      if (lane_id & lane_mask) {
        // Push first half
        auto push_offset = index_in_chunk;
        auto self_offset = index_in_chunk + chunk_size;
        pushed_avg = inp_avg[push_offset];
        pushed_var = inp_var[push_offset];
        self_avg = inp_avg[self_offset];
        self_var = inp_var[self_offset];
      } else {
        // Push second half
        auto push_offset = index_in_chunk + chunk_size;
        auto self_offset = index_in_chunk;
        pushed_avg = inp_avg[push_offset];
        pushed_var = inp_var[push_offset];
        self_avg = inp_avg[self_offset];
        self_var = inp_var[self_offset];
      }
      auto peer_avg = __shfl_xor_sync(0xffffffff, pushed_avg, lane_mask);
      auto peer_var = __shfl_xor_sync(0xffffffff, pushed_var, lane_mask);

      auto delta = peer_avg - self_avg;
      self_avg += delta * b_N_div_ab_N;
      self_var += peer_var + delta * delta * ((DataType)(inp_N)) * b_N_div_ab_N;

      inp_avg[index_in_chunk] = self_avg;
      inp_var[index_in_chunk] = self_var;
    }
    inp_N = updated_N;
  }

  // At this point, chunk_size is reduced to NumVals/wdimy as
  // mentioned above. Each thread has warp-reduced chunk_size values
  // in array inp. This chunk_size_post_reduction should be equal to
  // chunk_size at this point.
  constexpr int chunk_size_post_reduction = NumVals / wdimy;

  // More specifically, the warp_tidy of each thread defines
  // the chunk IDs held by the thread as follows:
  //
  // [warp_tidy * chunk_size_post_reduction, warp_tidy *
  // chunk_size_post_reduction + chunk_size_post_reduction]
  //
  // Each thread uploads the chunk_size_post_reduction values one by
  // one. Each chunk is spread by BDIMX * BDIMY values. The data
  // layout of the shared memory is:
  //
  // [chunk_size, wid, warp_tidy, TIDx]
  //
  // The remaining reduction is done on the WID
  // dimension. More specifically, we assign one warp per chunk (or
  // a value of the group). The wdimy threads of the same threadId.x
  // collectively reduce num_warps partial results, each of which is
  // stored with stride 32. This means that there will be wdimy-way
  // bank conflicts, so to avoid that, swizzling is also employed.
#pragma unroll
  for (int i = 0; i < chunk_size; ++i) {
    // Accumulating smem offset from the innermost dimension
    int smem_offset = 0;
    // TIDx
    smem_offset += threadIdx.x;
    // Warp_TIDy with swizzle
    smem_offset += ((warp_tidy + wid) % wdimy) * BDIMX;
    // WID
    smem_offset += wid * 32;
    // chunk_size
    smem_offset += i * BDIMX * BDIMY;
    smem[smem_offset] = inp_avg[i];
    smem[smem_var_offset + smem_offset] = inp_var[i];
    // Upload N only when threadIdx.x == 0 && chunk_index == 0
    if (threadIdx.x == 0 && i == 0 && warp_tidy == 0) {
      reinterpret_cast<nvfuser_index_t*>(smem + smem_N_offset)[wid] = inp_N;
    }
  }

  __syncthreads();

  // The next step is to let each thread of a warp independently
  // accumulate the partial results on the shared memory
  // reduction. A single warp is used to accumulate of the partial
  // results for a single chunk, so warp wid takes care of the wid-th
  // chunk.
  //
  // The starting offset of partial results of a chunk is:
  //
  // (wid % chunk_size_post_reduction) * BDIMX * BDIMY + (wid /
  // chunk_size_post_reduction) * BDIMX
  //
  // Note that each thread had chunk_size_post_reduction contiguous
  // chunks, so when uploaded to shmem, they are strided by
  // BDIMX*BDIMY, hence (wid % chunk_size_post_reduction) * BDIMX *
  // BDIMY.

  // The vector width is likely at least 4, so at least 4 warps should
  // be used, which is
  // enough to occupy an SM. When NumVals=8, it might be more
  // efficient to use just 4 warps with each warp taking care of two
  // groups, but the difference would be pretty small.

  // Also, the number of warps should be at least 8 and can be 16
  // too. NumVals should be 8 at largest, so it's always num_warps >=
  // NumVals.

  DataType avg = 0;
  DataType var = 0;
  nvfuser_index_t N = 0;

  static_assert(
      num_warps >= NumVals,
      "Number of warps must be at least as large as NumVals");

  if (wid < NumVals) {
#pragma unroll
    for (int i = warp_tidy; i < num_warps; i += wdimy) {
      int offset = 0;
      offset += threadIdx.x;
      // Offset to the partial results of the i-th warp
      offset += i * 32;
      // Offset to the chunk for this warp. Swizzled to avoid bank
      // conflicts.
      offset += ((wid / chunk_size + i) % wdimy) * BDIMX;
      offset += (wid % chunk_size) * BDIMX * BDIMY;

      DataType avg_smem = smem[offset];
      DataType var_smem = smem[smem_var_offset + offset];
      nvfuser_index_t N_smem =
          reinterpret_cast<nvfuser_index_t*>(&smem[smem_N_offset])[i];

      welfordCombine(avg, var, N, avg_smem, var_smem, N_smem);
    }
  }

  __syncthreads();

  // Nothing to do for warps whose wid is larger than NunVals
  if (wid >= NumVals) {
    WelfordTriplet<DataType> out = {0, 0, 0};
    return out;
  }

  // Standard binary-exchange reduction within wdimy intra-warp
  // threads.
#pragma unroll
  for (int lane_mask = 16; lane_mask >= BDIMX; lane_mask /= 2) {
    auto avg_peer = __shfl_xor_sync(0xffffffff, avg, lane_mask);
    auto var_peer = __shfl_xor_sync(0xffffffff, var, lane_mask);
    auto N_peer = __shfl_xor_sync(0xffffffff, N, lane_mask);

    welfordCombine(avg, var, N, avg_peer, var_peer, N_peer);
  }

  WelfordTriplet<DataType> out = {avg, var, N};
  return out;
}

} // namespace impl
} // namespace fused_reduction
