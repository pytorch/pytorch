namespace fused_reduction {

namespace impl {

// Utility struct to hold multiple values for grouped Welford. The
// count is uniform, so there's only one N value.
template <int NumVals, typename DataType>
struct WelfordTripletVector {
  Array<DataType, NumVals, NumVals> avg_;
  Array<DataType, NumVals, NumVals> var_;
  nvfuser_index_t N_;

  WelfordTripletVector() = default;

  __device__ WelfordTripletVector(
      const DataType avg[NumVals],
      const DataType var[NumVals],
      const nvfuser_index_t N) {
    memcpy(avg_.array, avg, sizeof(DataType) * NumVals);
    memcpy(var_.array, var, sizeof(DataType) * NumVals);
    N_ = N;
  }

  __device__ WelfordTripletVector& operator=(
      const WelfordTripletVector<NumVals, DataType>& other) {
    avg_ = other.avg_;
    var_ = other.var_;
    N_ = other.N_;
    return *this;
  }

  __device__ void init() {
    avg_.set((DataType)0);
    var_.set((DataType)0);
    N_ = 0;
  }

  __device__ DataType& avg(int idx) {
    return avg_[idx];
  }

  __device__ DataType avg(int idx) const {
    return avg_.array[idx];
  }

  __device__ DataType& var(int idx) {
    return var_[idx];
  }

  __device__ DataType var(int idx) const {
    return var_.array[idx];
  }

  __device__ nvfuser_index_t& N() {
    return N_;
  }

  __device__ nvfuser_index_t N() const {
    return N_;
  }
};

// The offset in smem buffer to broadcast final results within a
// thread block
template <int BDIMX>
__inline__ __device__ int getSmemGroupOffset(int iter_idx, int group_idx) {
  return group_idx * BDIMX + iter_idx;
}

// Upload the final results to smem for intra-block broadcasting
template <int NumVals, typename DataType, int BDIMX, int BDIMY>
__inline__ __device__ void copyFromTripletToSmem(
    DataType* smem,
    int iter_idx,
    int group_idx,
    const WelfordTriplet<DataType>& local_triplet) {
  int offset = getSmemGroupOffset<BDIMX>(iter_idx, group_idx);
  smem[offset] = local_triplet.avg;
  int smem_stride = BDIMX * NumVals;
  smem[smem_stride + offset] = local_triplet.var;
  if (iter_idx == 0 && group_idx == 0) {
    reinterpret_cast<nvfuser_index_t*>(smem + smem_stride * 2)[0] =
        local_triplet.N;
  }
}

// Fetch the final results from smem for intra-block broadcasting
template <int NumVals, typename DataType, int BDIMX, int BDIMY>
__inline__ __device__ void copyFromSmemToTriplet(
    WelfordTriplet<DataType>& local_triplet,
    const DataType* smem,
    int iter_idx,
    int group_idx) {
  int offset = getSmemGroupOffset<BDIMX>(iter_idx, group_idx);
  local_triplet.avg = smem[offset];
  int smem_stride = BDIMX * NumVals;
  local_triplet.var = smem[smem_stride + offset];
  local_triplet.N =
      reinterpret_cast<const nvfuser_index_t*>(smem + smem_stride * 2)[0];
}

// Per-thread accumulation of the per-block partial results in global
// memory. There's gridDim.y partial results, which is accumulated in
// parallel by threadIdx.y. This should be followed by a block reduction.
template <int NumVals, typename DataType, int BDIMX, int BDIMY>
__device__ __inline__ WelfordTripletVector<NumVals, DataType>
welfordGroupAccumulateGlobalBuffer(
    volatile DataType* global_buf_avg,
    volatile DataType* global_buf_var,
    volatile nvfuser_index_t* global_buf_N,
    bool flip) {
  const int grid_size = gridDim.x * gridDim.y;
  const int iter_idx = threadIdx.x;
  const int red_idx = threadIdx.y;
  const int num_threads_per_reduction = BDIMY;

  WelfordTripletVector<NumVals, DataType> results;

  results.init();

  // Reduction is done cooperatively with the thread blocks with the
  // same blockIdx.x. Thread blocks with the same blockIdx.x uses a
  // global buffer of size blockDim.x * gridDim.y for each value in a
  // group.

  // Advance the global buffer pointers to the location of the values
  // to accumulate for the first group value (i.e., gi == 0 in the
  // below NumVals loop)
  global_buf_avg += iter_idx + blockIdx.x * BDIMX * gridDim.y;
  global_buf_var += iter_idx + blockIdx.x * BDIMX * gridDim.y;
  global_buf_N += iter_idx + blockIdx.x * BDIMX * gridDim.y;

  if (flip) {
    global_buf_avg += BDIMX * grid_size * NumVals;
    global_buf_var += BDIMX * grid_size * NumVals;
    global_buf_N += BDIMX * grid_size * NumVals;
  }

  // Since there's gridDim.y elements to reduce using blockDim.y
  // threads, loop over gridDim.y with stride blockDim.y. First, just
  // grab the values in the global memory.

  if (red_idx < gridDim.y) {
    int work_buf_offset = red_idx * BDIMX;
    // N is constant across NumVals
    const auto g_N = global_buf_N[work_buf_offset];
    results.N() = g_N;

    // Just copy the first elements
#pragma unroll
    for (int gi = 0; gi < NumVals; ++gi) {
      auto& a_avg = results.avg(gi);
      auto& a_var = results.var(gi);
      auto b_avg = global_buf_avg[work_buf_offset];
      auto b_var = global_buf_var[work_buf_offset];
      work_buf_offset += grid_size * BDIMX;

      results.avg(gi) = b_avg;
      results.var(gi) = b_var;
    }
  }

  // Accumulate into results by looping over the remaining results in
  // the global buffer
  for (int ri = red_idx + num_threads_per_reduction; ri < gridDim.y;
       ri += num_threads_per_reduction) {
    int work_buf_offset = ri * BDIMX;
    // N is constant across NumVals
    const auto g_N = global_buf_N[work_buf_offset];
    nvfuser_index_t updated_N = results.N() + g_N;

    // Hoist the division by updated_N as it's invariant over the
    // NumVals loop
    DataType b_N_div_ab_N = updated_N != 0
        ? (((DataType)g_N) / ((DataType)updated_N))
        : (DataType)0;
    DataType a_N_b_N_div_ab_N = ((DataType)results.N()) * b_N_div_ab_N;

#pragma unroll
    for (int gi = 0; gi < NumVals; ++gi) {
      auto& a_avg = results.avg(gi);
      auto& a_var = results.var(gi);
      auto b_avg = global_buf_avg[work_buf_offset];
      auto b_var = global_buf_var[work_buf_offset];
      work_buf_offset += grid_size * BDIMX;

      auto delta = b_avg - a_avg;
      a_avg += delta * b_N_div_ab_N;
      a_var += b_var + delta * delta * a_N_b_N_div_ab_N;
    }
    results.N() = updated_N;
  }

  return results;
}

} // namespace impl

template <
    int X_BLOCK,
    int Y_BLOCK,
    int Z_BLOCK,
    int X_THREAD,
    int Y_THREAD,
    int Z_THREAD,
    bool PERSISTENT_REDUCTION,
    bool BROADCAST>
template <int NumVals, typename DataType, int BDIMX, int BDIMY>
__device__ __inline__ void ParallelReduce<
    X_BLOCK,
    Y_BLOCK,
    Z_BLOCK,
    X_THREAD,
    Y_THREAD,
    Z_THREAD,
    PERSISTENT_REDUCTION,
    BROADCAST>::
    welfordGroupOuter(
        DataType out_avg[NumVals],
        DataType out_var[NumVals],
        nvfuser_index_t out_N[NumVals],
        const DataType in_avg[NumVals],
        const DataType in_var[NumVals],
        nvfuser_index_t in_N,
        DataType* global_buf_avg,
        DataType* global_buf_var,
        nvfuser_index_t* global_buf_N,
        DataType* shared_buf,
        int64_t* global_sync_buffer) {
  using namespace fused_reduction::impl;

  static_assert(
      isIter(X_BLOCK) && isReduce(Y_BLOCK) && inactive(Z_BLOCK) &&
          isIter(X_THREAD) && isReduce(Y_THREAD) && inactive(Z_THREAD),
      "Invalid parallelization for outer welford reduction");

  static_assert(
      BDIMY % NumVals == 0, "blockDim.y must be divisible by group count");
  static_assert(BDIMX <= 32, "blockDim.x must be up to 32.");
  static_assert(
      (BDIMX * BDIMY) % 32 == 0, "Number of threads must be a multiple of 32.");
  static_assert(32 % BDIMX == 0, "blockDim.x must be able to divide 32.");
  static_assert(
      NumVals >= (32 / BDIMX), "Group count must be >= 32 / blockDim.x");

#pragma unroll
  for (int i = 0; i < NumVals; ++i) {
    out_avg[i] = in_avg[i];
    out_var[i] = in_var[i];
  }

  auto iter_tid = index_utils::
      maskedOffset<isIter(X_THREAD), isIter(Y_THREAD), isIter(Z_THREAD)>(
          threadIdx, blockDim);

  auto per_block_result =
      impl::blockWelfordOuter<NumVals, DataType, BDIMX, BDIMY>(
          out_avg, out_var, in_N, shared_buf);

  // At this point, threads with tid_in_group == 0 has valid partial
  // results. Store them to global buffer.

  const int grid_size = gridDim.x * gridDim.y;
  const int iter_idx = threadIdx.x;

  // Stores the partial results into the global work buffer. Only
  // threads with tid_in_group have the valid partial results
  const int wid = (threadIdx.x + threadIdx.y * BDIMX) / 32;
  constexpr int wdimy = 32 / BDIMX;
  const int warp_tidy = threadIdx.y % wdimy;
  const bool has_valid_block_reduction_result = warp_tidy == 0 && wid < NumVals;
  // Each valid result is held by a warp
  const int valid_group_idx = wid;

  if (has_valid_block_reduction_result) {
    int work_buf_offset = iter_idx + blockIdx.y * BDIMX +
        blockIdx.x * BDIMX * gridDim.y + valid_group_idx * BDIMX * grid_size;
    if (PERSISTENT_REDUCTION && flip) {
      auto global_buffer_size = BDIMX * grid_size * NumVals;
      work_buf_offset += global_buffer_size;
    }

    global_buf_avg[work_buf_offset] = per_block_result.avg;
    global_buf_var[work_buf_offset] = per_block_result.var;

    // the count values should be the same across the group, so just
    // store once
    if (valid_group_idx == 0) {
      global_buf_N[work_buf_offset] = per_block_result.N;
    }
  }

  flip = !flip;

  // -- GLOBAL BUFFER FILLED -- //

  bool last_block = index_utils::
      maskedIsLast<isReduce(X_BLOCK), isReduce(Y_BLOCK), isReduce(Z_BLOCK)>(
          blockIdx, gridDim);

  grid_sync::sync<
      isReduce(X_BLOCK),
      isReduce(Y_BLOCK),
      isReduce(Z_BLOCK),
      PERSISTENT_REDUCTION>(
      global_sync_buffer[blockIdx.x], gridDim.y, last_block);

  auto partial_results =
      welfordGroupAccumulateGlobalBuffer<NumVals, DataType, BDIMX, BDIMY>(
          global_buf_avg, global_buf_var, global_buf_N, !flip);

  auto per_block_final_result =
      impl::blockWelfordOuter<NumVals, DataType, BDIMX, BDIMY>(
          partial_results.avg_.array,
          partial_results.var_.array,
          partial_results.N_,
          shared_buf);

  // At this point, each thread of the groups with tid_in_group=0
  // has the final reduction result. We need to upload them to
  // shmem for broadcasting.
  if (has_valid_block_reduction_result) {
    copyFromTripletToSmem<NumVals, DataType, BDIMX, BDIMY>(
        shared_buf, iter_idx, valid_group_idx, per_block_final_result);
  }

  __syncthreads();

#pragma unroll
  for (int i = 0; i < NumVals; ++i) {
    WelfordTriplet<DataType> final_result;
    copyFromSmemToTriplet<NumVals, DataType, BDIMX, BDIMY>(
        final_result, shared_buf, iter_idx, i);
    out_avg[i] = final_result.avg;
    out_var[i] = final_result.var;
    in_N = final_result.N;
  }

#pragma unroll
  for (int i = 0; i < NumVals; ++i) {
    out_N[i] = in_N;
  }

  // Forward protect the smem buffer
  __syncthreads();
}

template <
    int X_BLOCK,
    int Y_BLOCK,
    int Z_BLOCK,
    int X_THREAD,
    int Y_THREAD,
    int Z_THREAD,
    bool PERSISTENT_REDUCTION,
    bool BROADCAST>
template <int NumVals, typename DataType, int BDIMX, int BDIMY>
__device__ __inline__ void ParallelReduce<
    X_BLOCK,
    Y_BLOCK,
    Z_BLOCK,
    X_THREAD,
    Y_THREAD,
    Z_THREAD,
    PERSISTENT_REDUCTION,
    BROADCAST>::
    welfordGroupOuter(
        DataType out_avg[NumVals],
        DataType out_var[NumVals],
        nvfuser_index_t out_N[NumVals],
        const DataType in_avg[NumVals],
        const DataType in_var[NumVals],
        nvfuser_index_t in_N,
        DataType* global_buf_avg,
        DataType* global_buf_var,
        nvfuser_index_t* global_buf_N,
        DataType* shared_buf,
        int64_t* global_sync_buffer,
        int64_t& cycles,
        int64_t& count) {
  int64_t start_counter = 0;

  if (isLastBlockInGrid() &&
      index_utils::maskedIsZero<true, true, true>(threadIdx)) {
    start_counter = readCycleCounter();
  }

  welfordGroupOuter<NumVals, DataType, BDIMX, BDIMY>(
      out_avg,
      out_var,
      out_N,
      in_avg,
      in_var,
      in_N,
      global_buf_avg,
      global_buf_var,
      global_buf_N,
      shared_buf,
      global_sync_buffer);

  if (isLastBlockInGrid() &&
      index_utils::maskedIsZero<true, true, true>(threadIdx)) {
    cycles += readCycleCounter() - start_counter;
    ++count;
  }
}

} // namespace fused_reduction
