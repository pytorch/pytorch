namespace fused_reduction {

namespace impl {

//! Implementation helper for welfordEach.
template <int ValIdx, typename Triplet0, typename Triplet1>
struct WelfordForEach {
  static __inline__ __device__ void call(
      Triplet0& triplet0,
      nvfuser_index_t offset0,
      const Triplet1& triplet1,
      nvfuser_index_t offset1) {
    static_assert(
        Triplet0::num_vals == Triplet1::num_vals, "Invalid Triplet types");
    static_assert(
        IsSameType<typename Triplet0::DataType, typename Triplet1::DataType>::
            value,
        "Invalid Triplet types");
    static_assert(
        IsSameType<typename Triplet0::IndexType, typename Triplet1::IndexType>::
            value,
        "Invalid Triplet types");

    using DataType = typename Triplet0::DataType;
    using IndexType = typename Triplet0::IndexType;

    WelfordForEach<ValIdx - 1, Triplet0, Triplet1>::call(
        triplet0, offset0, triplet1, offset1);
    welfordCombine<DataType, IndexType>(
        triplet0.avg.val<ValIdx>(offset0),
        triplet0.var.val<ValIdx>(offset0),
        triplet0.N.val<ValIdx>(offset0),
        triplet1.avg.val<ValIdx>(offset1),
        triplet1.var.val<ValIdx>(offset1),
        triplet1.N.val<ValIdx>(offset1));
  }
};

template <typename Triplet0, typename Triplet1>
struct WelfordForEach<-1, Triplet0, Triplet1> {
  __inline__ __device__ static void call(
      Triplet0& triplet0,
      nvfuser_index_t offset0,
      const Triplet1& triplet1,
      nvfuser_index_t offset1) {}
};

//! Call welfordCombine with each of the triplet tuples. This is a
//! welford version of reduceEach.
template <typename Triplet0, typename Triplet1>
__inline__ __device__ static void welfordEach(
    Triplet0& triplet0,
    nvfuser_index_t offset0,
    const Triplet1& triplet1,
    nvfuser_index_t offset1) {
  WelfordForEach<Triplet0::num_vals - 1, Triplet0, Triplet1>::call(
      triplet0, offset0, triplet1, offset1);
}

// Welford version of BlockReduceEach
template <
    int idx,
    bool BROADCAST,
    bool FORWARD_PROTECT_SMEM,
    typename LocalWelfordTripletTupleT>
struct BlockWelfordEach {
  __inline__ __device__ static void reduce(
      LocalWelfordTripletTupleT& block_result,
      const LocalWelfordTripletTupleT& partial_result,
      PtrTuple<
          typename LocalWelfordTripletTupleT::DataType,
          typename LocalWelfordTripletTupleT::DataType,
          typename LocalWelfordTripletTupleT::IndexType> shared_buf,
      bool has_block_result,
      int tid_in_reduction,
      int num_threads_per_reduction,
      int num_elements_per_reduction,
      int reduction_idx) {
    // Finish the reduction of each tuple value with a smaller offset
    BlockWelfordEach<idx - 1, BROADCAST, true, LocalWelfordTripletTupleT>::
        reduce(
            block_result,
            partial_result,
            shared_buf,
            has_block_result,
            tid_in_reduction,
            num_threads_per_reduction,
            num_elements_per_reduction,
            reduction_idx);

    if (num_elements_per_reduction == 1) {
      if (has_block_result) {
        copyWelfordTripletTuple(block_result, partial_result);
      }
      return;
    }

    using DataType = typename LocalWelfordTripletTupleT::DataType;
    using IndexType = typename LocalWelfordTripletTupleT::IndexType;

    LocalTuple<DataType, DataType, IndexType> block_result_i(
        partial_result.avg.val<idx>(0),
        partial_result.var.val<idx>(0),
        partial_result.N.val<idx>(0));

    const auto smem_offset =
        reduction_idx * num_threads_per_reduction + tid_in_reduction;

    const int np2 = 1 << (31 - __clz(num_elements_per_reduction));

    // Threads values are initialized, so all can participate here
    if (tid_in_reduction >= np2) {
      copyTuple(shared_buf, smem_offset, block_result_i);
    }

    block_sync::sync();
    if (tid_in_reduction < np2 &&
        tid_in_reduction + np2 < num_elements_per_reduction) {
      impl::reduceTuple(
          block_result_i,
          0,
          shared_buf,
          smem_offset + np2,
          welfordCombine<DataType, IndexType>);
    }

    if (tid_in_reduction < np2) {
      copyTuple(shared_buf, smem_offset, block_result_i);
    }

    // Always sync when communicating across smem
    block_sync::sync();

    // Reduce down to 2 values, last thread will do the final reduction and
    // can save a syncthreads this way
    for (int factor = np2 / 2; factor > 1; factor >>= 1) {
      if (tid_in_reduction < factor) {
        impl::reduceTuple(
            shared_buf,
            smem_offset,
            shared_buf,
            smem_offset + factor,
            welfordCombine<DataType, IndexType>);
      }
      block_sync::sync();
    }

    copyTuple(block_result_i, shared_buf, smem_offset);

    // Do the last reduction
    if (has_block_result) {
      impl::reduceTuple(
          block_result_i,
          0,
          shared_buf,
          smem_offset + 1,
          welfordCombine<DataType, IndexType>);
    }

    if (BROADCAST) {
      if (has_block_result) {
        // Put result back in shared memory, put in the first entry of the
        // reduction segment's buffer
        copyTuple(
            shared_buf,
            reduction_idx * num_threads_per_reduction,
            block_result_i);
      }

      // Sync threads to make sure result is in smem
      block_sync::sync();

      copyTuple(
          block_result_i,
          shared_buf,
          reduction_idx * num_threads_per_reduction);
    }

    block_result.avg.val<idx>(0) = block_result_i.val<0>(0);
    block_result.var.val<idx>(0) = block_result_i.val<1>(0);
    block_result.N.val<idx>(0) = block_result_i.val<2>(0);

    if (FORWARD_PROTECT_SMEM) {
      block_sync::sync();
    }
  }
};

// Specialization for idx == -1, i.e., no value to reduce.
template <
    bool BROADCAST,
    bool FORWARD_PROTECT_SMEM,
    typename LocalWelfordTripletTupleT>
struct BlockWelfordEach<
    -1,
    BROADCAST,
    FORWARD_PROTECT_SMEM,
    LocalWelfordTripletTupleT> {
  __inline__ __device__ static void reduce(
      LocalWelfordTripletTupleT& block_result,
      const LocalWelfordTripletTupleT& partial_result,
      PtrTuple<
          typename LocalWelfordTripletTupleT::DataType,
          typename LocalWelfordTripletTupleT::DataType,
          typename LocalWelfordTripletTupleT::IndexType> shared_buf,
      bool has_block_result,
      int tid_in_reduction,
      int num_threads_per_reduction,
      int num_elements_per_reduction,
      int reduction_idx) {}
};

//! Welford version of blockReduceEach. Perform block-parallel Welford
//! reduction of each Welford triplet.
template <
    bool BROADCAST,
    bool FORWARD_PROTECT_SMEM,
    typename LocalWelfordTripletTupleT>
__inline__ __device__ void blockWelfordEach(
    LocalWelfordTripletTupleT& block_result,
    const LocalWelfordTripletTupleT& partial_result,
    PtrTuple<
        typename LocalWelfordTripletTupleT::DataType,
        typename LocalWelfordTripletTupleT::DataType,
        typename LocalWelfordTripletTupleT::IndexType> shared_buf,
    bool has_block_result,
    int tid_in_reduction,
    int num_threads_per_reduction,
    int num_elements_per_reduction,
    int reduction_idx) {
  BlockWelfordEach<
      LocalWelfordTripletTupleT::num_vals - 1,
      BROADCAST,
      FORWARD_PROTECT_SMEM,
      LocalWelfordTripletTupleT>::
      reduce(
          block_result,
          partial_result,
          shared_buf,
          has_block_result,
          tid_in_reduction,
          num_threads_per_reduction,
          num_elements_per_reduction,
          reduction_idx);
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
template <int NumArgs, typename DataType, typename IndexType>
__device__ __inline__ void ParallelReduce<
    X_BLOCK,
    Y_BLOCK,
    Z_BLOCK,
    X_THREAD,
    Y_THREAD,
    Z_THREAD,
    PERSISTENT_REDUCTION,
    BROADCAST>::
    welfordGroup(
        typename MakeRefTuple<NumArgs, DataType>::type out_avg,
        typename MakeRefTuple<NumArgs, DataType>::type out_var,
        typename MakeRefTuple<NumArgs, IndexType>::type out_N,
        const typename MakeConstRefTuple<NumArgs, DataType>::type& inp_avg,
        const typename MakeConstRefTuple<NumArgs, DataType>::type& inp_var,
        const typename MakeConstRefTuple<NumArgs, IndexType>::type& inp_N,
        const typename MakeLocalTuple<NumArgs, DataType>::type& init_avg,
        const typename MakeLocalTuple<NumArgs, DataType>::type& init_var,
        const typename MakeLocalTuple<NumArgs, IndexType>::type& init_N,
        typename MakeVolatilePtrTuple<NumArgs, DataType>::type
            global_work_buffer_avg,
        typename MakeVolatilePtrTuple<NumArgs, DataType>::type
            global_work_buffer_var,
        typename MakeVolatilePtrTuple<NumArgs, IndexType>::type
            global_work_buffer_N,
        int64_t* global_sync_buffer,
        PtrTuple<DataType, DataType, IndexType> shared_buf,
        const typename MakeLocalTuple<NumArgs, bool>::type& read_preds,
        const typename MakeLocalTuple<NumArgs, bool>::type& write_preds) {
  const ConstRefWelfordTripletTuple<NumArgs, DataType, IndexType> inp(
      inp_avg, inp_var, inp_N);
  RefWelfordTripletTuple<NumArgs, DataType, IndexType> out(
      out_avg, out_var, out_N);

  // If no reduction needed, just return input
  if (!BLOCK_REDUCE && !GRID_REDUCE) {
    copyWelfordTripletTupleIf(out, inp, read_preds && write_preds);
    return;
  }

  // Don't read/write in temporary buffers if in a predicated dimension
  const bool block_reduce_participate = index_utils::
      maskedIsZero<isPred(X_THREAD), isPred(Y_THREAD), isPred(Z_THREAD)>(
          threadIdx);

  // Only threads that with id == 0 in the dimensions being reduced will
  // have a valid result
  const bool has_block_result = index_utils::
      maskedIsZero<isReduce(X_THREAD), isReduce(Y_THREAD), isReduce(Z_THREAD)>(
          threadIdx);

  LocalWelfordTripletTuple<NumArgs, DataType, IndexType> block_result(
      init_avg, init_var, init_N);

  // Initial per-block reduction. Result is broadcast if specified
  // and this call is block reduction only.
  welfordGroupBlock<!GRID_REDUCE && BROADCAST>(
      block_result, inp, shared_buf, read_preds, block_reduce_participate);

  // If block reduction only, save to out and exit
  if (!GRID_REDUCE) {
    copyWelfordTripletTupleIf(
        out,
        block_result,
        write_preds &&
            (block_reduce_participate && (BROADCAST || has_block_result)));

    // Need a block sync here as reduceGroupBlock does not
    // forward-protect the smem buffer. This block sync is not
    // necessary when a grid reduction follows since a block sync is
    // done just before the grid sync.
    block_sync::sync();
    return;
  }

  // -- START GRID REDUCTION -- //
  // Grid reductions are more challenging for two reasons, (1) the reduction
  // itself is 3D instead of 2D because we now have an iter domain space in
  // the grid dimension. (2) a tree reduction isn't performed, instead all
  // blocks will populate GMEM and one  block will finish the grid reduction.

  // What is the grid reduction size, block reduction already performed so
  // that doesn't have to be taken into consideration
  const auto grid_red_size = index_utils::
      maskedSize<isReduce(X_BLOCK), isReduce(Y_BLOCK), isReduce(Z_BLOCK)>(
          gridDim);

  // Which ID in the reduction is this block. Threads can participate in
  // multiple grid reductions, but the block will have the same relative index
  // in those reductions
  const auto idx_in_grid_red = index_utils::
      maskedOffset<isReduce(X_BLOCK), isReduce(Y_BLOCK), isReduce(Z_BLOCK)>(
          blockIdx, gridDim);

  // How many grid reductions have to be performed, in the grid dimension
  const auto num_block_iters = index_utils::
      maskedSize<isIter(X_BLOCK), isIter(Y_BLOCK), isIter(Z_BLOCK)>(gridDim);

  // Which grid reduction does this block participate in, in the grid
  // dimension
  const auto block_red_idx_offset = index_utils::
      maskedOffset<isIter(X_BLOCK), isIter(Y_BLOCK), isIter(Z_BLOCK)>(
          blockIdx, gridDim);

  // How many grid reductions have to be performed, in the block dimension
  const auto num_thread_iters = index_utils::
      maskedSize<isIter(X_THREAD), isIter(Y_THREAD), isIter(Z_THREAD)>(
          blockDim);

  // Which grid reduction does this thread participate in, in the block
  // dimension
  const auto thread_red_idx_offset = index_utils::
      maskedOffset<isIter(X_THREAD), isIter(Y_THREAD), isIter(Z_THREAD)>(
          threadIdx, blockDim);

  // 3D buffer of reductions:
  //    [reduction_offset(grid), iter_offset(grid), iter_offset(block)]
  // Offset into the work buffer
  auto work_buf_offset =
      (idx_in_grid_red * num_block_iters + block_red_idx_offset) *
          num_thread_iters +
      thread_red_idx_offset;

  // Don't read/write in temporary buffers if in a predicated dimension
  bool grid_reduce_participate = index_utils::
      maskedIsZero<isPred(X_BLOCK), isPred(Y_BLOCK), isPred(Z_BLOCK)>(blockIdx);

  VolatilePtrWelfordTripletTuple<NumArgs, DataType, IndexType>
      global_work_buffer(
          global_work_buffer_avg, global_work_buffer_var, global_work_buffer_N);

  if (PERSISTENT_REDUCTION && flip) {
    auto global_buffer_size =
        index_utils::
            maskedSize<isIter(X_BLOCK), isIter(Y_BLOCK), isIter(Z_BLOCK)>(
                gridDim) *
        index_utils::
            maskedSize<isIter(X_THREAD), isIter(Y_THREAD), isIter(Z_THREAD)>(
                blockDim) *
        grid_red_size;
    global_work_buffer += global_buffer_size;
  }
  flip = !flip;

  // Per-block partial reduction to global work buffer
  if (grid_reduce_participate && block_reduce_participate && has_block_result) {
    copyWelfordTripletTuple(global_work_buffer, work_buf_offset, block_result);
  }

  // -- GLOBAL BUFFER FILLED -- //

  bool last_block = index_utils::
      maskedIsLast<isReduce(X_BLOCK), isReduce(Y_BLOCK), isReduce(Z_BLOCK)>(
          blockIdx, gridDim);

  if (grid_reduce_participate) {
    // Don't need to sync up blocks that are not participating in this
    // reduction
    grid_sync::sync<
        isReduce(X_BLOCK),
        isReduce(Y_BLOCK),
        isReduce(Z_BLOCK),
        PERSISTENT_REDUCTION>(
        global_sync_buffer[block_red_idx_offset], grid_red_size, last_block);
  }

  // -- START BLOCK CLEANUP -- //
  welfordGroupLastBlock(
      out,
      global_work_buffer,
      LocalWelfordTripletTuple<NumArgs, DataType, IndexType>(
          init_avg, init_var, init_N),
      shared_buf,
      block_red_idx_offset,
      num_thread_iters,
      num_block_iters,
      thread_red_idx_offset,
      grid_red_size,
      write_preds,
      block_reduce_participate,
      grid_reduce_participate);

  // Forward protect the smem buffer
  block_sync::sync();
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
template <
    bool BLOCK_BROADCAST,
    int NumVals,
    typename DataType,
    typename IndexType>
__device__ __inline__ void ParallelReduce<
    X_BLOCK,
    Y_BLOCK,
    Z_BLOCK,
    X_THREAD,
    Y_THREAD,
    Z_THREAD,
    PERSISTENT_REDUCTION,
    BROADCAST>::
    welfordGroupBlock(
        LocalWelfordTripletTuple<NumVals, DataType, IndexType>& block_result,
        const ConstRefWelfordTripletTuple<NumVals, DataType, IndexType>& inp,
        PtrTuple<DataType, DataType, IndexType> shared_buf,
        const typename MakeLocalTuple<NumVals, bool>::type& read_preds,
        bool block_reduce_participate) {
  const bool has_block_result = index_utils::
      maskedIsZero<isReduce(X_THREAD), isReduce(Y_THREAD), isReduce(Z_THREAD)>(
          threadIdx);

  copyWelfordTripletTupleIf(
      block_result, inp, block_reduce_participate && read_preds);

  // Size of the block reduction segment, can be an int since it's limited
  // to number of threads
  const int block_reduction_size = index_utils::
      maskedSize<isReduce(X_THREAD), isReduce(Y_THREAD), isReduce(Z_THREAD)>(
          blockDim);

  // Index in the reduction segment, can be an int since it's limited to
  // number of threads
  const int tid_in_block_reduction = index_utils::
      maskedOffset<isReduce(X_THREAD), isReduce(Y_THREAD), isReduce(Z_THREAD)>(
          threadIdx, blockDim);

  // ID of the block reduction this thread is participating in
  //
  // If any of the parallel dimensions are predicated out, that means
  // they've already been reduced, so we only care about the first thread in
  // that dimension. Therefore don't expand the reduction_idx by that
  // dimension
  const int block_reduction_idx = index_utils::
      maskedOffset<isIter(X_THREAD), isIter(Y_THREAD), isIter(Z_THREAD)>(
          threadIdx, blockDim);

  // Do not protect the smem buffer as it's not always necessary.
  impl::blockWelfordEach<
      BLOCK_BROADCAST,
      false,
      LocalWelfordTripletTuple<NumVals, DataType, IndexType>>(
      block_result,
      block_result,
      shared_buf,
      has_block_result,
      tid_in_block_reduction,
      block_reduction_size,
      block_reduction_size,
      block_reduction_idx);
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
template <int NumVals, typename DataType, typename IndexType>
__device__ __inline__ void ParallelReduce<
    X_BLOCK,
    Y_BLOCK,
    Z_BLOCK,
    X_THREAD,
    Y_THREAD,
    Z_THREAD,
    PERSISTENT_REDUCTION,
    BROADCAST>::
    welfordGroupLastBlock(
        RefWelfordTripletTuple<NumVals, DataType, IndexType>& out,
        const VolatilePtrWelfordTripletTuple<NumVals, DataType, IndexType>&
            global_work_buffer,
        const LocalWelfordTripletTuple<NumVals, DataType, IndexType>& init_val,
        PtrTuple<DataType, DataType, IndexType> shared_buf,
        nvfuser_index_t block_red_idx_offset,
        nvfuser_index_t num_thread_iters,
        nvfuser_index_t num_block_iters,
        nvfuser_index_t thread_red_idx_offset,
        nvfuser_index_t grid_red_size,
        const typename MakeLocalTuple<NumVals, bool>::type& write_preds,
        bool block_reduce_participate,
        bool grid_reduce_participate) {
  // Initialize block result
  auto last_block_result = init_val;

  const bool last_block = index_utils::
      maskedIsLast<isReduce(X_BLOCK), isReduce(Y_BLOCK), isReduce(Z_BLOCK)>(
          blockIdx, gridDim);

  if ((PERSISTENT_REDUCTION || last_block) && grid_reduce_participate) {
    // Can use the last block to reduce all the values the blocks filled in.
    // Can use any thread that has been predicated, or has been reduced to do
    // this reduction, cannot use any block that's associated with an
    // iteration domain

    // Start with non-block reduction

    // Index in the reduction segment
    int tid_in_block_reduction = index_utils::maskedOffset<
        activeNotIter(X_THREAD),
        activeNotIter(Y_THREAD),
        activeNotIter(Z_THREAD)>(threadIdx, blockDim);

    int block_reduction_size = index_utils::maskedSize<
        activeNotIter(X_THREAD),
        activeNotIter(Y_THREAD),
        activeNotIter(Z_THREAD)>(blockDim);

    bool has_block_result = index_utils::maskedIsZero<
        activeNotIter(X_THREAD),
        activeNotIter(Y_THREAD),
        activeNotIter(Z_THREAD)>(threadIdx);

    // 3D buffer of reductions:
    //    [reduction_offset(grid), iter_offset(grid), iter_offset(block)]
    // Change the offset, we want to keep the last two dimensions, but the
    // first dimension is what we will reduce over
    const auto work_buf_offset =
        block_red_idx_offset * num_thread_iters + thread_red_idx_offset;
    for (auto reduction_i = tid_in_block_reduction; reduction_i < grid_red_size;
         reduction_i += block_reduction_size) {
      impl::welfordEach(
          last_block_result,
          0,
          global_work_buffer,
          work_buf_offset + reduction_i * num_block_iters * num_thread_iters);
    }

    // Which block reduction this thread is participating in
    int block_reduction_idx = index_utils::
        maskedOffset<isIter(X_THREAD), isIter(Y_THREAD), isIter(Z_THREAD)>(
            threadIdx, blockDim);

    impl::blockWelfordEach<
        BROADCAST,
        false,
        LocalWelfordTripletTuple<NumVals, DataType, IndexType>>(
        last_block_result,
        last_block_result,
        shared_buf,
        has_block_result,
        tid_in_block_reduction,
        block_reduction_size,
        min(grid_red_size, block_reduction_size),
        block_reduction_idx);

    copyWelfordTripletTupleIf(
        out,
        last_block_result,
        write_preds &&
            (block_reduce_participate && (BROADCAST || has_block_result)));
  }
}

} // namespace fused_reduction
