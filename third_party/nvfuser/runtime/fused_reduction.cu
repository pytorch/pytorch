namespace fused_reduction {

namespace impl {

//! Suppose f_i be the i-th function of the binary function
//! parameters. Call the function as: f_i(x, y)
template <int i, typename DataType, typename Func, typename... Funcs>
struct FuncSelector {
  static __device__ void call(
      DataType& x,
      const DataType y,
      Func f,
      Funcs... funcs) {
    // Here, i is guaranteed to be larger than 0 as there's a
    // specialization for i == 0 below. Recursively call FuncSelector
    // by dropping f and decrementing i.
    FuncSelector<i - 1, DataType, Funcs...>::call(x, y, funcs...);
  }
};

//! Specialization of FuncSelector when i == 0, so f_i is f.
template <typename DataType, typename Func, typename... Funcs>
struct FuncSelector<0, DataType, Func, Funcs...> {
  static __device__ void call(
      DataType& x,
      const DataType y,
      Func f,
      Funcs... funcs) {
    f(x, y);
  }
};

//! Call each of the first i+1 functions with the first i+1 values of
//! tuples. Here, i is guaranteed to be larger than -1 as there's a
//! specialization for i == -1.
template <int i, typename TupleType0, typename TupleType1, typename... Funcs>
struct FuncForEach {
  static __device__ void call(
      TupleType0& val0,
      nvfuser_index_t offset0,
      const TupleType1& val1,
      nvfuser_index_t offset1,
      Funcs... funcs) {
    static_assert(
        IsSameType<
            typename TupleType0::template ValType<i>,
            typename TupleType1::template ValType<i>>::value,
        "Invalid tuple types");
    // Process the first i functions first.
    FuncForEach<i - 1, TupleType0, TupleType1, Funcs...>::call(
        val0, offset0, val1, offset1, funcs...);
    // Call the i+1-th function
    FuncSelector<i, typename TupleType0::template ValType<i>, Funcs...>::call(
        val0.val<i>(offset0), val1.val<i>(offset1), funcs...);
  }
};

//! Specialization of FuncForEach when i == -1, which means no
//! function to call. Just for stopping the recursive pattern here.
template <typename TupleType0, typename TupleType1, typename... Funcs>
struct FuncForEach<-1, TupleType0, TupleType1, Funcs...> {
  static __device__ void call(
      TupleType0& val0,
      nvfuser_index_t offset0,
      const TupleType1& val1,
      nvfuser_index_t offset1,
      Funcs... funcs) {}
};

//! Reduce one value of a tuple using one of the reduction ops. The
//! value at val_idx is reduced by the function at func_idx.
template <
    int func_idx,
    int val_idx,
    typename TupleType0,
    typename TupleType1,
    typename... Funcs>
__inline__ __device__ static void reduceVal(
    TupleType0& val0,
    nvfuser_index_t offset0,
    const TupleType1& val1,
    nvfuser_index_t offset1,
    Funcs... reduction_ops) {
  static_assert(
      IsSameType<
          typename TupleType0::template ValType<val_idx>,
          typename TupleType1::template ValType<val_idx>>::value,
      "Invalid tuple types");
  FuncSelector<
      func_idx,
      typename TupleType0::template ValType<val_idx>,
      Funcs...>::
      call(
          val0.val<val_idx>(offset0),
          val1.val<val_idx>(offset1),
          reduction_ops...);
}

//! Accumulate each value of a given pair of tuples using its corresponding
//! function. Suppose f_i be the i-th reduciton function. Call f_i as:
//! f_i(val0.val<i>(offset0), val1.val<i>(offset1)).
template <typename TupleType0, typename TupleType1, typename... Funcs>
__inline__ __device__ static void reduceEach(
    TupleType0& val0,
    nvfuser_index_t offset0,
    const TupleType1& val1,
    nvfuser_index_t offset1,
    Funcs... reduction_ops) {
  constexpr int num_funcs = sizeof...(reduction_ops);
  FuncForEach<num_funcs - 1, TupleType0, TupleType1, Funcs...>::call(
      val0, offset0, val1, offset1, reduction_ops...);
}

template <typename TupleType0, typename TupleType1, typename Func, int num_vals>
struct TupleReduce {};

template <typename TupleType0, typename TupleType1, typename Func>
struct TupleReduce<TupleType0, TupleType1, Func, 1> {
  __inline__ __device__ static void reduce(
      TupleType0& val0,
      nvfuser_index_t offset0,
      const TupleType1& val1,
      nvfuser_index_t offset1,
      Func reduction_op) {
    static_assert(
        IsSameType<
            typename TupleType0::ValTypes,
            typename TupleType1::ValTypes>::value,
        "Invalid value types");
    reduction_op(val0.val<0>(offset0), val1.val<0>(offset1));
  }
};

template <typename TupleType0, typename TupleType1, typename Func>
struct TupleReduce<TupleType0, TupleType1, Func, 2> {
  __inline__ __device__ static void reduce(
      TupleType0& val0,
      nvfuser_index_t offset0,
      const TupleType1& val1,
      nvfuser_index_t offset1,
      Func reduction_op) {
    static_assert(
        IsSameType<
            typename TupleType0::ValTypes,
            typename TupleType1::ValTypes>::value,
        "Invalid value types");
    reduction_op(
        val0.val<0>(offset0),
        val0.val<1>(offset0),
        val1.val<0>(offset1),
        val1.val<1>(offset1));
  }
};

template <typename TupleType0, typename TupleType1, typename Func>
struct TupleReduce<TupleType0, TupleType1, Func, 3> {
  __inline__ __device__ static void reduce(
      TupleType0& val0,
      nvfuser_index_t offset0,
      const TupleType1& val1,
      nvfuser_index_t offset1,
      Func reduction_op) {
    static_assert(
        IsSameType<
            typename TupleType0::ValTypes,
            typename TupleType1::ValTypes>::value,
        "Invalid value types");
    reduction_op(
        val0.val<0>(offset0),
        val0.val<1>(offset0),
        val0.val<2>(offset0),
        val1.val<0>(offset1),
        val1.val<1>(offset1),
        val1.val<2>(offset1));
  }
};

//! Reduce all values of a tuple together. The reduction function must
//! have the same number of inputs as the number of values of each tuple.
template <typename TupleType0, typename TupleType1, typename Func>
__inline__ __device__ void reduceTuple(
    TupleType0& val0,
    nvfuser_index_t offset0,
    const TupleType1& val1,
    nvfuser_index_t offset1,
    Func reduction_op) {
  static_assert(
      TupleType0::num_vals == TupleType1::num_vals, "Invalid number of values");
  TupleReduce<TupleType0, TupleType1, Func, TupleType0::num_vals>::reduce(
      val0, offset0, val1, offset1, reduction_op);
}

// Reduces all of the first (idx+1) values by a thread block
template <
    int idx,
    bool BROADCAST,
    bool FORWARD_PROTECT_SMEM,
    typename LocalTupleT,
    typename... Funcs>
struct BlockReduceEach {
  __inline__ __device__ static void reduce(
      LocalTupleT& block_result,
      const LocalTupleT& partial_result,
      void* shared_mem,
      bool has_block_result,
      int tid_in_reduction,
      int num_threads_per_reduction,
      int num_elements_per_reduction,
      int reduction_idx,
      Funcs... funcs) {
    // Finish the reduction of each tuple value with a smaller offset
    BlockReduceEach<idx - 1, BROADCAST, true, LocalTupleT, Funcs...>::reduce(
        block_result,
        partial_result,
        shared_mem,
        has_block_result,
        tid_in_reduction,
        num_threads_per_reduction,
        num_elements_per_reduction,
        reduction_idx,
        funcs...);

    if (num_elements_per_reduction == 1) {
      if (has_block_result) {
        block_result.val<idx>(0) = partial_result.val<idx>(0);
      }
      return;
    }

    using DataType = typename LocalTupleT::template ValType<idx>;

    PtrTuple<DataType> shared_buf(static_cast<DataType*>(shared_mem));

    LocalTuple<DataType> block_result_i(partial_result.val<idx>(0));

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
      impl::reduceVal<idx, 0>(
          block_result_i, 0, shared_buf, smem_offset + np2, funcs...);
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
        impl::reduceVal<idx, 0>(
            shared_buf,
            smem_offset,
            shared_buf,
            smem_offset + factor,
            funcs...);
      }
      block_sync::sync();
    }

    copyTuple(block_result_i, shared_buf, smem_offset);

    // Do the last reduction
    if (has_block_result) {
      impl::reduceVal<idx, 0>(
          block_result_i, 0, shared_buf, smem_offset + 1, funcs...);
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

    block_result.val<idx>(0) = block_result_i.val<0>(0);

    if (FORWARD_PROTECT_SMEM) {
      block_sync::sync();
    }
  }
};

// Specialization for idx == -1, i.e., no value to reduce.
template <
    bool BROADCAST,
    bool FORWARD_PROTECT_SMEM,
    typename LocalTupleT,
    typename... Funcs>
struct BlockReduceEach<
    -1,
    BROADCAST,
    FORWARD_PROTECT_SMEM,
    LocalTupleT,
    Funcs...> {
  __inline__ __device__ static void reduce(
      LocalTupleT& block_result,
      const LocalTupleT& partial_result,
      void* shared_mem,
      bool has_block_result,
      int tid_in_reduction,
      int num_threads_per_reduction,
      int num_elements_per_reduction,
      int reduction_idx,
      Funcs... funcs) {}
};

//! Reduce each value of a tuple by a thread block.
//!
//! The final result is broadcast when BROADCAST is true.
//!
//! \param block_result result of the block reduction
//! \param partial_result Per-thread input tuple
//! \param shared_mem
//! \param has_block_result
//! \param tid_in_reduction
//! \param num_threads_per_reduction
//! \param num_elements_per_reduction
//! \param reduction_idx
//! \param reduction_ops
template <
    bool BROADCAST,
    bool FORWARD_PROTECT_SMEM,
    typename LocalTupleT,
    typename... Funcs>
__inline__ __device__ void blockReduceEach(
    LocalTupleT& block_result,
    const LocalTupleT& partial_result,
    void* shared_mem,
    bool has_block_result,
    int tid_in_reduction,
    int num_threads_per_reduction,
    int num_elements_per_reduction,
    int reduction_idx,
    Funcs... reduction_ops) {
  BlockReduceEach<
      LocalTupleT::num_vals - 1,
      BROADCAST,
      FORWARD_PROTECT_SMEM,
      LocalTupleT,
      Funcs...>::
      reduce(
          block_result,
          partial_result,
          shared_mem,
          has_block_result,
          tid_in_reduction,
          num_threads_per_reduction,
          num_elements_per_reduction,
          reduction_idx,
          reduction_ops...);
}

} // namespace impl

// We have 6 dimensions, 3 in the grid, 3 in the block
// They can be 1 of 3 states,
// Reduction Domain - TEMPLATE STATE 0
//   - Participating in the reduction, has values coming in, one value coming
//     out across the dimension
// Iteration Domain - TEMPLATE STATE 1
//   - Not participating in the reduction, has values across the dimension after
//     the reduction
// Collapsed Domain - TEMPLATE STATE 2
//   - Previously reduced, doesn't need to be reduced on that dimension, doesn't
//     have values across that dimension
constexpr __device__ bool isReduce(int STATE) {
  return STATE == 0;
}

constexpr __device__ bool isIter(int STATE) {
  return STATE == 1;
}

constexpr __device__ bool isPred(int STATE) {
  return STATE == 2;
}

constexpr __device__ bool inactive(int STATE) {
  return STATE == 3;
}

constexpr __device__ bool activeNotIter(int STATE) {
  return STATE != 3 && STATE != 1;
}

constexpr __device__ bool isReduceOrIter(int STATE) {
  return isReduce(STATE) || isIter(STATE);
}

// When generating an index into the reduction, we have to stride by iteration
// domains and reduction domains. Collapsed domains we can ignore, but we need
// to make sure they never read or write (need to be predicated to correct
// participation).

// All inclusive reduction with option to re-broadcast. This reduction class
// does not use predication of parallelization in the read or write predicates.
// Instead there are 3 states each dimension of parallelization can have,
// described above. Predication, indexing, and reduction will be done based on
// this information.
template <
    int X_BLOCK,
    int Y_BLOCK,
    int Z_BLOCK,
    int X_THREAD,
    int Y_THREAD,
    int Z_THREAD,
    bool PERSISTENT_REDUCTION,
    bool BROADCAST>
class ParallelReduce {
  static_assert(
      !BROADCAST || PERSISTENT_REDUCTION,
      "Broadcast requires persistent reduction");

  static constexpr bool BLOCK_REDUCE =
      isReduce(X_THREAD) || isReduce(Y_THREAD) || isReduce(Z_THREAD);

  static constexpr bool GRID_REDUCE =
      isReduce(X_BLOCK) || isReduce(Y_BLOCK) || isReduce(Z_BLOCK);

  // ping-pong between global buffers to avoid a second sync
  bool flip = false;

 public:
  __device__ ParallelReduce() {}

  // reduceGroup does not support Welford-style reductions that reduce
  // all values of a tuple together, so this is the only entry point
  // for Welford for now.
  template <typename Func, typename... Types>
  __device__ __inline__ void reduce(
      RefTuple<Types...> out,
      const ConstRefTuple<Types...>& inp,
      VolatilePtrTuple<Types...> global_work_buffer,
      int64_t* global_sync_buffer, // Allocated as product of all
                                   // non-participating Grid dimension
      PtrTuple<Types...> shared_buf,
      bool read_pred, // Prevent reading from out of bounds memory
      bool write_pred, // Prevent from writing out of bounds
      const LocalTuple<Types...>& init_val,
      Func reduction_op);

  //! Profiled version
  template <typename Func, typename... Types>
  __device__ __inline__ void reduce(
      RefTuple<Types...> out,
      const ConstRefTuple<Types...>& inp,
      VolatilePtrTuple<Types...> global_work_buffer,
      int64_t* global_sync_buffer, // Allocated as product of all
                                   // non-participating Grid dimension
      PtrTuple<Types...> shared_buf,
      bool read_pred, // Prevent reading from out of bounds memory
      bool write_pred, // Prevent from writing out of bounds
      const LocalTuple<Types...>& init_val,
      Func reduction_op,
      int64_t& cycles,
      int64_t& count);

  //! Each value of a tuple is independently reduced by the
  //! corresponding reduction op. Thus, Welford-like reductions are
  //! not supported by this interface.
  //!
  //! Note that out is purely used as the output parameter, and its
  //! initial value is not used but just overwritten. Since grid
  //! reductions do not allow serial reduction IterDomains, there is
  //! no need to accumulate into the out parameter.
  template <typename... DataTypes, typename... Funcs, typename... BoolTypes>
  __device__ __inline__ void reduceGroup(
      RefTuple<DataTypes...> out,
      const ConstRefTuple<DataTypes...>& inp,
      VolatilePtrTuple<DataTypes...> global_work_buffer,
      const LocalTuple<DataTypes...>& init_val,
      int64_t* global_sync_buffer,
      void* shared_mem,
      const LocalTuple<BoolTypes...>& read_preds,
      const LocalTuple<BoolTypes...>& write_preds,
      Funcs... funcs);

  //! Profiled version
  template <typename... DataTypes, typename... Funcs, typename... BoolTypes>
  __device__ __inline__ void reduceGroup(
      RefTuple<DataTypes...> out,
      const ConstRefTuple<DataTypes...>& inp,
      VolatilePtrTuple<DataTypes...> global_work_buffer,
      const LocalTuple<DataTypes...>& init_val,
      int64_t* global_sync_buffer,
      void* shared_mem,
      const LocalTuple<BoolTypes...>& read_preds,
      const LocalTuple<BoolTypes...>& write_preds,
      int64_t& cycles,
      int64_t& count,
      Funcs... funcs);

  template <int NumArgs, typename DataType, typename IndexType>
  __device__ __inline__ void welfordGroup(
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
      const typename MakeLocalTuple<NumArgs, bool>::type& write_preds);

 private:
  __device__ static bool isLastBlockInGrid() {
    return index_utils::maskedIsLast<
               isReduceOrIter(X_BLOCK),
               isReduceOrIter(Y_BLOCK),
               isReduceOrIter(Z_BLOCK)>(blockIdx, gridDim) &&
        index_utils::maskedIsZero<
               !isReduceOrIter(X_BLOCK),
               !isReduceOrIter(Y_BLOCK),
               !isReduceOrIter(Z_BLOCK)>(blockIdx);
  }

  //! Initial per-CTA reduction of each value of a tuple. Each value
  //! is reduced individually, so the shared memory buffer just needs
  //! to be large enough for each value. NOTE that the smem buffer is
  //! not forward protected.
  template <
      bool BLOCK_BROADCAST,
      typename... DataTypes,
      typename... Funcs,
      typename... BoolTypes>
  __device__ __inline__ static LocalTuple<DataTypes...> reduceGroupBlock(
      const ConstRefTuple<DataTypes...>& inp,
      const LocalTuple<DataTypes...>& init_val,
      void* shared_mem,
      const LocalTuple<BoolTypes...>& read_preds,
      bool block_reduce_participate,
      Funcs... funcs);

  //! Final reduction of partial results. Done by all blocks
  //! redundantly when BROADCAST is true, or just one block otherwise.
  //! The smem buffer is assumed synchronized when it is passed in,
  //! but it isn't synchronized when returning from this function.
  template <typename... DataTypes, typename... Funcs, typename... BoolTypes>
  __device__ __inline__ static void reduceGroupLastBlock(
      RefTuple<DataTypes...>& out,
      const VolatilePtrTuple<DataTypes...>& global_work_buffer,
      const LocalTuple<DataTypes...>& init_val,
      void* shared_mem,
      nvfuser_index_t block_red_idx_offset,
      nvfuser_index_t num_thread_iters,
      nvfuser_index_t num_block_iters,
      nvfuser_index_t thread_red_idx_offset,
      nvfuser_index_t grid_red_size,
      const LocalTuple<BoolTypes...>& write_preds,
      bool block_reduce_participate,
      bool grid_reduce_participate,
      Funcs... reduction_ops);

  //! Welford version of reduceGroupBlock
  template <
      bool BLOCK_BROADCAST,
      int NumVals,
      typename DataType,
      typename IndexType>
  __device__ __inline__ static void welfordGroupBlock(
      LocalWelfordTripletTuple<NumVals, DataType, IndexType>& block_result,
      const ConstRefWelfordTripletTuple<NumVals, DataType, IndexType>& inp,
      PtrTuple<DataType, DataType, IndexType> shared_buf,
      const typename MakeLocalTuple<NumVals, bool>::type& read_preds,
      bool block_reduce_participate);

  //! Welford version of reduceGrouplLastBlock
  template <int NumVals, typename DataType, typename IndexType>
  __device__ __inline__ static void welfordGroupLastBlock(
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
      bool grid_reduce_participate);

  // End Parallel reduce class
};

template <
    int X_BLOCK,
    int Y_BLOCK,
    int Z_BLOCK,
    int X_THREAD,
    int Y_THREAD,
    int Z_THREAD,
    bool PERSISTENT_REDUCTION,
    bool BROADCAST>
template <typename Func, typename... Types>
__device__ __inline__ void ParallelReduce<
    X_BLOCK,
    Y_BLOCK,
    Z_BLOCK,
    X_THREAD,
    Y_THREAD,
    Z_THREAD,
    PERSISTENT_REDUCTION,
    BROADCAST>::
    reduce(
        RefTuple<Types...> out,
        const ConstRefTuple<Types...>& inp,
        VolatilePtrTuple<Types...> global_work_buffer,
        int64_t* global_sync_buffer, // Allocated as product of all
        // non-participating Grid dimension
        PtrTuple<Types...> shared_buf,
        bool read_pred, // Prevent reading from out of bounds memory
        bool write_pred, // Prevent from writing out of bounds
        const LocalTuple<Types...>& init_val,
        Func reduction_op) {
  // If no reduction needed, just return input
  if (!BLOCK_REDUCE && !GRID_REDUCE) {
    if (read_pred && write_pred) {
      out = inp;
    }
    return;
  }

  // Don't read/write in temporary buffers if in a predicated dimension
  bool block_reduce_participate = index_utils::
      maskedIsZero<isPred(X_THREAD), isPred(Y_THREAD), isPred(Z_THREAD)>(
          threadIdx);

  // Initialize block result
  LocalTuple<Types...> block_result = init_val;

  // Grab input data if participating in the reduction, set to block_result in
  // the case there is no block reduction
  if (block_reduce_participate && read_pred) {
    block_result = inp;
  }

  // Only threads that with id == 0 in the dimensions being reduced will
  // have a valid result
  bool has_block_result = index_utils::
      maskedIsZero<isReduce(X_THREAD), isReduce(Y_THREAD), isReduce(Z_THREAD)>(
          threadIdx);

  if (BLOCK_REDUCE) {
    // -- START BLOCK REDUCTION -- //

    // Size of the block reduction segment, can be an int since it's limited
    // to number of threads
    int block_reduction_size = index_utils::
        maskedSize<isReduce(X_THREAD), isReduce(Y_THREAD), isReduce(Z_THREAD)>(
            blockDim);

    // Index in the reduction segment, can be an int since it's limited to
    // number of threads
    int tid_in_block_reduction = index_utils::maskedOffset<
        isReduce(X_THREAD),
        isReduce(Y_THREAD),
        isReduce(Z_THREAD)>(threadIdx, blockDim);

    // ID of the block reduction this thread is participating in
    //
    // If any of the parallel dimensions are predicated out, that means
    // they've already been reduced, so we only care about the first thread in
    // that dimension. Therefore don't expand the reduction_idx by that
    // dimension
    int block_reduction_idx = index_utils::
        maskedOffset<isIter(X_THREAD), isIter(Y_THREAD), isIter(Z_THREAD)>(
            threadIdx, blockDim);

    // Shared memory buffer is 2D
    // [iter dimension, reduction dimension]

    // Offset into smem for the current thread
    int block_reduce_smem_offset =
        block_reduction_idx * block_reduction_size + tid_in_block_reduction;

    // Initialize shared memory
    if (block_reduce_participate) {
      copyTuple(shared_buf, block_reduce_smem_offset, block_result);
    }

    // Sync to make sure smem is completely initialized
    block_sync::sync();

    // Round reduction size down to nearest power of 2
    int np2 = 1 << (31 - __clz(block_reduction_size));

    // Perform an initial reduction leaving np2 elements
    if (block_reduce_participate && tid_in_block_reduction < np2 &&
        tid_in_block_reduction + np2 < block_reduction_size) {
      impl::reduceTuple(
          shared_buf,
          block_reduce_smem_offset,
          shared_buf,
          block_reduce_smem_offset + np2,
          reduction_op);
    }

    // Always need to sync while operating on shared memory
    block_sync::sync();

    // Reduce down until 2 values, leaving 2 values allows us to manually
    // perform the last reduction and avoid a syncthreads
    for (int factor = np2 / 2; factor > 1; factor >>= 1) {
      if (tid_in_block_reduction < factor && block_reduce_participate) {
        impl::reduceTuple(
            shared_buf,
            block_reduce_smem_offset,
            shared_buf,
            block_reduce_smem_offset + factor,
            reduction_op);
      }
      block_sync::sync();
    }

    // Accumulate that last valid result
    if (has_block_result) {
      copyTuple(block_result, shared_buf, block_reduce_smem_offset);
      if (block_reduction_size > 1) {
        impl::reduceTuple(
            block_result,
            0,
            shared_buf,
            block_reduce_smem_offset + 1,
            reduction_op);
      }
    }

    // ===== BLOCK REDUCTION CLEANUP =======
    if (!GRID_REDUCE) {
      // If no grid reduction, we don't have to continue. Either broadcast
      // back across the block or return the correct reduction
      if (has_block_result && write_pred) {
        impl::reduceTuple(block_result, 0, out, 0, reduction_op);
        out = block_result;
      }
      if (BROADCAST) {
        // No grid reduce, but need to broadcast, perform block broadcast
        if (has_block_result && write_pred) {
          // Put result back in shared memory, put in the first entry of the
          // reduction segment's buffer
          copyTuple(
              shared_buf,
              block_reduction_idx * block_reduction_size,
              block_result);
        }

        // Sync threads to make sure result is in smem
        block_sync::sync();
        // If the thread is participating, and is not attempting to write out
        // of bounds, return the broadcasted value.
        if (block_reduce_participate && write_pred) {
          copyTuple(
              out, shared_buf, block_reduction_idx * block_reduction_size);
        }
      }

      // Forward protect shared memory, don't want threads to continue to
      // another reduction/broadcast and pollute shared memory before the
      // reduction is completely finished.
      //
      // This could be avoided in some cases if we added thread syncs from
      // block reductions in the syncthread insertion pass.
      block_sync::sync();
      return;
    }
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
  const auto work_buf_offset =
      (idx_in_grid_red * num_block_iters + block_red_idx_offset) *
          num_thread_iters +
      thread_red_idx_offset;

  // Don't read/write in temporary buffers if in a predicated dimension
  bool grid_reduce_participate = index_utils::
      maskedIsZero<isPred(X_BLOCK), isPred(Y_BLOCK), isPred(Z_BLOCK)>(blockIdx);

  if (grid_reduce_participate && block_reduce_participate) {
    if (has_block_result) {
      copyTuple(global_work_buffer, work_buf_offset, block_result);
    }
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
  // All blocks perform the last cleanup, so every block, and every thread
  // will have the final result

  // Initialize block result
  LocalTuple<Types...> last_block_result(init_val);

  if ((PERSISTENT_REDUCTION || last_block) && grid_reduce_participate) {
    // Can use the last block to reduce all the values the blocks filled in.
    // Can use any thread that has been predicated, or has been reduced to do
    // this reduction, cannot use any block that's associated with an
    // iteration domain

    // Start with non-block reduction

    // Index in the reduction segment
    int tid_in_block_reduction_2 = index_utils::maskedOffset<
        activeNotIter(X_THREAD),
        activeNotIter(Y_THREAD),
        activeNotIter(Z_THREAD)>(threadIdx, blockDim);

    int block_reduction_size_2 = index_utils::maskedSize<
        activeNotIter(X_THREAD),
        activeNotIter(Y_THREAD),
        activeNotIter(Z_THREAD)>(blockDim);

    // 3D buffer of reductions:
    //    [reduction_offset(grid), iter_offset(grid), iter_offset(block)]
    // Change the offset, we want to keep the last two dimensions, but the
    // first dimension is what we will reduce over
    const auto work_buf_offset_2 =
        block_red_idx_offset * num_thread_iters + thread_red_idx_offset;
    for (auto reduction_i = tid_in_block_reduction_2;
         reduction_i < grid_red_size;
         reduction_i += block_reduction_size_2) {
      impl::reduceTuple(
          last_block_result,
          0,
          global_work_buffer,
          work_buf_offset_2 +
              reduction_i * num_block_iters *
                  num_thread_iters, // Iterating over the outer most
          // dimension, so need to stride by the
          // total number of grid reductions. Could
          // come back and change it so this is the
          // contiguous dimension
          reduction_op);
    }

    // -- START LAST BLOCK - BLOCK REDUCTION -- //

    // Reduced so we have one value per thread, we need to further reduce any
    // dimension that is not an iter dimension

    // Which block reduction this thread is participating in
    int block_reduction_idx = index_utils::
        maskedOffset<isIter(X_THREAD), isIter(Y_THREAD), isIter(Z_THREAD)>(
            threadIdx, blockDim);

    // Offset in smem for this thread's result
    auto smem_offset =
        block_reduction_idx * block_reduction_size_2 + tid_in_block_reduction_2;

    // Similar as before, reduce down to nearest power of 2 so we can do a
    // tree reduction
    int np2 = 1 << (31 - __clz(min(block_reduction_size_2, grid_red_size)));

    // Threads values are initialized, so all can participate here
    if (tid_in_block_reduction_2 >= np2) {
      copyTuple(shared_buf, smem_offset, last_block_result);
    }

    block_sync::sync();

    if (tid_in_block_reduction_2 < np2 &&
        tid_in_block_reduction_2 + np2 <
            min(block_reduction_size_2, grid_red_size)) {
      impl::reduceTuple(
          last_block_result, 0, shared_buf, smem_offset + np2, reduction_op);
    }

    if (tid_in_block_reduction_2 < np2) {
      copyTuple(shared_buf, smem_offset, last_block_result);
    }

    // Always sync when communicating across smem
    block_sync::sync();

    // Reduce down to 2 values, last thread will do the final reduction and
    // can save a syncthreads this way
    for (int factor = np2 / 2; factor > 1; factor >>= 1) {
      if (tid_in_block_reduction_2 < factor) {
        impl::reduceTuple(
            shared_buf,
            smem_offset,
            shared_buf,
            smem_offset + factor,
            reduction_op);
      }
      block_sync::sync();
    }

    // If this thread in each block has the final result before broadcasting
    // to all other threads in block
    bool has_block_result_2 = index_utils::maskedIsZero<
        activeNotIter(X_THREAD),
        activeNotIter(Y_THREAD),
        activeNotIter(Z_THREAD)>(threadIdx);
    // Do the last reduction, protected by the write predicate
    copyTuple(last_block_result, shared_buf, smem_offset);
    if (has_block_result && grid_reduce_participate) {
      impl::reduceTuple(last_block_result, 0, out, 0, reduction_op);
      if (min(block_reduction_size_2, grid_red_size) > 1) {
        impl::reduceTuple(
            last_block_result, 0, shared_buf, smem_offset + 1, reduction_op);
      }
    }
    if (grid_reduce_participate && PERSISTENT_REDUCTION) {
      // If persistent reduction, always broadcast reduced values
      copyTuple(shared_buf, smem_offset, last_block_result);
      block_sync::sync();
      if (write_pred && block_reduce_participate) {
        copyTuple(
            out, shared_buf, block_reduction_idx * block_reduction_size_2);
      }
      // For persistent kernels we double the global buffer allocation so we
      // don't need to protect those buffers every iteration preventing the
      // need of an additional grid_sync. Since we flip back and forth between
      // sections of the buffer, the one grid sync protects the other part of
      // the buffer.
    } else {
      if (grid_reduce_participate) {
        if (last_block && has_block_result && block_reduce_participate &&
            write_pred) {
          copyTuple(
              out, shared_buf, block_reduction_idx * block_reduction_size_2);
        }
      }
    }
    // Forward protect the smem used in this reduction
    block_sync::sync();
  }
}

//! Profiled version
template <
    int X_BLOCK,
    int Y_BLOCK,
    int Z_BLOCK,
    int X_THREAD,
    int Y_THREAD,
    int Z_THREAD,
    bool PERSISTENT_REDUCTION,
    bool BROADCAST>
template <typename Func, typename... Types>
__device__ __inline__ void ParallelReduce<
    X_BLOCK,
    Y_BLOCK,
    Z_BLOCK,
    X_THREAD,
    Y_THREAD,
    Z_THREAD,
    PERSISTENT_REDUCTION,
    BROADCAST>::
    reduce(
        RefTuple<Types...> out,
        const ConstRefTuple<Types...>& inp,
        VolatilePtrTuple<Types...> global_work_buffer,
        int64_t* global_sync_buffer, // Allocated as product of all
        // non-participating Grid dimension
        PtrTuple<Types...> shared_buf,
        bool read_pred, // Prevent reading from out of bounds memory
        bool write_pred, // Prevent from writing out of bounds
        const LocalTuple<Types...>& init_val,
        Func reduction_op,
        int64_t& cycles,
        int64_t& count) {
  int64_t start_counter = 0;

  if (isLastBlockInGrid() &&
      index_utils::maskedIsZero<true, true, true>(threadIdx)) {
    start_counter = readCycleCounter();
  }

  reduce(
      out,
      inp,
      global_work_buffer,
      global_sync_buffer,
      shared_buf,
      read_pred,
      write_pred,
      init_val,
      reduction_op);

  if (isLastBlockInGrid() &&
      index_utils::maskedIsZero<true, true, true>(threadIdx)) {
    cycles += readCycleCounter() - start_counter;
    ++count;
  }
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
template <typename... DataTypes, typename... Funcs, typename... BoolTypes>
__device__ __inline__ void ParallelReduce<
    X_BLOCK,
    Y_BLOCK,
    Z_BLOCK,
    X_THREAD,
    Y_THREAD,
    Z_THREAD,
    PERSISTENT_REDUCTION,
    BROADCAST>::
    reduceGroup(
        RefTuple<DataTypes...> out,
        const ConstRefTuple<DataTypes...>& inp,
        VolatilePtrTuple<DataTypes...> global_work_buffer,
        const LocalTuple<DataTypes...>& init_val,
        int64_t* global_sync_buffer,
        void* shared_mem,
        const LocalTuple<BoolTypes...>& read_preds,
        const LocalTuple<BoolTypes...>& write_preds,
        Funcs... funcs) {
  static_assert(
      sizeof...(DataTypes) == sizeof...(Funcs),
      "Mismatched number of Tuple values and functions");
  static_assert(
      sizeof...(DataTypes) == sizeof...(BoolTypes),
      "Mismatched number of Tuple values and predicate values");

  // If no reduction needed, just return input
  if (!BLOCK_REDUCE && !GRID_REDUCE) {
    copyTupleIf(out, inp, read_preds && write_preds);
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

  // Initial per-block reduction. Result is broadcast if specified
  // and this call is block reduction only.
  const auto block_result = reduceGroupBlock < !GRID_REDUCE &&
      BROADCAST > (inp,
                   init_val,
                   shared_mem,
                   read_preds,
                   block_reduce_participate,
                   funcs...);
  // If block reduction only, save to out and exit
  if (!GRID_REDUCE) {
    copyTupleIf(
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
  const auto work_buf_offset =
      (idx_in_grid_red * num_block_iters + block_red_idx_offset) *
          num_thread_iters +
      thread_red_idx_offset;

  // Don't read/write in temporary buffers if in a predicated dimension
  bool grid_reduce_participate = index_utils::
      maskedIsZero<isPred(X_BLOCK), isPred(Y_BLOCK), isPred(Z_BLOCK)>(blockIdx);

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
    copyTuple(global_work_buffer, work_buf_offset, block_result);
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
  reduceGroupLastBlock(
      out,
      global_work_buffer,
      init_val,
      shared_mem,
      block_red_idx_offset,
      num_thread_iters,
      num_block_iters,
      thread_red_idx_offset,
      grid_red_size,
      write_preds,
      block_reduce_participate,
      grid_reduce_participate,
      funcs...);

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
template <typename... DataTypes, typename... Funcs, typename... BoolTypes>
__device__ __inline__ void ParallelReduce<
    X_BLOCK,
    Y_BLOCK,
    Z_BLOCK,
    X_THREAD,
    Y_THREAD,
    Z_THREAD,
    PERSISTENT_REDUCTION,
    BROADCAST>::
    reduceGroup(
        RefTuple<DataTypes...> out,
        const ConstRefTuple<DataTypes...>& inp,
        VolatilePtrTuple<DataTypes...> global_work_buffer,
        const LocalTuple<DataTypes...>& init_val,
        int64_t* global_sync_buffer,
        void* shared_mem,
        const LocalTuple<BoolTypes...>& read_preds,
        const LocalTuple<BoolTypes...>& write_preds,
        int64_t& cycles,
        int64_t& count,
        Funcs... funcs) {
  int64_t start_counter = 0;

  if (isLastBlockInGrid() &&
      index_utils::maskedIsZero<true, true, true>(threadIdx)) {
    start_counter = readCycleCounter();
  }

  reduceGroup(
      out,
      inp,
      global_work_buffer,
      init_val,
      global_sync_buffer,
      shared_mem,
      read_preds,
      write_preds,
      funcs...);

  if (isLastBlockInGrid() &&
      index_utils::maskedIsZero<true, true, true>(threadIdx)) {
    cycles += readCycleCounter() - start_counter;
    ++count;
  }
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
    typename... DataTypes,
    typename... Funcs,
    typename... BoolTypes>
__device__ __inline__ LocalTuple<DataTypes...> ParallelReduce<
    X_BLOCK,
    Y_BLOCK,
    Z_BLOCK,
    X_THREAD,
    Y_THREAD,
    Z_THREAD,
    PERSISTENT_REDUCTION,
    BROADCAST>::
    reduceGroupBlock(
        const ConstRefTuple<DataTypes...>& inp,
        const LocalTuple<DataTypes...>& init_val,
        void* shared_mem,
        const LocalTuple<BoolTypes...>& read_preds,
        bool block_reduce_participate,
        Funcs... funcs) {
  const bool has_block_result = index_utils::
      maskedIsZero<isReduce(X_THREAD), isReduce(Y_THREAD), isReduce(Z_THREAD)>(
          threadIdx);

  // Initialize block result
  LocalTuple<DataTypes...> block_result = init_val;

  copyTupleIf(block_result, inp, block_reduce_participate && read_preds);

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
  impl::blockReduceEach<
      BLOCK_BROADCAST,
      false,
      LocalTuple<DataTypes...>,
      Funcs...>(
      block_result,
      block_result,
      shared_mem,
      has_block_result,
      tid_in_block_reduction,
      block_reduction_size,
      block_reduction_size,
      block_reduction_idx,
      funcs...);

  return block_result;
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
template <typename... DataTypes, typename... Funcs, typename... BoolTypes>
__device__ __inline__ void ParallelReduce<
    X_BLOCK,
    Y_BLOCK,
    Z_BLOCK,
    X_THREAD,
    Y_THREAD,
    Z_THREAD,
    PERSISTENT_REDUCTION,
    BROADCAST>::
    reduceGroupLastBlock(
        RefTuple<DataTypes...>& out,
        const VolatilePtrTuple<DataTypes...>& global_work_buffer,
        const LocalTuple<DataTypes...>& init_val,
        void* shared_mem,
        nvfuser_index_t block_red_idx_offset,
        nvfuser_index_t num_thread_iters,
        nvfuser_index_t num_block_iters,
        nvfuser_index_t thread_red_idx_offset,
        nvfuser_index_t grid_red_size,
        const LocalTuple<BoolTypes...>& write_preds,
        bool block_reduce_participate,
        bool grid_reduce_participate,
        Funcs... reduction_ops) {
  // Initialize block result
  LocalTuple<DataTypes...> last_block_result(init_val);

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
      impl::reduceEach(
          last_block_result,
          0,
          global_work_buffer,
          work_buf_offset +
              reduction_i * num_block_iters *
                  num_thread_iters, // Iterating over the outer most
                                    // dimension, so need to stride by the
                                    // total number of grid reductions. Could
                                    // come back and change it so this is the
                                    // contiguous dimension
          reduction_ops...);
    }

    // Which block reduction this thread is participating in
    int block_reduction_idx = index_utils::
        maskedOffset<isIter(X_THREAD), isIter(Y_THREAD), isIter(Z_THREAD)>(
            threadIdx, blockDim);

    impl::blockReduceEach<BROADCAST, false, LocalTuple<DataTypes...>, Funcs...>(
        last_block_result,
        last_block_result,
        shared_mem,
        has_block_result,
        tid_in_block_reduction,
        block_reduction_size,
        min(grid_red_size, block_reduction_size),
        block_reduction_idx,
        reduction_ops...);

    copyTupleIf(
        out,
        last_block_result,
        write_preds &&
            (block_reduce_participate && (BROADCAST || has_block_result)));
  }
}

} // namespace fused_reduction
