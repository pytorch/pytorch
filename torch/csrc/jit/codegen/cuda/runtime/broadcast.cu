namespace broadcast {

template <bool X_THREAD, bool Y_THREAD, bool Z_THREAD>
__host__ __device__ unsigned offset_of_source(
    const dim3& block_dim,
    const dim3& thread_idx) {
  unsigned offset = 0;
  if (!Z_THREAD)
    offset = offset * block_dim.z + thread_idx.z;
  if (!Y_THREAD)
    offset = offset * block_dim.y + thread_idx.y;
  if (!X_THREAD)
    offset = offset * block_dim.x + thread_idx.x;
  return offset;
}

// Broadcasts within partitioned groups of threads.
//
// X_THREAD: Broadcast from threadIdx.x == 0 if true
// Y_THREAD: Broadcast from threadIdx.y == 0 if true
// Z_THREAD: Broadcast from threadIdx.z == 0 if true
// inp_val: Per-thread source value. Only valid when the thread is a source.
// out: Per-thread output location
//
template <bool X_THREAD, bool Y_THREAD, bool Z_THREAD, typename T>
__device__ void blockBroadcast(T& out, T inp_val, T* shared_mem) {
  const bool has_valid_data = (!X_THREAD || threadIdx.x == 0) &&
      (!Y_THREAD || threadIdx.y == 0) && (!Z_THREAD || threadIdx.z == 0);

  const auto shared_offset =
      offset_of_source<X_THREAD, Y_THREAD, Z_THREAD>(blockDim, threadIdx);

  if (has_valid_data)
    shared_mem[shared_offset] = inp_val;

  __syncthreads();

  out = shared_mem[shared_offset];
}

} // namespace broadcast
