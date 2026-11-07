
// arity-2 with one broadcast input: data1 is always the contiguous operand and
// data2 always the broadcast one, `swap_tensor` tells which of them is the
// functor's first argument. The caller swaps the type arguments and the data
// pointers together, so a single kernel covers both broadcast positions.
template <
    int nt,
    int vt,
    int tws,
    typename res_t,
    typename cont_t,
    typename bcast_t,
    typename func_t,
    bool swap_tensor = false>
C10_LAUNCH_BOUNDS_1(nt)
__global__ void elementwise_kernel_1_2_broadcast_in2_dim0(
    int N,
    char* data0,
    const char* data1,
    const char* data2,
    func_t f) {
  using LoadT = memory::aligned_vector<cont_t, vt>;
  using StoreT = memory::aligned_vector<res_t, vt>;
  static_assert(
      tws % vt == 0, "the workload per thread must be a multiple of vt");
  constexpr int loop_size = tws / vt;
  constexpr int block_work = nt * tws;
  const int base = block_work * static_cast<int>(blockIdx.x);
  const int remaining = N - base;

  const bcast_t ld_2 = c10::load(reinterpret_cast<const bcast_t*>(data2));

  if (remaining < block_work) {
    const cont_t* in = reinterpret_cast<const cont_t*>(data1) + base;
    res_t* out = reinterpret_cast<res_t*>(data0) + base;
    int idx = static_cast<int>(threadIdx.x);
#pragma unroll
    for (int i = 0; i < tws; i++) {
      if (idx < remaining) {
        const cont_t ld_1 = c10::load(in + idx);
        // if constexpr, not a plain if: for functors whose two arguments have
        // different types the discarded call must not even be instantiated.
        if constexpr (swap_tensor) {
          out[idx] = f(ld_2, ld_1);
        } else {
          out[idx] = f(ld_1, ld_2);
        }
        idx += nt;
      }
    }
  } else {
    // Same three-phase shape as elementwise_kernel_helper + the `vectorized`
    // policy: issue every load first, then compute, then issue every store.
    // Doing load/compute/store per chunk instead serializes the chunks on their
    // own dependency chain and measurably loses throughput.
    const LoadT* in =
        reinterpret_cast<const LoadT*>(data1) + nt * loop_size * blockIdx.x;
    StoreT* out = reinterpret_cast<StoreT*>(data0) + nt * loop_size * blockIdx.x;

    // load
    cont_t args[tws];
#pragma unroll
    for (int j = 0; j < loop_size; j++) {
      const LoadT v = in[threadIdx.x + j * nt];
#pragma unroll
      for (int k = 0; k < vt; k++) {
        args[vt * j + k] = v.val[k];
      }
    }

    // compute
    res_t results[tws];
#pragma unroll
    for (int i = 0; i < tws; i++) {
      if constexpr (swap_tensor) {
        results[i] = f(ld_2, args[i]);
      } else {
        results[i] = f(args[i], ld_2);
      }
    }

    // store
#pragma unroll
    for (int j = 0; j < loop_size; j++) {
      StoreT v;
#pragma unroll
      for (int k = 0; k < vt; k++) {
        v.val[k] = results[vt * j + k];
      }
      out[threadIdx.x + j * nt] = v;
    }
  }
}


// Launcher for the generic 1D arity-2 broadcast kernel defined above as
// elementwise_kernel_1_2_broadcast_in2_dim0<nt, vt, tws, ...>: data1 is the
// contiguous operand, data2 the stride-0 scalar; swap_tensor=true restores
// the functor argument order when the broadcast operand is the functor's
// first argument (the caller swaps the data pointers and the type arguments
// together). tws is the per-thread workload in elements and must be a
// multiple of vt (enforced by the kernel's static_assert).
template <int nt, int vt, int tws, typename res_t, typename cont_t, typename bcast_t, typename func_t, bool swap_tensor = false>
static void launch_elementwise_kernel_1_2_broadcast_in2_dim0(
    int64_t N,
    char* data0, char* data1, char* data2,
    const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  constexpr int block_work = nt * tws;
  dim3 block(nt);
  dim3 grid((N + block_work - 1) / block_work);
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_1_2_broadcast_in2_dim0<nt, vt, tws, res_t, cont_t, bcast_t, func_t, swap_tensor>
      <<<grid, block, 0, stream>>>(static_cast<int>(N), data0, data1, data2, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
