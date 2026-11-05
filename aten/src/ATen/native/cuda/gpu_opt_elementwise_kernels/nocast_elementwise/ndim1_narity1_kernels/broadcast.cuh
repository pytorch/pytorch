// arity-1 with a broadcast input: every thread evaluates the functor once on
// the broadcast value and writes it out, out[i] = f(bcast[0]). This holds for
// any unary functor, not just copy: the single input is the same element for
// every i.
template <int nt, int vt, int tws, typename res_t, typename bcast_t, typename func_t>
C10_LAUNCH_BOUNDS_1(nt)
__global__ void elementwise_kernel_1_1_broadcast(
    int N,
    char* data0,
    const char* data1,
    func_t f) {
  using StoreT = memory::aligned_vector<res_t, vt>;
  static_assert(
      tws % vt == 0, "the workload per thread must be a multiple of vt");
  constexpr int loop_size = tws / vt;
  constexpr int block_work = nt * tws;
  const int base = block_work * static_cast<int>(blockIdx.x);
  const int remaining = N - base;

  // The broadcast value has to be dereferenced on device: passing it by value
  // from the host would require a device-to-host sync.
  const res_t val = f(c10::load(reinterpret_cast<const bcast_t*>(data1)));

  if (remaining < block_work) {
    res_t* out = reinterpret_cast<res_t*>(data0) + base;
    int idx = static_cast<int>(threadIdx.x);
#pragma unroll
    for (int i = 0; i < tws; i++) {
      if (idx < remaining) {
        out[idx] = val;
        idx += nt;
      }
    }
  } else {
    StoreT ld_out;
#pragma unroll
    for (int i = 0; i < vt; i++) {
      ld_out.val[i] = val;
    }
    StoreT* out = reinterpret_cast<StoreT*>(data0) + nt * loop_size * blockIdx.x;
#pragma unroll
    for (int j = 0; j < loop_size; j++) {
      out[threadIdx.x + j * nt] = ld_out;
    }
  }
}

template <int nt, int vt, int tws, typename res_t, typename bcast_t, typename func_t>
static void launch_gpu_1_1_broadcast(
    int64_t N,
    char* data0,
    const char* data1,
    const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  const int64_t grid = (N + static_cast<int64_t>(nt) * tws - 1) /
      (static_cast<int64_t>(nt) * tws);
  dim3 block(nt);
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_1_1_broadcast<nt, vt, tws, res_t, bcast_t, func_t>
      <<<grid, block, 0, stream>>>(static_cast<int>(N), data0, data1, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}