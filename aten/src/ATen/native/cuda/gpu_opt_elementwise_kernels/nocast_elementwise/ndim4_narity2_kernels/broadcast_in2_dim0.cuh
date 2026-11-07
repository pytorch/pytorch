
// IN1 dim0-contiguous (vector read), IN2 dim0-broadcast (scalar read per
// vector, reused across vt elements). Covers layouts like (128,94,16,32)
// with a (94,16,32)-shaped tensor broadcast over dim0.
template <int nt, int vt, typename res_t, typename arg0_t, typename arg1_t, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_4_2_broadcast_in2_dim0(
    int64_t N,
    char* data0, char* data1, char* data2,
    int64_t size0, int64_t size1, int64_t size2, int64_t size3,
    int64_t stride00, int64_t stride01, int64_t stride02,
    int64_t stride10, int64_t stride11, int64_t stride12,
    int64_t stride20, int64_t stride21, int64_t stride22,
    int64_t stride30, int64_t stride31, int64_t stride32,
    c10::IntDivider<uint32_t> div0, c10::IntDivider<uint32_t> div1,
    c10::IntDivider<uint32_t> div2, func_t f) {
  int64_t linear_idx = (static_cast<int64_t>(blockIdx.x) * nt + threadIdx.x) * vt;
  if (linear_idx >= N) {
    return;
  }
  const auto x = div0.divmod(static_cast<uint32_t>(linear_idx));
  const auto y = div1.divmod(x.div);
  const auto z = div2.divmod(y.div);
  const int64_t idx0 = x.mod, idx1 = y.mod, idx2 = z.mod, idx3 = z.div;
  const int64_t offset0 = idx0 * stride00 + idx1 * stride10 + idx2 * stride20 + idx3 * stride30;
  const int64_t offset1 = idx0 * stride01 + idx1 * stride11 + idx2 * stride21 + idx3 * stride31;
  // stride02 == 0: IN2 is invariant along dim0, so one scalar load feeds all
  // vt elements of this thread.
  const int64_t offset2 = idx1 * stride12 + idx2 * stride22 + idx3 * stride32;

  using vec_res = at::native::memory::aligned_vector<res_t, vt>;
  using vec_arg0 = at::native::memory::aligned_vector<arg0_t, vt>;
  vec_arg0 in1 = *reinterpret_cast<const vec_arg0*>(data1 + offset1);
  // IN2 dim0 broadcast: scalar read, reused across vt elements.
  const arg1_t in2 = c10::load(reinterpret_cast<const arg1_t*>(data2 + offset2));
  vec_res out;
#pragma unroll
  for (int i = 0; i < vt; i++) {
    out.val[i] = f(in1.val[i], in2);
  }
  *reinterpret_cast<vec_res*>(data0 + offset0) = out;
}


template <int nt, int vt, typename res_t, typename arg0_t, typename arg1_t, typename func_t>
static void launch_elementwise_kernel_4_2_broadcast_in2_dim0(
    int64_t N,
    char* data0, char* data1, char* data2,
    int64_t size0, int64_t size1, int64_t size2, int64_t size3,
    int64_t stride00, int64_t stride01, int64_t stride02,
    int64_t stride10, int64_t stride11, int64_t stride12,
    int64_t stride20, int64_t stride21, int64_t stride22,
    int64_t stride30, int64_t stride31, int64_t stride32,
    const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_4_2_broadcast_in2_dim0<nt, vt, res_t, arg0_t, arg1_t, func_t>
      <<<grid, block, 0, stream>>>(
          N, data0, data1, data2, size0, size1, size2, size3,
          stride00, stride01, stride02, stride10, stride11, stride12,
          stride20, stride21, stride22, stride30, stride31, stride32,
                    c10::IntDivider<uint32_t>(static_cast<uint32_t>(size0)),
                    c10::IntDivider<uint32_t>(static_cast<uint32_t>(size1)),
                    c10::IntDivider<uint32_t>(static_cast<uint32_t>(size2)), f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
