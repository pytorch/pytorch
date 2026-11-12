
#include "../../utils/common.cuh"

// IN is permuted-contiguous with the chain (dim4, dim0, dim2, dim1, dim3):
// every dim3 slice (fixed l) is one contiguous es*size4*size0*size2*size1
// byte segment; OUT is fully contiguous. One block handles `chunk` dim3
// slices (A2: grid compaction, div hoisting out of the slice loop); per
// slice, 16-byte vector reads fill the slice's shared-memory segment
// linearly, then each (fixed l, m) OUT block of size0*size1*size2 elements
// (contiguous in OUT) is written with 8-byte vectors gathered from scalar
// smem positions (m + i*size4 + k*size4*size0 + j*size4*size0*size2 in
// elements). Write-phase (m,q,i,j,k) indices are derived once per thread
// and carried by increments (A1): no per-vector int64 div/mod in the hot
// loop (fallback re-derivation only when q crosses a block boundary).
// Dynamic smem = chunk * slice_bytes <= 32KB. tag p_e_gpu_5_1_pcb.
template <int nt, typename res_t, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_5_1_permuted_contiguous_block(
    char* data0, char* data1,
    int64_t size0, int64_t size1, int64_t size2, int64_t size3, int64_t size4,
    int64_t stride30, int64_t stride40,
    int64_t stride31,
    int64_t slice_bytes,  // es * size4 * size0 * size1 * size2
    int64_t slice_elems,  // size4 * size0 * size1 * size2
    func_t f, int64_t chunk) {
  extern __shared__ char smem[];
  const int64_t l0 = blockIdx.x * chunk;
  const int64_t tid = threadIdx.x;
  constexpr int vt = per_op_vector_width<res_t>();
  using Vec = at::native::memory::aligned_vector<res_t, vt>;
  using Vec8 = at::native::memory::aligned_vector<res_t, 4>;
  const int64_t nvec = slice_bytes / 16;
  // Write phase: per (l, m) OUT block of block_elems contiguous elements;
  // 8-byte vector stores (4 elements) require block_elems % 4 == 0.
  const int64_t block_elems = size0 * size1 * size2;
  const int64_t nv8 = (size4 * block_elems) / 4;
  const int64_t bv = block_elems / 4;  // 8B vectors per OUT block

  // A1: the per-thread write-vector walk (v8 = tid, tid+nt, ...) is

  // once outside the slice loop and re-derived only when q crosses an OUT
  // block boundary (once per bv/nt vectors, amortized).
  const int64_t v8_init = tid;
  const bool has_write = v8_init < nv8;
  int64_t m_i, q_i, e0_i, i0_i, j0_i, k0_i;
  if (has_write) {
    m_i = v8_init / bv;
    q_i = v8_init - m_i * bv;
    e0_i = q_i * 4;
    i0_i = e0_i % size0;
    const int64_t jk = e0_i / size0;
    j0_i = jk % size1;
    k0_i = jk / size1;
  }

  for (int64_t c = 0; c < chunk; c++) {
    const int64_t l = l0 + c;
    if (l >= size3) {
      break;
    }
    char* smem_c = smem + c * slice_bytes;
    // Read phase: linear 16-byte vector fill of the contiguous IN slice.
    const char* in_slice = data1 + l * stride31;
    for (int64_t v = tid; v < nvec; v += nt) {
      *reinterpret_cast<Vec*>(smem_c + v * 16) =
          *reinterpret_cast<const Vec*>(in_slice + v * 16);
    }
    __syncthreads();
    if (has_write) {
      int64_t v8 = v8_init;
      int64_t m = m_i, q = q_i, e0 = e0_i, i0 = i0_i, j0 = j0_i, k0 = k0_i;
      for (; v8 < nv8; v8 += nt) {
        Vec8 out;
#pragma unroll
        for (int t = 0; t < 4; t++) {
          // Element walk e = e0 + t: carry via compare/subtract, no div.
          int64_t i = i0 + t;
          int64_t j = j0;
          int64_t k = k0;
          while (i >= size0) { i -= size0; j += 1; }
          while (j >= size1) { j -= size1; k += 1; }
          const int64_t smem_elem = m + i * size4 + k * size4 * size0 + j * size4 * size0 * size2;
          out.val[t] = f(*reinterpret_cast<const res_t*>(smem_c + smem_elem * sizeof(res_t)));
        }
        *reinterpret_cast<Vec8*>(data0 + l * stride30 + m * stride40 + e0 * sizeof(res_t)) = out;
        // Advance to the next write vector: q += nt, carry across OUT blocks.
        q += nt;
        if (q >= bv) {
          while (q >= bv) { q -= bv; m += 1; }

          e0 = q * 4;
          i0 = e0 % size0;
          const int64_t jk = e0 / size0;
          j0 = jk % size1;
          k0 = jk / size1;
        } else {
          e0 = q * 4;
          i0 += 4 * nt;
          const int64_t c1 = i0 / size0;
          i0 -= c1 * size0;
          j0 += c1;
          const int64_t c2 = j0 / size1;
          j0 -= c2 * size1;
          k0 += c2;
        }
      }
    }
    __syncthreads();
  }
}


template <int nt, typename res_t, typename func_t>
static void launch_elementwise_kernel_5_1_permuted_contiguous_block(
    char* data0, char* data1,
    int64_t size0, int64_t size1, int64_t size2, int64_t size3, int64_t size4,
    int64_t stride30, int64_t stride40, int64_t stride31,
    int64_t slice_bytes, int64_t slice_elems, const func_t& f, int64_t chunk) {
  dim3 block(nt);
  dim3 grid((size3 + chunk - 1) / chunk);
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_5_1_permuted_contiguous_block<nt, res_t, func_t>
      <<<grid, block, chunk * slice_bytes, stream>>>(
          data0, data1, size0, size1, size2, size3, size4,
          stride30, stride40, stride31, slice_bytes, slice_elems, f, chunk);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
