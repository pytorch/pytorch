#define TORCH_ASSERT_NO_OPERATOR
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/LinearAlgebra.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/cuda/block_reduce.cuh>
#include <c10/core/Scalar.h>

namespace at { namespace native {

namespace {

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ householder_orthogonalization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

__device__ __forceinline__ void wait_barrier(int* barrier, int target){
    if (threadIdx.x == 0){
        int counter;
        do {
            asm volatile ("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(counter): "l"(barrier) );
        }
        while (counter < target);
    }
    __syncthreads();
}

__device__ __forceinline__ void set_barrier(int* barrier, int value){
    if(threadIdx.x == 0)
        asm volatile ("st.global.cg.s32 [%0], %1;" :: "l"(barrier), "r"(value));
}

template <int BLOCK_THREADS, typename scalar_t>
__device__  __forceinline__ scalar_t dot(scalar_t *a, scalar_t *b, uint length){
    __shared__ scalar_t tmp_storage[(BLOCK_THREADS - 1) / 32 + 1];

    int tx = threadIdx.x;
    uint unroll = ceil( (float)length / (float)BLOCK_THREADS );
    uint idx = (tx & -32u)*unroll + (tx & 31);

    scalar_t local_prod = 0;
    for (uint i = 0; i < unroll; ++i){
        local_prod += (idx < length)? a[idx] * b[idx] : (scalar_t) 0;
        idx += 32;
    }

    scalar_t reduce = cuda_utils::BlockReduceSum(local_prod, tmp_storage);

     __shared__ scalar_t dot;
    if (tx == 0) 
        dot = reduce;
    __syncthreads();

    return dot;
}

template <int BLOCK_THREADS, typename scalar_t> 
__global__ void reflections(scalar_t *R, scalar_t *vs, const uint m, const uint n, int *barriers){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    scalar_t *v = &vs[bx * n + bx];
    uint v_len = n - bx;

    wait_barrier(&barriers[bx], bx);

    for (uint idx = tx; idx < v_len; idx += BLOCK_THREADS)
        v[idx] = - R[bx * n + bx + idx];

    scalar_t norm_v_sq = dot<BLOCK_THREADS, scalar_t>(v, v, v_len);
    if (tx == 0) 
        v[0] += copysign(sqrt(norm_v_sq), v[0]);
    
    scalar_t norm_v = sqrt(dot<BLOCK_THREADS, scalar_t>(v, v, v_len));
    for (uint idx = tx; idx < v_len; idx += BLOCK_THREADS)
        v[idx] /= norm_v;

    for (uint row = bx + 1; row < m; ++row){
        wait_barrier(&barriers[row], bx);

        scalar_t dot_value = dot<BLOCK_THREADS, scalar_t>(&R[row * n + bx], v, v_len);
        
        for (uint idx = tx; idx < v_len; idx += BLOCK_THREADS)
            R[row * n + bx + idx] -= 2.0 * v[idx] * dot_value;

        __syncthreads();
        set_barrier(&barriers[row], bx + 1);
    }
}

template <int BLOCK_THREADS, typename scalar_t> 
__global__ void q_loop(scalar_t *Q, scalar_t *vs, const uint n, const uint m){
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    for (int v_idx = 0; v_idx < m; ++v_idx){
        scalar_t *v = &vs[v_idx * n + v_idx];
        uint v_len = n - v_idx;
    
        scalar_t dot_value = dot<BLOCK_THREADS, scalar_t>(v, &Q[bx * n + v_idx], v_len);

        for (uint idx = tx; idx < v_len ; idx += BLOCK_THREADS)
            Q[bx * n + v_idx + idx] -= 2.0 * v[idx] * dot_value;

        __syncthreads();
    }
}

template <int BLOCK_THREADS, typename scalar_t> 
void householder_main(const Tensor& R, Tensor& Q, Tensor& vs, const uint m, const uint n){
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int barriers[m];
    
    reflections<BLOCK_THREADS, scalar_t><<<m, BLOCK_THREADS, 0, stream>>>(
        R.data_ptr<scalar_t>(), 
        vs.data_ptr<scalar_t>(), 
        m, 
        n, 
        barriers
    );

    Q.fill_(0);
    Q.fill_diagonal_(1);
    q_loop<BLOCK_THREADS, scalar_t><<<m, BLOCK_THREADS, 0, stream>>>(
        Q.data_ptr<scalar_t>(), 
        vs.data_ptr<scalar_t>(), 
        n, 
        m
    );
}

void householder_orthogonalization_cuda_impl(Tensor& R, Tensor& out, Tensor& vs){
    const uint m = R.size(0);
    const uint n = R.size(1);

    AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16,
    R.scalar_type(), "qr_orthogonalization", ([&] {
      if (n < 512)
          householder_main<256, scalar_t>(R, out, vs, m, n);
      else if (n < 1024)
          householder_main<512, scalar_t>(R, out, vs, m, n);
      else
          householder_main<1024, scalar_t>(R, out, vs, m, n);
    })
    );
    
}

REGISTER_DISPATCH(householder_orthogonalization_stub, &householder_orthogonalization_cuda_impl);

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void addr_kernel_cuda(TensorIterator &iter, const Scalar& beta, const Scalar& alpha) {
  if (iter.dtype() == ScalarType::Bool) {
    using scalar_t = bool;
    auto beta_val = beta.to<scalar_t>();
    auto alpha_val = alpha.to<scalar_t>();

    // when beta is false, values in self should be ignored,
    // nans and infs in self should not propagate.
    if (beta_val == false) {
      gpu_kernel(
        iter,
        [=] GPU_LAMBDA (scalar_t self_val,
                        scalar_t vec1_val, scalar_t vec2_val) -> scalar_t {
          return alpha_val && vec1_val && vec2_val;
        }
      );
    } else {
      gpu_kernel(
        iter,
        [=] GPU_LAMBDA (scalar_t self_val,
                        scalar_t vec1_val, scalar_t vec2_val) -> scalar_t {
          return (beta_val && self_val) || (alpha_val && vec1_val && vec2_val);
        }
      );
    }
    return;
  }

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf,
                                         iter.dtype(), "addr_cuda", [&] {
    auto beta_val = beta.to<scalar_t>();
    auto alpha_val = alpha.to<scalar_t>();

    scalar_t zero_val(0);
    // when beta==0, values in self should be ignored,
    // nans and infs in self should not propagate.
    if (beta_val == zero_val) {
      gpu_kernel(
        iter,
        [=] GPU_LAMBDA (scalar_t self_val,
                        scalar_t vec1_val, scalar_t vec2_val) -> scalar_t {
          return alpha_val * vec1_val * vec2_val;
        }
      );
    } else {
      gpu_kernel(
        iter,
        [=] GPU_LAMBDA (scalar_t self_val,
                        scalar_t vec1_val, scalar_t vec2_val) -> scalar_t {
          return beta_val * self_val + alpha_val * vec1_val * vec2_val;
        }
      );
    }
  });
}


template <int n_threads, int n_elems_per_thread, typename func_t>
C10_LAUNCH_BOUNDS_2(n_threads, n_elems_per_thread)
__global__ void _elementwise_kernel(int total_n_elems, func_t f) {
  constexpr int total_work_block = n_threads * n_elems_per_thread;
  int idx = total_work_block * blockIdx.x + threadIdx.x;

  #pragma unroll
  for (int i = 0; i < n_elems_per_thread; ++i) {
    if (idx < total_n_elems) {
      f(idx);
      idx += n_threads;
    }
  }
}

template <int n_threads, int n_elems_per_thread, typename func_t>
static void _launch_kernel(int total_n_elems, func_t f) {
  TORCH_INTERNAL_ASSERT(
    total_n_elems >= 0 && total_n_elems <= std::numeric_limits<int32_t>::max()
  );

  dim3 block(n_threads);
  constexpr int total_work_block = n_threads * n_elems_per_thread;
  dim3 grid((total_n_elems + total_work_block - 1) / total_work_block);

  auto stream = at::cuda::getCurrentCUDAStream();
  _elementwise_kernel<n_threads, n_elems_per_thread, func_t>
    <<<grid, block, 0, stream>>>(total_n_elems, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void _unpack_pivots_internal_kernel(
  TensorIterator& iter,
  int64_t dim_size
) {
  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      _unpack_pivots_internal_kernel(sub_iter, dim_size);
    }
    return;
  }

  auto offset_calculator = make_offset_calculator<2>(iter);

  char* unpacked_pivots_ptr = reinterpret_cast<char*>(iter.data_ptr(0));
  const char* const __restrict__ pivots_ptr = reinterpret_cast<const char*>(iter.data_ptr(1));

  auto loop = [=]C10_DEVICE(int i) {
    auto offsets = offset_calculator.get(i);

    auto* unpacked_pivots_data = reinterpret_cast<int32_t*>(
      unpacked_pivots_ptr + offsets[0]);
    const auto* const __restrict__ pivots_data = reinterpret_cast<const int32_t*>(
      pivots_ptr + offsets[1]);

    // QUESTION: can we mix 64bit offsets with 32bit Iterator indexing?
    for (int64_t i = 0; i < dim_size; ++i) {
      thrust::swap(
        unpacked_pivots_data[i],
        unpacked_pivots_data[pivots_data[i]]
      );
    }
  };

  _launch_kernel<num_threads(), thread_work_size()>(iter.numel(), loop);
}

void unpack_pivots_cuda_kernel(
  TensorIterator& iter,
  int64_t dim_size
) {
  _unpack_pivots_internal_kernel(iter, dim_size);
}

} // anonymous namespace

REGISTER_DISPATCH(addr_stub, &addr_kernel_cuda);
REGISTER_DISPATCH(unpack_pivots_stub, &unpack_pivots_cuda_kernel);

}}
