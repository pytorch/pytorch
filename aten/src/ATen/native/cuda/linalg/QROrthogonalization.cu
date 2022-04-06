#include <ATen/ATen.h>
#include <ATen/cuda/cub.cuh>
#include <ATen/cuda/CUDAContext.h>

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
    typedef cub::BlockReduce<scalar_t, BLOCK_THREADS, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tmp_storage;

    int tx = threadIdx.x;
    uint unroll = ceil( (float)length / (float)BLOCK_THREADS );
    uint idx = (tx & -32u)*unroll + (tx & 31);

    scalar_t local_prod = 0;
    for (uint i = 0; i < unroll; ++i){
        local_prod += (idx < length)? a[idx] * b[idx] : (scalar_t) 0;
        idx += 32;
    }

    scalar_t reduce = BlockReduce(tmp_storage).Sum(local_prod);

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
__global__  void q_loop(scalar_t *Q, scalar_t *vs, const uint n, const uint m){
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
void qr_main(const at::Tensor& A, at::Tensor& Q, const uint m, const uint n, const float epsilon){
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto options = at::TensorOptions().dtype(torch::kInt32).device(A.device());
    at::Tensor barriers = at::zeros({m}, options);
    
    at::Tensor vs = at::zeros_like(A);
    at::Tensor R = A.clone();
    R.diagonal().add_((scalar_t) epsilon);
    
    reflections<BLOCK_THREADS, scalar_t><<<m, BLOCK_THREADS, 0, stream>>>(
        R.data_ptr<scalar_t>(), 
        vs.data_ptr<scalar_t>(), 
        m, 
        n, 
        barriers.data_ptr<int>()
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


