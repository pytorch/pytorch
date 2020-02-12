#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/TensorUtils.h>
#include <c10/util/Exception.h>

#include <ATen/cuda/CUDAContext.h>
#include <THC/THCDeviceUtils.cuh>
#include <THC/THCTensorMathReduce.cuh>

namespace at {
namespace native {
namespace {

// Block size for weight_norm_*_first_dim_kernel.
// Currently, kernels are non-persistent.
// Dialing up the block size to, say 1024, can improve performance by
// increase the amount of cache available per block, which can improve cache hit rate.
// However, this is less efficient for short rows.  256 is pretty versatile.
// May be worth implementing heuristics later.
#define BLOCK 256

// Block size for weight_norm_*_last_dim_kernel.
// This is tricker than the first_dim case because we must make blocks
// at least 16 fast elements wide to ensure fully-coalesced half-precision accesses.
// Since output-element parallelism is along the fast dimension, this reduces the number of
// blocks we can launch by 16X.
#define TILE_W 16
// Somewhat versatile strategy: max out intra-block parallelism by extending
// blocks across the slow dimension up to the hardware-max block size of 1024.
#define TILE_H 64

template<typename T, typename ReduceOp>
__device__ __forceinline__ void reduce_block_into_lanes
  (T *x,
   T val,
   int lanes, // lanes is intended to be <= 32.
   ReduceOp reduceOp)
{
  int tid = threadIdx.x + threadIdx.y*blockDim.x;
  int blockSize = blockDim.x*blockDim.y; // blockSize is intended to be a multiple of 32.

  if(blockSize >= 64)
  {
    x[tid] = val;
    __syncthreads();
  }

#ifndef __HIP_PLATFORM_HCC__
  #pragma unroll
#endif
  for(int i = (blockSize >> 1); i >= 64; i >>= 1)
  {
    if(tid < i)
      x[tid] = reduceOp(x[tid], x[tid+i]);
    __syncthreads();
  }

  if(tid < 32)
  {
    T final;
    if(blockSize >= 64)
      final = reduceOp(x[tid], x[tid+32]);
    else
      final = val;
    // __SYNCWARP();

#ifndef __HIP_PLATFORM_HCC__
    #pragma unroll
#endif
    for(int i = 16; i >= lanes; i >>= 1)
      final = reduceOp(final, WARP_SHFL_DOWN(final, i));

    if(tid < lanes)
      x[tid] = final; // EpilogueOp
  }

  // Make sure the smem result is visible to all warps.
  __syncthreads();
}

template
  <typename scalar_t,
   typename accscalar_t>
__global__ void weight_norm_fwd_first_dim_kernel
  (scalar_t* __restrict__ w,
   accscalar_t* __restrict__ norms,
   const scalar_t* __restrict__ v,
   const scalar_t* __restrict__ g,
   const int rowSize)
{
  // We are norming each slowest-dim row of the tensor separately.
  // For now, assign one block to each row.
  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  const int stride = blockDim.x;

  // Logical index offset for this flattened row
  const int rowStart = row*rowSize;

  // Hack to get around nvcc complaining when an smem array is declared with the same name
  // but different types in different kernels (in this case different instantiations)
  // extern __shared__ accscalar_t s[]; // error: declaration is incompatible with previous "s"
  extern __shared__ char buf[];
  accscalar_t* s = (accscalar_t*)buf;

  accscalar_t thread_sum = 0.f;
  for(int i = tid; i < rowSize; i += stride )
  {
    accscalar_t val_f = scalar_cast<accscalar_t>(v[i+rowStart]);
    thread_sum += val_f*val_f; // AccumOp, could do Kahan here
  }

  reduce_block_into_lanes(s, thread_sum, 1, ReduceAdd<accscalar_t>());
  accscalar_t result = s[0];

  result = sqrtf(result);

  if(tid == 0)
    norms[row] = result;

  // Broadcast load, could use shared memory instead.
  accscalar_t g_this_row = scalar_cast<accscalar_t>(g[row]);

  accscalar_t rnorm = 1.f/result; // for consistency with backward kernel

  // Write data to output
  for(int i = tid; i < rowSize; i += stride )
  {
    accscalar_t val_f = scalar_cast<accscalar_t>(v[i+rowStart]);
    w[i+rowStart] = scalar_cast<scalar_t>(g_this_row*val_f*rnorm);
  }
}

template
  <typename scalar_t,
   typename accscalar_t>
__global__ void weight_norm_fwd_last_dim_kernel
(
  scalar_t* __restrict__ w,
  accscalar_t* __restrict__ norms,
  const scalar_t* __restrict__ v,
  const scalar_t* __restrict__ g,
  const int fast_dim_size,
  const int slower_dims_size
)
{
  const int fast_dim_location = threadIdx.x + blockIdx.x*blockDim.x;

  extern __shared__ char buf[];
  accscalar_t* alloc = (accscalar_t*)buf;
  accscalar_t* s = &alloc[0];
  accscalar_t* rnorms_this_block = &alloc[blockDim.x*blockDim.y];

  accscalar_t thread_sum = 0.f;

  int slower_dims_location = threadIdx.y;
  int currentIdx = fast_dim_location + fast_dim_size*slower_dims_location;
  if(fast_dim_location < fast_dim_size)
    while(slower_dims_location < slower_dims_size)
    {
      accscalar_t val_f = scalar_cast<accscalar_t>(v[currentIdx]);
      thread_sum += val_f*val_f; // AccumOp, could do Kahan here
      currentIdx += blockDim.y*fast_dim_size;
      slower_dims_location += blockDim.y;
    }

  reduce_block_into_lanes(s, thread_sum, blockDim.x, ReduceAdd<accscalar_t>());

  // Better to pass an EpilogueOp to reduce_block_into_lanes?
  if(threadIdx.y == 0)
  {
    accscalar_t result = s[threadIdx.x];
    accscalar_t norm_this_col = sqrtf(result);
    norms[fast_dim_location] = norm_this_col;
    rnorms_this_block[threadIdx.x] = 1.f/norm_this_col;
  }

  __syncthreads();

  accscalar_t g_this_col = scalar_cast<accscalar_t>(g[fast_dim_location]);
  accscalar_t rnorm = rnorms_this_block[threadIdx.x];

  slower_dims_location = threadIdx.y;
  currentIdx = fast_dim_location + fast_dim_size*slower_dims_location;
  if(fast_dim_location < fast_dim_size)
    while(slower_dims_location < slower_dims_size)
    {
      accscalar_t val_f = scalar_cast<accscalar_t>(v[currentIdx]);
      w[currentIdx] = scalar_cast<scalar_t>(g_this_col*val_f*rnorm);
      currentIdx += blockDim.y*fast_dim_size;
      slower_dims_location += blockDim.y;
    }
}

template
  <typename scalar_t,
   typename accscalar_t>
__global__ void weight_norm_bwd_first_dim_kernel
  (scalar_t* __restrict__ grad_v,
   scalar_t* __restrict__ grad_g,
   const scalar_t* __restrict__ grad_w,
   const scalar_t* __restrict__ saved_v,
   const scalar_t* __restrict__ saved_g,
   const accscalar_t* __restrict__ saved_norms,
   const int rowSize)
{
  // For now, assign one block to each row.
  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  const int stride = blockDim.x;

  // Logical index offset for this flattened row
  const int rowStart = row*rowSize;

  // Hack to get around nvcc complaining when an smem array is declared with the same name
  // but different types in different kernels (in this case different instantiations)
  // extern __shared__ accscalar_t s[]; // error: declaration is incompatible with previous "s"
  extern __shared__ char buf[];
  accscalar_t* s = (accscalar_t*)buf;

  accscalar_t thread_sum = 0.f;
  for(int i = tid; i < rowSize; i += stride )
  {
    accscalar_t grad_wi = scalar_cast<accscalar_t>(grad_w[i+rowStart]);
    accscalar_t saved_vi = scalar_cast<accscalar_t>(saved_v[i+rowStart]);
    thread_sum += grad_wi*saved_vi; // AccumOp, could do Kahan here
  }

  reduce_block_into_lanes(s, thread_sum, 1, ReduceAdd<accscalar_t>());
  accscalar_t result = s[0];

  // Could choose to save reciprocal of norm instead I suppose, but norms is probably
  // more handy to keep around.
  // Broadcast load; could use shared memory instead.
  accscalar_t rnorm = 1.f/saved_norms[row];
  accscalar_t rnorm3 = rnorm*rnorm*rnorm;

  // Write g gradients.
  if(tid == 0)
    grad_g[row] = scalar_cast<scalar_t>(result*rnorm);

  // Broadcast load, could use shared memory instead.
  accscalar_t g_this_row = scalar_cast<accscalar_t>(saved_g[row]);

  // Write v gradients.  We are reusing values that were loaded earlier, so there
  // is an optimization opportunity here (store values persistently).
  for(int j = tid; j < rowSize; j += stride )
  {
    accscalar_t grad_wj = scalar_cast<accscalar_t>(grad_w[j+rowStart]);
    accscalar_t saved_vj = scalar_cast<accscalar_t>(saved_v[j+rowStart]);
    accscalar_t grad_vj = g_this_row*(rnorm*grad_wj - rnorm3*saved_vj*result);
    grad_v[j+rowStart] = scalar_cast<scalar_t>(grad_vj);
  }
}

template
  <typename scalar_t,
   typename accscalar_t>
__global__ void weight_norm_bwd_last_dim_kernel
  (scalar_t* __restrict__ grad_v,
   scalar_t* __restrict__ grad_g,
   const scalar_t* __restrict__ grad_w,
   const scalar_t* __restrict__ saved_v,
   const scalar_t* __restrict__ saved_g,
   const accscalar_t* __restrict__ saved_norms,
   const int fast_dim_size,
   const int slower_dims_size)
{
  const int fast_dim_location = threadIdx.x + blockIdx.x*blockDim.x;

  extern __shared__ char buf[];
  accscalar_t* s = (accscalar_t*)buf;

  accscalar_t thread_sum = 0.f;

  int slower_dims_location = threadIdx.y;
  int currentIdx = fast_dim_location + fast_dim_size*slower_dims_location;
  if(fast_dim_location < fast_dim_size)
    while(slower_dims_location < slower_dims_size)
    {
      accscalar_t grad_wi = scalar_cast<accscalar_t>(grad_w[currentIdx]);
      accscalar_t saved_vi = scalar_cast<accscalar_t>(saved_v[currentIdx]);
      thread_sum += grad_wi*saved_vi; // AccumOp, could do Kahan here
      currentIdx += blockDim.y*fast_dim_size;
      slower_dims_location += blockDim.y;
    }

  reduce_block_into_lanes(s, thread_sum, blockDim.x, ReduceAdd<accscalar_t>());
  accscalar_t result = s[threadIdx.x];

  // Broadcast load; could use shared memory instead.
  accscalar_t rnorm = 1.f/saved_norms[fast_dim_location];
  accscalar_t rnorm3 = rnorm*rnorm*rnorm;

  // Write g gradients.
  if(threadIdx.y == 0)
    grad_g[fast_dim_location] = scalar_cast<scalar_t>(result*rnorm);

  // Entire block pulls these values, could use shared memory instead.
  accscalar_t g_this_col = scalar_cast<accscalar_t>(saved_g[fast_dim_location]);

  // Write v gradients.
  slower_dims_location = threadIdx.y;
  currentIdx = fast_dim_location + fast_dim_size*slower_dims_location;
  if(fast_dim_location < fast_dim_size)
    while(slower_dims_location < slower_dims_size)
    {
      accscalar_t grad_wj = scalar_cast<accscalar_t>(grad_w[currentIdx]);
      accscalar_t saved_vj = scalar_cast<accscalar_t>(saved_v[currentIdx]);
      accscalar_t grad_vj = g_this_col*(rnorm*grad_wj - rnorm3*saved_vj*result);
      grad_v[currentIdx] = scalar_cast<scalar_t>(grad_vj);
      currentIdx += blockDim.y*fast_dim_size;
      slower_dims_location += blockDim.y;
    }
}

} // anonymous namespace

std::tuple<Tensor,Tensor> weight_norm_cuda
  (const Tensor & v,
   const Tensor & g,
   int64_t dim)
{
  auto w = at::empty_like(v, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  // weight_norm_fused does have a derivative defined in derivatives.yaml, therefore, VariableType.cpp
  // sends the unpacked g.data() as the argument.  In other words, we expect "g" is a bare Tensor here.

  // norms is only needed to stash for backward.
  // g.scalar_type() may be at::ScalarType::Double, Float, or Half.
  // If Half, stash norms as float.
  at::ScalarType AccType = g.scalar_type() == at::ScalarType::Half ?
                           at::ScalarType::Float : g.scalar_type();
  // Will this create norms on the same device as g, regardless of what the thread's default
  // current device is?  I believe so, because Type::* functions are DeviceGuard()ed.
  auto norms = at::empty_strided(g.sizes(), g.strides(), g.options().dtype(AccType));

  const int ndims = v.dim();

  if(dim == 0)
  {
    // Find logical size of each flattened slowest-dim row
    int rowSize = 1;
    for(int i = ndims - 1; i > 0; i--)
      rowSize *= v.size(i);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF
      (v.scalar_type(),
       "weight_norm_fwd_first_dim_kernel",
       [&]
       {
         using accscalar_t = acc_type<scalar_t, true>;

         weight_norm_fwd_first_dim_kernel<scalar_t, accscalar_t>
           <<<v.size(0),
              BLOCK,
              BLOCK*sizeof(accscalar_t),
              stream>>>
           (w.data_ptr<scalar_t>(),
            norms.data_ptr<accscalar_t>(),
            v.data_ptr<scalar_t>(),
            g.data_ptr<scalar_t>(),
            rowSize);
       });
  }
  else if(dim == ndims - 1)
  {
    // Precompute slower_dims_size and fast_dim_size
    int slower_dims_size = 1;
    for(int i = 0; i < ndims - 1; i++)
      slower_dims_size *= v.size(i);

    int fast_dim_size = v.size(ndims-1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF
      (v.scalar_type(),
       "weight_norm_fwd_last_dim_kernel",
       [&]
       {
         using accscalar_t = acc_type<scalar_t, true>;

         weight_norm_fwd_last_dim_kernel<scalar_t, accscalar_t>
           <<<(fast_dim_size+TILE_W-1)/TILE_W,
              dim3(TILE_W,TILE_H),
              (TILE_W*TILE_H + TILE_W)*sizeof(accscalar_t),
              stream>>>
           (w.data_ptr<scalar_t>(),
            norms.data_ptr<accscalar_t>(),
            v.data_ptr<scalar_t>(),
            g.data_ptr<scalar_t>(),
            fast_dim_size,
            slower_dims_size);
       });
  }

  // The kernel execution is asynchronous, so this will only catch errors on the kernel launch,
  // not the kernel's execution.  Errors in kernel execution aren't guaranteed to be caught
  // until a later error check on a synchronizing CUDA call.  Unfortunately, without manually
  // synchronizing here, this is the best we can do.
  THCudaCheck(cudaGetLastError());

  return std::tuple<Tensor, Tensor>{w, norms};
}

std::tuple<Tensor, Tensor> weight_norm_cuda_backward
  (const Tensor & grad_w,
   const Tensor & saved_v,
   const Tensor & saved_g,
   const Tensor & saved_norms,
   int64_t dim)
{
  // These checks should always succeed, because weight_norm_fused_backward should only
  // ever be recorded in the autograd graph via weight_norm, which passes contiguous v and g.
  TORCH_CHECK(saved_v.is_contiguous(), "saved_v must be contiguous");
  TORCH_CHECK(saved_g.is_contiguous(), "saved_g must be contiguous");
  TORCH_CHECK(saved_norms.is_contiguous(), "saved_norms must be contiguous");
  TORCH_CHECK(dim == 0 || dim == saved_v.dim() - 1, "fused kernels can only be applied for first or last dim")

  auto grad_v = at::empty_like(saved_v, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto grad_g = at::empty_like(saved_g, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  const int ndims = saved_v.dim();

  if(dim == 0)
  {
    // Find logical size of each flattened slowest-dim row
    int rowSize = 1;
    for(int i = ndims - 1; i > 0; i--)
      rowSize *= saved_v.size(i);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF
      (saved_v.scalar_type(),
       "weight_norm_bwd_first_dim_kernel",
       [&]
       {
         using accscalar_t = acc_type<scalar_t, true>;

         weight_norm_bwd_first_dim_kernel<scalar_t, accscalar_t>
           <<<grad_w.size(0),
              BLOCK,
              BLOCK*sizeof(accscalar_t),
              stream>>>
           (grad_v.data_ptr<scalar_t>(),
            grad_g.data_ptr<scalar_t>(),
            grad_w.data_ptr<scalar_t>(),
            saved_v.data_ptr<scalar_t>(),
            saved_g.data_ptr<scalar_t>(),
            saved_norms.data_ptr<accscalar_t>(),
            rowSize);
       });
  }
  else if(dim == ndims - 1)
  {
    // Precompute slower_dims_size and fast_dim_size because they involve dynamically indexing an array.
    int slower_dims_size = 1;
    for(int i = 0; i < ndims - 1; i++)
      slower_dims_size *= saved_v.size(i);

    int fast_dim_size = saved_v.size(ndims-1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF
      (saved_v.scalar_type(),
       "weight_norm_bwd_last_dim_kernel",
       [&]
       {
         using accscalar_t = acc_type<scalar_t, true>;

         weight_norm_bwd_last_dim_kernel<scalar_t, accscalar_t>
           <<<(fast_dim_size+TILE_W-1)/TILE_W,
              dim3(TILE_W,TILE_H),
              (TILE_W*TILE_H + TILE_W)*sizeof(accscalar_t),
              stream>>>
           (grad_v.data_ptr<scalar_t>(),
            grad_g.data_ptr<scalar_t>(),
            grad_w.data_ptr<scalar_t>(),
            saved_v.data_ptr<scalar_t>(),
            saved_g.data_ptr<scalar_t>(),
            saved_norms.data_ptr<accscalar_t>(),
            fast_dim_size,
            slower_dims_size);
       });
  }

  // The kernel execution is asynchronous, so this will only catch errors on the kernel launch,
  // not the kernel's execution.  Errors in kernel execution aren't guaranteed to be caught
  // until a later error check on a synchronizing CUDA call.  Unfortunately, without manually
  // synchronizing here, this is the best we can do.
  THCudaCheck(cudaGetLastError());

  return std::tuple<Tensor, Tensor>{grad_v, grad_g};
}

#undef BLOCK
#undef TILE_W
#undef TILE_H

} // namespace native
} // namespace at
