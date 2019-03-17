#include <THCUNN/THCUNN.h>
#include <THCUNN/common.h>
#include <THC/THCThrustAllocator.cuh>
#include <thrust/unique.h>
#include <TH/THHalf.h>
#include <THCUNN/THCHalfAutoNumerics.cuh>
#include <THC/THCTensorSort.cuh>
#include <THC/THCTensorMathReduce.cuh>

#ifdef __HIP_PLATFORM_HCC__
const int WARP_SIZE = 64;
#else
const int WARP_SIZE = 32;
#endif

template
  <typename Dtype,
   typename Acctype>
__global__ void cunn_LookupTable_accGradParametersKernelByFeature
  (int64_t *indices,
   Dtype *grad,
   Dtype *grad_weight,
   Dtype scale,
   ptrdiff_t n,
   int64_t stride,
   int padding_idx)
{
  extern __shared__ char buf[];
  Acctype* smem = (Acctype*)buf;
  Acctype* my_s = smem + WARP_SIZE*threadIdx.y;
  int* indices_batch = (int*)(buf + sizeof(Acctype)*WARP_SIZE*blockDim.y);

  const int s = (int)stride; // OK to make int, we don't expect 2 billion+ embedding row size

  const int f = threadIdx.x + blockIdx.x*blockDim.x; // feature_dim

  for(int batch_start = 0; batch_start < n; batch_start += blockDim.x*blockDim.y)
  {
    // Entire block cooperates to load a batch of 1024 indices to process
    int tid = threadIdx.x + threadIdx.y*blockDim.x;
    if(batch_start + tid < n)
      indices_batch[tid] = (int)(indices[batch_start + tid]);

    // Loop over the batch of <= 1024 loaded indices in chunks of blockDim.y = 32
    for(int chunk_start = batch_start; chunk_start < n; chunk_start += blockDim.y)
    {
      // This does double duty:  it makes sure indices_batch is ready, and it makes sure match-group
      // leaders are done with their accumulates before other warps start loading again.
      __syncthreads();

      int n_this_chunk = (n - chunk_start) < blockDim.y ? (n - chunk_start) : blockDim.y;

      int src_row = chunk_start + threadIdx.y;
      int dst_row = indices_batch[src_row - batch_start]; // This warp's target row in grad_weight

      // All warps load their smem segments with incoming grad data
      if(src_row < n && f < s && dst_row != padding_idx)
        my_s[threadIdx.x] =  ScalarConvert<Dtype, Acctype>::to(scale*grad[src_row*stride + f]);

      __syncthreads();

      // To ensure determinism, we can't just have each warp add its grad data to its dst_row.
      // We need to check if any other warps pulled grad data targeting dst_row.
      // If so, we elect the first warp in each matching group as the leader.
      // Each leader warp serializes the accumulates targeting dst_row in shared memory,
      // then finishes by adding the accumulated buffer to dst_row in grad_weight.
      if(dst_row != padding_idx && src_row < n) // Per-warp exit condition
      {
        int match_found_this_thread =
          (dst_row == indices_batch[chunk_start - batch_start + threadIdx.x]);
        if(threadIdx.x >= n_this_chunk)
          match_found_this_thread = 0;
#ifdef __HIP_PLATFORM_HCC__
        unsigned long long int matchmask = WARP_BALLOT(match_found_this_thread);
        int first_remaining_peer = __ffsll(matchmask) - 1;
#else
        unsigned int matchmask = WARP_BALLOT(match_found_this_thread);
        int first_remaining_peer = __ffs(matchmask) - 1;
#endif

        if(threadIdx.y == first_remaining_peer) // Nominate lowest-indexed warp as the leader
        {
          matchmask ^= (1 << first_remaining_peer);
          while(matchmask)
          {
#ifdef __HIP_PLATFORM_HCC__
            first_remaining_peer = __ffsll(matchmask) - 1;
#else
            first_remaining_peer = __ffs(matchmask) - 1;
#endif
	    my_s[threadIdx.x] += smem[threadIdx.x + WARP_SIZE*first_remaining_peer];
            matchmask ^= (1 << first_remaining_peer);
          }
          if(f < s)
            grad_weight[dst_row*stride + f] += ScalarConvert<Acctype, Dtype>::to(my_s[threadIdx.x]);
        }
      }
    }
  }
}

template <typename Dtype, typename Acctype>
__global__ void cunn_LookupTable_accGradParametersKernel(
  int64_t *input, int64_t *indices, Dtype *gradOutput, Dtype *gradWeight,
  int64_t *count, Dtype defaultScale, ptrdiff_t numel, int64_t stride, int paddingValue) {

  int idx = blockIdx.x * 4 + threadIdx.y;

  // Each warp is responsible for an input into the LookupTable.
  // If the preceding input has the same as this input, then the warp
  // exits immediately. The warp also processes subsequent inputs with the
  // same value.
  //
  // Input Warp
  // 1     <warp 1>
  // 1     <warp 1> (<warp 2> exits without doing any work)
  // 5     <warp 3>
  // 8     <warp 4>

  // Number of values proceessed by each thread (grain size)
  const int SZ = 4;

  if (idx < numel
      && (idx == 0 || input[idx] != input[idx - 1])
      && input[idx] != paddingValue) {
    do {
      const int startFeature = threadIdx.x + blockIdx.y * blockDim.x * SZ;
      const int weightRow = ((int) input[idx]) * stride;
      const int gradOutputRow = ((int) indices[idx]) * stride;
      const Acctype scale = count ? ScalarConvert<Dtype, Acctype>::to(defaultScale) / count[idx] : ScalarConvert<Dtype, Acctype>::to(defaultScale);

      Acctype gradient[SZ];
      Acctype weight[SZ];

      #pragma unroll
      for (int ii = 0; ii < SZ; ii++)
      {
        int featureDim = startFeature + ii * WARP_SIZE;
        if (featureDim < stride)
        {
          gradient[ii] = ScalarConvert<Dtype, Acctype>::to(gradOutput[gradOutputRow + featureDim]);
          weight[ii] = ScalarConvert<Dtype, Acctype>::to(gradWeight[weightRow + featureDim]);
        }
      }

      #pragma unroll
      for (int ii = 0; ii < SZ; ii++)
      {
        weight[ii] += gradient[ii] * scale;
      }

      #pragma unroll
      for (int ii = 0; ii < SZ; ii++)
      {
        int featureDim = startFeature + ii * WARP_SIZE;
        if (featureDim < stride)
        {
          gradWeight[weightRow + featureDim] = ScalarConvert<Acctype, Dtype>::to(weight[ii]);
        }
      }

      idx++;
    } while (idx < numel && input[idx] == input[idx - 1]);
  }
}

template <typename DType, typename AccType, int Norm>
struct FastPow
{
  __host__ __device__
  static inline AccType pow(DType x, AccType norm) {
    AccType xA = ScalarConvert<DType, AccType>::to(x);
    return std::pow(std::abs(xA), norm);
  }
};

template <typename DType, typename AccType>
struct FastPow<DType, AccType, 1>
{
  __host__ __device__
  static inline AccType pow(DType x, AccType _) {
    AccType xA = ScalarConvert<DType, AccType>::to(x);
    return std::abs(xA);
  }
};

template <typename DType, typename AccType>
struct FastPow<DType, AccType, 2>
{
  __host__ __device__
  static inline AccType pow(DType x, AccType _) {
    AccType xA = ScalarConvert<DType, AccType>::to(x);
    return xA * xA;
  }
};

/* Calculate norms of the rows of weight_ptr given by idx_ptr and capture them in norms */
template <typename DType, typename AccType, typename IndexType, int Norm>
__global__
void calculate_norms_and_renorm(DType *weights,
                                THCIndex_t *indices,
                                AccType normType,
                                AccType maxNorm,
                                IndexType dim)
{
  // Some casting hacks since dynamic shared memory and templates don't work together:
  extern __shared__ unsigned char smem[];
  AccType *sdata = reinterpret_cast<AccType *>(smem);

  IndexType tid = threadIdx.x;
  IndexType baseIndex = (indices[blockIdx.x]) * dim;

  AccType accZero = ScalarConvert<int, AccType>::to(0);
  AccType v = accZero;
  for (IndexType i = tid; i < dim; i += blockDim.x) {
    v += FastPow<DType, AccType, Norm>::pow(weights[baseIndex + i], normType);
  }

  v = reduceBlock<AccType, ReduceAdd<AccType>>
        (sdata, blockDim.x, v, ReduceAdd<AccType>(), accZero);

  if (tid == 0) {
    sdata[0] = std::pow(v,
        THCNumerics<AccType>::div(ScalarConvert<int, AccType>::to(1), normType)
    );
  }
  __syncthreads();
  // now we renormalize the blocks that need it
  if (sdata[0] > maxNorm) {
    DType factor = ScalarConvert<AccType, DType>::to(maxNorm / (sdata[0] + 1e-7));
    for (IndexType i = tid; i < dim; i += blockDim.x) {
      weights[baseIndex + i] *= factor;
    }
  }

}

#include <THCUNN/generic/LookupTable.cu>
#include <THC/THCGenerateFloatTypes.h>
