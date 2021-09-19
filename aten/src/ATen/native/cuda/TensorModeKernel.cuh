#pragma once

#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/SortingCommon.cuh>
#include <ATen/native/cuda/block_reduce.cuh>

namespace at {
namespace native {

// Used for a segmented reduction
struct ModeUnsignedBoolPair {
  unsigned int val;
  bool flag;
};

// In the kernel below, we have a common pattern of reducing (unsigned int,
// unsigned int) pairs of data
struct ModeUnsignedPair {
  unsigned int val;
  unsigned int index;
};

// Inclusive Scan via an upsweep/downsweep mechanism. Assumes:
//
// 1. Power2ScanSize is a power of 2. This code still works for collections that
// do not exactly contain a power of 2 number of elements, simply round up to
// the nearest power of 2 and then call.
//
// 2. That there are two-elements per thread, i.e. the size of the smem storage
// is 2 * blockDim.x * sizeof(T).
//
// Consider a (+)-Scan on the following elements:
//
// Upsweep:
//
//    0  1  2  3  4  5  6  7
//       1     5     9    13
//             6          22
//                        28
//
// Downsweep:
//                  15
//         3     10    21
template <int Power2ScanSize, typename T, class BinaryOp>
__device__ void inclusivePrefixScan(T* smem, BinaryOp binop) {
  // Reduce step ("upsweep")
#pragma unroll
  for (int stride = 1; stride < Power2ScanSize; stride <<= 1) {
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if (index < Power2ScanSize) {
      smem[index] = binop(smem[index], smem[index - stride]);
    }
    __syncthreads();
  }

  // Post-reduce step ("downsweep")
#pragma unroll
  for (int stride = Power2ScanSize / 4; stride > 0; stride >>= 1) {
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if ((index + stride) < Power2ScanSize) {
      smem[index + stride] = binop(smem[index + stride], smem[index]);
    }
    __syncthreads();
  }
}

// Block-wide reduction where each thread locally reduces N
// values before letting a single warp take over - assumes
// threadVals is in registers, not shared memory
//
// If smem is not used again, there is no need to __syncthreads before this
// call. However, if smem will be used, e.g., this function is called in a loop,
// then __syncthreads is needed either before or afterwards to prevent non-0
// threads overriding smem in the next loop before num-0 thread reads from it.
template <int N, typename T, typename ReduceOp>
__device__ T reduceBlockWithNThreadLocalReductions(
    T* smem,
    T threadVals[N],
    const unsigned int numVals,
    ReduceOp reduceOp,
    T init) {
  int offset = threadIdx.x * N;
  T local = offset < numVals ? threadVals[0] : init;

#pragma unroll
  for (int i = 1; i < N; ++i) {
    ++offset;
    T next = offset < numVals ? threadVals[i] : init;
    local = reduceOp.combine(local, next);
  }

  return cuda_utils::BlockReduce(local, reduceOp, init, smem);
}

template <typename T>
__device__ inline void swapVars(T& t1, T& t2) {
  T tmp = t1;
  t1 = t2;
  t2 = tmp;
}

template <typename Comparator, typename K, typename V>
__device__ inline void bitonicSwap(
    K& kA,
    V& vA,
    bool& validA,
    K& kB,
    V& vB,
    bool& validB,
    bool dir,
    const Comparator& comp) {
  // Invalid entries always sort to the end
  bool swap = (comp(kA, kB) && validA) || !validB;
  if (swap == dir) {
    swapVars(kA, kB);
    swapVars(vA, vB);
    swapVars(validA, validB);
  }
};

template <typename Comparator, typename K>
__device__ inline void bitonicSwapKeys(
    K& kA,
    bool& validA,
    K& kB,
    bool& validB,
    bool dir,
    const Comparator& comp) {
  bool swap = (comp(kA, kB) && validA) || !validB;
  if (swap == dir) {
    swapVars(kA, kB);
    swapVars(validA, validB);
  }
}

template <
    typename K,
    typename IndexType,
    int Power2SortSize,
    typename Comparator>
__device__ inline void bitonicSortKeys(
    K keys[Power2SortSize],
    bool valid[Power2SortSize],
    const Comparator& comp) {
#ifndef __HIP_PLATFORM_HCC__
#pragma unroll
#endif
  for (unsigned int size = 2; size < Power2SortSize; size *= 2) {
    bool flag = ((threadIdx.x & (size / 2)) != 0);

#ifndef __HIP_PLATFORM_HCC__
#pragma unroll
#endif
    for (unsigned int stride = size / 2; stride > 0; stride /= 2) {
      __syncthreads();

      unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      bitonicSwapKeys<Comparator, K>(
          keys[pos],
          valid[pos],
          keys[pos + stride],
          valid[pos + stride],
          flag,
          comp);
    }
  }

#ifndef __HIP_PLATFORM_HCC__
#pragma unroll
#endif
  for (unsigned int stride = Power2SortSize / 2; stride > 0; stride /= 2) {
    __syncthreads();

    unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
    bitonicSwapKeys<Comparator, K>(
        keys[pos],
        valid[pos],
        keys[pos + stride],
        valid[pos + stride],
        false,
        comp);
  }

  __syncthreads();
}

// The mode kernel has the following characteristics: It uses internal shared
// memory buffers of Power2Size, which must be greater than the number of
// elements. Additionally, there is one block for every slice to calculate the
// mode for, and in each block there is one thread for every two elements.
//
// Both sorted and positions are assumed to be contiguous Tensors with the mode
// dimension as the innermost dim, such that we can get the particular slice for
// a Tensor via its linear block dimension * the slice size.
template <typename T, unsigned int Power2Size>
__global__ void compute_mode(
    T* input,
    at::cuda::detail::TensorInfo<T, unsigned int> values,
    at::cuda::detail::TensorInfo<int64_t, unsigned int> indices,
    int64_t sliceSize,
    int64_t slices) {
  int tidx = threadIdx.x;
  int stidx = blockDim.x + threadIdx.x; // Second index this thread responsible for

  // First, we need to calculate the offset into the sorted Tensor that
  // represents the start of the slice for this block to calculate the mode for.
  // This offset is a combination of the gridIndices, and the number of elements
  // in the slice.
  unsigned int blockId = getLinearBlockId<unsigned int>();
  unsigned int linearOffset = blockId * sliceSize;

  if (blockId >= slices) {
      return;
  }

  // shmem is a dynamically sized buffer we will use throughout the kernel to
  // handle computation efficiently. The size of this shmem must be
  // sizeof(T) * Power2Size + (2 * sizeof(unsigned int) * Power2Size)
  //
  // Initially, the buffer will be organized as follows:
  //
  // [smem (slice elements) | bmem (valid indices) | <scratch space>]
  extern __shared__ char shmem[];

  // smem represents a proportion of the shared memory buffer that is used to
  // store the elements from the slice:
  T* smem = reinterpret_cast<T*>(shmem);

  // Each thread loads up to two elements from the Tensor into shared memory
  if (tidx < sliceSize) {
    smem[tidx] = input[linearOffset + tidx];
  }
  if (stidx < sliceSize) {
    smem[stidx] = input[linearOffset + stidx];
  }

  // Next, we initialize a boolean region of the buffer, offset by the loaded
  // element smem region
  bool* bmem = reinterpret_cast<bool*>(&smem[Power2Size]);

  // The first use of this region stores bmem[i] = i < sliceSize to mark the
  // valid components in the smem buffer
  bmem[tidx] = tidx < sliceSize;
  bmem[stidx] = stidx < sliceSize;
  __syncthreads(); // barrier for smem, bmem initialization

  // First, sort the input slice in ascending order. smem contains the input
  // elements, and bmem marks the valid indices
  bitonicSortKeys<T, unsigned int, Power2Size>(
      smem, bmem, [&] GPU_LAMBDA(const auto& a, const auto& b) {
        return a < b;
      });
  __syncthreads(); // make no assumptions that the sort syncs at end

  // The next step of our algorithm is performing a block-wide comparison of
  // neighboring elements. In particular, given an sorted input slice A, we
  // produce an output slice B, such that B[i] = 1 if A[i-i] != A[i], otherwise
  // 0.
  //
  // Given the input A = [0, 0, 1, 1, 2, 2, 2, 4, 5, 6, 6, 7, 8]
  //                 B = [1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1]
  //
  // In particular, we can think of B[i] true indicating the start of a sequence
  // of equal values in the sorted list. Similarly, we will also store the
  // negation of B, which we'll call C. In particular, we can think of C[i] =
  // true iff A[i-1] == A[i] in our original sorted slice.
  //
  //                 C = [0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0]

  // We overwrite bmem, and treat the rest of shared memory as a buffer of
  // (index, flag) pairs where the index represents values from C, and the flag
  // represents values from B.
  //
  // [smem (sorted slice) | ubpmem (index, flag pairs)]

  struct ModeUnsignedBoolPair* ubpmem =
      reinterpret_cast<struct ModeUnsignedBoolPair*>(&smem[Power2Size]);

  if (tidx == 0) {
    ubpmem[0].flag = true;
    ubpmem[0].val = 0;
  }

  // Compares elements (0, 1), (2, 3), ... and sets 1, 3, ...
  ubpmem[tidx * 2 + 1].flag =
      smem[tidx * 2] != smem[tidx * 2 + 1]; // (0, 1), (1, 2), etc.
  ubpmem[tidx * 2 + 1].val = !ubpmem[tidx * 2 + 1].flag;

  // Compares elements (1, 2), (3, 4), ... and sets 2, 4, ...
  if (((tidx + 1) * 2) < Power2Size) {
    ubpmem[(tidx + 1) * 2].flag =
        smem[((tidx + 1) * 2) - 1] != smem[(tidx + 1) * 2];
    ubpmem[(tidx + 1) * 2].val = !ubpmem[(tidx + 1) * 2].flag;
  }
  __syncthreads(); // barrier for ubpmem initialization

  // Next, we perform a segmented prefix sum on the neighboring elements, where
  // the presence of a one indicates the start of a segment. In this case B acts
  // as the segment start flags, and C is the buffer to be summed:
  //
  // Input  (C)  = [0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0]
  // Flag   (B)  = [1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1]
  // Output (C)  = [0, 1, 0, 1, 0, 1, 2, 0, 0, 0, 1, 0, 0]
  //
  // Afterwards, the (index) components of the ubpmem buffer contain the lengths
  // of the segments (minus 1), i.e. the counts of each element in the original
  // input.
  inclusivePrefixScan<Power2Size>(
      ubpmem, [=] GPU_LAMBDA(const auto& a, const auto& b) {
        ModeUnsignedBoolPair c;
        c.val = a.flag ? a.val : a.val + b.val;
        c.flag = a.flag | b.flag;
        return c;
      });
  // assumes scan syncs at the end

  // Next, we reinterpret the ubpmem buffer as pairs of unsigned integers (i.e.
  // we treat the boolean flag regions as integers). We initialize these to
  // represent indices, and we'll call this buffer I
  struct ModeUnsignedPair* uupmem =
      reinterpret_cast<struct ModeUnsignedPair*>(ubpmem);

  // At this point, we need to find the maximum element in lengths buffer C.
  // This element will represent the count (-1) of the mode. Because of the
  // way we have set up the problem, the index where this mode occurs will
  // also be the location of the mode value in the sorted array, e.g.
  //
  // smem = [0, 0, 1, 1, 1, 2]
  // C    = [0, 1, 0, 1, 2, 0]
  // I    = [0, 1, 2, 3, 4, 5]
  //                     ^
  //                     maximum value, also aligned with mode = 1
  //
  // We perform a block wide max-reduction of the C buffer, but we also need the
  // indices to come along with it, so we utilize the uupmem construction.
  //
  // At the end we need to return the ModeUnsignedPair containing index = 4, val
  // = 2, which represents the max

  // In practice, we will make each thread locally reduce 2 values in its
  // registers prior to the global block-wide reduction. Note that instead of
  // tidx/stidx, we utilize tidx * 2, tidx * 2 + 1, so each thread deals with
  // adjacent elements. This is because the reduce code below relies on thread
  // elements to be adjacent.
  struct ModeUnsignedPair uup[2];
  uup[0].index = tidx * 2;
  uup[0].val = ubpmem[tidx * 2].val;
  uup[1].index = tidx * 2 + 1;
  uup[1].val = ubpmem[tidx * 2 + 1].val;
  __syncthreads();

  struct ModeUnsignedPair max = {0, 0};

  struct MaxOp {
    inline __device__ ModeUnsignedPair combine(ModeUnsignedPair a, ModeUnsignedPair b) const {
      return b.val > a.val ? b : a;
    }

    inline __device__ ModeUnsignedPair warp_shfl_down(ModeUnsignedPair acc, int offset) const {
      ModeUnsignedPair ret;
      ret.index = WARP_SHFL_DOWN(acc.index, offset);
      ret.val = WARP_SHFL_DOWN(acc.val, offset);
      return ret;
    }
  } max_op;

  max = reduceBlockWithNThreadLocalReductions<2>(
      uupmem,
      uup,
      sliceSize,
      max_op,
      max);

  // Store the mode in shared memory for use in finding the mode in the input
  // slice
  __shared__ T mode;

  // Given the above constraints, the mode is the value at the reduced index in
  // the original sorted element buffer
  if (tidx == 0) {
    mode = smem[max.index];
  }
  __syncthreads(); // broadcast mode

  // Finally, we need to find "an" index of the mode in the input
  // Tensor. The API does not constrain which index we pick, but here
  // we always pick the largest index. We store the index if the value
  // is the mode, or 0 otherwise. Then find the maximum value.
  //
  // Again we reduce 2 elements in the thread's registers prior to the
  // block-wide reduction
  unsigned mode_index[2] = {0u, 0u};
  if (tidx * 2 < sliceSize) {
    const unsigned idx = tidx * 2;
    mode_index[0] = input[linearOffset + idx] == mode ? idx : 0u;
  }
  if (tidx * 2 + 1 < sliceSize) {
    const unsigned idx = tidx * 2 + 1;
    mode_index[1] = input[linearOffset + idx] == mode ? idx : 0u;
  }

  struct MaxIndexOp {
    inline __device__ unsigned combine(unsigned a, unsigned b) const {
      return b > a ? b : a;
    }

    inline __device__ unsigned warp_shfl_down(unsigned acc, int offset) const {
      return WARP_SHFL_DOWN(acc, offset);
    }
  } max_index_op;

  int64_t index = reduceBlockWithNThreadLocalReductions<2>(
      reinterpret_cast<unsigned*>(&shmem[0]),
      mode_index,
      sliceSize,
      max_index_op,
      0u);

  // Finally, we have the mode, and an index where it occurs. We use a single
  // thread to place this in the appropriate output position
  if (tidx == 0) {
    unsigned int outputOffset =
        at::cuda::detail::IndexToOffset<T, unsigned int, -1>::get(
            blockId, values);
    values.data[outputOffset] = mode;
    indices.data[outputOffset] = index;
  }
}

} // namespace native
} // namespace at
