#ifndef THC_TENSOR_MODE_CUH
#define THC_TENSOR_MODE_CUH

#include <THC/THCNumerics.cuh>
#include <THC/THCSortUtils.cuh>
#include <THC/THCScanUtils.cuh>

struct ThrustHalfLess
{
  __host__ __device__ inline bool operator()(const at::Half& lhs, const at::Half& rhs) {
    return THCNumerics<at::Half>::lt(lhs, rhs);
  }
};

struct ThrustHalfNotEqualTo
{
  __host__ __device__ inline bool operator()(const at::Half& lhs, const at::Half& rhs) {
    return THCNumerics<at::Half>::ne(lhs, rhs);
  }
};

struct ThrustHalfEqualTo
{
  __host__ __device__ inline bool operator()(const at::Half& lhs, const at::Half& rhs) {
    return THCNumerics<at::Half>::eq(lhs, rhs);
  }
};

struct ThrustHalfEqualToPredicate
{
  ThrustHalfEqualToPredicate(at::Half val): val_(val) {}
  __host__ __device__ inline bool operator()(at::Half x) {
    return THCNumerics<at::Half>::eq(val_, x);
  }

  at::Half val_;
};

template <typename T>
struct BinaryAddOp {
  __host__ __device__ inline T operator()(const T a, const T b) {
    return THCNumerics<T>::add(a, b);
  }
};

template <>
struct BinaryAddOp<unsigned int> {
  __host__ __device__ inline unsigned int operator()(const unsigned int a, const unsigned int b) {
    return a + b;
  }
};

// Used for a segmented reduction
struct ModeUnsignedBoolPair {
  unsigned int val;
  bool flag;
};

// In the kernel below, we have a common pattern of reducing (unsigned int, unsigned int)
// pairs of data
struct ModeUnsignedPair {
  unsigned int val;
  unsigned int index;
};

template <typename T>
struct MaxReduceOp {
  __host__ __device__ inline T operator()(const T& a, const T& b) {
    return b.val > a.val ? b : a;
  }
};

template <typename T>
struct MatchReduceOp {
  __host__ __device__ inline T operator()(const T& a, const T& b) {
    return b.flag ? b : a;
  }
};

// The mode kernel has the following characteristics: It uses internal shared memory
// buffers of Power2Size, which must be greater than the number of elements. Additionally,
// there is one block for every slice to calculate the mode for, and in each block there
// is one thread for every two elements.
//
// Both sorted and positions are assumed to be contiguous Tensors with the mode dimension
// as the innermost dim, such that we can get the particular slice for a Tensor via its
// linear block dimension * the slice size.
template <typename T, unsigned int Power2Size>
__global__ void computeMode(
    T *input,
    TensorInfo<T, unsigned int> values,
    TensorInfo<int64_t, unsigned int> indices,
    int64_t sliceSize)
{
  int tidx = threadIdx.x;
  int stidx = blockDim.x + threadIdx.x; // Second index this thread responsible for

  // First, we need to calculate the offset into the sorted Tensor that represents
  // the start of the slice for this block to calculate the mode for. This offset
  // is a combination of the gridIndices, and the number of elements in the slice.
  unsigned int blockId = getLinearBlockId<unsigned int>();
  unsigned int linearOffset = blockId * sliceSize;

  // shmem is a dynamically sized buffer we will use throughout the kernel to
  // handle computation efficiently. The size of this shmem must be
  // sizeof(T) * Power2Size + (2 * sizeof(unsigned int) * Power2Size)
  //
  // Initially, the buffer will be organized as follows:
  //
  // [smem (slice elements) | bmem (valid indices) | <scratch space>]
  extern __shared__ char shmem[];

  // smem represents a proportion of the shared memory buffer that is used to store
  // the elements from the slice:
  T *smem = reinterpret_cast<T *>(shmem);

  // Each thread loads up to two elements from the Tensor into shared memory
  if (tidx < sliceSize) {
    smem[tidx] = input[linearOffset + tidx];
  }
  if (stidx < sliceSize) {
    smem[stidx] = input[linearOffset + stidx];
  }

  // Next, we initialize a boolean region of the buffer, offset by the loaded element
  // smem region
  bool *bmem = reinterpret_cast<bool *>(&smem[Power2Size]);

  // The first use of this region stores bmem[i] = i < sliceSize to mark the valid
  // components in the smem buffer
  bmem[tidx] = tidx < sliceSize;
  bmem[stidx] = stidx < sliceSize;
  __syncthreads(); // barrier for smem, bmem initialization

  // First, sort the input slice in ascending order. smem contains the input
  // elements, and bmem marks the valid indices
  bitonicSortKeys<LTComp<T>, T, unsigned int, Power2Size>(smem, bmem, LTComp<T>());
  __syncthreads(); // make no assumptions that the sort syncs at end

  // The next step of our algorithm is performing a block-wide comparison of
  // neighboring elements. In particular, given an sorted input slice A, we
  // produce an output slice B, such that B[i] = 1 if A[i-i] != A[i], otherwise 0.
  //
  // Given the input A = [0, 0, 1, 1, 2, 2, 2, 4, 5, 6, 6, 7, 8]
  //                 B = [1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1]
  //
  // In particular, we can think of B[i] true indicating the start of a sequence of
  // equal values in the sorted list. Similarly, we will also store the negation of B,
  // which we'll call C. In particular, we can think of C[i] = true iff A[i-1] == A[i]
  // in our original sorted slice.
  //
  //                 C = [0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0]

  // We overwrite bmem, and treat the rest of shared memory as a buffer of (index, flag) pairs
  // where the index represents values from C, and the flag represents values from B.
  //
  // [smem (sorted slice) | ubpmem (index, flag pairs)]

  struct ModeUnsignedBoolPair *ubpmem = reinterpret_cast<struct ModeUnsignedBoolPair *>(
      &smem[Power2Size]);

  if (tidx == 0) {
    ubpmem[0].flag = true;
    ubpmem[0].val = 0;
  }

  // Compares elements (0, 1), (2, 3), ... and sets 1, 3, ...
  ubpmem[tidx * 2 + 1].flag = THCNumerics<T>::ne(smem[tidx * 2], smem[tidx * 2 + 1]); // (0, 1), (1, 2), etc.
  ubpmem[tidx * 2 + 1].val = !ubpmem[tidx * 2 + 1].flag;

  // Compares elements (1, 2), (3, 4), ... and sets 2, 4, ...
  if (((tidx + 1) * 2) < Power2Size) {
    ubpmem[(tidx + 1) * 2].flag = THCNumerics<T>::ne(smem[((tidx + 1) * 2) - 1], smem[(tidx + 1) * 2]);
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
  // Afterwards, the (index) components of the ubpmem buffer contain the lengths of the
  // segments (minus 1), i.e. the counts of each element in the original input.

  inclusivePrefixScan<
    struct ModeUnsignedBoolPair,
    struct SegmentedScanOp<struct ModeUnsignedBoolPair, BinaryAddOp<unsigned int> >,
    Power2Size>(
        ubpmem,
        SegmentedScanOp<struct ModeUnsignedBoolPair, BinaryAddOp<unsigned int> >(BinaryAddOp<unsigned int>()));
  // assumes scan syncs at the end

  // Next, we reinterpret the ubpmem buffer as pairs of unsigned integers (i.e. we treat the
  // boolean flag regions as integers). We initialize these to represent indices, and we'll call
  // this buffer I
  struct ModeUnsignedPair *uupmem = reinterpret_cast<struct ModeUnsignedPair *>(ubpmem);

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
  // At the end we need to return the ModeUnsignedPair containing index = 4, val = 2,
  // which represents the max

  // In practice, we will make each thread locally reduce 2 values in its registers prior
  // to the global block-wide reduction. Note that instead of tidx/stidx, we utilize tidx * 2,
  // tidx * 2 + 1, so each thread deals with adjacent elements. This is because the reduce
  // code below relies on thread elements to be adjacent.
  struct ModeUnsignedPair uup[2];
  uup[0].index = tidx * 2;
  uup[0].val = ubpmem[tidx * 2].val;
  uup[1].index = tidx * 2 + 1;
  uup[1].val = ubpmem[tidx * 2 + 1].val;
  __syncthreads();

  struct ModeUnsignedPair max = {0, 0};

  max = reduceBlockWithNThreadLocalReductions<struct ModeUnsignedPair, MaxReduceOp<struct ModeUnsignedPair>, 2>
    (uupmem, uup, sliceSize, MaxReduceOp<struct ModeUnsignedPair>(), max);

  // Store the mode in shared memory for use in finding the mode in the input slice
  __shared__ T  mode;

  // Given the above constraints, the mode is the value at the reduced index in the
  // original sorted element buffer
  if (tidx == 0) {
    mode = smem[max.index];
  }
  __syncthreads(); // broadcast mode

  // Finally, we need to find the "an" index of the mode in the input Tensor. The API does
  // not constrain which index we pick, so it can be any of the indices that contain the mode.
  // We will do a reduction to find the index. We go back to using the (index, flag) buffer
  // arrangement. First, we mark indices that are equal to the mode, i.e B[i] = true if
  // input[i] == mode, and initialize C[i] to be the index
  //
  // Again we reduce 2 elements in the thread's registers prior to the block-wide reduction
  struct ModeUnsignedBoolPair ubpp[2];
  if (tidx * 2 < sliceSize) {
    ubpp[0].flag = THCNumerics<T>::eq(input[linearOffset + (tidx * 2)], mode);
    ubpp[0].val = tidx * 2;
  }
  if (tidx * 2 + 1 < sliceSize) {
    ubpp[1].flag = THCNumerics<T>::eq(input[linearOffset + (tidx * 2 + 1)], mode);
    ubpp[1].val = tidx * 2 + 1;
  }

  // Then we perform a similar reduction to the one above, except this time we update
  // the element if the element at the base position is not equal to the mode and
  // the element at the offset position is. At the end, C[0] will contain an index
  // with the mode.
  struct ModeUnsignedBoolPair match = {0, false};

  match = reduceBlockWithNThreadLocalReductions<struct ModeUnsignedBoolPair, MatchReduceOp<struct ModeUnsignedBoolPair>, 2>
    (ubpmem, ubpp, sliceSize, MatchReduceOp<struct ModeUnsignedBoolPair>(), match);

  // Finally, we have the mode, and an index where it occurs. We use a single thread
  // to place this in the appropriate output position
  if (tidx == 0) {
    int64_t index = match.val;

    unsigned int outputOffset = IndexToOffset<T, unsigned int, -1>::get(blockId, values);
    values.data[outputOffset] = mode;
    indices.data[outputOffset] = index;
  }
}

#endif // THC_TENSOR_MODE_CUH
