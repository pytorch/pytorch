#ifndef THC_OFFSET_INFO_INC
#define THC_OFFSET_INFO_INC

#include "THCIntegerDivider.cuh"
#include "THCTensorInfo.cuh"

// A faster implementation of IndexToOffset that pre-computes the increments to
// the indices along each dimension.
//
// Consider a kernel with the following loop:
//
//      for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
//           linearIndex < totalElements;
//           linearIndex += gridDim.x * blockDim.x) { ... }
//
// Note that the increment value (gridDim.x * blockDim.x) is fixed and known
// before the kernel starts.  Hence, we can pre-compute how the coordinates
// change.  For example, assume that the increment is 1024 and the tensor has
// size [5, 70, 10] and strides [2000, 20, 1].  During the execution of a
// particular thread, linearIndex may change from 1205 to 2229:
//
//      Before: linear index 1205 = 1 * 70 * 10 + 50 * 10 + 5
//              coordinate (1, 50, 5)
//              offset = 1 * 2000 + 50 * 20 + 5 * 1 = 3005
//
//      Linear index increment = 1024 = 1 * 70 * 10 + 32 * 10 + 4
//      Coordinate increment = (1, 32, 4)
//
//      After: linear index 2029
//             coordinate (1, 50, 5) + (1, 32, 4) = (2, 82, 9)
//                        = (3, 12, 9)   // Handle carry over.
//             offset = 3 * 2000 + 12 * 20 + 9 * 1 = 6249
//
// Thus, by pre-computing "coordinate increment" (1, 32, 4), we can compute the
// next coordinate and offset without costly division.  We also pre-compute the
// following values, for each dimension k (0 <= k < Dims):
//
//      increments[k] == (increment to coordinate #k at each step)
//      scaledIncrs[k] == increments[k] * strides[k]
//      carryDelta[k] == strides[k - 1] - (size[k] * strides[k])  (for k > 0)
//
// (carryDelta[k] is the change of offset when the addition to dimension #k
// "carries over" to #k-1.  In general, carryDelta can be "negative" even though
// IndexType is unsigned: it is always added to the offset, so the result will
// be correct.)
//
// So, the example shown above will be stored as:
//
//      increments  = (1, 32, 4)
//      scaledIncrs = (2000, 640, 4)
//      carryDelta  = (_, 600, 10)
//
// Finally, we can also optimize the initial coordinate computation by
// pre-computing "magic dividers", thus replacing the initial div/mod operations
// by multiplication.  (See IntDivider for details.)

// Helper function that increments 'indices' and returns the updated offset.
template <typename IndexType>
__host__ __device__ __forceinline__
IndexType incrementIdx(int dims, IndexType offset,
                       const IntDivider<IndexType> sizes[],
                       const IndexType increments[],
                       const IndexType scaledIncrs[],
                       const IndexType carryDelta[],
                       IndexType indices[])
{
  bool carry = false;

  for (int i = dims - 1; i > 0; --i) {
    IndexType index = indices[i] + increments[i] + (IndexType) carry;
    offset += scaledIncrs[i];

    // Note that 'index' here may reach (2 * sizes[i] - 1), because indices[i]
    // and increments[i] can each be up to (sizes[i] - 1), and there can be a
    // carry.  Assuming 32-bit indices, sizes[i] can be at most UINT32_MAX =
    // 0xffffffff (otherwise we will be using 64-bit indices), so the "true
    // value" of 'index' can be at most 0x1fffffffd, larger than UINT32_MAX.
    //
    // The second condition below checks if 'index' had an overflow.
    carry = (index >= sizes[i].divisor) || (index < indices[i]);
    if (carry) {
      index -= sizes[i].divisor;
      offset += carryDelta[i];
    }

    indices[i] = index;
  }

  // Dimension 0 is special because we are guaranteed to stay within bound.
  offset += scaledIncrs[0];

  return offset;
}

// Helper class that keeps track of the linear index.
template <typename IndexType> struct LinearIdIterator {
  // NOTE: The typecasting in 'index' and 'step' is essential when IndexType is
  //       64-bit; without casting, results are silently truncated to 32 bits!
  __device__ explicit LinearIdIterator(IndexType limit)
    : index((IndexType) blockIdx.x * blockDim.x + threadIdx.x),
      step((IndexType) gridDim.x * blockDim.x),
      limit(limit), hasNext(index < limit) { }

  // Explicit version for testing.
  __host__ __device__
  LinearIdIterator(IndexType start, IndexType step, IndexType limit)
    : index(start), step(step), limit(limit), hasNext(start < limit) { }

  // Assumes 'hasNext' is true before called.
  __host__ __device__ void increment() {
    IndexType next = index + step;

    // The second condition is necessary to handle overflow (e.g., when step is
    // 2GB and limit is 3GB, assuming 32-bit index).
    hasNext = (next < limit) && (next >= index);
    index = next;
  }

  IndexType index;
  const IndexType step, limit;
  bool hasNext;
};

template <typename T, typename IndexType, int Dims>
struct OffsetInfo {
  OffsetInfo(const TensorInfo<T, IndexType>& tinfo, IndexType step) {
    assert(tinfo.dims == Dims);
    data = tinfo.data;

    for (int i = Dims - 1; i >= 0; --i) {
      IndexType size = tinfo.sizes[i];
      IndexType stride = tinfo.strides[i];

      sizes[i] = IntDivider<IndexType>(size);
      strides[i] = stride;

      increments[i] = step % size;
      step /= size;

      scaledIncrs[i] = increments[i] * stride;
      if (i > 0)
        carryDelta[i] = tinfo.strides[i - 1] - (size * stride);
    }
  }

  T* data;
  IntDivider<IndexType> sizes[Dims];
  IndexType strides[Dims];

  // increments[0] is filled in but not actually used; carryDelta[0] is unused.
  IndexType increments[Dims];
  IndexType scaledIncrs[Dims];
  IndexType carryDelta[Dims];
};

template <typename T, typename IndexType, int Dims>
struct OffsetIterator {
  typedef OffsetInfo<T, IndexType, Dims> Info;

  // Initializes the current offset and coordinates given the initial linearId.
  __host__ __device__
  OffsetIterator(const Info &info, const LinearIdIterator<IndexType> &linear) {
    IndexType index = linear.index;
    offset = 0;

    for (int i = Dims - 1; i > 0; --i) {
      DivMod<IndexType> divmod = info.sizes[i].divmod(index);
      index = divmod.div;
      indices[i] = divmod.mod;
      offset += divmod.mod * info.strides[i];
    }

    offset += index * info.strides[0];
  }

  // Return the pointer to the current element.
  // 'linear' is ignored here: it is used only for the contiguous tensor case.
  __host__ __device__ T* get(const Info &info,
                             const LinearIdIterator<IndexType> &linear) {
    return &info.data[offset];
  }

  // Increment the offset.
  __host__ __device__ void increment(const Info &info) {
    offset = incrementIdx(Dims, offset,
                          info.sizes, info.increments,
                          info.scaledIncrs, info.carryDelta,
                          indices);
  }

  // indices[0] is not used.
  IndexType indices[Dims];

  // The current offset.
  IndexType offset;
};

// For contiguous tensors, offset equals linear index.
template <typename T, typename IndexType>
struct OffsetInfo<T, IndexType, -2> {
  OffsetInfo(const TensorInfo<T, IndexType>& tinfo, IndexType step)
    : data(tinfo.data) {
    assert(tinfo.isContiguous());
  }

  T* data;
};

template <typename T, typename IndexType>
struct OffsetIterator<T, IndexType, -2> {
  typedef OffsetInfo<T, IndexType, -2> Info;

  __host__ __device__
  OffsetIterator(const Info &info,
                 const LinearIdIterator<IndexType> &linear) { }

  // Return the pointer to the current element.
  __host__ __device__ T* get(const Info &info,
                             const LinearIdIterator<IndexType> &linear) {
    return &info.data[linear.index];
  }

  __host__ __device__ void increment(const Info &info) { }
};

// Dims=-1 is used when the dimension is unknown at compile time.
//
// Unfortunately, pre-computation does not work here, because of a bug in nvcc
// (tested on CUDA 8.0): if a kernel argument contains an array that is
// dynamically accessed, the whole array is first copied into the local memory.
// (That is, every kernel thread makes its own copy of the argument, even if it
// is never updated.)  Pre-computation makes it worse because now we have more
// data to copy.
//
// So let's fall back to vanilla division approach.

template <typename T, typename IndexType>
struct OffsetInfo<T, IndexType, -1> {
  OffsetInfo(const TensorInfo<T, IndexType>& tinfo, IndexType step)
    : tinfo(tinfo) { }

  TensorInfo<T, IndexType> tinfo;
};

template <typename T, typename IndexType>
struct OffsetIterator<T, IndexType, -1> {
  typedef OffsetInfo<T, IndexType, -1> Info;

  __host__ __device__
  OffsetIterator(const Info &info,
                 const LinearIdIterator<IndexType> &linear) { }

  // Return the pointer to the current element.
  __host__ __device__ T* get(const Info &info,
                             const LinearIdIterator<IndexType> &linear) {
    const TensorInfo<T, IndexType> &tinfo = info.tinfo;
    IndexType index = linear.index;
    IndexType offset = 0;

    for (int i = tinfo.dims - 1; i > 0; --i) {
      IndexType curDimIndex = index % tinfo.sizes[i];
      index /= tinfo.sizes[i];
      offset += curDimIndex * tinfo.strides[i];
    }

    offset += index * tinfo.strides[0];

    return &tinfo.data[offset];
  }

  __host__ __device__ void increment(const Info &info) { }
};

#endif // THC_OFFSET_INFO_INC
