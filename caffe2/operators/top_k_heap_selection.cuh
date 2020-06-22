#ifndef CAFFE2_OPERATORS_TOP_K_HEAP_SELECTION_H_
#define CAFFE2_OPERATORS_TOP_K_HEAP_SELECTION_H_

#include "caffe2/utils/GpuBitonicSort.cuh"
#include "caffe2/utils/GpuDefs.cuh"
#include "caffe2/utils/math.h"
#include <cuda_runtime.h>

namespace caffe2 {

template <typename K, typename V>
struct LTComp {
  __device__ inline bool
  operator()(const K& kA, const V& vA, const K& kB, const V& vB) const {
    // FIXME: adding value comparison is slow
    return (kA < kB) || ((kA == kB) && (vA < vB));
  }
};

template <typename K, typename V>
struct GTComp {
  __device__ inline bool
  operator()(const K& kA, const V& vA, const K& kB, const V& vB) const {
    // FIXME: adding value comparison is slow
    // FIXME: it's vA < vB because the sorting order for V (aka
    // indices) is different in our use case
    return (kA > kB) || ((kA == kB) && (vA < vB));
  }
};

constexpr size_t getHeapSmemSize(
    size_t keySize,
    size_t valueSize,
    int numThreads,
    int heapSize) {
  return (numThreads / kWarpSize) * heapSize * (keySize + valueSize);
}

// Per-warp heap structure in shared memory:
// [key_0, ..., key_(HeapSize-2)], [empty element] (warp 0)
// ...
// [key_0, ..., key_(HeapSize-2)], [empty element] (warp n-1)
// [value_0, ..., value_(HeapSize-2)], [empty element] (warp 0)
// ...
// [value_0, ..., value_(HeapSize-2)], [empty element] (warp n-1)

// Dir == true means we are selecting the largest values, thus
// the heap is a min-heap
template <typename K, typename V, int HeapSize, bool Dir>
__device__ inline void warpHeapInsert(K k, V v, K* keyHeap, V* valueHeap) {
  // Replace head if we are < head
  bool valid = Dir ? (k > keyHeap[0]) : (k < keyHeap[0]);

  // Even though this is the single-thread case, another preceding
  // thread in the warp may have inserted in a new element that
  // supersedes our element and thus our attempt at an insert would do
  // nothing.
  if (!valid) {
    return;
  }

  // Swap with head if valid
  K currentKey = k;
  V currentValue = v;

  keyHeap[0] = currentKey;
  valueHeap[0] = currentValue;

  // The number of interior nodes in the heap is log2(HeapSize / 2):
  // heap size 8 means there are 7 elements in the heap, indices 0-6
  // (0 12 3456)
  // log2(8 / 2) = 2 levels of interior nodes for heap size 8 (0 and 12)
  int i = 0;
#ifndef __HIP_PLATFORM_HCC__
#pragma unroll
#endif
  for (int levels = 0; levels < math::IntegerLog2(HeapSize / 2); ++levels) {
    int leftChild = i * 2 + 1;
    int rightChild = leftChild + 1;
    K leftKey = keyHeap[leftChild];
    K rightKey = keyHeap[rightChild];

    // What child might we want to swap with (max heap = larger child;
    // min heap = smaller child)
    bool swap = Dir ? (leftKey < rightKey) : (leftKey > rightKey);
    int childToSwap = swap ? leftChild : rightChild;
    K keyChildToSwap = swap ? leftKey : rightKey;

    // If we're bigger than both children (max heap), or smaller than
    // both children (min heap), then we do nothing for the rest of
    // the iterations
    valid =
        Dir ? !(currentKey < keyChildToSwap) : !(currentKey > keyChildToSwap);

    // Swap with childToSwap if still valid
    keyHeap[i] = valid ? keyChildToSwap : currentKey;
    valueHeap[i] = valid ? valueHeap[childToSwap] : currentValue;

    keyHeap[childToSwap] = valid ? currentKey : keyChildToSwap;
    valueHeap[childToSwap] = valid ? currentValue : valueHeap[childToSwap];

    i = childToSwap;

    // This is our new element to potentially downheap
    currentKey = keyHeap[i];
    currentValue = valueHeap[i];
  }
}

template <typename K, typename V, int HeapSize, bool Dir>
__device__ inline void
warpHeap(K k, V v, K& keyHeapHead, K* keyHeap, V* valueHeap) {
  // All threads in the warp have elements
  bool wantInsert = Dir ? (k > keyHeapHead) : (k < keyHeapHead);

  // Find out all the lanes that have elements to add to the heap
#if defined(__HIP_PLATFORM_HCC__)
  unsigned long long int vote = __ballot(wantInsert);

  if (!vote) {
    // Everything the warp has is smaller than our heap
    return;
  }

  // Otherwise, we want to serialize execution of the threads
  // that have elements
  int index = __popcll(getLaneMaskLt() & vote);
  int total = __popcll(vote);
#else
  unsigned int vote = __ballot_sync(__activemask(), wantInsert);

  if (!vote) {
    // Everything the warp has is smaller than our heap
    return;
  }

  // Otherwise, we want to serialize execution of the threads
  // that have elements
  int index = __popc(getLaneMaskLt() & vote);
  int total = __popc(vote);
#endif  // __HIP_PLATFORM_HCC__

  // FIXME: try switch statement and explicitly handle cases
  // FIXME: how do cases work?
  for (int i = 0; i < total; ++i) {
    if (index == i && wantInsert) {
      // Insert into our heap
      warpHeapInsert<K, V, HeapSize, Dir>(k, v, keyHeap, valueHeap);

      // Make sure all smem writes are visible
      __threadfence_block();
    }
  }

  // Re-broadcast the new heap head
  // FIXME: consider each updater above will broadcast its value with
  // a shuffle instead?
  keyHeapHead = keyHeap[0];
}

template <typename K, typename V, int ThreadsPerBlock, int HeapSize, bool Dir>
class Heap {
 public:
  static constexpr size_t getSmemSize() {
    return getHeapSmemSize(sizeof(K), sizeof(V), ThreadsPerBlock, HeapSize);
  }

  __device__ Heap(void* smem, K initKey, V initVal) {
    heapBase = smem;

    int warpId = threadIdx.x / kWarpSize;
    int laneId = getLaneId();

    auto kStart = getKeyStart();
    heapK = &kStart[warpId * HeapSize];
    auto vStart = getValueStart();
    heapV = &vStart[warpId * HeapSize];

    heapHead = initKey;

    if (HeapSize < kWarpSize) {
      if (laneId < HeapSize) {
        heapK[laneId] = initKey;
        heapV[laneId] = initVal;
      }
    } else {
#pragma unroll
      for (int i = 0; i < HeapSize / kWarpSize; ++i) {
        heapK[laneId + i * kWarpSize] = initKey;
        heapV[laneId + i * kWarpSize] = initVal;
      }
    }
  }

  // Returns a pointer to the start of our block-wide key storage
  inline __device__ K* getKeyStart() {
    return (K*)heapBase;
  }

  // Returns a pointer to the start of our block-wide value storage
  inline __device__ V* getValueStart() {
    constexpr int warpsPerBlock = ThreadsPerBlock / kWarpSize;
    return (V*)&getKeyStart()[warpsPerBlock * HeapSize];
  }

  // Returns a pointer past the end of our block-wide heap storage
  inline __device__ void* getStorageEnd() {
    constexpr int warpsPerBlock = ThreadsPerBlock / kWarpSize;
    return getValueStart() + (warpsPerBlock * HeapSize);
  }

  inline __device__ void add(K k, V v) {
    warpHeap<K, V, HeapSize, Dir>(k, v, heapHead, heapK, heapV);
  }

  // Reduce all per-warp heaps to a unified, sorted list
  inline __device__ void reduceHeaps() {
    constexpr int allHeapSize = (ThreadsPerBlock / kWarpSize) * HeapSize;

    if (Dir) {
      bitonicSort<GTComp<K, V>, K, V, allHeapSize, ThreadsPerBlock>(
          getKeyStart(), getValueStart(), GTComp<K, V>());
    } else {
      bitonicSort<LTComp<K, V>, K, V, allHeapSize, ThreadsPerBlock>(
          getKeyStart(), getValueStart(), LTComp<K, V>());
    }
  }

 private:
  void* heapBase;
  K heapHead;
  K* heapK;
  V* heapV;
};

template <
    typename V,
    typename IndexType,
    typename OutIndexType,
    int ThreadsPerBlock,
    int HeapSize,
    bool Dir>
__global__ void selectRowsViaHeap(
    const V* input, // m x n
    V* outKeys, // m x k
    OutIndexType* outIndices, // m x k
    V initVal,
    IndexType initIndex,
    int m,
    int n,
    int k) {
  extern __shared__ float smem[];

  Heap<V, IndexType, ThreadsPerBlock, HeapSize, Dir> heap(
      smem, initVal, initIndex);

  auto inputStart = &input[blockIdx.x * n];

  // FIXME choose desired unroll level
  constexpr int Unroll = 1;
  V vals[Unroll];

  for (int i = threadIdx.x; i < n; i += blockDim.x * Unroll) {
#ifndef __HIP_PLATFORM_HCC__
#pragma unroll
#endif
    for (int j = 0; j < Unroll; ++j) {
      vals[j] = inputStart[i + j * blockDim.x];
    }

#ifndef __HIP_PLATFORM_HCC__
#pragma unroll
#endif
    for (int j = 0; j < Unroll; ++j) {
      heap.add(vals[j], (IndexType)i + j * blockDim.x);
    }
  }

  // When finished, we restructure the heaps in shared memory
  // The heaps are actually of size HeapSize - 1 (e.g., 32 -> 31); the
  // extra element should have remained untouched, so we can still
  // sort things in-place as a power of 2.
  __syncthreads();

  heap.reduceHeaps();

  auto outKeysStart = &outKeys[blockIdx.x * k];
  auto outIndicesStart = &outIndices[blockIdx.x * k];

  auto heapK = heap.getKeyStart();
  auto heapV = heap.getValueStart();

  // Write out the final k-selected values; they should be all
  // together
  for (int i = threadIdx.x; i < n && i < k; i += blockDim.x) {
    outKeysStart[i] = heapK[i];
    outIndicesStart[i] = (OutIndexType)heapV[i];
  }
}

} // namespace caffe2

#endif // CAFFE2_OPERATORS_TOP_K_HEAP_SELECTION_H_
