#include <stdint.h>
#include <memory>
#include <vector>

#include "THC/THCOffsetInfo.cuh"
#include "THC/THCTensorInfo.cuh"

#include "test_assert.h"

using std::vector;

int *dummyData_;

const int MAX_STEPS = 20;

template<typename IndexType, int Dims>
static void testOffset(const TensorInfo<int, IndexType>& tinfo,
                       IndexType start, IndexType step)
{
  uint64_t totalElements = 1;
  uint64_t maxOffset = 0;
  for (int i = 0; i < tinfo.dims; i++) {
    totalElements *= tinfo.sizes[i];
    maxOffset += (tinfo.sizes[i] - 1) * tinfo.strides[i];
  }

  // Bail out if the tensor does not fit in IndexType.
  if (sizeof(IndexType) == 4 &&
      (totalElements >= UINT32_MAX || maxOffset >= UINT32_MAX))
    return;

  OffsetInfo<int, IndexType, Dims> offset(tinfo, step);
  LinearIdIterator<IndexType> linear(start, step, totalElements);
  OffsetIterator<int, IndexType, Dims> iter(offset, linear);

  // For verification.
  vector<uint64_t> expectedIndices(tinfo.dims);

  // Use 64-bit linear index to verify overflow handling.
  int steps = 0;
  for (uint64_t linearId = start; linearId < totalElements; linearId += step) {
    ASSERT(linear.hasNext);
    ASSERT(linear.index == linearId);

    // Let's not repeat too many times.
    if (++steps >= MAX_STEPS) return;

    // Build and verify offset.
    IndexType index = linearId;
    IndexType expectedOffset = 0;
    for (int i = tinfo.dims - 1; i >= 0; --i) {
      IndexType curDimIndex = index % tinfo.sizes[i];
      index /= tinfo.sizes[i];
      expectedOffset += curDimIndex * tinfo.strides[i];
    }

    // Strictly speaking, C++ standard requires that tinfo.data[] be a valid
    // array with at least 'expectedOffset' elements; otherwise the pointer is
    // undefined.  However, in practice, I think this code should work with
    // dummy pointers.
    ASSERT(&tinfo.data[expectedOffset] == iter.get(offset, linear));

    linear.increment();
    iter.increment(offset);
  }

  ASSERT(!linear.hasNext);
}

// Helper function for constructing TensorInfo: the input vector 'data' is split
// into sizes (first half) and strides (second half).
template<typename IndexType>
static TensorInfo<int, IndexType> BuildTI(const vector<IndexType>& data)
{
  ASSERT(data.size() % 2 == 0);
  int dims = data.size() / 2;
  IndexType *sizes = const_cast<IndexType *>(&data[0]);
  IndexType *strides = const_cast<IndexType *>(&data[dims]);
  return TensorInfo<int, IndexType>(dummyData_, dims, sizes, strides);
}

template<typename IndexType>
static void runTest()
{
  // Testing the example shown in the comments of THCOffsetInfo.cuh.
  {
    TensorInfo<int, IndexType> tinfo =
        BuildTI(vector<IndexType>{ 5, 70, 10, 2000, 20, 1 });
    OffsetInfo<int, IndexType, 3> offset(tinfo, 1024);
    LinearIdIterator<IndexType> linear(1205, 1024, 5 * 70 * 10);
    OffsetIterator<int, IndexType, 3> iter(offset, linear);

    assert(iter.offset == 3005);
    // indices[0] is unused.
    assert(iter.indices[1] == 50);
    assert(iter.indices[2] == 5);

    linear.increment();
    iter.increment(offset);

    assert(iter.offset == 6249);
    // indices[0] is unused.
    assert(iter.indices[1] == 12);
    assert(iter.indices[2] == 9);
  }

  printf("Testing contiguous tensors.\n");
  for (unsigned size = 1; size < 1000; size++) {
    TensorInfo<int, IndexType> tinfo =
      BuildTI(vector<IndexType>{ size, 1 });

    for (int start = 0; start <= size + 100; start++) {
      for (int step : { 1, 2, 5, 10, 20, 50, 100, 1000, 10000 }) {
        testOffset<IndexType, -2>(tinfo, start, step);
      }
    }
  }

  printf("Testing 1-dimensional tensors.\n");
  for (unsigned sz1 : { 1, 2, 5, 10, 100, 1000, 100000 }) {
    for (unsigned st1 : { 0, 1, 2, 3, 5, 10, 20, 50, 100, 1000, 100000 }) {
      TensorInfo<int, IndexType> tinfo =
        BuildTI(vector<IndexType>{ sz1, st1 });

      for (int start = 0; start <= sz1 + 100; start *= 1.1, start++) {
        for (int step = 1; step <= sz1 + 100; step *= 1.5, step++) {
          testOffset<IndexType, 1>(tinfo, start, step);
          testOffset<IndexType, -1>(tinfo, start, step);
        }
      }
    }
  }

  printf("Testing 2-dimensional tensors.\n");
  for (unsigned sz1 : { 1, 2, 5, 10, 100, 1000 }) {
    for (unsigned st1 : { 0, 1, 2, 3, 5, 10, 20, 50, 100 }) {
      for (unsigned sz2 : { 1, 2, 5, 10, 100, 1000 }) {
        for (unsigned st2 : { 0, 1, 2, 3, 5, 10, 20, 50, 100 }) {
          TensorInfo<int, IndexType> tinfo =
            BuildTI(vector<IndexType>{ sz1, sz2, st1, st2 });

          unsigned sz = sz1 * sz2;

          for (int step = 1; step <= sz + 100; step *= 1.2, step++) {
            for (int start = 0; start <= sz + 100; start *= 1.5, start++) {
              testOffset<IndexType, 2>(tinfo, start, step);
              testOffset<IndexType, -1>(tinfo, start, step);
            }

            // Test starting almost at the end of the tensor.
            for (int start = sz - step * 20;
                 start <= sz + 100; start += step * 1.5) {
              testOffset<IndexType, 2>(tinfo, start, step);
              testOffset<IndexType, -1>(tinfo, start, step);
            }
          }
        }
      }
    }
  }

  printf("Testing 3-dimensional tensors.\n");
  for (unsigned sz1 : { 1, 5, 10, 100, 1000 }) {
    for (unsigned st1 : { 0, 1, 2, 3, 10, 100 }) {
      for (unsigned sz2 : { 1, 3, 5, 7, 200 }) {
        for (unsigned st2 : { 0, 1, 2, 3, 10, 100 }) {
          for (unsigned sz3 : { 1, 5, 10, 100, 1000 }) {
            for (unsigned st3 : { 0, 1, 2, 3, 10, 100 }) {
              TensorInfo<int, IndexType> tinfo =
                BuildTI(vector<IndexType>{ sz1, sz2, sz3, st1, st2, st3 });

              unsigned sz = sz1 * sz2 * sz3;

              for (int start = 0; start <= sz + 100; start *= 1.5, start++) {
                for (int step = 1; step <= sz + 100; step *= 1.5, step++) {
                  testOffset<IndexType, 3>(tinfo, start, step);
                  testOffset<IndexType, -1>(tinfo, start, step);
                }
              }
            }
          }
        }
      }
    }
  }

  printf("Testing near-overflow cases.\n");
  for (unsigned st1 = 1; st1 < 64; st1++) {
    unsigned sz1 = 0xffffffff / st1;
    TensorInfo<int, IndexType> tinfo = BuildTI(vector<IndexType>{ sz1, st1 });

    testOffset<IndexType, 1>(tinfo, sz1 - 100, 1);
    for (uint64_t step = sz1 - 5; step < sz1 + 5; step++) {
      if (step <= UINT32_MAX)
        testOffset<IndexType, 1>(tinfo, 0, step);
    }
  }

  for (unsigned sz1 = 1; sz1 <= 3; sz1++) {
    for (unsigned st2 = 1; st2 < 64; st2++) {
      unsigned sz2 = 0xffffffff / (sz1 * st2);
      TensorInfo<int, IndexType> tinfo =
        BuildTI(vector<IndexType>{ sz1, sz2, 1, st2 });

      unsigned sz = sz1 * sz2;

      testOffset<IndexType, 2>(tinfo, sz - 100, 1);
      testOffset<IndexType, -1>(tinfo, sz - 100, 1);
      for (uint64_t step = sz - 5; step < sz + 5; step++) {
        if (step <= UINT32_MAX)
          testOffset<IndexType, 2>(tinfo, 0, step);
          testOffset<IndexType, -1>(tinfo, 0, step);
      }
    }
  }
}

int main()
{
  // Not actually used, but we do need a valid pointer.
  dummyData_ = new int[1000000];

  printf("=== IndexType = uint32_t\n");
  runTest<uint32_t>();
  printf("=== IndexType = uint64_t\n");
  runTest<uint64_t>();

  return 0;
}
