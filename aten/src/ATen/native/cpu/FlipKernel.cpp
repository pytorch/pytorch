#include <ATen/native/TensorTransformations.h>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/AtomicAddFloat.h>
#include <cmath>
#include <iostream>

namespace at {
namespace native {
namespace {

struct Indexer {
  Indexer(int64_t num_indexers, char** indexers, const int64_t* indexer_strides)
      : num_indexers(num_indexers),
        indexers(indexers),
        indexer_strides(indexer_strides) {}

  int64_t num_indexers;
  char** indexers;
  const int64_t* indexer_strides;

  int64_t get(int64_t idx) {
    int64_t offset = *(int64_t*)&indexers[0][idx * indexer_strides[0]];
    for (int j = 1; j < num_indexers; j++) {
      offset += *(int64_t*)&indexers[j][idx * indexer_strides[j]];
    }
    return offset;
  }
};

template <typename scalar_t>
void flip_cpu_kernel(TensorIterator& iter) {
  int ntensor = iter.ntensors();
  // When launch the index parallel version, set a relative samll grain size
  // less than the INTERNAL::GRAIN_SIZE to make the whole available thread
  // numbers get more balanced work load and a better cache location. The grain
  // size here is chosen by the op benchmark to overcome the thread launch
  // overhead This value was taken from the AdvancedIndexing kernel.
  const int index_parallel_grain_size = 3000;
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    auto indexer = Indexer(ntensor - 2, &data[2], &strides[2]);
    char* dst = data[0];
    char* src = data[1];

    for (int64_t i = 0; i < n; i++) {
      int64_t offset = indexer.get(i);
      *(scalar_t*)(dst + strides[0] * i) =
          *(scalar_t*)(src + strides[1] * i + offset);
    }
  };

  iter.for_each(loop, index_parallel_grain_size);
}

void flip_kernel(TensorIterator& iter, const Tensor& input) {
  if (input.is_quantized()) {
    AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(
        input.scalar_type(), "flip_quantized_cpu", [&] {
          flip_cpu_kernel<scalar_t>(iter);
        });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        kBool, kHalf, kBFloat16, input.scalar_type(), "flip_cpu", [&] {
          flip_cpu_kernel<scalar_t>(iter);
        });
  }
}

} // Anonymous namespace

REGISTER_DISPATCH(flip_stub, &flip_cpu_kernel);

} // namespace native
} // namespace at