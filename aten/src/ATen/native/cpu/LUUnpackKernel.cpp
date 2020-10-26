#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/cpu/Loops.h>

namespace at {
namespace native {

namespace {

void unpack_pivots_cpu_kernel(
  TensorIterator& iter,
  int64_t dim_size
) {
  if (iter.numel() == 0) {
    return;
  }

  auto loop = [&](char** data, const int64_t* strides, int64_t nelems) {
    auto* unpacked_pivots_ptr = data[0];
    auto* pivots_ptr = data[1];

    for (int64_t elem = 0; elem < nelems; ++elem) {
      // WARNING: torch.lu returns int32 pivots,
      // this behavior could change in the future.
      auto* unpacked_pivots_data = reinterpret_cast<int32_t*>(unpacked_pivots_ptr);
      auto* pivots_data = reinterpret_cast<int32_t*>(pivots_ptr);

      for (int64_t i = 0; i < dim_size; ++i) {
        std::swap(
          unpacked_pivots_data[i],
          unpacked_pivots_data[pivots_data[i]]
        );
      }

      unpacked_pivots_ptr += strides[0];
      pivots_ptr += strides[1];
    }
  };

  iter.for_each(loop);
}

}

REGISTER_DISPATCH(unpack_pivots_stub, &unpack_pivots_cpu_kernel);

}} // namespace at::native
