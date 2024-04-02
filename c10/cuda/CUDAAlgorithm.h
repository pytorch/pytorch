#ifdef THRUST_DEVICE_LOWER_BOUND_WORKS
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#endif
namespace c10 {
namespace cuda {
#ifdef THRUST_DEVICE_LOWER_BOUND_WORKS
template <typename Iter, typename Scalar>
__forceinline__ __device__ Iter
lower_bound(Iter start, Iter end, Scalar value) {
  return thrust::lower_bound(thrust::device, start, end, value);
}
#else
// thrust::lower_bound is broken on device, see
// https://github.com/NVIDIA/thrust/issues/1734 Implementation inspired by
// https://github.com/pytorch/pytorch/blob/805120ab572efef66425c9f595d9c6c464383336/aten/src/ATen/native/cuda/Bucketization.cu#L28
template <typename Iter, typename Scalar>
__device__ Iter lower_bound(Iter start, Iter end, Scalar value) {
  while (start < end) {
    auto mid = start + ((end - start) >> 1);
    if (*mid < value) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }
  return end;
}
#endif // THRUST_DEVICE_LOWER_BOUND_WORKS
} // namespace cuda
} // namespace c10
