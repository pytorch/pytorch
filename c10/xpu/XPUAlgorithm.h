#ifdef ONEDPL_DEVICE_LOWER_BOUND_WORKS
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/functional>
#include <oneapi/dpl/execution>
#endif
using namespace oneapi;
using namespace sycl;
namespace c10::xpu {
#ifdef ONEDPL_DEVICE_LOWER_BOUND_WORKS
template <typename Iter, typename Scalar>
Iter __attribute__((always_inline))  
lower_bound(Iter start, Iter end, Scalar value) {
  return dpl::lower_bound(start, end, value, dpl::greater<int>());
}
#else
// thrust::lower_bound is broken on device, see
// https://github.com/NVIDIA/thrust/issues/1734 Implementation inspired by
// https://github.com/pytorch/pytorch/blob/805120ab572efef66425c9f595d9c6c464383336/aten/src/ATen/native/cuda/Bucketization.cu#L28
template <typename Iter, typename Scalar>
__attribute__((always_inline)) Iter lower_bound(Iter start, Iter end, Scalar value) {
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
#endif // ONEDPL_DEVICE_LOWER_BOUND_WORKS
} // namespace c10::xpu