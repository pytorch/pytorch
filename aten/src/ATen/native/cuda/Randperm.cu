#include <ATen/ATen.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/cuda/cub.cuh>
#include <ATen/native/cuda/Randperm.cuh>

#include <limits>

namespace at {
namespace native {

// [Algorithm of randperm]
//
// randperm is implemented by sorting an arange tensor of size n with randomly
// generated keys. When random keys are different from each other, all different
// permutations have the same probability.
//
// However, there is a pitfall here:
// For better performance, these N random keys are generated independently,
// and there is no effort to make sure they are different at the time of generation.
// When two keys are identical, stable sorting algorithms will not permute these two keys.
// As a result, (0, 1) will appear more often than (1, 0).
//
// To overcome this pitfall we first carefully choose the number of bits in these keys,
// so that the probability of having duplicate keys is under a threshold. Let q be the
// threshold probability for having non-duplicate keys, then it can be proved that[1]
// the number of bits required is: ceil(log2(n - (6 n^2 + 1) / (12 log(q))))
//
// Then after sort, we lauch a separate kernel that additionally shuffles any islands
// of values whose keys matched. The algorithm of this kernel is as follows:
// Each thread reads its key and the keys of its neighbors to tell if it's part of an island.
// For each island, the first thread in the island sees a key match at index i+1 but not index i-1.
// This thread considers itself the "island leader". The island leader then reads more indices to
// the right to figure out how big the island is. Most likely, the island will be very small,
// just a few values. The island leader then rolls that many RNG, uses them to additionally
// shuffle values within the island using serial Fisher-Yates, and writes them out.
//
// Reference
// [1] https://osf.io/af2hy/

// The kernels are templated on an opaque, self-aligned type of the correct
// size to avoid redundant kernels for different types of the same size.
namespace {
template <int N> struct alignas(N) OpaqueType { char data[N]; };
}

Tensor& randperm_out_cuda(int64_t n, c10::optional<Generator> generator, Tensor& result) {
  TORCH_CHECK(n >= 0, "n must be non-negative, got", n);
  TORCH_CHECK(n <= std::numeric_limits<int>::max(),
    "randperm of tensors larger than INT_MAX is not supported yet in pytorch");

  check_supported_max_int_with_precision(n, result);

  result.resize_({n});

  auto range = at::arange(n, result.options());

  // shuffled_data points to the underlying data of the output tensor if the tensor is contiguous; otherwise it
  // points to a new tensor.
  Tensor shuffled;
  void *shuffled_data;
  if (result.is_contiguous()) {
    shuffled_data = result.data_ptr();
  } else {
    shuffled = at::empty(n, result.options());
    shuffled_data = shuffled.data_ptr();
  }

  auto opt = TensorOptions().device(result.device());

  // See note [Algorithm of randperm]
  const double log_threshold_12 = std::log(0.9) * 12;
  double nd = static_cast<double>(n);

  int bits = std::min(64,
    static_cast<int>(std::ceil(std::log2(nd - (6 * nd * nd + 1) / log_threshold_12))));

  if (n == 0) {
    return result;
  } else if (bits <= 32) {
    // For asserting device type match of the generator and result,
    // we deligate that to the 'random_' function below.

    auto keys = at::empty(result.sizes(), opt.dtype(kInt)).random_(
      std::numeric_limits<int>::min(), std::numeric_limits<int>::max(), generator);
    auto keys_tmp = at::empty_like(keys);
    auto keys_out = keys_tmp.data_ptr<int>();
    AT_DISPATCH_ALL_TYPES_AND(kHalf, result.scalar_type(), "randperm_out_cuda", [&] {
      using dtype = OpaqueType<sizeof(scalar_t)>;
      auto shuffled_data_ = reinterpret_cast<dtype*>(shuffled_data);
      dtype* range_data = reinterpret_cast<dtype*>(range.data_ptr());
      at::cuda::cub::sort_pairs<int, dtype>(
        keys.data_ptr<int>(), keys_out,
        range_data, shuffled_data_,
        n, false, 0, bits);

      randperm_handle_duplicate_keys(keys_out, shuffled_data_, bits, n, generator);
    });
  } else {
    auto keys = at::empty(result.sizes(), opt.dtype(kLong)).random_(
      std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max(), generator);
    auto keys_tmp = at::empty_like(keys);
    auto keys_out = keys_tmp.data_ptr<int64_t>();
    AT_DISPATCH_ALL_TYPES_AND(kHalf, result.scalar_type(), "randperm_out_cuda", [&] {
      using dtype = OpaqueType<sizeof(scalar_t)>;
      auto shuffled_data_ = reinterpret_cast<dtype*>(shuffled_data);
      dtype* range_data = reinterpret_cast<dtype*>(range.data_ptr());
      at::cuda::cub::sort_pairs<int64_t, dtype>(
        keys.data_ptr<int64_t>(), keys_out,
        range_data, shuffled_data_,
        n, false, 0, bits);

      randperm_handle_duplicate_keys(keys_out, shuffled_data_, bits, n, generator);
    });
  }

  if (!result.is_contiguous()) {
    result.copy_(shuffled);
  }

  return result;
}

}} // namespace at::native
