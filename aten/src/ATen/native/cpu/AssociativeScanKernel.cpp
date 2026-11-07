#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/TensorBase.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/AssociativeScanKernel.h>
#include <ATen/native/AssociativeScanUtils.h>

#include <c10/util/irange.h>
#include <c10/util/Load.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace at::native {

namespace {

constexpr int64_t kChunkSize = 4096;

// 2-pass work-efficient parallel prefix scan over the innermost (contiguous)
// dimension of a tensor viewed as [M, N]:
//   1. upsweep:   reduce each block of kChunkSize elements into a partial
//   2. prefix:    inclusive-scan the block partials along the scan dim
//   3. downsweep: inclusive-scan each block, seeded with the prefix of the
//                 preceding blocks.
// `self`/`result` hold L parallel arrays of N x M contiguous elements.
template <typename scalar_t, int L, typename Combine>
void scan_impl(
    const scalar_t* const* self,
    scalar_t* const* result,
    int64_t N,
    int64_t M) {
  const int64_t num_chunks = (N + kChunkSize - 1) / kChunkSize;

  if (num_chunks == 1) {
    // Short scan: a single sequential pass per row, parallelized over rows.
    at::parallel_for(0, M, internal::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
      for (const auto m : c10::irange(begin, end)) {
        auto acc = Combine::identity();
        for (const auto i : c10::irange(N)) {
          ScanVec<scalar_t, L> v;
          for (int l = 0; l < L; ++l) {
            v.v[l] = c10::load(&self[l][m * N + i]);
          }
          acc = Combine::combine(acc, v);
          for (int l = 0; l < L; ++l) {
            result[l][m * N + i] = acc.v[l];
          }
        }
      }
    });
    return;
  }

  // Block partial reductions (upsweep): partials[num_chunks][M] elements.
  std::vector<scalar_t> partials(num_chunks * M * L);
  at::parallel_for(
      0, num_chunks * M, internal::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (const auto idx : c10::irange(begin, end)) {
          const int64_t c = idx / M;
          const int64_t m = idx % M;
          const int64_t lo = c * kChunkSize;
          const int64_t hi = std::min(lo + kChunkSize, N);
          auto acc = Combine::identity();
          for (const auto i : c10::irange(lo, hi)) {
            ScanVec<scalar_t, L> v;
            for (int l = 0; l < L; ++l) {
              v.v[l] = c10::load(&self[l][m * N + i]);
            }
            acc = Combine::combine(acc, v);
          }
          scalar_t* p = partials.data() + (c * M + m) * L;
          for (int l = 0; l < L; ++l) {
            p[l] = acc.v[l];
          }
        }
      });

  // Inclusive scan of the block partials (few chunks -> sequential per row).
  std::vector<scalar_t> prefix(num_chunks * M * L);
  at::parallel_for(0, M, internal::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
    for (const auto m : c10::irange(begin, end)) {
      auto acc = Combine::identity();
      for (const auto c : c10::irange(num_chunks)) {
        ScanVec<scalar_t, L> v;
        for (int l = 0; l < L; ++l) {
          v.v[l] = partials[(c * M + m) * L + l];
        }
        acc = Combine::combine(acc, v);
        for (int l = 0; l < L; ++l) {
          prefix[(c * M + m) * L + l] = acc.v[l];
        }
      }
    }
  });

  // Downsweep: scan each block seeded with the prefix of all preceding blocks.
  at::parallel_for(
      0, num_chunks * M, internal::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (const auto idx : c10::irange(begin, end)) {
          const int64_t c = idx / M;
          const int64_t m = idx % M;
          const int64_t lo = c * kChunkSize;
          const int64_t hi = std::min(lo + kChunkSize, N);
          auto acc = Combine::identity();
          if (c > 0) {
            for (int l = 0; l < L; ++l) {
              acc.v[l] = prefix[((c - 1) * M + m) * L + l];
            }
          }
          for (const auto i : c10::irange(lo, hi)) {
            ScanVec<scalar_t, L> v;
            for (int l = 0; l < L; ++l) {
              v.v[l] = c10::load(&self[l][m * N + i]);
            }
            acc = Combine::combine(acc, v);
            for (int l = 0; l < L; ++l) {
              result[l][m * N + i] = acc.v[l];
            }
          }
        }
      });
}

template <typename scalar_t>
void dispatch_combine(
    const TensorBase& result,
    const TensorBase& self,
    const std::string& combine_mode) {
  const int64_t N = self.size(-1);
  const int64_t M = self.numel() / N;
  const scalar_t* self_ptr = self.const_data_ptr<scalar_t>();
  scalar_t* result_ptr = result.mutable_data_ptr<scalar_t>();

  if (combine_mode == "add") {
    const scalar_t* ptrs[1] = {self_ptr};
    scalar_t* rptrs[1] = {result_ptr};
    scan_impl<scalar_t, 1, CombineAdd<scalar_t, 1>>(ptrs, rptrs, N, M);
  } else if (combine_mode == "mul") {
    const scalar_t* ptrs[1] = {self_ptr};
    scalar_t* rptrs[1] = {result_ptr};
    scan_impl<scalar_t, 1, CombineMul<scalar_t, 1>>(ptrs, rptrs, N, M);
  } else if (combine_mode == "max") {
    const scalar_t* ptrs[1] = {self_ptr};
    scalar_t* rptrs[1] = {result_ptr};
    scan_impl<scalar_t, 1, CombineMax<scalar_t>>(ptrs, rptrs, N, M);
  } else if (combine_mode == "min") {
    const scalar_t* ptrs[1] = {self_ptr};
    scalar_t* rptrs[1] = {result_ptr};
    scan_impl<scalar_t, 1, CombineMin<scalar_t>>(ptrs, rptrs, N, M);
  } else {
    TORCH_INTERNAL_ASSERT(false, "unsupported combine_mode: ", combine_mode);
  }
}

void associative_scan_cpu_kernel(
    const TensorBase& result,
    const TensorBase& self,
    const std::string& combine_mode) {
  if (self.numel() == 0) {
    return;
  }
  if (combine_mode == "add" || combine_mode == "mul") {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
        kHalf,
        kBFloat16,
        self.scalar_type(),
        "associative_scan_cpu",
        [&] { dispatch_combine<scalar_t>(result, self, combine_mode); });
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(
        kHalf,
        kBFloat16,
        self.scalar_type(),
        "associative_scan_cpu",
        [&] { dispatch_combine<scalar_t>(result, self, combine_mode); });
  }
}

void associative_scan_tensor_list_cpu_kernel(
    const std::vector<TensorBase>& result,
    const std::vector<TensorBase>& self,
    const std::string& combine_mode) {
  if (self.empty() || self[0].numel() == 0) {
    return;
  }
  const int64_t N = self[0].size(-1);
  const int64_t M = self[0].numel() / N;
  AT_DISPATCH_ALL_TYPES_AND2(
      kHalf,
      kBFloat16,
      self[0].scalar_type(),
      "associative_scan_tensor_list_cpu",
      [&] {
        const scalar_t* in_ptrs[2] = {
            self[0].const_data_ptr<scalar_t>(),
            self[1].const_data_ptr<scalar_t>(),
        };
        scalar_t* out_ptrs[2] = {
            result[0].mutable_data_ptr<scalar_t>(),
            result[1].mutable_data_ptr<scalar_t>(),
        };
        scan_impl<scalar_t, 2, CombineLinearRecurrence<scalar_t>>(
            in_ptrs, out_ptrs, N, M);
      });
}

} // namespace

REGISTER_DISPATCH(associative_scan_stub, &associative_scan_cpu_kernel)
REGISTER_DISPATCH(
    associative_scan_tensor_list_stub,
    &associative_scan_tensor_list_cpu_kernel)

} // namespace at::native
