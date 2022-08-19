#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>

#include <ATen/Dispatch.h>
#include <ATen/native/SpmmReduce.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <c10/util/irange.h>

namespace at { namespace native {

namespace {

template <typename scalar_t, bool has_optional_value>
void spmm_sum_kernel_impl(
    const Tensor& result,
    const Tensor& rowptr,
    const Tensor& col,
    const c10::optional<Tensor>& optional_value,
    const Tensor& mat) {

  scalar_t* result_data = result.data_ptr<scalar_t>();
  int64_t* rowptr_data = rowptr.data_ptr<int64_t>();
  int64_t* col_data = col.data_ptr<int64_t>();
  scalar_t* value_data = has_optional_value ? optional_value.value().data_ptr<scalar_t>() : nullptr;
  scalar_t* mat_data = mat.data_ptr<scalar_t>();

  int64_t M = rowptr.numel() - 1;
  int64_t N = mat.size(-2);
  int64_t K = mat.size(-1);
  int64_t B = mat.numel() / (N * K);

  // directly parallel on `B * M` may lead to load imbalance,
  // statically determine thread partition here to average payload
  // for each thread.
  int num_threads = at::get_num_threads();
  std::vector<int64_t> thread_splits(num_threads + 1, B * M);
  int64_t thread_averge_payload = (rowptr_data[M] - rowptr_data[0]) / num_threads;

  thread_splits[0] = 0;
  int64_t sum = 0;
  int64_t t = 1;
  for (const auto m : c10::irange(M)) {
    int64_t row_start = rowptr_data[m];
    int64_t row_end = rowptr_data[m + 1];
    sum += row_end - row_start;
    if (sum > t * thread_averge_payload) {
      thread_splits[t] = B * m;
      t++;
    }
  }
  // need to restore the last index,
  // due to rounding error when calculating `thread_averge_payload`.
  thread_splits[num_threads] = B * M;

  // TODO: add bfloat16 support here
  using Vec = vec::Vectorized<scalar_t>;
  at::parallel_for(0, num_threads, 1, [&](int64_t cbegin, int64_t cend) {
    int tid = at::get_thread_num();
    int64_t begin = thread_splits[tid];
    int64_t end = thread_splits[tid + 1];

    int64_t row_start, row_end, b, m, c;
    for (const auto i : c10::irange(begin, end)) {
      b = i / M;
      m = i % M;
      row_start = rowptr_data[m];
      row_end = rowptr_data[m + 1];

      scalar_t* result_ptr = result_data + i * K;

      constexpr int64_t kVecSize = Vec::size();
      constexpr int64_t kVLEN = kVecSize * 4;
      constexpr int64_t CHUNK_SIZE = 16;

      // init the output lane
      vec::map<scalar_t>([](Vec x) { return Vec(0); }, result_ptr, result_ptr, K);

      // blocking on rowwise to reduce write memory bandwidth
      for (int64_t e0 = row_start; e0 < row_end; e0 += CHUNK_SIZE) {
        int64_t e1 = std::min(e0 + CHUNK_SIZE, row_end);

        // unrolling by 4
        int64_t k = 0;
        for (; k < K - (K % kVLEN); k += kVLEN) {
          Vec out_vec0 = Vec::loadu(result_ptr + k);
          Vec out_vec1 = Vec::loadu(result_ptr + k + kVecSize);
          Vec out_vec2 = Vec::loadu(result_ptr + k + kVecSize * 2);
          Vec out_vec3 = Vec::loadu(result_ptr + k + kVecSize * 3);
          for (const auto e : c10::irange(e0, e1)) {
            c = col_data[e];
            scalar_t val = has_optional_value ? value_data[e] : scalar_t(1);
            scalar_t* mat_ptr = mat_data + b * N * K + c * K + k;

            out_vec0 += Vec::loadu(mat_ptr) * Vec(val);
            out_vec1 += Vec::loadu(mat_ptr + kVecSize) * Vec(val);
            out_vec2 += Vec::loadu(mat_ptr + kVecSize * 2) * Vec(val);
            out_vec3 += Vec::loadu(mat_ptr + kVecSize * 3) * Vec(val);
          }
          out_vec0.store(result_ptr + k);
          out_vec1.store(result_ptr + k + kVecSize);
          out_vec2.store(result_ptr + k + kVecSize * 2);
          out_vec3.store(result_ptr + k + kVecSize * 3);
        }
        for (; k < K - (K % Vec::size()); k += Vec::size()) {
          Vec out_vec = Vec::loadu(result_ptr + k);
          for (const auto e : c10::irange(e0, e1)) {
            c = col_data[e];
            scalar_t val = has_optional_value ? value_data[e] : scalar_t(1);
            scalar_t* mat_ptr = mat_data + b * N * K + c * K;
            out_vec += Vec::loadu(mat_ptr + k) * Vec(val);
          }
          out_vec.store(result_ptr + k);
        }
        for (; k < K; k++) {
          scalar_t out_val = result_ptr[k];
          for (const auto e : c10::irange(e0, e1)) {
            c = col_data[e];
            scalar_t val = has_optional_value ? value_data[e] : scalar_t(1);
            scalar_t* mat_ptr = mat_data + b * N * K + c * K;
            out_val += mat_ptr[k] * val;
          }
          result_ptr[k] = out_val;
        }
      }
    }
  });
}

void spmm_sum_kernel(
    const Tensor& result,
    const Tensor& rowptr,
    const Tensor& col,
    const c10::optional<Tensor>& optional_value,
    const Tensor& mat) {
  AT_DISPATCH_FLOATING_TYPES(result.scalar_type(), "spmm_sum_kernel", [&]() {
    if (optional_value.has_value()) {
      spmm_sum_kernel_impl<scalar_t, true>(result, rowptr, col, optional_value, mat);
    } else {
      spmm_sum_kernel_impl<scalar_t, false>(result, rowptr, col, optional_value, mat);
    }
  });
}

} // anonymous namespace

REGISTER_DISPATCH(spmm_sum_stub, &spmm_sum_kernel);

}} // at::native
