#pragma once

#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <c10/util/irange.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>

namespace at::native {
inline namespace CPU_CAPABILITY {
template <typename scalar_t>
int64_t vec_log_softmax_lastdim_chunk_size(int64_t grain_size, int64_t outer_size, int64_t dim_size) {
  // Coincidentally, at::internal::GRAIN_SIZE is 32768, which is equal to the
  // size of L1D cache on many processors. Some processors have 48 KB L1D cache
  // nowadays, so maybe in the future, we can leverage the knowledge of a
  // machine's L1D cache size.
  int64_t MAX_CHUNK_SIZE = std::max<int64_t>(
      1,
      grain_size / (sizeof(scalar_t) * dim_size));
  return std::min<int64_t>(MAX_CHUNK_SIZE, outer_size);
}

template <typename scalar_t>
void serial_vec_log_softmax_lastdim_range(
    const scalar_t* input_data_base,
    scalar_t* output_data_base,
    int64_t dim_size,
    int64_t chunk_size,
    int64_t begin,
    int64_t end) {
  if (end <= begin) {
    return;
  }
  using Vec = vec::Vectorized<vec::vec_scalar_t<scalar_t>>;
  // MSVC requires such a declaration of dynamic arrays
  // Source: https://stackoverflow.com/a/33423538
  auto tmp_sum_scalar = std::make_unique<scalar_t[]>(chunk_size);
  auto max_input_arr = std::make_unique<scalar_t[]>(chunk_size);
  for (int64_t ii = begin; ii < end; ii += chunk_size) {
    int64_t loop_end = chunk_size;
    if (ii + chunk_size > end) {
      loop_end = end - ii;
    }
    for (const auto j : c10::irange(loop_end)) {
      int64_t i = ii + j;
      const scalar_t* input_data = input_data_base + i * dim_size;
      max_input_arr[j] = vec::reduce_all<scalar_t>(
          [](Vec& x, Vec& y) { return vec::maximum(x, y); },
          input_data,
          dim_size);
    }
    for (const auto j : c10::irange(loop_end)) {
      int64_t i = ii + j;
      const scalar_t* input_data = input_data_base + i * dim_size;
      scalar_t max_input = max_input_arr[j];
      tmp_sum_scalar[j] = vec::map_reduce_all<scalar_t>(
          [max_input](Vec x) { return (x - Vec(max_input)).exp(); },
          [](Vec x, Vec y) { return x + y; },
          input_data,
          dim_size);
    }
    // See [Note AVX-SSE transitions] for why this should call the
    // vectorized version (aside from perf improvements).
    vec::map(
        [](Vec x) { return x.log(); },
        tmp_sum_scalar.get(),
        tmp_sum_scalar.get(),
        loop_end);
    for (const auto j : c10::irange(loop_end)) {
      int64_t i = ii + j;
      const scalar_t* input_data = input_data_base + i * dim_size;
      scalar_t* output_data = output_data_base + i * dim_size;
      scalar_t tmp_sum = tmp_sum_scalar[j];
      scalar_t max_input = max_input_arr[j];

      // It's necessary to keep the order of the operations below.
      // In some cases that input is large digits and the difference
      // is small, if we compute `max_input` plus `tmp_sum` before,
      // there would be a numerical problem. See an example in
      // https://github.com/pytorch/pytorch/issues/11752#issuecomment-422883379
      vec::map(
          [tmp_sum, max_input](Vec x) {
            return x - Vec(max_input) - Vec(tmp_sum);
          },
          output_data,
          input_data,
          dim_size);
    }
  }
}

// Can't include ATen/Parallel.h.
// TODO: find a way to have only one copy of divup.
inline int64_t divup(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

template <typename scalar_t, int64_t BLOCK_SIZE = 128 * 1024>
std::pair<int64_t,int64_t> vec_logsoftmax_chunk_size_and_num_chunks(int64_t inner_size, int64_t dim_size) {
  using Vec = vec::Vectorized<scalar_t>;
  int64_t MAX_CHUNK_SIZE = std::max<int64_t>(BLOCK_SIZE / dim_size / sizeof(scalar_t), Vec::size());
  MAX_CHUNK_SIZE = MAX_CHUNK_SIZE / Vec::size() * Vec::size();
  int64_t CHUNK_SIZE = std::min<int64_t>(MAX_CHUNK_SIZE, inner_size);
  int64_t num_chunks = divup(inner_size, CHUNK_SIZE);
  return {CHUNK_SIZE, num_chunks};
}

template <typename scalar_t>
std::enable_if_t<std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
serial_vec_logsoftmax_range(
    const scalar_t* input_data_base,
    scalar_t* output_data_base,
    int64_t inner_size,
    int64_t chunk_size,
    int64_t num_chunks,
    int64_t dim_size,
    int64_t begin,
    int64_t end) {
  using Vec = vec::Vectorized<scalar_t>;
  // thread local temp buffer which holds vertical reduction result: max and sum.
  auto buffer = std::make_unique<scalar_t []>(chunk_size * 2);
  scalar_t* input_max_data = buffer.get();
  scalar_t* tmp_sum_data = buffer.get() + chunk_size;

  for (int64_t i = begin; i < end; i++) {
    int64_t outer_idx = i / num_chunks;
    int64_t k = i % num_chunks;
    int64_t inner_idx_begin = k * chunk_size;
    int64_t size = std::min(chunk_size, inner_size - inner_idx_begin);

    // init
    Vec zero_vec = Vec(scalar_t(0));
    Vec min_vec = Vec(-std::numeric_limits<scalar_t>::infinity());
    int64_t d0 = 0;
    for (; d0 < size - (size % Vec::size()); d0 += Vec::size()) {
      min_vec.store(input_max_data + d0);
      zero_vec.store(tmp_sum_data + d0);
    }
    for (; d0 < size; d0++) {
      input_max_data[d0] = -std::numeric_limits<scalar_t>::infinity();
      tmp_sum_data[d0] = scalar_t(0);
    }

    // compute max
    for (int64_t dim_idx = 0; dim_idx < dim_size; dim_idx++) {
      const scalar_t* input_ptr = input_data_base + outer_idx * dim_size * inner_size
          + dim_idx * inner_size + inner_idx_begin;

      int64_t d1 = 0;
      for (; d1 < size - (size % Vec::size()); d1 += Vec::size()) {
        Vec data_vec = Vec::loadu(input_ptr + d1);
        Vec max_vec = Vec::loadu(input_max_data + d1);
        max_vec = Vec::blendv(max_vec, data_vec, data_vec > max_vec);
        max_vec.store(input_max_data + d1);
      }
      for (; d1 < size; d1++) {
        scalar_t data_val = input_ptr[d1];
        scalar_t max_val = input_max_data[d1];
        input_max_data[d1] = data_val > max_val ? data_val : max_val;
      }
    }

    // compute sum of (x - max).exp()
    for (int64_t dim_idx = 0; dim_idx < dim_size; dim_idx++) {
      const scalar_t* input_ptr = input_data_base + outer_idx * dim_size * inner_size
          + dim_idx * inner_size + inner_idx_begin;

      int64_t d2 = 0;
      for (; d2 < size - (size % Vec::size()); d2 += Vec::size()) {
        Vec data_vec = Vec::loadu(input_ptr + d2);
        Vec sum_vec = Vec::loadu(tmp_sum_data + d2);
        Vec max_vec = Vec::loadu(input_max_data + d2);
        sum_vec += (data_vec - max_vec).exp();
        sum_vec.store(tmp_sum_data + d2);
      }
      for (; d2 < size; d2++) {
        scalar_t data_val = input_ptr[d2];
        scalar_t max_val = input_max_data[d2];
        tmp_sum_data[d2] += std::exp(data_val - max_val);
      }
    }

    // apply log
    vec::map([](Vec x) { return x.log(); }, tmp_sum_data, tmp_sum_data, size);

    // compute x - max - sum
    for (int64_t dim_idx = 0; dim_idx < dim_size; dim_idx++) {
      int64_t offset = outer_idx * dim_size * inner_size + dim_idx * inner_size + inner_idx_begin;
      const scalar_t* input_ptr = input_data_base + offset;
      scalar_t* output_ptr = output_data_base + offset;

      int64_t d3 = 0;
      for (; d3 < size - (size % Vec::size()); d3 += Vec::size()) {
        Vec data_vec = Vec::loadu(input_ptr + d3);
        Vec max_vec = Vec::loadu(input_max_data + d3);
        Vec sum_vec = Vec::loadu(tmp_sum_data + d3);
        Vec out_vec = data_vec - max_vec - sum_vec;
        out_vec.store(output_ptr + d3);
      }
      for (; d3 < size; d3++) {
        output_ptr[d3] = input_ptr[d3] - input_max_data[d3] - tmp_sum_data[d3];
      }
    }
  }
}

template <typename scalar_t>
std::enable_if_t<!std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
serial_vec_logsoftmax_range(
    const scalar_t* input_data_base,
    scalar_t* output_data_base,
    int64_t inner_size,
    int64_t chunk_size,
    int64_t num_chunks,
    int64_t dim_size,
    int64_t begin,
    int64_t end) {
  using Vec = vec::Vectorized<scalar_t>;
  using fVec = vec::Vectorized<float>;
  auto buffer = std::make_unique<float []>(chunk_size * 2);
  float* input_max_data = buffer.get();
  float* tmp_sum_data = buffer.get() + chunk_size;

  // thread local buffer that holds input data in float32 to save next 2 dtype conversion
  auto input_buffer = std::make_unique<float []>(dim_size * chunk_size);
  float* input_buffer_data = input_buffer.get();

  // init
  for (int64_t i = begin; i < end; i++) {
    int64_t outer_idx = i / num_chunks;
    int64_t k = i % num_chunks;
    int64_t inner_idx_begin = k * chunk_size;
    int64_t size = std::min(chunk_size, inner_size - inner_idx_begin);

    fVec zero_fvec = fVec(float(0));
    fVec min_fvec = fVec(-std::numeric_limits<float>::infinity());
    int64_t d0 = 0;
    for (; d0 < size - (size % Vec::size()); d0 += Vec::size()) {
      min_fvec.store(input_max_data + d0);
      min_fvec.store(input_max_data + d0 + fVec::size());
      zero_fvec.store(tmp_sum_data + d0);
      zero_fvec.store(tmp_sum_data + d0 + fVec::size());
    }
    for (; d0 < size; d0++) {
      input_max_data[d0] = -std::numeric_limits<float>::infinity();
      tmp_sum_data[d0] = float(0);
    }

    // compute max
    for (int64_t dim_idx = 0; dim_idx < dim_size; dim_idx++) {
      const scalar_t* input_ptr = input_data_base + outer_idx * dim_size * inner_size
          + dim_idx * inner_size + inner_idx_begin;
      float* input_buffer_ptr = input_buffer_data + dim_idx * chunk_size;

      int64_t d1 = 0;
      for (; d1 < size - (size % Vec::size()); d1 += Vec::size()) {
        Vec data_vec = Vec::loadu(input_ptr + d1);
        auto [data_fvec0, data_fvec1] = vec::convert_to_float<scalar_t>(data_vec);
        fVec max_fvec0 = fVec::loadu(input_max_data + d1);
        fVec max_fvec1 = fVec::loadu(input_max_data + d1 + fVec::size());
        max_fvec0 = fVec::blendv(max_fvec0, data_fvec0, data_fvec0 > max_fvec0);
        max_fvec1 = fVec::blendv(max_fvec1, data_fvec1, data_fvec1 > max_fvec1);
        max_fvec0.store(input_max_data + d1);
        max_fvec1.store(input_max_data + d1 + fVec::size());

        // cache the 'converted' float input
        data_fvec0.store(input_buffer_ptr + d1);
        data_fvec1.store(input_buffer_ptr + d1 + fVec::size());
      }
      for (; d1 < size; d1++) {
        float data_val = float(input_ptr[d1]);
        float max_val = input_max_data[d1];
        input_max_data[d1] = data_val > max_val ? data_val : max_val;
        input_buffer_ptr[d1] = data_val;
      }
    }

    // compute sum of (x - max).exp()
    for (int64_t dim_idx = 0; dim_idx < dim_size; dim_idx++) {
      float* input_buffer_ptr = input_buffer_data + dim_idx * chunk_size;

      int64_t d2 = 0;
      for (; d2 < size - (size % Vec::size()); d2 += Vec::size()) {
        fVec data_fvec0 = fVec::loadu(input_buffer_ptr + d2);
        fVec data_fvec1 = fVec::loadu(input_buffer_ptr + d2 + fVec::size());
        fVec sum_fvec0 = fVec::loadu(tmp_sum_data + d2);
        fVec sum_fvec1 = fVec::loadu(tmp_sum_data + d2 + fVec::size());
        fVec max_fvec0 = fVec::loadu(input_max_data + d2);
        fVec max_fvec1 = fVec::loadu(input_max_data + d2 + fVec::size());
        sum_fvec0 += (data_fvec0 - max_fvec0).exp();
        sum_fvec1 += (data_fvec1 - max_fvec1).exp();
        sum_fvec0.store(tmp_sum_data + d2);
        sum_fvec1.store(tmp_sum_data + d2 + fVec::size());
      }
      for (; d2 < size; d2++) {
        float data_val = input_buffer_ptr[d2];
        float max_val = input_max_data[d2];
        tmp_sum_data[d2] += std::exp(data_val - max_val);
      }
    }

    // apply log
    vec::map([](fVec x) { return x.log(); }, tmp_sum_data, tmp_sum_data, size);

    // compute x - max - sum
    for (int64_t dim_idx = 0; dim_idx < dim_size; dim_idx++) {
      float* input_buffer_ptr = input_buffer_data + dim_idx * chunk_size;
      scalar_t* output_ptr = output_data_base + outer_idx * dim_size * inner_size
          + dim_idx * inner_size + inner_idx_begin;

      int64_t d3 = 0;
      for (; d3 < size - (size % Vec::size()); d3 += Vec::size()) {
        fVec data_fvec0 = fVec::loadu(input_buffer_ptr + d3);
        fVec data_fvec1 = fVec::loadu(input_buffer_ptr + d3 + fVec::size());
        fVec max_fvec0 = fVec::loadu(input_max_data + d3);
        fVec max_fvec1 = fVec::loadu(input_max_data + d3 + fVec::size());
        fVec sum_fvec0 = fVec::loadu(tmp_sum_data + d3);
        fVec sum_fvec1 = fVec::loadu(tmp_sum_data + d3 + fVec::size());
        fVec out_fvec0 = data_fvec0 - max_fvec0 - sum_fvec0;
        fVec out_fvec1 = data_fvec1 - max_fvec1 - sum_fvec1;
        Vec out_vec = vec::convert_from_float<scalar_t>(out_fvec0, out_fvec1);
        out_vec.store(output_ptr + d3);
      }
      for (; d3 < size; d3++) {
        output_ptr[d3] = scalar_t(input_buffer_ptr[d3] - input_max_data[d3] - tmp_sum_data[d3]);
      }
    }
  }
} // namespace CPU_CAPABILITY
}} // namespace at::native
