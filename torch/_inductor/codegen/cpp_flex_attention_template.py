# mypy: allow-untyped-defs
import contextlib
import functools
import logging
import math
import os
import re
from unittest.mock import patch

import sympy

import torch
import torch.utils

from ...utils._ordered_set import OrderedSet
from .. import config, cpu_vec_isa, ir
from ..ir import TensorBox
from ..ops_handler import DefaultHandler
from ..select_algorithm import DataProcessorTemplateWrapper
from ..utils import parallel_num_threads
from ..virtualized import OpsValue, V
from .cpp_template import CppTemplate
from .cpp_utils import GemmBlocking


log = logging.getLogger(__name__)

# TODO: reuse cpp codegen to generate below pointwise/reduction kernels
SOFTMAX_FUSIONS = r"""
// 1) out = exp(a - val)
// 2) val = sum(out)
template <typename T1, typename T2>
inline void {{kernel_name}}_exp_reduce_sum_fusion_kernel(
    T1* a,
    const int& size,
    T2* out,
    T1& val) {
  auto vec_size = at::vec::Vectorized<T1>::size();
  auto vec_max = at::vec::Vectorized<T1>(val);
  T1 tmp_sum = 0;
  auto vec_tmp_sum = at::vec::Vectorized<T1>(tmp_sum);
  for (long i = 0; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = at::vec::Vectorized<T1>::loadu(a + i);
    auto tmp1 = tmp0 - vec_max;
    auto tmp2 = tmp1.exp_u20();
    vec_tmp_sum += tmp2;
    at::native::_store(out + i, tmp2);
  }
  tmp_sum = at::vec::vec_reduce_all<T1>(
      [](at::vec::Vectorized<T1>& x, at::vec::Vectorized<T1>& y) {
        return x + y;
      },
      vec_tmp_sum);
  for (long i = vec_size * (size / vec_size); i < size; i++) {
    auto tmp0 = a[i];
    auto tmp1 = tmp0 - val;
    auto tmp2 = exp(tmp1);
    tmp_sum += tmp2;
    out[i] = tmp2;
  }
  val = tmp_sum;
}

// 1) out = a * scale
// 2) max = max(out)
template <typename scalar_t>
inline void {{kernel_name}}_mul_reduce_max_fusion_kernel(
    const scalar_t* a,
    const scalar_t& scale,
    const int& size,
    scalar_t* out,
    scalar_t& max) {
  auto vec_size = at::vec::Vectorized<scalar_t>::size();
  auto vec_scale = at::vec::Vectorized<scalar_t>(scale);
  scalar_t tmp_max = -std::numeric_limits<scalar_t>::infinity();
  auto vec_tmp_max = at::vec::Vectorized<scalar_t>(tmp_max);
  for (long i = 0; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(a + i);
    auto tmp1 = tmp0 * vec_scale;
    vec_tmp_max = at::vec::maximum(vec_tmp_max, tmp1);
    at::native::_store(out + i, tmp1);
  }
  for (long i = vec_size * (size / vec_size); i < size; i++) {
    auto tmp0 = a[i];
    auto tmp1 = tmp0 * scale;
    tmp_max = std::max(tmp_max, tmp1);
    out[i] = tmp1;
  }
  max = std::max(
      tmp_max,
      at::vec::vec_reduce_all<scalar_t>(
          [](at::vec::Vectorized<scalar_t>& x, at::vec::Vectorized<scalar_t>& y) {
            return at::vec::maximum(x, y);
          },
          vec_tmp_max));
}

template <typename scalar_t>
static inline scalar_t* {{kernel_name}}_conditional_data_ptr(scalar_t* ptr, scalar_t* ptr2) {
  TORCH_CHECK(ptr2 == nullptr);
  return ptr;
}

template <typename scalar_t,
          typename std::enable_if_t<c10::is_reduced_floating_point_v<scalar_t>, int> = 0>
static inline scalar_t* {{kernel_name}}_conditional_data_ptr(float* ptr, scalar_t* ptr2) {
  return ptr2;
}

template <typename scalar_t>
inline void {{kernel_name}}_fill_stub(scalar_t* data, scalar_t val, int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  Vec data_vec = Vec(val);
  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    data_vec.store(data + d);
  }
  #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
  # pragma unroll
  #endif
  for (; d < size; d++) {
    data[d] = val;
  }
}

// out = a * scale
template <typename scalar_t>
inline void {{kernel_name}}_mul_scale_kernel(
    scalar_t* a,
    scalar_t scale,
    int64_t size) {
  auto vec_size = at::vec::Vectorized<scalar_t>::size();
  auto vec_scale = at::vec::Vectorized<scalar_t>(scale);
  for (int64_t i = 0; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(a + i);
    auto tmp1 = tmp0 * vec_scale;
    at::native::_store(a + i, tmp1);
  }
  for (int64_t i = vec_size * (size / vec_size); i < size; i++) {
    auto tmp0 = a[i];
    auto tmp1 = tmp0 * scale;
    a[i] = tmp1;
  }
}

"""

BRGEMM_PACK_FUNCTIONS = r"""
template <typename scalar_t>
inline void {{kernel_name}}_copy_value_with_pad(
    const scalar_t* value_ptr,
    scalar_t* dst_ptr,
    int64_t rows,
    int64_t cols,
    int64_t prows,
    int64_t pcols,
    int64_t ldi) {
  auto vec_size = at::vec::Vectorized<scalar_t>::size();
  int64_t i = 0;
  for (; i < rows; i++) {
    int64_t j = 0;
    for (; j < cols - (cols % vec_size); j += vec_size) {
      auto vec_v =
          at::vec::Vectorized<scalar_t>::loadu(value_ptr + i * ldi + j);
      vec_v.store(dst_ptr + i * pcols + j);
    }

    if (j < cols) {
      auto vec_v = at::vec::Vectorized<scalar_t>::loadu(
          value_ptr + i * ldi + j, cols - j);
      vec_v.store(dst_ptr + i * pcols + j, cols - j);
    }

    // col padding
    auto psize = pcols - cols;
    if (psize > 0) {
      auto zero_vec = at::vec::Vectorized<scalar_t>(0);
      int64_t pj = 0;
      for (; pj < psize - (psize % vec_size); pj += vec_size) {
        zero_vec.store(dst_ptr + i * pcols + cols + pj);
      }
      if (pj < psize) {
        zero_vec.store(dst_ptr + i * pcols + cols + pj, psize - pj);
      }
    }
  }
  // row padding
  for (; i < prows; i++) {
    auto zero_vec = at::vec::Vectorized<scalar_t>(0);
    int64_t j = 0;
    for (; j < pcols - (pcols % vec_size); j += vec_size) {
      zero_vec.store(dst_ptr + i * pcols + j);
    }
    if (j < pcols) {
      zero_vec.store(dst_ptr + i * pcols + j, pcols - j);
    }

  }
}
"""

MICRO_GEMM_TEMPLATE = r"""
GEMM_DEFINE
"""

ALLOCATE_BUFFER = r"""
  int64_t {{buffer_name}}_dtype_itemsize = c10::is_reduced_floating_point_v<{{buffer_dtype}}> ? 2 : 4;
  auto& {{buffer_name}}_allocator = *at::getCPUAllocator();
  auto {{buffer_name}}_work_data = {{buffer_name}}_allocator.allocate({{buffer_size}}*{{buffer_name}}_dtype_itemsize);
  void* {{buffer_name}}_data_ptr = {{buffer_name}}_work_data.get();
  {{buffer_dtype}}* {{buffer_name}} = ({{buffer_dtype}}*){{buffer_name}}_data_ptr;
"""

FLEX_ATTENTION_TEMPLATE = r"""
{{template.header().getvalue()}}
#include <ATen/native/cpu/utils.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/Context.h>
{{template.codegen_micro_gemm(kernel.kernel_name)}}
{{template.codegen_softmax_fusion(kernel.kernel_name)}}
{{template.codegen_brgemm_pack_function(kernel.kernel_name)}}
{%- set kernel_args = {"query": query, "key": key, "value": value,
                       "kv_num_blocks": kv_num_blocks, "kv_indices": kv_indices,
                       "full_kv_num_blocks": full_kv_num_blocks, "full_kv_indices": full_kv_indices } %}
{%- set kernel_args = template.update_kernel_args(kernel_args) %}

extern "C"
{{kernel.def_kernel(inputs=kernel_args, outputs={"output": output}, extra_sizevars=template.extra_sizevars)}}
{
  {{ kernel.maybe_codegen_profile() }}
  int64_t qBlockSize = {{qBlockSize}};
  int64_t kvBlockSize = {{kvBlockSize}};
  int64_t num_thread = {{num_thread}};

  // dtypes of kernel and internal buffers
  using scalar_t = {{kernel.dtype(query)}};
  constexpr bool is_reduced_type = c10::is_reduced_floating_point_v<scalar_t>;
  using accum_t = at::opmath_type<{{kernel.dtype(query)}}>;
  using Vec = at::vec::Vectorized<accum_t>;
  accum_t scaling_factor = {{scale}};
  int64_t batchSize = {{kernel.size(query, 0)}};
  int64_t qSize = {{kernel.size(query, 1)}};
  int64_t num_head = {{kernel.size(query, 2)}};
  int64_t headSize = {{kernel.size(query, 3)}};
  int64_t batchSize_k = {{kernel.size(key, 0)}};
  int64_t num_head_k = {{kernel.size(key, 2)}};
  int64_t headSize_v = {{kernel.size(value, 3)}};
  bool is_broadcast_bs_kv = batchSize != batchSize_k;
  bool is_broadcast_head_kv = num_head != num_head_k;
  int64_t gqa_shards = num_head / num_head_k;
  int64_t bs_shards = batchSize / batchSize_k;

  int64_t batchSize_kvi = {{kernel.size(kv_indices, 0)}};
  int64_t num_head_kvi = {{kernel.size(kv_indices, 1)}};
  int64_t block_num_kvi = {{kernel.size(kv_indices, 3)}};
  bool is_broadcast_bs_kvi = batchSize != batchSize_kvi;
  bool is_broadcast_head_kvi = num_head != num_head_kvi;
  int64_t gqa_shards_kvi = num_head / num_head_kvi;
  int64_t bs_shards_kvi = batchSize / batchSize_kvi;

  int64_t kviStrideB = {{kernel.stride(kv_indices, 0)}};
  int64_t kviStrideH = {{kernel.stride(kv_indices, 1)}};
  int64_t kviStrideQ = {{kernel.stride(kv_indices, 2)}};

  int64_t num_kviStrideB = {{kernel.stride(kv_num_blocks, 0)}};
  int64_t num_kviStrideH = {{kernel.stride(kv_num_blocks, 1)}};

{%- if has_full_kv_block %}
  int64_t full_kviStrideB = {{kernel.stride(full_kv_indices, 0)}};
  int64_t full_kviStrideH = {{kernel.stride(full_kv_indices, 1)}};
  int64_t full_kviStrideQ = {{kernel.stride(full_kv_indices, 2)}};

  int64_t full_num_kviStrideB = {{kernel.stride(full_kv_num_blocks, 0)}};
  int64_t full_num_kviStrideH = {{kernel.stride(full_kv_num_blocks, 1)}};
  auto full_kv_indices_data = full_kv_indices;
  auto full_kv_num_blocks_data = full_kv_num_blocks;
{%- endif %}

  auto kv_num_blocks_data = kv_num_blocks;
  auto kv_indices_data = kv_indices;

  // Strides
  int64_t qStrideB = {{kernel.stride(query, 0)}};
  int64_t qStrideM = {{kernel.stride(query, 1)}};
  int64_t qStrideH = {{kernel.stride(query, 2)}};
  int64_t kStrideB = {{kernel.stride(key, 0)}};
  int64_t kStrideN = {{kernel.stride(key, 1)}};
  int64_t kStrideH = {{kernel.stride(key, 2)}};
  int64_t vStrideB = {{kernel.stride(value, 0)}};
  int64_t vStrideN = {{kernel.stride(value, 1)}};
  int64_t vStrideH = {{kernel.stride(value, 2)}};
  int64_t oStrideB = {{kernel.stride(output, 0)}};
  int64_t oStrideM = {{kernel.stride(output, 2)}};
  int64_t oStrideH = {{kernel.stride(output, 1)}};

  int64_t kvSize = {{kernel.size(key, 1)}};

  int64_t qSplitSize = qBlockSize;
  int64_t kvSplitSize = kvBlockSize;


  qSplitSize = qSplitSize > qSize ? qSize : qSplitSize;
  kvSplitSize = kvSplitSize > kvSize ? kvSize : kvSplitSize;
  int64_t qSlice = (qSize + qSplitSize - 1) / qSplitSize;
  int64_t kvSlice = (kvSize + kvSplitSize - 1) / kvSplitSize;
  int64_t kvTail = (kvSize - 1) % kvSplitSize + 1;

  bool need_pack = false;
  // Whether pack is needed for BFloat16/Half
  if (is_reduced_type) {
    // check platform ability
    need_pack = std::is_same_v<scalar_t, at::BFloat16> ? at::native::cpublas::could_pack(at::kBFloat16)
                                                       : at::native::cpublas::could_pack(at::kHalf);
  }
  if (need_pack) {
    // When the number of gemm is greater than the number of pack,
    // the pack overhead can be overlapped.
    int64_t thresh_size = 64;
    need_pack = kvSize >= thresh_size && qSize >= thresh_size;
    if (need_pack) {
      double pack_size = batchSize * num_head * kvSize * headSize;
      double qs_per_thread = (batchSize * num_head * qSlice + num_thread - 1) / num_thread;
      double gemm_size_per_thread = qs_per_thread * qSplitSize * kvSize * headSize;
      need_pack = gemm_size_per_thread / pack_size >= 4;
    }
  }
  // Pad is needed for packing when K is not even
  bool headSize_even = headSize % 2 == 0;
  int64_t eheadSize = need_pack && !headSize_even ? headSize + 1: headSize;
  int64_t ekvSplitSize = need_pack && (kvSplitSize % 2 != 0) ? kvSplitSize + 1 : kvSplitSize;
  int64_t ekvTail = need_pack && (kvTail % 2 != 0) ? kvTail + 1 : kvTail;
  int64_t kv_padding_size = (kvSize - 1) / kvSplitSize * ekvSplitSize + ekvTail;

  // Allocate per thread temp buf (accumulate type)
  int64_t _size_per_thread =
      /* qk     */ qSplitSize * kvSplitSize +
      /* qk_max */ qSplitSize +
      /* qk_sum */ qSplitSize +
      /* dst    */ qSplitSize * headSize_v;

  // Inputs/outputs buffers
  const scalar_t* q_data = query;
  const scalar_t* k_data = key;
  const scalar_t* v_data = value;
  scalar_t* out_data = output;

  // Buffers to store accum results, padding query and transpose/packing key/value
  {{template.codegen_allocate_buffer("buf_data", "accum_t", "num_thread*_size_per_thread")}}
  {{template.codegen_allocate_buffer("buf_reduced_data", "scalar_t", "num_thread*qSplitSize*ekvSplitSize")}}
  {{template.codegen_allocate_buffer("key_reorder_ptr", "scalar_t", "batchSize_k*num_head_k*eheadSize*kvSize")}}
  {{template.codegen_allocate_buffer("value_reorder_ptr", "scalar_t", "batchSize_k*num_head_k*kv_padding_size*headSize_v")}}
  {{template.codegen_allocate_buffer("transpose_buffer_ptr", "scalar_t", "num_thread*kvSplitSize*headSize")}}
  {{template.codegen_allocate_buffer("query_padding_ptr", "scalar_t", "num_thread*qSplitSize*eheadSize")}}
  if (need_pack) {
    // Pack K, V
    at::parallel_for(0, batchSize_k * num_head_k * kvSlice, 1, [&](int64_t begin, int64_t end) {
      int ompIdx = at::get_thread_num();
      int64_t i = 0, j = 0, l = 0, n = 0;
      scalar_t* transpose_ptr = need_pack? transpose_buffer_ptr + ompIdx * kvSplitSize * headSize : nullptr;
      at::native::data_index_init(begin, i, batchSize_k, j, num_head_k, l, kvSlice);
      for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
        n = l * kvSplitSize;
        int64_t cur_kvSplitSize = std::min(kvSplitSize, kvSize - n);
        auto k_addr =
              k_data + i * kStrideB + j * kStrideH + n * kStrideN;
        auto v_addr =
              v_data + i * vStrideB + j * vStrideH + n * vStrideN;
        // transpose [cur_kvSplitSize, headSize] -> [headSize, cur_kvSplitSize]
        at::native::utils::transpose<uint16_t>(
          cur_kvSplitSize,
          headSize,
          /* src_ptr */
          reinterpret_cast<const uint16_t*>(k_addr),
          /* ld_src */ kStrideN,
          /* dst */ reinterpret_cast<uint16_t*>(transpose_ptr),
          /* ld_dst */ cur_kvSplitSize);

        // Pack [headSize, cur_kvSplitSize]
        at::vec::pack_vnni2(
          /* src */ reinterpret_cast<const uint16_t*>(transpose_ptr),
          /* dst */ reinterpret_cast<uint16_t*>(key_reorder_ptr + i * num_head_k * eheadSize * kvSize +
                  j * eheadSize * kvSize + n * eheadSize),
          /* ld_src */ cur_kvSplitSize,
          /* K */ headSize,
          /* N */ cur_kvSplitSize);

        // Pack [cur_kvSplitSize, headSize_v]
        at::vec::pack_vnni2(
          /* src */ reinterpret_cast<const uint16_t*>(v_addr),
          /* dst */ reinterpret_cast<uint16_t*>(value_reorder_ptr +
                  i * num_head_k * kv_padding_size * headSize_v +
                  j * kv_padding_size * headSize_v + n * headSize_v),
          /* ld_src */ vStrideN,
          /* K */ cur_kvSplitSize,
          /* N */ headSize_v);
      // Move to the next query
      at::native::data_index_step(i, batchSize_k, j, num_head_k, l, kvSlice);
      }
    });
  }
  // Attention loop below
  at::parallel_for(0, batchSize * num_head * qSlice, 1, [&](int64_t begin, int64_t end) {
    int64_t i = 0, j = 0, k = 0;
    at::native::data_index_init(begin, i, batchSize, j, num_head, k, qSlice);
    int ompIdx = at::get_thread_num();
    accum_t* buf_ptr = buf_data + ompIdx * _size_per_thread;
    accum_t* qk_data = buf_ptr;
    accum_t* qk_max_data = qk_data + qSplitSize * kvSplitSize;
    accum_t* qk_sum_data = qk_max_data + qSplitSize;
    accum_t* dst_data = qk_sum_data + qSplitSize;
    scalar_t *qk_reduced_data =
        is_reduced_type
            ? buf_reduced_data + ompIdx * qSplitSize * ekvSplitSize
            : nullptr;
    scalar_t* query_t_padding_ptr = (!headSize_even && need_pack)
            ? query_padding_ptr + ompIdx * qSplitSize * eheadSize
            : nullptr;

    for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
      auto i_kvi = is_broadcast_bs_kvi ? i/bs_shards_kvi : i;
      auto j_kvi = is_broadcast_head_kvi ? j/gqa_shards_kvi : j;
      auto kv_logical_num_data = kv_num_blocks_data + i_kvi * num_kviStrideB +
                              j_kvi * num_kviStrideH + k;
      int kv_indice_num = *kv_logical_num_data;
      std::vector<int> kv_indice_list(kv_indice_num);
      for(int kv_i = 0; kv_i < kv_indice_num; kv_i++){
        auto kv_logical_data = kv_indices_data + i_kvi * kviStrideB +
                                  j_kvi * kviStrideH + k*kviStrideQ + kv_i;
        kv_indice_list[kv_i] = *kv_logical_data;
      }
      bool is_skip_kv = kv_indice_num > 0 ? false : true;
{%- if has_full_kv_block %}
      auto full_kv_logical_num_data = full_kv_num_blocks_data + i_kvi * num_kviStrideB +
                              j_kvi * num_kviStrideH + k;
      int full_kv_indice_num = *full_kv_logical_num_data;
      std::vector<int> full_kv_indice_list(full_kv_indice_num);
      for(int kv_i = 0; kv_i < full_kv_indice_num; kv_i++){
        auto full_kv_logical_data = full_kv_indices_data + i_kvi * full_kviStrideB +
                                  j_kvi * full_kviStrideH + k*full_kviStrideQ + kv_i;
        full_kv_indice_list[kv_i] = *full_kv_logical_data;
      }
      is_skip_kv = kv_indice_num + full_kv_indice_num > 0 ? false : true;
{%- endif %}
      int64_t m = k * qSplitSize;
      int64_t cur_qSplitSize = std::min(qSplitSize, qSize - m);
      if (!is_skip_kv){
        // Initialize max and sum
        {{kernel.kernel_name}}_fill_stub(qk_max_data,
            -std::numeric_limits<accum_t>::infinity(), cur_qSplitSize);
        {{kernel.kernel_name}}_fill_stub(qk_sum_data,
            static_cast<accum_t>(0), cur_qSplitSize);

        if (!headSize_even && need_pack) {
          // Pad query if headSize is not even
          {{kernel.kernel_name}}_copy_value_with_pad<scalar_t>(
            q_data + i * qStrideB + j * qStrideH + m * qStrideM,
            query_t_padding_ptr,
            cur_qSplitSize,
            headSize,
            cur_qSplitSize,
            eheadSize,
            qStrideM
          );
        }
      }

{%- if has_full_kv_block %}
      for (int64_t n_idx = 0; n_idx < kv_indice_num + full_kv_indice_num ; n_idx += 1) {
        auto n = n_idx < kv_indice_num ? kv_indice_list[n_idx]*kvSplitSize : full_kv_indice_list[n_idx - kv_indice_num]*kvSplitSize;
{%- else %}
      for (int64_t n_idx = 0; n_idx < kv_indice_num ; n_idx += 1) {
        auto n = kv_indice_list[n_idx]*kvSplitSize;
{%- endif %}

        auto cur_n = n/kvSplitSize;
        int64_t cur_kvSplitSize = std::min(kvSplitSize, kvSize - n);
        int64_t cur_ekvSplitSize = (need_pack && cur_kvSplitSize % 2 != 0) ? cur_kvSplitSize + 1 : cur_kvSplitSize;

        // Calculate scale * q @ k.T
        auto i_kv = is_broadcast_bs_kv ? i/bs_shards : i;
        auto j_kv = is_broadcast_head_kv ? j/gqa_shards : j;

        if (!need_pack) {
          auto k_addr =
              k_data + i_kv * kStrideB + j_kv * kStrideH + n * kStrideN;

          {{kernel.kernel_name}}_kernel_micro_gemm_transpose_b<static_cast<bool>(false)>(
              q_data + i * qStrideB + j * qStrideH +
                  m * qStrideM,
              k_addr,
              qk_data,
              cur_qSplitSize,
              cur_kvSplitSize,
              headSize,
              qStrideM,
              kStrideN,
              cur_kvSplitSize);

        } else {
          at::native::cpublas::brgemm(
              cur_qSplitSize,
              cur_kvSplitSize,
              eheadSize,
              headSize_even ? qStrideM : eheadSize,
              cur_kvSplitSize,
              cur_kvSplitSize,
              false,
              !headSize_even
                  ? query_t_padding_ptr
                  : q_data + i * qStrideB + j * qStrideH + m * qStrideM,
              key_reorder_ptr + i_kv * num_head_k * eheadSize * kvSize +
                  j_kv * eheadSize * kvSize + n * eheadSize,
              qk_data,
              need_pack);
        }

        {{kernel.kernel_name}}_mul_scale_kernel<accum_t>(qk_data, scaling_factor, cur_qSplitSize*cur_kvSplitSize);

{%- if score_mod and mask_mod %}
        // TODO: reduce the number of calls of q_idx and kv_idx initialization
        std::vector<int64_t> q_idx(cur_qSplitSize);
        for (int64_t i = 0; i < cur_qSplitSize; ++i) {
          q_idx[i] = m + i;
        }

        std::vector<int64_t> kv_idx(cur_kvSplitSize);
        for (int64_t i = 0; i < cur_kvSplitSize; ++i) {
          kv_idx[i] = n + i;
        }

        std::vector<int64_t> b_idx = {i};
        std::vector<int64_t> h_idx = {j};

        accum_t* in_ptr0 = qk_data;

        auto in_ptr1 = b_idx.data();
        auto in_ptr2 = h_idx.data();
        auto in_ptr3 = q_idx.data();
        auto in_ptr4 = kv_idx.data();

        // apply score mod function
        {
            {{ template.generate_other_buffer("score_others", 0, "len_score_other", kernel.args) }}
            accum_t* out_ptr{{score_buf_idx}} = in_ptr0;
            {{ template.modification(score_mod, score_buf_name, score_buf_idx)|indent(12, false) }}
        }

        if ((std::find(kv_indice_list.begin(), kv_indice_list.end(), cur_n) != kv_indice_list.end()) ){
          // Apply block mask, fill unused with -inf
          {
              {{ template.generate_other_buffer("mask_others", -1, "len_mask_other", kernel.args) }}
              accum_t* out_ptr{{mask_buf_idx}} = in_ptr0;
              {{ template.modification(mask_mod, mask_buf_name, mask_buf_idx)|indent(12, false) }}
          }
        }

{%- endif %}
        // Update coefficients with Softmax
        accum_t tmp_max = 0, tmp_sum = 0, exp_tmp = 0;
        for (int64_t row = 0; row < cur_qSplitSize; ++row) {
          // apply scaling factor and max per row in fusion
          {{kernel.kernel_name}}_mul_reduce_max_fusion_kernel(
              qk_data + row * cur_kvSplitSize,
              static_cast<accum_t>(1),
              cur_kvSplitSize,
              qk_data + row * cur_kvSplitSize,
              tmp_max);
          tmp_max = qk_max_data[row] > tmp_max ? qk_max_data[row] : tmp_max;
          if (tmp_max == -std::numeric_limits<accum_t>::infinity()) {
            // to avoid `nan = exp2f(-inf - (-inf))`
            {{kernel.kernel_name}}_fill_stub(
              {{kernel.kernel_name}}_conditional_data_ptr(qk_data, qk_reduced_data) + row * cur_ekvSplitSize,
              static_cast<scalar_t>(0), cur_kvSplitSize);
          } else {
            tmp_sum = tmp_max;
            // qk <- exp(qk - max) and sum per row
            {{kernel.kernel_name}}_exp_reduce_sum_fusion_kernel(
              qk_data + row * cur_kvSplitSize, cur_kvSplitSize,
              {{kernel.kernel_name}}_conditional_data_ptr(qk_data, qk_reduced_data) + row * cur_ekvSplitSize,
              tmp_sum);
            // exp_tmp <- exp(max[row] - max)
            exp_tmp = std::exp(qk_max_data[row] - tmp_max);
            // sum[row] <- sum + exp_tmp * sum[row]
            qk_sum_data[row] = tmp_sum + exp_tmp * qk_sum_data[row];
            // max[row] <- max
            qk_max_data[row] = tmp_max;
            // dst <- dst * exp_tmp
            if (n_idx > 0) {
              at::vec::map<accum_t>(
              [exp_tmp](Vec x) { return x * Vec(exp_tmp); },
              dst_data + row * headSize_v,
              dst_data + row * headSize_v,
              headSize_v);
            }
          }
          if (need_pack && cur_kvSplitSize % 2 != 0) {
            // Pad: [qSplitSize, cur_kvSplitSize] -> [qSplitSize, cur_kvSplitSize + 1]
            *(qk_reduced_data + row * (1 + cur_kvSplitSize) + cur_kvSplitSize) = scalar_t(0);
          }
        }
        // Calculate Softmax(q @ k.T) @ v
        if (!need_pack) {
          auto v_addr =
              v_data + i_kv * vStrideB + j_kv * vStrideH + n * vStrideN;
          // Fallback Half brgemm is slower than micro gemm
          if (!std::is_same_v<scalar_t, at::Half>) {
            at::native::cpublas::brgemm(
                  cur_qSplitSize,
                  headSize_v,
                  cur_ekvSplitSize,
                  cur_ekvSplitSize,
                  vStrideN,
                  headSize_v,
                  n_idx > 0,
                  {{kernel.kernel_name}}_conditional_data_ptr(qk_data, qk_reduced_data),
                  v_addr,
                  dst_data,
                  need_pack);
          } else {
            if (n_idx > 0) {
              {{kernel.kernel_name}}_kernel_micro_gemm<static_cast<bool>(true)>(
                {{kernel.kernel_name}}_conditional_data_ptr(qk_data, qk_reduced_data),
                v_addr,
                dst_data,
                cur_qSplitSize,
                headSize_v,
                cur_ekvSplitSize,
                cur_ekvSplitSize,
                vStrideN,
                headSize_v);
            } else {
              {{kernel.kernel_name}}_kernel_micro_gemm<static_cast<bool>(false)>(
                {{kernel.kernel_name}}_conditional_data_ptr(qk_data, qk_reduced_data),
                v_addr,
                dst_data,
                cur_qSplitSize,
                headSize_v,
                cur_ekvSplitSize,
                cur_ekvSplitSize,
                vStrideN,
                headSize_v);
            }
          }
        } else {
          int64_t psize = n / kvSplitSize * ekvSplitSize;
          at::native::cpublas::brgemm(
              cur_qSplitSize,
              headSize_v,
              cur_ekvSplitSize,
              cur_ekvSplitSize,
              headSize_v,
              headSize_v,
              n_idx > 0,
              qk_reduced_data,
              value_reorder_ptr +
                  i_kv * num_head_k * kv_padding_size * headSize_v +
                  j_kv * kv_padding_size * headSize_v + psize * headSize_v,
              dst_data,
              need_pack);
        }
      }

      // dst <- dst / sum[row]
      // reorder MHA output with strides
      for (int64_t row = 0; row < cur_qSplitSize; ++row) {
        // Row sums for full masked out rows are 0, we set them to 1
        // in order to avoid NaNs in the output and instead set fully
        // masked out rows to 0
        qk_max_data[row] = qk_max_data[row] == -std::numeric_limits<accum_t>::infinity() ? 0 : qk_max_data[row];
        qk_sum_data[row] = qk_sum_data[row] == 0 ? 1 : qk_sum_data[row];
        accum_t sum_reciprocal = 1 / qk_sum_data[row];
        at::vec::map<scalar_t>(
            [sum_reciprocal, is_skip_kv](Vec x) { return  is_skip_kv ? Vec(0.0) : x * Vec(sum_reciprocal); },
            out_data + i * oStrideB + j * oStrideH + m * oStrideM + row * oStrideM,
            dst_data + row * headSize_v,
            headSize_v);
      }

      // Move to the next query
      at::native::data_index_step(i, batchSize, j, num_head, k, qSlice);
    }

    at::native::cpublas::brgemm_release(need_pack);

  });
}
"""


class CppFlexAttentionTemplate(CppTemplate):
    def __init__(
        self,
        input_nodes,
        layout: ir.Layout,
        scale,
        score_mod,
        mask_mod,
        kv_block_size,
        q_block_size,
        has_other_buffer,
        no_full_kv_block,
        fake_buffers,
        len_score_other,
        len_mask_other,
        kernel_input_name_to_buffer,
        block_vars,
    ) -> None:
        assert layout.dtype in [torch.float, torch.bfloat16, torch.float16]
        super().__init__("flex_attention", input_nodes, layout, parallel_num_threads())
        self.scale = scale
        self.score_mod = score_mod
        self.mask_mod = mask_mod
        self.score_buf_name = (
            V.graph.register_buffer(self.score_mod) if self.score_mod else None
        )
        self.mask_buf_name = (
            V.graph.register_buffer(self.mask_mod) if self.mask_mod else None
        )

        def get_idx(buf_name):
            match = re.search(r"\d+", buf_name)
            assert match, f"incorrect score buf name: {buf_name}"
            return match.group()

        self.score_buf_idx = (
            get_idx(self.score_buf_name) if self.score_buf_name else None
        )
        self.mask_buf_idx = get_idx(self.mask_buf_name) if self.mask_buf_name else None
        self.kv_block_size = kv_block_size
        self.q_block_size = q_block_size
        self.has_other_buffer = has_other_buffer
        self.no_full_kv_block = no_full_kv_block
        self.other_buffer_input_offset = 2
        if self.no_full_kv_block:
            self.other_buffer_input_offset = 0
        self.fake_buffers = fake_buffers
        self.len_score_other = len_score_other
        self.len_mask_other = len_mask_other
        self.kernel_input_name_to_buffer = kernel_input_name_to_buffer
        self.block_vars = block_vars
        self.extra_sizevars = list(
            OrderedSet(
                val
                for val in self.kernel_input_name_to_buffer.values()
                if isinstance(val, sympy.Symbol)
            )
        )
        self.other_buf_start_idx = 5
        self.score_mod_other_buffers = (
            self.input_nodes[
                self.other_buf_start_idx
                + self.other_buffer_input_offset : self.other_buf_start_idx
                + self.other_buffer_input_offset
                + self.len_score_other
            ]
            if self.has_other_buffer
            else None
        )
        self.mask_mod_other_buffers = (
            self.input_nodes[
                self.other_buf_start_idx
                + self.other_buffer_input_offset
                + self.len_score_other :
            ]
            if self.has_other_buffer
            else None
        )
        self.other_ptr_data = {}  # type: ignore[var-annotated]

    def update_kernel_args(self, kernel_args):
        kernel_args.update(
            {
                key: value
                for key, value in self.kernel_input_name_to_buffer.items()
                if not isinstance(value, sympy.Symbol)
            }
        )
        return kernel_args

    def generate_other_buffer(self, buf_list, start_offset, len_attr, kernel_args):
        kernel_input_name_to_buffer_name = {
            key: value if isinstance(value, sympy.Symbol) else value.get_name()
            for key, value in self.kernel_input_name_to_buffer.items()
        }

        def get_arg(name):
            return kernel_input_name_to_buffer_name.get(name)

        def get_arg_name(name):
            if isinstance(get_arg(name), sympy.Symbol):
                return kernel_args.sizevars.get(get_arg(name))
            return kernel_args.input_buffers.get(get_arg(name))

        if not self.has_other_buffer:
            return ""

        if start_offset == -1:
            start_offset = self.len_score_other

        length = getattr(self, len_attr)
        for i in range(length):
            pointer = f"in_ptr{self.other_buf_start_idx + start_offset + i}"
            buffer_key = f"{buf_list}_{i}"
            if pointer not in self.other_ptr_data:
                self.other_ptr_data[pointer] = (
                    get_arg_name(buffer_key),
                    get_arg(buffer_key),
                )

        return "\n".join(
            f"auto {ptr} = {name};" for ptr, (name, _) in self.other_ptr_data.items()
        )

    def modification(self, subgraph_buffer, output_name, output_idx):
        assert isinstance(subgraph_buffer, ir.ComputedBuffer)
        subgraph_buffer_data = subgraph_buffer.data
        from ..loop_body import LoopBody
        from ..utils import sympy_index_symbol_with_prefix, SymT
        from ..virtualized import V
        from .cpp import CppKernelProxy, KernelGroup, ParallelDepth

        kernel_group = KernelGroup()
        kernel_input_args = {
            "score": "in_ptr0",
            "b": "in_ptr1",
            "h": "in_ptr2",
            "q_idx": "in_ptr3",
            "kv_idx": "in_ptr4",
        }
        if self.has_other_buffer:
            kernel_input_args.update(
                {arg: ptr for ptr, (_, arg) in self.other_ptr_data.items()}
            )

        kernel_output_args = {output_name: f"out_ptr{output_idx}"}

        args = kernel_group.args
        for name, inp in kernel_input_args.items():
            args.input_buffers[name] = inp

        for name, inp in kernel_output_args.items():
            args.output_buffers[name] = inp

        for name in self.extra_sizevars:
            args.sizevars[name] = f"k{name}"

        kernel_group.args = args

        cpp_kernel_proxy = CppKernelProxy(kernel_group)
        bodies = []
        var_sizes_list = []
        var_sizes = tuple(subgraph_buffer.get_size())
        var_ranges = {
            sympy_index_symbol_with_prefix(SymT.INDEX, i): sz
            for i, sz in enumerate(var_sizes)
        }

        dst_layout = subgraph_buffer.get_layout()
        output_index = dst_layout.make_indexer()([*var_ranges.keys()])

        def fn(*args):
            V.ops.store(
                output_name,
                output_index,
                subgraph_buffer_data.make_loader()(args).value,
            )

        body = LoopBody(
            fn,
            (list(var_ranges.keys())),
            var_ranges,
            list(var_ranges.keys()),
            tuple(),
        )

        from ..loop_body import MemoryUsageType

        assert all(
            mem.buffer_name in kernel_group.args.input_buffers
            for mem in body.memory_usage[MemoryUsageType.LOAD]
        ), (
            "All the buffers in the score and mask subgraph should be in kernel_group.args.input_buffers"
        )

        bodies.append(body)
        var_sizes_list.append((var_sizes, ()))

        cpp_kernel_proxy.codegen_loop_bodies(bodies, var_sizes_list)

        def max_parallel_depth():
            return ParallelDepth(parallel_depth=0, start_depth=0)

        # This loop is not parallelized since it is not the outermost loop.
        with patch.object(
            cpp_kernel_proxy.loop_nest, "max_parallel_depth", max_parallel_depth
        ):
            kernel_group.finalize_kernel(cpp_kernel_proxy, [])
        output_code = kernel_group.loops_code.getvalue()

        var_q_symbol, var_kv_symbol = self.block_vars
        # See [Note] Handle the case where the split sizes are not statically known.
        # We don't know the value of qBlockSize and rkvBlockSize during compilation time
        # thus we've represented them by symbols.
        # We change the symbol strings back to "cur_qSplitSize" and "cur_kvSplitSize"
        # in the generated code thus they'll be filled with the real value during runtime.
        if var_q_symbol in kernel_group.args.sizevars:
            output_code = output_code.replace(
                kernel_group.args.sizevars[var_q_symbol], "cur_qSplitSize"
            )
        if var_kv_symbol in kernel_group.args.sizevars:
            output_code = output_code.replace(
                kernel_group.args.sizevars[var_kv_symbol], "cur_kvSplitSize"
            )

        return output_code

    @staticmethod
    def add_choices(
        choices,
        input_nodes,
        layout,
        scale,
        score_mod,
        mask_mod,
        kv_block_size,
        q_block_size,
        has_other_buffer,
        no_full_kv_block,
        fake_buffers,
        len_score_other,
        len_mask_other,
        kernel_input_name_to_buffer,
        block_vars,
    ):
        # Drop-in router (Part 3): the cases the causal template serves exactly go to
        # it; everything else continues below, unchanged.
        use_causal, causal_offset = _causal_template_route(
            score_mod, mask_mod, has_other_buffer, len_score_other, len_mask_other
        )
        if use_causal:
            return CppCausalAttentionCpuTemplate.add_choices(
                choices=choices,
                input_nodes=input_nodes,
                layout=layout,
                scale=scale,
                kv_block_size=kv_block_size,
                q_block_size=q_block_size,
                no_full_kv_block=no_full_kv_block,
                causal_offset=causal_offset,
            )

        def preprocessor(input_nodes, layout):
            return input_nodes, layout

        def postprocessor(output):
            return output

        template = DataProcessorTemplateWrapper(
            CppFlexAttentionTemplate,
            preprocessor,
            postprocessor,
            input_nodes=input_nodes,
            layout=layout,
            scale=scale,
            score_mod=score_mod,
            mask_mod=mask_mod,
            kv_block_size=kv_block_size,
            q_block_size=q_block_size,
            has_other_buffer=has_other_buffer,
            no_full_kv_block=no_full_kv_block,
            fake_buffers=fake_buffers,
            len_score_other=len_score_other,
            len_mask_other=len_mask_other,
            kernel_input_name_to_buffer=kernel_input_name_to_buffer,
            block_vars=block_vars,
        )
        template.maybe_append_choice(choices)
        return template

    def apply_score_mod(self, score, b, h, q_idx, kv_idx):
        return self.score_mod.graph_module(score, b, h, q_idx, kv_idx).item()

    def render(  # type: ignore[override,return]
        self,
        kernel,
        template_buffer_node: ir.CppTemplateBuffer | None = None,
        epilogue_nodes: list[ir.IRNode] | None = None,
        **kwargs,
    ) -> str:
        if epilogue_nodes is not None and epilogue_nodes != []:
            raise NotImplementedError(
                "Unsupported for `epilogue_nodes` in CppFlexAttentionTemplate."
            )
        # Query (Batch x Num_heads  x Q_seq_len  x Dim_per_head)
        #     -> (Batch x Q_seq_len  x Num_heads  x Dim_per_head)
        #  Key   (Batch x Num_heads  x KV_seq_len x Dim_per_head)
        #     -> (Batch x KV_seq_len x Num_heads  x Dim_per_head)
        #  Value (Batch x Num_heads  x KV_seq_len x Dim_per_head)
        #     -> (Batch x KV_seq_len x Num_heads  x Dim_per_head)

        query = kernel.permute(self.input_nodes[0], [0, 2, 1, 3])
        key = kernel.permute(self.input_nodes[1], [0, 2, 1, 3])
        value = kernel.permute(self.input_nodes[2], [0, 2, 1, 3])
        self.accumulate_dtype = torch.float
        self.input_dtype = query.layout.dtype

        num_threads = parallel_num_threads()
        assert isinstance(self.output_node, ir.IRNode)
        buf_out = TensorBox.create(self.output_node)
        if template_buffer_node is not None:
            buf_out = template_buffer_node
        options = dict(
            query=query,
            key=key,
            value=value,
            kv_num_blocks=self.input_nodes[3],
            kv_indices=self.input_nodes[4],
            full_kv_num_blocks=(
                self.input_nodes[5] if not self.no_full_kv_block else None
            ),
            full_kv_indices=self.input_nodes[6] if not self.no_full_kv_block else None,
            score_mod_other_buffers=self.score_mod_other_buffers,
            mask_mod_other_buffers=self.mask_mod_other_buffers,
            scale=self.scale,
            has_full_kv_block=not self.no_full_kv_block,
            accumulate_dtype=self.accumulate_dtype,
            query_dtype=self.input_dtype,
            kvBlockSize=self.kv_block_size,
            qBlockSize=self.q_block_size,
            template=self,
            output=buf_out,
            kernel=kernel,
            num_thread=num_threads,
            score_mod=self.score_mod,
            mask_mod=self.mask_mod,
            score_buf_name=self.score_buf_name,
            mask_buf_name=self.mask_buf_name,
            score_buf_idx=self.score_buf_idx,
            mask_buf_idx=self.mask_buf_idx,
        )
        with contextlib.ExitStack() as stack:
            for buf in self.fake_buffers:
                stack.enter_context(
                    patch.object(V.graph, "get_dtype", self._fake_get_dtype(buf))
                )
            return self._template_from_string(FLEX_ATTENTION_TEMPLATE).render(**options)

    def codegen_softmax_fusion(self, kernel_name: str):
        # TODO: use inductor IR to rewrite those fusions
        return self._template_from_string(SOFTMAX_FUSIONS).render(
            dict(kernel_name=kernel_name)
        )

    def codegen_brgemm_pack_function(self, kernel_name: str):
        # TODO: make them general for common bmm templates
        return self._template_from_string(BRGEMM_PACK_FUNCTIONS).render(
            dict(kernel_name=kernel_name)
        )

    def codegen_allocate_buffer(self, buffer_name: str, buffer_dtype, buffer_size):
        return self._template_from_string(ALLOCATE_BUFFER).render(
            dict(
                buffer_name=buffer_name,
                buffer_dtype=buffer_dtype,
                buffer_size=buffer_size,
            )
        )

    def micro_gemm_define(self, kernel_name: str):
        from torch._inductor.codegen.cpp_gemm_template import (
            CppTemplateKernel,
            parallel_num_threads,
        )
        from torch._inductor.codegen.cpp_micro_gemm import CppMicroGemmFP32Vec
        from torch._inductor.virtualized import V

        micro_gemm_trans = CppMicroGemmFP32Vec(
            kernel_name + "_kernel_micro_gemm_transpose_b",
            self.input_dtype,
            self.input_dtype,
            self.accumulate_dtype,
            self.accumulate_dtype,
            GemmBlocking(1, 16, 1),
            1,
            True,
            True,
        )

        micro_gemm = CppMicroGemmFP32Vec(
            kernel_name + "_kernel_micro_gemm",
            self.input_dtype,
            self.input_dtype,
            self.accumulate_dtype,
            self.accumulate_dtype,
            GemmBlocking(1, 16, 1),
            1,
            True,
            False,
        )

        with V.set_graph_handler(V.graph):
            kernel = CppTemplateKernel("cpp_micro_gemm", parallel_num_threads())
            code_trans = micro_gemm_trans.codegen_define(kernel)
            code = micro_gemm.codegen_define(kernel)
        return code + code_trans

    def codegen_micro_gemm(self, kernel_name: str):
        micro_gemm = self.micro_gemm_define(kernel_name)
        GEMM_SOURCE_CODE = MICRO_GEMM_TEMPLATE.replace("GEMM_DEFINE", micro_gemm)
        return self._template_from_string(GEMM_SOURCE_CODE).render()


# ===========================================================================
# The optimized causal attention on CPU
#
# TESTED PLATFORMS:
#   Xeon 6767P (Granite Rapids)   AVX-512 + AMX bf16, 2 MiB L2/core, ~336 MiB L3/socket.
#   Core Ultra 265K (Arrow Lake)  AVX2 only, no AMX, no SMT, hybrid 8P+12E, ~30 MiB L3.
#
#   L1D and L3 are read from sysfs; only _l2_bytes still carries a literal default.
# ===========================================================================


def causal_enabled():
    # Whether to RECOGNISE a causal mask at all (the recognition itself lives in
    # flex_cpu.py, which is where the mask graph is). Default ON.
    return os.environ.get("TORCHINDUCTOR_CPP_CPU_CAUSAL_ATTENTION_CAUSAL", "1") == "1"


def _causal_balance():
    # Causal LOAD BALANCE: the cost-model work queue plus the parallel granularity
    return os.environ.get("TORCHINDUCTOR_CPP_CPU_CAUSAL_ATTENTION_CAUSAL_BALANCE", "1") == "1"


@functools.cache
def _cache_bytes_sysfs(level, ctype, default):
    # Check the cache byte sizes depends on level - level=3 means L3
    for idx in range(8):
        base = f"/sys/devices/system/cpu/cpu0/cache/index{idx}"
        try:
            with open(f"{base}/level") as f:
                if f.read().strip() != level:
                    continue
            with open(f"{base}/type") as f:
                if f.read().strip() != ctype:
                    continue
            with open(f"{base}/size") as f:
                size = f.read().strip()
        except OSError:
            continue
        return int(size.rstrip("KM")) * {"K": 1024, "M": 1024 * 1024}.get(size[-1:], 1)
    return default


def _l3_bytes():
    # Per-socket L3
    override = os.environ.get("TORCHINDUCTOR_CPP_CPU_CAUSAL_ATTENTION_L3_BYTES")
    return (
        int(override)
        if override
        else _cache_bytes_sysfs("3", "Unified", 432 * 1024 * 1024)
    )


def _l3_fraction():
    return float(os.environ.get("TORCHINDUCTOR_CPP_CPU_CAUSAL_ATTENTION_L3_FRACTION", "0.8"))


def _nkv_override():
    # 0 = use the analytic size.
    return int(os.environ.get("TORCHINDUCTOR_CPP_CPU_CAUSAL_ATTENTION_NKV", "0"))


def _jkv_chunk_override():
    # 0 = analytic: the most WHOLE heads whose K+V fits the L3 budget
    return int(os.environ.get("TORCHINDUCTOR_CPP_CPU_CAUSAL_ATTENTION_JKV", "0"))


def _kv_split_size():
    kvb = int(os.environ.get("TORCHINDUCTOR_CPP_CPU_CAUSAL_ATTENTION_KV_SPLIT", "256"))
    # Must be a multiple of 32, for two independent reasons.
    if kvb <= 0 or kvb % 32:
        raise ValueError(
            f"KV_SPLIT must be a positive multiple of 32 (packed-V block stride and the "
            f"VNNI2 K scatter both require it), got {kvb}"
        )
    return kvb


def _q_split_size():
    qb = int(os.environ.get("TORCHINDUCTOR_CPP_CPU_CAUSAL_ATTENTION_Q_SPLIT", "0"))
    if qb and qb % 32:
        raise ValueError(
            f"Q_SPLIT must be a multiple of 32 (AMX M constraint), got {qb}"
        )
    return qb


def _l2_bytes():
    # L2 per physical CORE. Default 2 MiB, which holds for every Granite Rapids part here
    return int(
        os.environ.get("TORCHINDUCTOR_CPP_CPU_CAUSAL_ATTENTION_L2_BYTES", str(2 * 1024 * 1024))
    )


def _l1d_bytes_sysfs():
    # L1D per core. Fallback 48 KiB, which is every Granite Rapids part here.
    return _cache_bytes_sysfs("1", "Data", 48 * 1024)


def _l1d_bytes():
    override = os.environ.get("TORCHINDUCTOR_CPP_CPU_CAUSAL_ATTENTION_L1D_BYTES")
    return int(override) if override else _l1d_bytes_sysfs()


def _l2_fraction():
    return float(os.environ.get("TORCHINDUCTOR_CPP_CPU_CAUSAL_ATTENTION_L2_FRACTION", "0.65"))


def _l2_group_override():
    # 0 = analytic from the L2 budget
    return int(os.environ.get("TORCHINDUCTOR_CPP_CPU_CAUSAL_ATTENTION_L2_GROUP", "0"))


def _softmax_fuse_scale():
    return int(os.environ.get("TORCHINDUCTOR_CPP_CPU_CAUSAL_ATTENTION_SOFTMAX_FUSE_SCALE", "1"))


def _softmax_ilv():
    # Interleave the AVX-512 softmax with the AMX QK GEMM
    return int(os.environ.get("TORCHINDUCTOR_CPP_CPU_CAUSAL_ATTENTION_SOFTMAX_ILV", "1"))


def _amx_gemm():
    # Use intrinsic AMX bf16 GEMM or not
    override = os.environ.get("TORCHINDUCTOR_CPP_FLEX_AMX_GEMM")
    if override is not None:
        return override == "1"
    return os.environ.get("TORCHINDUCTOR_CPP_CPU_CAUSAL_ATTENTION_AMX_GEMM", "1") == "1"


def _qk_blas():
    # QK^T GEMM backend. Uses MKL GEMM if it is 1
    return os.environ.get("TORCHINDUCTOR_CPP_CPU_CAUSAL_ATTENTION_QK_BLAS", "1") == "1"


def _softmax_ilv_rb():
    # Row-block height for the woven schedule: 16, 32, or 0 = analytic from L1D.
    rb = int(os.environ.get("TORCHINDUCTOR_CPP_CPU_CAUSAL_ATTENTION_SOFTMAX_ILV_RB", "0"))
    if rb not in (0, 16, 32):
        raise ValueError(f"SOFTMAX_ILV_RB must be 0 (analytic), 16 or 32, got {rb}")
    return rb


def _static_size(v):
    # Sizes are sympy, and symbolic under dynamic shapes. The analytic row-block height
    # needs a real int and falls back to the shipped 16 when it cannot have one.
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _softmax_ilv_rb_analytic(q_split_size, kv_split_size):
    # ---- Analytic woven row-block height, sized against L1D ----
    rb = _softmax_ilv_rb()
    if rb:
        return rb
    # Symbolic q/kvSize under dynamic shapes: no int to size against, so take the default.
    if q_split_size is None or kv_split_size is None:
        return 16
    # A row block keeps two score tiles live: the previous one in bf16, the current in fp32.
    data_size = 2 + 4
    for rb in (32, 16):
        if rb * kv_split_size * data_size >= _l1d_bytes():
            continue
        if q_split_size >= 2 * rb:
            return rb
    return 16


def _softmax_ilv_pv():
    # Weave the TAIL row-block's exp pass into PV
    return int(os.environ.get("TORCHINDUCTOR_CPP_CPU_CAUSAL_ATTENTION_SOFTMAX_ILV_PV", "1"))


def _amx_b_nontemporal():
    # Load the B operand (packed K for QK, packed V for PV) with TILELOADDT1 instead of
    # TILELOADD
    return (
        os.environ.get("TORCHINDUCTOR_CPP_CPU_CAUSAL_ATTENTION_AMX_B_NONTEMPORAL", "1") == "1"
    )


def _amx_interleave_stores():
    # Retire each C tile right after its own last TMUL instead of bursting all four
    # tilestores at the end of the tile group.
    return os.environ.get("TORCHINDUCTOR_CPP_FLEX_AMX_INTERLEAVE_STORES", "1") == "1"


# ---------------------------------------------------------------------------
# Generated-C++ source blocks. Copied from cpp_flex_attention_template
# so this template has no dependency on the flash template
# ---------------------------------------------------------------------------

# TODO: reuse cpp codegen to generate below pointwise/reduction kernels
CAUSAL_SOFTMAX_FUSIONS = r"""
{%- if base2_softmax %}
// Base-2 softmax. The QK scale carries an extra log2(e) factor (see
// scaling_factor in INIT_PARAMS), so every score reaching softmax is already in
// base-2 units and exp(x) becomes exp2(x) with no per-element conversion.
{%- if exp2_deg3 %}
constexpr float {{kernel_name}}_exp2_c[] = {
    0.999928074f, 0.693260985f, 0.242611122f, 0.0551716691f};
{%- else %}
constexpr float {{kernel_name}}_exp2_c[] = {
    1.0f, 0.69314718f, 0.24022651f, 0.05550411f, 0.00961812f, 0.00133336f};
{%- endif %}

#if defined(CPU_CAPABILITY_AVX512)
inline __m512 {{kernel_name}}_exp2_ps(__m512 x) {
  const __m512 c0 = _mm512_set1_ps({{kernel_name}}_exp2_c[0]);
  const __m512 c1 = _mm512_set1_ps({{kernel_name}}_exp2_c[1]);
  const __m512 c2 = _mm512_set1_ps({{kernel_name}}_exp2_c[2]);
  const __m512 c3 = _mm512_set1_ps({{kernel_name}}_exp2_c[3]);
{%- if not exp2_deg3 %}
  const __m512 c4 = _mm512_set1_ps({{kernel_name}}_exp2_c[4]);
  const __m512 c5 = _mm512_set1_ps({{kernel_name}}_exp2_c[5]);
{%- endif %}
  // Flush what 2^x underflows to zero anyway; also maps -inf (fully masked
  // rows) to 0 instead of letting scalef see a NaN-producing exponent.
  const __m512 min_exp2 = _mm512_set1_ps(-126.0f);
  auto underflow = _mm512_cmp_ps_mask(x, min_exp2, _CMP_LT_OS);
  auto f = _mm512_reduce_ps(x, 0);          // f = x - round(x), |f| <= 0.5
  auto n = _mm512_sub_ps(x, f);             // exact integer
{%- if exp2_deg3 %}
  auto p = _mm512_fmadd_ps(f, c3, c2);
  p = _mm512_fmadd_ps(f, p, c1);
  p = _mm512_fmadd_ps(f, p, c0);
{%- else %}
  auto p = _mm512_fmadd_ps(f, c5, c4);
  p = _mm512_fmadd_ps(f, p, c3);
  p = _mm512_fmadd_ps(f, p, c2);
  p = _mm512_fmadd_ps(f, p, c1);
  p = _mm512_fmadd_ps(f, p, c0);
{%- endif %}
  return _mm512_mask_blend_ps(
      underflow, _mm512_scalef_ps(p, n), _mm512_setzero_ps());
}
#endif

inline float {{kernel_name}}_exp2_scalar(float x) {
  return std::exp2(x);
}

{%- if softmax_ilv %}
// Resumable exp pass, for weaving softmax into the AMX TMUL stream.
template <typename scalar_t, typename accum_t>
struct {{kernel_name}}_ExpCursor {
  const accum_t* row_max;  // per-row max (final, from the block max pass)
  accum_t* row_sum_out;    // per-row sum out
  int vecs_per_row;        // zmm steps per row = ceil(cols/16)
  int nsteps, idx;         // flat cursor over rows*vecs_per_row
  int cur_row, cur_vec;
  __m512 row_sums[32];     // one accumulator per row of a 32-row block
  __m512 cur_row_sum;

  inline void begin(accum_t* score, scalar_t* prob, int rows, int vecs_per_row_,
                    int64_t score_row_stride_, int64_t prob_row_stride_) {
    vecs_per_row = vecs_per_row_; idx = 0; nsteps = rows * vecs_per_row_;
    cur_row = 0; cur_vec = 0;
    score_ptr = score; prob_ptr = prob;
    score_next_row = score_row_stride_ - (int64_t)(vecs_per_row_ - 1) * 16;
    prob_next_row = prob_row_stride_ - (int64_t)(vecs_per_row_ - 1) * 16;
    cur_row_sum = _mm512_setzero_ps();
    for (int r = 0; r < rows; ++r) row_sums[r] = _mm512_setzero_ps();
  }
  accum_t* score_ptr;      // == begin()'s score + cur_row*score_row_stride + cur_vec*16
  scalar_t* prob_ptr;      // == begin()'s prob  + cur_row*prob_row_stride  + cur_vec*16
  int64_t score_next_row, prob_next_row;  // row-advance deltas, precomputed in begin()
  __attribute__((always_inline)) inline void step() {
    if (idx >= nsteps) return;
    const int row = cur_row;
    __m512 e = {{kernel_name}}_exp2_ps(
        _mm512_sub_ps(_mm512_loadu_ps(score_ptr), _mm512_set1_ps(row_max[row])));
    if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(prob_ptr), (__m256i)_mm512_cvtneps_pbh(e));
    } else {
      _mm512_storeu_ps(score_ptr, e);
      _mm512_storeu_ps(reinterpret_cast<float*>(prob_ptr), e);
    }
    cur_row_sum = _mm512_add_ps(cur_row_sum, e);
    ++idx;
    // Row complete: park the partial and reset.
    if (++cur_vec == vecs_per_row) {
      row_sums[cur_row] = cur_row_sum; cur_row_sum = _mm512_setzero_ps();
      cur_vec = 0; ++cur_row;
      score_ptr += score_next_row; prob_ptr += prob_next_row;
    } else {
      score_ptr += 16; prob_ptr += 16;
    }
  }
  __attribute__((always_inline)) inline void drain() { while (idx < nsteps) step(); }
  inline void finish(int rows) {
    // A partially-walked row
    if (cur_vec != 0 && cur_row < rows) row_sums[cur_row] = cur_row_sum;
    for (int r = 0; r < rows; ++r) row_sum_out[r] = _mm512_reduce_add_ps(row_sums[r]);
  }
};

// In place scale + row max for a block of rows
template <typename accum_t>
inline void {{kernel_name}}_scale_max_block(
    accum_t* score, int64_t score_row_stride, int rows, int cols, accum_t scale,
    accum_t* row_max) {
  const __m512 vs = _mm512_set1_ps(scale);
  for (int r = 0; r < rows; ++r) {
    accum_t* srow = score + (int64_t)r * score_row_stride;
    __m512 vmax = _mm512_set1_ps(-std::numeric_limits<accum_t>::infinity());
    int j = 0;
    for (; j + 16 <= cols; j += 16) {
      __m512 v = _mm512_mul_ps(_mm512_loadu_ps(srow + j), vs);
      _mm512_storeu_ps(srow + j, v);
      vmax = _mm512_max_ps(vmax, v);
    }
    accum_t m = _mm512_reduce_max_ps(vmax);
    for (; j < cols; ++j) { srow[j] *= scale; m = std::max(m, srow[j]); }
    row_max[r] = m;
  }
}
{%- endif %}

// 1) out = exp2(a - val)
// 2) val = sum(out)
template <typename T1, typename T2>
inline void {{kernel_name}}_exp_reduce_sum_fusion_kernel(
    T1* a,
    const int& size,
    T2* out,
    T1& val) {
  auto vec_size = at::vec::Vectorized<T1>::size();
  auto vec_max = at::vec::Vectorized<T1>(val);
  T1 tmp_sum = 0;
  auto vec_tmp_sum = at::vec::Vectorized<T1>(tmp_sum);
  for (long i = 0; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = at::vec::Vectorized<T1>::loadu(a + i);
    auto tmp1 = tmp0 - vec_max;
    auto tmp2 = at::vec::Vectorized<T1>({{kernel_name}}_exp2_ps(tmp1));
    vec_tmp_sum += tmp2;
    at::native::_store(out + i, tmp2);
  }
  tmp_sum = at::vec::vec_reduce_all<T1>(
      [](at::vec::Vectorized<T1>& x, at::vec::Vectorized<T1>& y) {
        return x + y;
      },
      vec_tmp_sum);
  for (long i = vec_size * (size / vec_size); i < size; i++) {
    auto tmp0 = a[i];
    auto tmp1 = tmp0 - val;
    auto tmp2 = {{kernel_name}}_exp2_scalar(tmp1);
    tmp_sum += tmp2;
    out[i] = tmp2;
  }
  val = tmp_sum;
}
{%- else %}
// 1) out = exp(a - val)
// 2) val = sum(out)
template <typename T1, typename T2>
inline void {{kernel_name}}_exp_reduce_sum_fusion_kernel(
    T1* a,
    const int& size,
    T2* out,
    T1& val) {
  auto vec_size = at::vec::Vectorized<T1>::size();
  auto vec_max = at::vec::Vectorized<T1>(val);
  T1 tmp_sum = 0;
  auto vec_tmp_sum = at::vec::Vectorized<T1>(tmp_sum);
  // fexp_u20 is 13 per-element vector ops against exp_u20's 18, and is what SDPA's
  // own _exp_reduce_sum_fusion_kernel picks for the same dtype pair. Its accuracy
  // argument is the one exp2_deg3 relies on: the result is rounded to bf16/fp16 for
  // the PV GEMM, so with an fp32 output there is no such rounding and exp_u20 stands.
  auto exp_vec = [](const at::vec::Vectorized<T1>& v) {
    if constexpr (std::is_same_v<T1, float> && c10::is_reduced_floating_point_v<T2>) {
      return v.fexp_u20();
    } else {
      return v.exp_u20();
    }
  };
  for (long i = 0; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = at::vec::Vectorized<T1>::loadu(a + i);
    auto tmp1 = tmp0 - vec_max;
    auto tmp2 = exp_vec(tmp1);
    vec_tmp_sum += tmp2;
    at::native::_store(out + i, tmp2);
  }
  tmp_sum = at::vec::vec_reduce_all<T1>(
      [](at::vec::Vectorized<T1>& x, at::vec::Vectorized<T1>& y) {
        return x + y;
      },
      vec_tmp_sum);
  for (long i = vec_size * (size / vec_size); i < size; i++) {
    auto tmp0 = a[i];
    auto tmp1 = tmp0 - val;
    auto tmp2 = exp(tmp1);
    tmp_sum += tmp2;
    out[i] = tmp2;
  }
  val = tmp_sum;
}
{%- endif %}

// 1) out = a * scale
// 2) max = max(out)
template <typename scalar_t>
inline void {{kernel_name}}_mul_reduce_max_fusion_kernel(
    const scalar_t* a,
    const scalar_t& scale,
    const int& size,
    scalar_t* out,
    scalar_t& max) {
  auto vec_size = at::vec::Vectorized<scalar_t>::size();
  auto vec_scale = at::vec::Vectorized<scalar_t>(scale);
  scalar_t tmp_max = -std::numeric_limits<scalar_t>::infinity();
  auto vec_tmp_max = at::vec::Vectorized<scalar_t>(tmp_max);
  for (long i = 0; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(a + i);
    auto tmp1 = tmp0 * vec_scale;
    vec_tmp_max = at::vec::maximum(vec_tmp_max, tmp1);
    at::native::_store(out + i, tmp1);
  }
  for (long i = vec_size * (size / vec_size); i < size; i++) {
    auto tmp0 = a[i];
    auto tmp1 = tmp0 * scale;
    tmp_max = std::max(tmp_max, tmp1);
    out[i] = tmp1;
  }
  max = std::max(
      tmp_max,
      at::vec::vec_reduce_all<scalar_t>(
          [](at::vec::Vectorized<scalar_t>& x, at::vec::Vectorized<scalar_t>& y) {
            return at::vec::maximum(x, y);
          },
          vec_tmp_max));
}

template <typename scalar_t>
static inline scalar_t* {{kernel_name}}_conditional_data_ptr(scalar_t* ptr, scalar_t* ptr2) {
  TORCH_CHECK(ptr2 == nullptr);
  return ptr;
}

template <typename scalar_t,
          typename std::enable_if_t<c10::is_reduced_floating_point_v<scalar_t>, int> = 0>
static inline scalar_t* {{kernel_name}}_conditional_data_ptr(float* ptr, scalar_t* ptr2) {
  return ptr2;
}

template <typename scalar_t>
inline void {{kernel_name}}_fill_stub(scalar_t* data, scalar_t val, int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  Vec data_vec = Vec(val);
  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    data_vec.store(data + d);
  }
  #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
  # pragma unroll
  #endif
  for (; d < size; d++) {
    data[d] = val;
  }
}

{%- if causal %}
// Causal mask for a DIAGONAL score block: row-local keep count `keep` (>= 0),
// everything from there on set to -inf so the softmax drops it.
// `keep0` is the keep count of the FIRST row;
template <typename accum_t>
inline void {{kernel_name}}_causal_mask_block(
    accum_t* score, int64_t score_row_stride, int64_t rows, int64_t cols, int64_t keep0) {
  const accum_t ninf = -std::numeric_limits<accum_t>::infinity();
  for (int64_t r = 0; r < rows; ++r) {
    int64_t keep = keep0 + r;
    if (keep < 0) keep = 0;
    if (keep >= cols) break;  // this row and every later one keeps everything
    {{kernel_name}}_fill_stub(score + r * score_row_stride + keep, ninf, cols - keep);
  }
}
{%- endif %}

{%- if need_mul_scale %}
// out = a * scale. Only reachable when the scale is not folded into the max pass.
template <typename scalar_t>
inline void {{kernel_name}}_mul_scale_kernel(
    scalar_t* a,
    scalar_t scale,
    int64_t size) {
  auto vec_size = at::vec::Vectorized<scalar_t>::size();
  auto vec_scale = at::vec::Vectorized<scalar_t>(scale);
  for (int64_t i = 0; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(a + i);
    auto tmp1 = tmp0 * vec_scale;
    at::native::_store(a + i, tmp1);
  }
  for (int64_t i = vec_size * (size / vec_size); i < size; i++) {
    auto tmp0 = a[i];
    auto tmp1 = tmp0 * scale;
    a[i] = tmp1;
  }
}
{%- endif %}

"""

# AMX bf16 GEMM at the 2x2 tile decomposition
# A row-major [M, K] (an A-tile row is 32 bf16 = 64 B), B VNNI-packed
# [K/2, N, 2] from at::vec::pack_vnni2, C row-major f32.
AMX_GEMM_FUNCTIONS = r"""
#if defined(CPU_CAPABILITY_AVX512)
#include <immintrin.h>

// _tile_* intrinsics take the tile index as an instruction immediate, so these
// must be macros -- an enum or const int fails to assemble.
#define {{kernel_name}}_TC00 0
#define {{kernel_name}}_TC01 1
#define {{kernel_name}}_TC10 2
#define {{kernel_name}}_TC11 3
#define {{kernel_name}}_TA0  4
#define {{kernel_name}}_TA1  5
#define {{kernel_name}}_TB0  6
#define {{kernel_name}}_TB1  7

// B-operand tile load. TILELOADDT1 (_tile_stream_loadd) is the NON-TEMPORAL form
{%- if amx_b_nontemporal %}
#define {{kernel_name}}_TILELOAD_B(dst, src, stride) _tile_stream_loadd(dst, src, stride)
{%- else %}
#define {{kernel_name}}_TILELOAD_B(dst, src, stride) _tile_loadd(dst, src, stride)
{%- endif %}

// A and C stay TEMPORAL
#define {{kernel_name}}_TILELOAD_A(dst, src, stride) _tile_loadd(dst, src, stride)
#define {{kernel_name}}_TILELOAD_C(dst, src, stride) _tile_loadd(dst, src, stride)

struct alignas(64) {{kernel_name}}_tilecfg_t {
  uint8_t palette_id;
  uint8_t start_row;
  uint8_t reserved[14];
  uint16_t colsb[16];
  uint8_t rows[16];
};

// One palette for the whole kernel: 8 tiles of 16 rows x 64 B, shared by both GEMMs.
//
// Note never issue a raw LDTILECFG without brgemm_release() first. The tile config
// is a per-thread hardware resource that oneDNN's brgemm also owns, and brgemm caches the
// config it installed in a thread_local, so our _tile_loadconfig is invisible to it.
inline void {{kernel_name}}_amx_tile_init() {
  at::native::cpublas::brgemm_release(/*is_vnni=*/true);
  {{kernel_name}}_tilecfg_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.palette_id = 1;
  for (int t = 0; t < 8; ++t) { cfg.colsb[t] = 64; cfg.rows[t] = 16; }
  _tile_loadconfig(&cfg);
}

// C[0:32, 0:32] (+)= A[0:32, 0:K] * Bp[0:K, 0:32], one 2x2 tile group.
// A: row-major bf16, row stride lda elements.
// Bp: VNNI [K/2, N, 2]; the tile for (k0, n0) starts at Bp + k0*N + n0*2 with a
//     row stride of 2*N elements, i.e. 4*N bytes.
// C: row-major f32, row stride ldc elements.
template <bool ACC>
inline void {{kernel_name}}_amx_gemm_32x32(
    const uint16_t* A, int64_t lda,
    const uint16_t* Bp, int64_t N,
    float* C, int64_t ldc, int64_t K) {
  const int64_t As = lda * (int64_t)sizeof(uint16_t);
  const int64_t Bs = 2 * N * (int64_t)sizeof(uint16_t);
  const int64_t Cs = ldc * (int64_t)sizeof(float);
  if (ACC) {
    {{kernel_name}}_TILELOAD_C({{kernel_name}}_TC00, C, Cs);
    {{kernel_name}}_TILELOAD_C({{kernel_name}}_TC01, C + 16, Cs);
    {{kernel_name}}_TILELOAD_C({{kernel_name}}_TC10, C + 16 * ldc, Cs);
    {{kernel_name}}_TILELOAD_C({{kernel_name}}_TC11, C + 16 * ldc + 16, Cs);
  } else {
    _tile_zero({{kernel_name}}_TC00);
    _tile_zero({{kernel_name}}_TC01);
    _tile_zero({{kernel_name}}_TC10);
    _tile_zero({{kernel_name}}_TC11);
  }
{%- if amx_interleave_stores %}
  const int64_t k_last = K - 32;
{%- endif %}
  for (int64_t k0 = 0; k0 < K; k0 += 32) {
    {{kernel_name}}_TILELOAD_A({{kernel_name}}_TA0, A + k0, As);
    {{kernel_name}}_TILELOAD_A({{kernel_name}}_TA1, A + 16 * lda + k0, As);
    {{kernel_name}}_TILELOAD_B({{kernel_name}}_TB0, Bp + k0 * N, Bs);
    {{kernel_name}}_TILELOAD_B({{kernel_name}}_TB1, Bp + k0 * N + 32, Bs);
{%- if amx_interleave_stores %}
    // On the LAST K-step each accumulator is final right after its own TMUL, so
    // retire it immediately.
    if (k0 == k_last) {
      _tile_dpbf16ps({{kernel_name}}_TC00, {{kernel_name}}_TA0, {{kernel_name}}_TB0);
      _tile_stored({{kernel_name}}_TC00, C, Cs);
      _tile_dpbf16ps({{kernel_name}}_TC01, {{kernel_name}}_TA0, {{kernel_name}}_TB1);
      _tile_stored({{kernel_name}}_TC01, C + 16, Cs);
      _tile_dpbf16ps({{kernel_name}}_TC10, {{kernel_name}}_TA1, {{kernel_name}}_TB0);
      _tile_stored({{kernel_name}}_TC10, C + 16 * ldc, Cs);
      _tile_dpbf16ps({{kernel_name}}_TC11, {{kernel_name}}_TA1, {{kernel_name}}_TB1);
      _tile_stored({{kernel_name}}_TC11, C + 16 * ldc + 16, Cs);
      return;
    }
{%- endif %}
    _tile_dpbf16ps({{kernel_name}}_TC00, {{kernel_name}}_TA0, {{kernel_name}}_TB0);
    _tile_dpbf16ps({{kernel_name}}_TC01, {{kernel_name}}_TA0, {{kernel_name}}_TB1);
    _tile_dpbf16ps({{kernel_name}}_TC10, {{kernel_name}}_TA1, {{kernel_name}}_TB0);
    _tile_dpbf16ps({{kernel_name}}_TC11, {{kernel_name}}_TA1, {{kernel_name}}_TB1);
  }
  _tile_stored({{kernel_name}}_TC00, C, Cs);
  _tile_stored({{kernel_name}}_TC01, C + 16, Cs);
  _tile_stored({{kernel_name}}_TC10, C + 16 * ldc, Cs);
  _tile_stored({{kernel_name}}_TC11, C + 16 * ldc + 16, Cs);
}

// Full [M, N] (+)= A[M, K] * Bp[K, N]. M and N must be multiples of 32 and K a
// multiple of 32; the caller checks this and falls back to brgemm otherwise.
template <bool ACC>
inline void {{kernel_name}}_amx_gemm(
    const uint16_t* A, int64_t lda,
    const uint16_t* Bp, int64_t N,
    float* C, int64_t ldc, int64_t M, int64_t K) {
  for (int64_t m0 = 0; m0 < M; m0 += 32)
    for (int64_t n0 = 0; n0 < N; n0 += 32)
      {{kernel_name}}_amx_gemm_32x32<ACC>(
          A + m0 * lda, lda, Bp + n0 * 2, N, C + m0 * ldc + n0, ldc, K);
}

{%- if softmax_ilv and softmax_ilv_rb == 32 %}
// 32x32 group with a CURSOR interleave between the individual TMULs
template <bool ACC, typename Cursor>
__attribute__((always_inline)) inline void {{kernel_name}}_amx_gemm_32x32_woven(
    const uint16_t* A, int64_t lda,
    const uint16_t* Bp, int64_t N,
    float* C, int64_t ldc, int64_t K, Cursor& cur) {
  const int64_t As = lda * (int64_t)sizeof(uint16_t);
  const int64_t Bs = 2 * N * (int64_t)sizeof(uint16_t);
  const int64_t Cs = ldc * (int64_t)sizeof(float);
  if (ACC) {
    {{kernel_name}}_TILELOAD_C({{kernel_name}}_TC00, C, Cs);
    {{kernel_name}}_TILELOAD_C({{kernel_name}}_TC01, C + 16, Cs);
    {{kernel_name}}_TILELOAD_C({{kernel_name}}_TC10, C + 16 * ldc, Cs);
    {{kernel_name}}_TILELOAD_C({{kernel_name}}_TC11, C + 16 * ldc + 16, Cs);
  } else {
    _tile_zero({{kernel_name}}_TC00);
    _tile_zero({{kernel_name}}_TC01);
    _tile_zero({{kernel_name}}_TC10);
    _tile_zero({{kernel_name}}_TC11);
  }
  for (int64_t k0 = 0; k0 < K; k0 += 32) {
    {{kernel_name}}_TILELOAD_A({{kernel_name}}_TA0, A + k0, As);
    {{kernel_name}}_TILELOAD_A({{kernel_name}}_TA1, A + 16 * lda + k0, As);
    {{kernel_name}}_TILELOAD_B({{kernel_name}}_TB0, Bp + k0 * N, Bs);
    {{kernel_name}}_TILELOAD_B({{kernel_name}}_TB1, Bp + k0 * N + 32, Bs);
    _tile_dpbf16ps({{kernel_name}}_TC00, {{kernel_name}}_TA0, {{kernel_name}}_TB0);
    cur.step(); cur.step(); cur.step();
    _tile_dpbf16ps({{kernel_name}}_TC01, {{kernel_name}}_TA0, {{kernel_name}}_TB1);
    cur.step(); cur.step(); cur.step();
    _tile_dpbf16ps({{kernel_name}}_TC10, {{kernel_name}}_TA1, {{kernel_name}}_TB0);
    cur.step(); cur.step(); cur.step();
    _tile_dpbf16ps({{kernel_name}}_TC11, {{kernel_name}}_TA1, {{kernel_name}}_TB1);
    cur.step(); cur.step();
  }
  _tile_stored({{kernel_name}}_TC00, C, Cs);
  _tile_stored({{kernel_name}}_TC01, C + 16, Cs);
  _tile_stored({{kernel_name}}_TC10, C + 16 * ldc, Cs);
  _tile_stored({{kernel_name}}_TC11, C + 16 * ldc + 16, Cs);
}

// One 32-row block, woven: the n0 sweep with a cursor threaded through.
template <bool ACC, typename Cursor>
__attribute__((always_inline)) inline void {{kernel_name}}_amx_gemm_rowblock_woven(
    const uint16_t* A, int64_t lda,
    const uint16_t* Bp, int64_t N,
    float* C, int64_t ldc, int64_t m0, int64_t K, Cursor& cur) {
  for (int64_t n0 = 0; n0 < N; n0 += 32)
    {{kernel_name}}_amx_gemm_32x32_woven<ACC>(
        A + m0 * lda, lda, Bp + n0 * 2, N, C + m0 * ldc + n0, ldc, K, cur);
}

{%- endif %}
{%- if softmax_ilv and softmax_ilv_rb == 16 %}
// ---- 16-row woven block (RB=16) ----
template <bool ACC, typename Cursor>
__attribute__((always_inline)) inline void {{kernel_name}}_amx_gemm_16x32_woven(
    const uint16_t* A, int64_t lda,
    const uint16_t* Bp, int64_t N,
    float* C, int64_t ldc, int64_t K, Cursor& cur) {
  const int64_t As = lda * (int64_t)sizeof(uint16_t);
  const int64_t Bs = 2 * N * (int64_t)sizeof(uint16_t);
  const int64_t Cs = ldc * (int64_t)sizeof(float);
  if (ACC) {
    {{kernel_name}}_TILELOAD_C({{kernel_name}}_TC00, C, Cs);
    {{kernel_name}}_TILELOAD_C({{kernel_name}}_TC01, C + 16, Cs);
  } else {
    _tile_zero({{kernel_name}}_TC00);
    _tile_zero({{kernel_name}}_TC01);
  }
  for (int64_t k0 = 0; k0 < K; k0 += 32) {
    {{kernel_name}}_TILELOAD_A({{kernel_name}}_TA0, A + k0, As);
    {{kernel_name}}_TILELOAD_B({{kernel_name}}_TB0, Bp + k0 * N, Bs);
    {{kernel_name}}_TILELOAD_B({{kernel_name}}_TB1, Bp + k0 * N + 32, Bs);
    _tile_dpbf16ps({{kernel_name}}_TC00, {{kernel_name}}_TA0, {{kernel_name}}_TB0);
    cur.step(); cur.step(); cur.step();
    _tile_dpbf16ps({{kernel_name}}_TC01, {{kernel_name}}_TA0, {{kernel_name}}_TB1);
    cur.step(); cur.step(); cur.step();
  }
  _tile_stored({{kernel_name}}_TC00, C, Cs);
  _tile_stored({{kernel_name}}_TC01, C + 16, Cs);
}

// 16-row woven block: the n0 sweep, 32 columns at a time.
template <bool ACC, typename Cursor>
__attribute__((always_inline)) inline void {{kernel_name}}_amx_gemm_rowblock16_woven(
    const uint16_t* A, int64_t lda,
    const uint16_t* Bp, int64_t N,
    float* C, int64_t ldc, int64_t m0, int64_t K, Cursor& cur) {
  for (int64_t n0 = 0; n0 < N; n0 += 32)
    {{kernel_name}}_amx_gemm_16x32_woven<ACC>(
        A + m0 * lda, lda, Bp + n0 * 2, N, C + m0 * ldc + n0, ldc, K, cur);
}

// Non-woven 16-row block, for the FIRST row-block (which has no predecessor's
// softmax to weave in). Same tile sequence, no cursor.
template <bool ACC>
inline void {{kernel_name}}_amx_gemm_rowblock16(
    const uint16_t* A, int64_t lda,
    const uint16_t* Bp, int64_t N,
    float* C, int64_t ldc, int64_t m0, int64_t K) {
  const int64_t As = lda * (int64_t)sizeof(uint16_t);
  const int64_t Bs = 2 * N * (int64_t)sizeof(uint16_t);
  const int64_t Cs = ldc * (int64_t)sizeof(float);
  for (int64_t n0 = 0; n0 < N; n0 += 32) {
    const uint16_t* Ab = A + m0 * lda;
    const uint16_t* Bb = Bp + n0 * 2;
    float* Cb = C + m0 * ldc + n0;
    if (ACC) {
      {{kernel_name}}_TILELOAD_C({{kernel_name}}_TC00, Cb, Cs);
      {{kernel_name}}_TILELOAD_C({{kernel_name}}_TC01, Cb + 16, Cs);
    } else {
      _tile_zero({{kernel_name}}_TC00);
      _tile_zero({{kernel_name}}_TC01);
    }
    for (int64_t k0 = 0; k0 < K; k0 += 32) {
      {{kernel_name}}_TILELOAD_A({{kernel_name}}_TA0, Ab + k0, As);
      {{kernel_name}}_TILELOAD_B({{kernel_name}}_TB0, Bb + k0 * N, Bs);
      {{kernel_name}}_TILELOAD_B({{kernel_name}}_TB1, Bb + k0 * N + 32, Bs);
      _tile_dpbf16ps({{kernel_name}}_TC00, {{kernel_name}}_TA0, {{kernel_name}}_TB0);
      _tile_dpbf16ps({{kernel_name}}_TC01, {{kernel_name}}_TA0, {{kernel_name}}_TB1);
    }
    _tile_stored({{kernel_name}}_TC00, Cb, Cs);
    _tile_stored({{kernel_name}}_TC01, Cb + 16, Cs);
  }
}

{%- endif %}
{%- if softmax_ilv and softmax_ilv_rb == 32 %}
// One 32-ROW BLOCK of the above: C[m0:m0+32, 0:N] (+)= A[m0:m0+32, :] * Bp.
// Exists so the caller can interleave other work (the previous row-block's softmax)
// between the row-blocks of a GEMM
template <bool ACC>
inline void {{kernel_name}}_amx_gemm_rowblock(
    const uint16_t* A, int64_t lda,
    const uint16_t* Bp, int64_t N,
    float* C, int64_t ldc, int64_t m0, int64_t K) {
  for (int64_t n0 = 0; n0 < N; n0 += 32)
    {{kernel_name}}_amx_gemm_32x32<ACC>(
        A + m0 * lda, lda, Bp + n0 * 2, N, C + m0 * ldc + n0, ldc, K);
}

{%- endif %}
// The 2x2 group needs M, N multiples of 32 and K a multiple of 32 (the VNNI tile
// depth). Everything else keeps using brgemm.
inline bool {{kernel_name}}_amx_gemm_applicable(int64_t M, int64_t N, int64_t K) {
  return (M % 32 == 0) && (N % 32 == 0) && (K % 32 == 0) && M > 0 && N > 0 && K > 0;
}
#endif  // CPU_CAPABILITY_AVX512
"""

CAUSAL_BRGEMM_PACK_FUNCTIONS = r"""
template <typename scalar_t>
inline void {{kernel_name}}_copy_value_with_pad(
    const scalar_t* value_ptr,
    scalar_t* dst_ptr,
    int64_t rows,
    int64_t cols,
    int64_t prows,
    int64_t pcols,
    int64_t ldi) {
  auto vec_size = at::vec::Vectorized<scalar_t>::size();
  int64_t i = 0;
  for (; i < rows; i++) {
    int64_t j = 0;
    for (; j < cols - (cols % vec_size); j += vec_size) {
      auto vec_v =
          at::vec::Vectorized<scalar_t>::loadu(value_ptr + i * ldi + j);
      vec_v.store(dst_ptr + i * pcols + j);
    }

    if (j < cols) {
      auto vec_v = at::vec::Vectorized<scalar_t>::loadu(
          value_ptr + i * ldi + j, cols - j);
      vec_v.store(dst_ptr + i * pcols + j, cols - j);
    }

    // col padding
    auto psize = pcols - cols;
    if (psize > 0) {
      auto zero_vec = at::vec::Vectorized<scalar_t>(0);
      int64_t pj = 0;
      for (; pj < psize - (psize % vec_size); pj += vec_size) {
        zero_vec.store(dst_ptr + i * pcols + cols + pj);
      }
      if (pj < psize) {
        zero_vec.store(dst_ptr + i * pcols + cols + pj, psize - pj);
      }
    }
  }
  // row padding
  for (; i < prows; i++) {
    auto zero_vec = at::vec::Vectorized<scalar_t>(0);
    int64_t j = 0;
    for (; j < pcols - (pcols % vec_size); j += vec_size) {
      zero_vec.store(dst_ptr + i * pcols + j);
    }
    if (j < pcols) {
      zero_vec.store(dst_ptr + i * pcols + j, pcols - j);
    }

  }
}

// K^T GEMM for the !need_pack BLAS QK path: brgemm cannot transpose its B operand, so K
// is transposed once per (head, kv-block) in the head block pre-pass and reused by every
// q-block.
template <typename scalar_t>
inline void {{kernel_name}}_transpose_kv(
    int64_t M,
    int64_t N,
    const scalar_t* src,
    int64_t ld_src,
    scalar_t* dst,
    int64_t ld_dst) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    at::native::utils::transpose<float>(M, N, src, ld_src, dst, ld_dst);
  } else {
    at::native::utils::transpose<uint16_t>(
        M, N, reinterpret_cast<const uint16_t*>(src), ld_src,
        reinterpret_cast<uint16_t*>(dst), ld_dst);
  }
}
"""

CAUSAL_ALLOCATE_BUFFER = r"""
  int64_t {{buffer_name}}_dtype_itemsize = c10::is_reduced_floating_point_v<{{buffer_dtype}}> ? 2 : 4;
  auto& {{buffer_name}}_allocator = *at::getCPUAllocator();
  auto {{buffer_name}}_work_data = {{buffer_name}}_allocator.allocate({{buffer_size}}*{{buffer_name}}_dtype_itemsize);
  void* {{buffer_name}}_data_ptr = {{buffer_name}}_work_data.get();
  {{buffer_dtype}}* {{buffer_name}} = ({{buffer_dtype}}*){{buffer_name}}_data_ptr;
"""


ANALYTIC_L3_TEMPLATE = r"""
  // ---- Analytic KV-chunk sizing ----
  int64_t l3_budget_bytes = (int64_t)({{l3_bytes}} * (double){{l3_fraction}});
  int64_t bytes_per_kv_token = (eheadSize + headSize_v) * (int64_t)sizeof(scalar_t);

  int64_t total_head_instances = batchSize * num_head;
  int64_t heads_in_flight = {{jkv_chunk_override}};

  // K/V are packed once per KV head
  if (heads_in_flight <= 0) {
    int64_t kv_bytes_per_head = kvSize * bytes_per_kv_token;
    heads_in_flight = kv_bytes_per_head > 0
        ? q_heads_per_kv_head * (l3_budget_bytes / kv_bytes_per_head)
        : total_head_instances;
    // Clamp to one whole KV group
    if (heads_in_flight < q_heads_per_kv_head) heads_in_flight = q_heads_per_kv_head;
    // Never fewer q-block groups than threads - parallelism issue
    int64_t min_heads_in_flight = (num_thread + qSlice - 1) / qSlice;
    // Set to minimum required
    if (heads_in_flight < min_heads_in_flight) heads_in_flight = min_heads_in_flight;
  }

  if (heads_in_flight > total_head_instances) heads_in_flight = total_head_instances;
  // Divide the head count evenly so the LAST head block is not a straggler
  {
    int64_t head_block_count_estimate = (total_head_instances + heads_in_flight - 1) / heads_in_flight;
    heads_in_flight = (total_head_instances + head_block_count_estimate - 1) / head_block_count_estimate;
  }
  // Restore the alignment invariant, rounding UP so the block count does not grow.
  heads_in_flight = ((heads_in_flight + q_heads_per_kv_head - 1) / q_heads_per_kv_head)
      * q_heads_per_kv_head;
  if (heads_in_flight > total_head_instances) heads_in_flight = total_head_instances;
  int64_t kv_heads_in_flight = heads_in_flight / q_heads_per_kv_head;

  // KV chunk size. Round down to accommodate the whole KV blocks so a chunk never splits one.
  int64_t kv_chunk_tokens = {{nkv_override}};
  if (kv_chunk_tokens <= 0) {
    int64_t budget_kv_tokens = l3_budget_bytes / (kv_heads_in_flight * bytes_per_kv_token);
    int64_t budget_kv_blocks = budget_kv_tokens / kvSplitSize;
    if (budget_kv_blocks < 1) budget_kv_blocks = 1;
    kv_chunk_tokens = budget_kv_blocks * kvSplitSize;
  }

  kv_chunk_tokens = (kv_chunk_tokens / kvSplitSize) * kvSplitSize;
  // Clamping
  if (kv_chunk_tokens < kvSplitSize) kv_chunk_tokens = kvSplitSize;
  if (kv_chunk_tokens > kvSize) kv_chunk_tokens = kvSize;
  int64_t kv_chunk_count = (kvSize + kv_chunk_tokens - 1) / kv_chunk_tokens;
"""


ANALYTIC_L2_GROUP_TEMPLATE = r"""
  // ---- Analytic L2 group sizing ----
  int64_t l2_budget_bytes = (int64_t)({{l2_bytes}} * (double){{l2_fraction}});

  // The size that one q-group occupies: the Q rows of each q-block plus an O/max/sum that must survive
  // every KV step and every chunk
  int64_t l2_bytes_per_qblock = qSplitSize * (eheadSize * (int64_t)sizeof(scalar_t)
      + headSize_v * (int64_t)sizeof(accum_t) + 2 * (int64_t)sizeof(accum_t));

  // The size of one step of K,V and one block of score
  int64_t l2_bytes_shared_by_group =
      kvSplitSize * (eheadSize + headSize_v) * (int64_t)sizeof(scalar_t)
      + qSplitSize * ekvSplitSize * (int64_t)(sizeof(accum_t) + sizeof(scalar_t));
  int64_t qblocks_per_kv_step = {{l2_group_override}};

  // ---Heuristics---
  // Clamping
  if (qblocks_per_kv_step <= 0) {
    qblocks_per_kv_step = (l2_budget_bytes - l2_bytes_shared_by_group) / l2_bytes_per_qblock;
    if (qblocks_per_kv_step < 1) qblocks_per_kv_step = 1;
  }
  if (qblocks_per_kv_step > qSlice) qblocks_per_kv_step = qSlice;

  // Guarantee parallelism - do not starve the threads
  while (qblocks_per_kv_step > 1 && heads_in_flight * ((qSlice + qblocks_per_kv_step - 1) / qblocks_per_kv_step) < num_thread) {
    qblocks_per_kv_step--;
  }

  // Balance the last wave for the parallel for. Shrink until the last wave is >=80% packed
  while (qblocks_per_kv_step > 1) {
    int64_t parallel_item_count = heads_in_flight * ((qSlice + qblocks_per_kv_step - 1) / qblocks_per_kv_step);
    int64_t barrier_wave_count = (parallel_item_count + num_thread - 1) / num_thread;
    if (parallel_item_count * 5 >= barrier_wave_count * num_thread * 4) break;
    qblocks_per_kv_step--;
  }

{%- if causal_balance %}
  // causal only - needs >= 2*num_thread items so the work queue has spare work for workload balancing
  while (qblocks_per_kv_step > 2 && heads_in_flight * ((qSlice + qblocks_per_kv_step - 1) / qblocks_per_kv_step) < 2 * num_thread) {
    qblocks_per_kv_step--;
  }
{%- endif %}
  int64_t qblock_group_count = (qSlice + qblocks_per_kv_step - 1) / qblocks_per_kv_step;
"""

PACK_KV_TEMPLATE = r"""
{#- Spliced into the head-block loop of CAUSAL_ATTENTION_TEMPLATE, so it shares that scope.
    CONSUMES  need_pack, qk_blas, kv_head_block_size, head_block_first_instance,
              kvSlice / kvSplitSize / kvSize, headSize / eheadSize / headSize_v,
              key_transpose_scratch, k_data / v_data and their strides, the broadcast
              flags, padded_kv_token_count, q_heads_per_kv_head
    WRITES    packed_key_block, packed_value_block  (one slot per KV HEAD)
    DECLARES  nothing that outlives it -- the parallel_for lambda owns every local #}
    // ---- Prepare this head block's K/V ----
    // Iterates KV HEADS
    if (need_pack || qk_blas) {
      at::parallel_for(0, kv_head_block_size * kvSlice, 1, [&](int64_t begin, int64_t end) {
        int thread_idx = at::get_thread_num();
        scalar_t* key_transpose_tile = key_transpose_scratch + thread_idx * kvSplitSize * headSize;
        int64_t kv_in_block = 0, kv_block_idx = 0;
        at::native::data_index_init(begin, kv_in_block, kv_head_block_size, kv_block_idx, kvSlice);
        for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
          // kv_instance = head_instance / q_heads_per_kv_head is EXACT:
          // head_instance = b*num_head + h and num_head = num_head_k*rep, so
          // head_instance/rep == b*num_head_k + h/rep. The alignment invariant on
          // heads_in_flight guarantees head_block_first_instance is a multiple of rep,
          // so no head block ever splits a KV group.
          int64_t kv_instance = head_block_first_instance / q_heads_per_kv_head + kv_in_block;
          int64_t batch_idx = kv_instance / num_head_k;
          int64_t head_idx_kv = kv_instance % num_head_k;
          int64_t batch_idx_kv = is_broadcast_bs_kv ? batch_idx / q_batch_per_kv_batch : batch_idx;
          int64_t kv_step_begin = kv_block_idx * kvSplitSize;
          int64_t cur_kvSplitSize = std::min(kvSplitSize, kvSize - kv_step_begin);
          auto k_addr = k_data + batch_idx_kv * kStrideB + head_idx_kv * kStrideH + kv_step_begin * kStrideN;
          auto v_addr = v_data + batch_idx_kv * vStrideB + head_idx_kv * vStrideH + kv_step_begin * vStrideN;
          if (need_pack) {
            // ---- ONE pass K Pack - scatter into the VNNI2 layout ----
            // pack_vnni2(src[K][N], dst, ld_src, K, N) sends source element (k, n) to
            //     dst[(k & ~1) * N + 2*n + (k & 1)]
            const bool fused_k_ok = (headSize % 2 == 0) && (cur_kvSplitSize % 32 == 0)
                && ((reinterpret_cast<uintptr_t>(k_addr) & 3) == 0) && (kStrideN % 2 == 0);
            if (fused_k_ok) {
              uint32_t* __restrict__ dstU = reinterpret_cast<uint32_t*>(
                  packed_key_block + kv_in_block * eheadSize * kvSize + kv_step_begin * eheadSize);
              const int64_t hj = headSize / 2;         // uint32 per token row
              const int64_t ldU = cur_kvSplitSize;     // uint32 per k-pair row
              for (int64_t n0 = 0; n0 < cur_kvSplitSize; n0 += 16) {
                for (int64_t n = n0; n < n0 + 16; ++n) {
                  const uint32_t* __restrict__ srcU =
                      reinterpret_cast<const uint32_t*>(k_addr + n * kStrideN);
                  for (int64_t j = 0; j < hj; ++j) {
                    dstU[j * ldU + n] = srcU[j];
                  }
                }
              }
            } else {
              // ORIGINAL two-pass path: ragged tail (pack_vnni2 zero-pads it), odd
              // headSize, or an unaligned operand.
              at::native::utils::transpose<uint16_t>(
                  cur_kvSplitSize, headSize,
                  reinterpret_cast<const uint16_t*>(k_addr), kStrideN,
                  reinterpret_cast<uint16_t*>(key_transpose_tile), cur_kvSplitSize);
              at::vec::pack_vnni2(
                  reinterpret_cast<const uint16_t*>(key_transpose_tile),
                  reinterpret_cast<uint16_t*>(packed_key_block + kv_in_block * eheadSize * kvSize + kv_step_begin * eheadSize),
                  cur_kvSplitSize, headSize, cur_kvSplitSize);
            }
            at::vec::pack_vnni2(
                reinterpret_cast<const uint16_t*>(v_addr),
                reinterpret_cast<uint16_t*>(packed_value_block + kv_in_block * padded_kv_token_count * headSize_v +
                    kv_step_begin * headSize_v),
                vStrideN, cur_kvSplitSize, headSize_v);
          } else {
            {{kernel_name}}_transpose_kv<scalar_t>(
                cur_kvSplitSize, headSize, k_addr, kStrideN,
                packed_key_block + kv_in_block * eheadSize * kvSize + kv_step_begin * eheadSize,
                cur_kvSplitSize);
          }
          at::native::data_index_step(kv_in_block, kv_head_block_size, kv_block_idx, kvSlice);
        }
      });
    }
"""


ONLINE_SOFTMAX_TEMPLATE = r"""
{#- Spliced into the (q-block, KV-step) body of CAUSAL_ATTENTION_TEMPLATE, so it shares
    that scope and captures its locals by reference.
    CONSUMES  qk_data, qk_reduced_data, cur_kvSplitSize, cur_ekvSplitSize, need_pack,
              block_row_max, block_row_sum, output_accum_rows, output_accum_unwritten,
              scaling_factor, headSize_v
    DECLARES  softmax_rows(), qk_was_woven, and under softmax_ilv also woven_row_max[32],
              woven_row_sum[32], commit_woven_row_block() and (under softmax_ilv_pv)
              pending_exp_row_block / pending_exp_row_count -- ALL read by the QK and PV
              sites below it, which is a wider contract than either ANALYTIC_* block #}
            // ---- Online softmax ----
            // A lambda over a row range so the softmax_ilv schedule can run
            // it in 32-row slices interleaved into the QK TMUL stream. Every step
            // below is per-row with no cross-row state (the running max/sum and the
            // O rescale are all indexed by `row`)
            accum_t tmp_max = 0, tmp_sum = 0, exp_tmp = 0;
            auto softmax_rows = [&](int64_t row_beg, int64_t row_end)
                __attribute__((always_inline)) {
            for (int64_t row = row_beg; row < row_end; ++row) {
              // The scale rides along in the max pass (which already multiplies by
              // its `scale` argument), so the separate mul_scale pass over the
              // whole score tile is skipped -- one fewer full RMW of 64 KiB.
              {{kernel_name}}_mul_reduce_max_fusion_kernel(
                  qk_data + row * cur_kvSplitSize,
{%- if softmax_fuse_scale %}
                  scaling_factor, cur_kvSplitSize,
{%- else %}
                  static_cast<accum_t>(1), cur_kvSplitSize,
{%- endif %}
                  qk_data + row * cur_kvSplitSize, tmp_max);
              const accum_t prev_row_max = block_row_max[row];
              tmp_max = prev_row_max > tmp_max ? prev_row_max : tmp_max;
              // Lazy rescale (see rescale_skip_thresh): keeping the carried max makes
              // exp_tmp exactly 1 below, so the O rescale is skipped for this step.
              if (!output_accum_unwritten && rescale_skip_thresh > accum_t(0) &&
                  prev_row_max != -std::numeric_limits<accum_t>::infinity() &&
                  tmp_max - prev_row_max <= rescale_skip_thresh) {
                tmp_max = prev_row_max;
              }
              if (tmp_max == -std::numeric_limits<accum_t>::infinity()) {
                {{kernel_name}}_fill_stub(
                    {{kernel_name}}_conditional_data_ptr(qk_data, qk_reduced_data) + row * cur_ekvSplitSize,
                    static_cast<scalar_t>(0), cur_kvSplitSize);
              } else {
                tmp_sum = tmp_max;
                {{kernel_name}}_exp_reduce_sum_fusion_kernel(
                    qk_data + row * cur_kvSplitSize, cur_kvSplitSize,
                    {{kernel_name}}_conditional_data_ptr(qk_data, qk_reduced_data) + row * cur_ekvSplitSize,
                    tmp_sum);
{%- if base2_softmax %}
                // block_row_max is in base-2 units (log2(e) folded into the QK
                // scale), so the running-max rescale must be exp2 too.
                exp_tmp = std::exp2(block_row_max[row] - tmp_max);
{%- else %}
                exp_tmp = std::exp(block_row_max[row] - tmp_max);
{%- endif %}
                block_row_sum[row] = tmp_sum + exp_tmp * block_row_sum[row];
                block_row_max[row] = tmp_max;
                // Rescale the carried O. Unlike the flash template's
                // `kv_step_idx > 0`, this must also apply on the first block of a
                // next chunk, where O/max/sum are non-trivial from the previous
                // chunk
                if (!output_accum_unwritten && exp_tmp != accum_t(1)) {
                  at::vec::map<accum_t>(
                      [exp_tmp](Vec x) { return x * Vec(exp_tmp); },
                      output_accum_rows + row * headSize_v,
                      output_accum_rows + row * headSize_v,
                      headSize_v);
                }
              }
              if (need_pack && cur_kvSplitSize % 2 != 0) {
                *(qk_reduced_data + row * (1 + cur_kvSplitSize) + cur_kvSplitSize) = scalar_t(0);
              }
            }
            };
            bool qk_was_woven = false;
{%- if softmax_ilv %}
            // Scratch for the woven schedule: per-row max and sum of ONE row-block.
            alignas(64) accum_t woven_row_max[32];
            alignas(64) accum_t woven_row_sum[32];
{%- if softmax_ilv_pv %}
            int64_t pending_exp_row_block = -1;
            int pending_exp_row_count = 0;
{%- endif %}
            auto commit_woven_row_block = [&](int64_t row0, int rows) __attribute__((always_inline)) {
              for (int r = 0; r < rows; ++r) {
                const int64_t row = row0 + r;
                const accum_t new_max = woven_row_max[r];
                if (new_max == -std::numeric_limits<accum_t>::infinity()) {
                  {{kernel_name}}_fill_stub(
                      {{kernel_name}}_conditional_data_ptr(qk_data, qk_reduced_data)
                          + row * cur_ekvSplitSize,
                      static_cast<scalar_t>(0), cur_kvSplitSize);
                  continue;
                }
{%- if base2_softmax %}
                const accum_t carry = std::exp2(block_row_max[row] - new_max);
{%- else %}
                const accum_t carry = std::exp(block_row_max[row] - new_max);
{%- endif %}
                block_row_sum[row] = woven_row_sum[r] + carry * block_row_sum[row];
                block_row_max[row] = new_max;
                if (!output_accum_unwritten && carry != accum_t(1)) {
                  at::vec::map<accum_t>(
                      [carry](Vec x) { return x * Vec(carry); },
                      output_accum_rows + row * headSize_v,
                      output_accum_rows + row * headSize_v, headSize_v);
                }
                if (need_pack && cur_kvSplitSize % 2 != 0) {
                  *(qk_reduced_data + row * (1 + cur_kvSplitSize) + cur_kvSplitSize) = scalar_t(0);
                }
              }
            };
{%- endif %}
"""


INIT_PARAMS = r"""
{{template.header().getvalue()}}
#include <ATen/native/cpu/utils.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/Context.h>
#include <cstdio>
#include <cstdlib>
{%- if causal_balance %}
{#- Only the balanced schedule needs these, and emitting them unconditionally would
    change the DENSE kernel's text -- byte-identity of the dense codegen is the
    cheapest proof this file's causal work cannot touch the dense path. #}
#include <algorithm>
#include <atomic>
#include <vector>
{%- endif %}
{{template.codegen_micro_gemm(kernel.kernel_name)}}
{{template.codegen_softmax_fusion(kernel.kernel_name, softmax_ilv, need_mul_scale)}}
{{template.codegen_brgemm_pack_function(kernel.kernel_name)}}
{%- if amx_gemm %}
{{template.codegen_amx_gemm(kernel.kernel_name, softmax_ilv, softmax_ilv_rb)}}
{%- endif %}
{#- The block mask is part of the lowering's signature, so its buffers stay in the
    kernel's parameter list, but nothing below reads them: the KV loop steps a
    contiguous [chunk_kv_begin, chunk_kv_end) and never indexes kv_indices. #}
{%- set kernel_args = {"query": query, "key": key, "value": value,
                       "kv_num_blocks": kv_num_blocks, "kv_indices": kv_indices,
                       "full_kv_num_blocks": full_kv_num_blocks, "full_kv_indices": full_kv_indices } %}

extern "C"
{{kernel.def_kernel(inputs=kernel_args, outputs={"output": output})}}
{
  {{ kernel.maybe_codegen_profile() }}

  // dtypes
  using scalar_t = {{kernel.dtype(query)}};
  constexpr bool is_reduced_type = c10::is_reduced_floating_point_v<scalar_t>;
  using accum_t = at::opmath_type<{{kernel.dtype(query)}}>;
  using Vec = at::vec::Vectorized<accum_t>;
{%- if base2_softmax %}
  // log2(e) folded into the QK scale so softmax runs in base 2 (exp -> exp2).
  // Free: this multiply already existed. Softmax is invariant to it because a
  // positive rescale of all logits commutes with max and cancels in the
  // normalization, so 2^(s-m)/sum(2^(s-m)) == e^(s'-m')/sum(e^(s'-m')).
  accum_t scaling_factor = accum_t({{scale}}) * accum_t(M_LOG2E);
{%- else %}
  accum_t scaling_factor = {{scale}};
{%- endif %}

  // sizes
  int64_t qBlockSize = {{qBlockSize}};
  int64_t kvBlockSize = {{kvBlockSize}};
  int64_t num_thread = {{num_thread}};
  int64_t batchSize = {{kernel.size(query, 0)}};
  int64_t qSize = {{kernel.size(query, 1)}};
  int64_t num_head = {{kernel.size(query, 2)}};
  int64_t headSize = {{kernel.size(query, 3)}};
  int64_t batchSize_k = {{kernel.size(key, 0)}};
  int64_t num_head_k = {{kernel.size(key, 2)}};
  int64_t headSize_v = {{kernel.size(value, 3)}};
  bool is_broadcast_bs_kv = batchSize != batchSize_k;
  bool is_broadcast_head_kv = num_head != num_head_k;
  int64_t q_heads_per_kv_head = num_head / num_head_k;
  int64_t q_batch_per_kv_batch = batchSize / batchSize_k;

  int64_t kvSize = {{kernel.size(key, 1)}};

{%- if causal %}
  // Causal: q row m attends to kv [0, m + causal_offset].
  const int64_t causal_offset = {{causal_offset}};
{%- endif %}

  int64_t qSplitSize = qBlockSize;
  int64_t kvSplitSize = kvBlockSize;

  qSplitSize = qSplitSize > qSize ? qSize : qSplitSize;
  kvSplitSize = kvSplitSize > kvSize ? kvSize : kvSplitSize;
  int64_t qSlice = (qSize + qSplitSize - 1) / qSplitSize;
  int64_t kvSlice = (kvSize + kvSplitSize - 1) / kvSplitSize;
  int64_t kvTail = (kvSize - 1) % kvSplitSize + 1;

  // Strides
  int64_t qStrideB = {{kernel.stride(query, 0)}};
  int64_t qStrideM = {{kernel.stride(query, 1)}};
  int64_t qStrideH = {{kernel.stride(query, 2)}};
  int64_t kStrideB = {{kernel.stride(key, 0)}};
  int64_t kStrideN = {{kernel.stride(key, 1)}};
  int64_t kStrideH = {{kernel.stride(key, 2)}};
  int64_t vStrideB = {{kernel.stride(value, 0)}};
  int64_t vStrideN = {{kernel.stride(value, 1)}};
  int64_t vStrideH = {{kernel.stride(value, 2)}};
  int64_t oStrideB = {{kernel.stride(output, 0)}};
  int64_t oStrideM = {{kernel.stride(output, 2)}};
  int64_t oStrideH = {{kernel.stride(output, 1)}};

  // Inputs/outputs buffers
  const scalar_t* q_data = query;
  const scalar_t* k_data = key;
  const scalar_t* v_data = value;
  scalar_t* out_data = output;

"""

CAUSAL_ATTENTION_TEMPLATE = r"""
  // init
  kvSplitSize = {{kv_split_size}};
  if (kvSplitSize > kvSize) kvSplitSize = kvSize;
  kvSlice = (kvSize + kvSplitSize - 1) / kvSplitSize;
  kvTail = (kvSize - 1) % kvSplitSize + 1;

{%- if q_split_size %}
  qSplitSize = {{q_split_size}};
  if (qSplitSize > qSize) qSplitSize = qSize;
  qSlice = (qSize + qSplitSize - 1) / qSplitSize;
{%- endif %}

  bool need_pack = false;
{%- if vnni_pack %}
  if (is_reduced_type) {
    need_pack = std::is_same_v<scalar_t, at::BFloat16> ? at::native::cpublas::could_pack(at::kBFloat16)
                                                       : at::native::cpublas::could_pack(at::kHalf);
  }
{%- endif %}

  // QK^T through cpublas::gemm instead of the micro_gemm_transpose_b microkernel.
  const bool qk_blas = {{ 1 if qk_blas else 0 }} && !need_pack;

  // Pad is needed for packing when K is odd.
  bool headSize_even = headSize % 2 == 0;
  int64_t eheadSize = need_pack && !headSize_even ? headSize + 1 : headSize;
  int64_t ekvSplitSize = need_pack && (kvSplitSize % 2 != 0) ? kvSplitSize + 1 : kvSplitSize;
  int64_t ekvTail = need_pack && (kvTail % 2 != 0) ? kvTail + 1 : kvTail;
  int64_t padded_kv_token_count = (kvSize - 1) / kvSplitSize * ekvSplitSize + ekvTail;

{{template.codegen_l3_chunk_sizing()}}

{{template.codegen_l2_group_sizing(causal_balance)}}
{%- if causal_balance %}
  // For calculating the cost
  std::vector<int64_t> group_tile_cost(qblock_group_count);
  std::vector<int32_t> groups_by_cost_desc(qblock_group_count);
{%- endif %}

  // ---- Init buffers ----
  bool single_chunk = (kv_chunk_count == 1);
  const accum_t rescale_skip_thresh = []() -> accum_t {
    const char* s = std::getenv("TORCHINDUCTOR_CPP_CPU_CAUSAL_ATTENTION_RESCALE_SKIP");
    return s ? static_cast<accum_t>(std::atof(s)) : static_cast<accum_t>(0);
  }();
  int64_t head_block_row_count = single_chunk
      ? num_thread * qblocks_per_kv_step * qSplitSize
      : heads_in_flight * qSize;
  {{template.codegen_allocate_buffer("output_accum", "accum_t", "head_block_row_count*headSize_v")}}
  {{template.codegen_allocate_buffer("running_row_max", "accum_t", "head_block_row_count")}}
  {{template.codegen_allocate_buffer("running_row_sum", "accum_t", "head_block_row_count")}}

  // Per-thread transient scratch: one qk score tile at a time.
  int64_t score_tile_elems = qSplitSize * kvSplitSize;
  {{template.codegen_allocate_buffer("score_tile_pool", "accum_t", "num_thread*score_tile_elems")}}
  {{template.codegen_allocate_buffer("prob_tile_pool", "scalar_t", "num_thread*qSplitSize*ekvSplitSize")}}
  {{template.codegen_allocate_buffer("query_padded_pool", "scalar_t", "num_thread*qSplitSize*eheadSize")}}
  {{template.codegen_allocate_buffer("key_transpose_scratch", "scalar_t", "num_thread*kvSplitSize*headSize")}}

  // Packed K/V for the head block in flight only, ONE SLOT PER KV HEAD.
  {{template.codegen_allocate_buffer("packed_key_block", "scalar_t", "kv_heads_in_flight*eheadSize*kvSize")}}
  {{template.codegen_allocate_buffer("packed_value_block", "scalar_t", "kv_heads_in_flight*padded_kv_token_count*headSize_v")}}

  int64_t head_block_count = (total_head_instances + heads_in_flight - 1) / heads_in_flight;

  for (int64_t head_block_idx = 0; head_block_idx < head_block_count; head_block_idx++) {
    int64_t head_block_first_instance = head_block_idx * heads_in_flight;
    int64_t head_block_size = std::min(heads_in_flight, total_head_instances - head_block_first_instance);
    // Whole KV groups by the invariant, so this division is exact even for a short
    // trailing block.
    int64_t kv_head_block_size = head_block_size / q_heads_per_kv_head;

{{template.codegen_pack_kv(kernel.kernel_name)}}

    // ---- KV-chunk loop (sequential; parallel_for's join is the barrier) ----
    for (int64_t chunk_idx = 0; chunk_idx < kv_chunk_count; chunk_idx++) {
      int64_t chunk_kv_begin = chunk_idx * kv_chunk_tokens;
      int64_t chunk_kv_end = std::min(chunk_kv_begin + kv_chunk_tokens, kvSize);
{%- if causal_balance %}

      // ---- Per-chunk cost model, then a descending-cost queue ----
      for (int64_t q_group_idx = 0; q_group_idx < qblock_group_count; ++q_group_idx) {
        const int64_t first_qblock_of_group = q_group_idx * qblocks_per_kv_step;
        // Find how many q blocks in this group
        const int64_t group_qblock_count = std::min(qblocks_per_kv_step, qSlice - first_qblock_of_group);
        int64_t tile_count = 0;
        for (int64_t qblock_in_group = 0; qblock_in_group < group_qblock_count; ++qblock_in_group) {
          // The index of Q - [0, Q sequence length)
          const int64_t q_row_begin = (first_qblock_of_group + qblock_in_group) * qSplitSize;
          const int64_t q_block_kv_end =
              std::min(chunk_kv_end, q_row_begin + std::min(qSplitSize, qSize - q_row_begin) + causal_offset);
          // Calculate the tile cost if not completely skipped
          if (q_block_kv_end > chunk_kv_begin) tile_count += (q_block_kv_end - chunk_kv_begin + kvSplitSize - 1) / kvSplitSize;
        }
        group_tile_cost[q_group_idx] = tile_count;
        groups_by_cost_desc[q_group_idx] = (int32_t)q_group_idx;
      }
      std::stable_sort(groups_by_cost_desc.begin(), groups_by_cost_desc.end(),
          [&](int32_t a, int32_t b) { return group_tile_cost[a] > group_tile_cost[b]; });
      int64_t live_group_count = qblock_group_count;

      // Drops the zero-cost groups off the queue
      if (chunk_idx > 0) {
        while (live_group_count > 0 && group_tile_cost[groups_by_cost_desc[live_group_count - 1]] == 0) {
          --live_group_count;
        }
      }
      const int64_t work_item_count = head_block_size * live_group_count;
      // Own cache line: this counter is written by every thread on every work_item.
      alignas(64) std::atomic<int64_t> work_queue_cursor{0};

      at::parallel_for(0, num_thread, 1, [&](int64_t, int64_t) {
{%- else %}

      at::parallel_for(0, head_block_size * qblock_group_count, 1, [&](int64_t begin, int64_t end) {
{%- endif %}
        int thread_idx = at::get_thread_num();
        accum_t* qk_data = score_tile_pool + thread_idx * score_tile_elems;
        scalar_t* qk_reduced_data = is_reduced_type
            ? prob_tile_pool + thread_idx * qSplitSize * ekvSplitSize
            : nullptr;
        scalar_t* query_padded_tile = (!headSize_even && need_pack)
            ? query_padded_pool + thread_idx * qSplitSize * eheadSize
            : nullptr;
{%- if causal_balance %}
        int64_t head_in_block = 0;

        for (;;) {
          const int64_t work_item = work_queue_cursor.fetch_add(1, std::memory_order_relaxed);
          if (work_item >= work_item_count) break;
          head_in_block = work_item % head_block_size;
          const int64_t q_group_idx = groups_by_cost_desc[work_item / head_block_size];
{%- else %}
        int64_t head_in_block = 0, q_group_idx = 0;
        at::native::data_index_init(begin, head_in_block, head_block_size, q_group_idx, qblock_group_count);

        for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
{%- endif %}
          int64_t head_instance = head_block_first_instance + head_in_block;
          int64_t batch_idx = head_instance / num_head;
          int64_t head_idx = head_instance % num_head;
          int64_t first_qblock_of_group = q_group_idx * qblocks_per_kv_step;
          int64_t qblocks_in_group = std::min(qblocks_per_kv_step, qSlice - first_qblock_of_group);

          // Where this q-block's online-softmax state lives. Under single_chunk it is a
          // per-thread slot reused across work items
          auto accum_base_of = [&](int64_t qblock_in_group, int64_t q_row_begin) {
            return single_chunk
                ? (thread_idx * qblocks_per_kv_step + qblock_in_group) * qSplitSize
                : head_in_block * qSize + q_row_begin;
          };

          if (chunk_idx == 0) {
            for (int64_t qblock_in_group = 0; qblock_in_group < qblocks_in_group; ++qblock_in_group) {
              int64_t init_q_row = (first_qblock_of_group + qblock_in_group) * qSplitSize;
              int64_t init_q_row_count = std::min(qSplitSize, qSize - init_q_row);
              int64_t init_base = accum_base_of(qblock_in_group, init_q_row);
              {{kernel.kernel_name}}_fill_stub(running_row_max + init_base,
                  -std::numeric_limits<accum_t>::infinity(), init_q_row_count);
              {{kernel.kernel_name}}_fill_stub(running_row_sum + init_base,
                  static_cast<accum_t>(0), init_q_row_count);
            }
          }

{%- if causal %}
          // ---- Causal: bound the KV sweep for this GROUP of q-blocks ----
          int64_t group_last_q_row =
              std::min((first_qblock_of_group + qblocks_in_group) * qSplitSize, qSize) - 1;
          int64_t group_kv_end = std::min(chunk_kv_end, group_last_q_row + causal_offset + 1);
{%- endif %}
          // KV step OUTER, the group's q-blocks INNER: the step's K/V stay resident in
          // L2 while all qblocks_in_group q-blocks consume them, so each step is
          // fetched from L3 once per GROUP instead of once per q-block.
          for (int64_t kv_step_begin = chunk_kv_begin; kv_step_begin < {{kv_hi}}; kv_step_begin += kvSplitSize) {
            int64_t kv_step_idx = (kv_step_begin - chunk_kv_begin) / kvSplitSize;
            bool output_accum_unwritten = (chunk_idx == 0 && kv_step_idx == 0);
            int64_t cur_kvSplitSize = std::min(kvSplitSize, kvSize - kv_step_begin);
            int64_t cur_ekvSplitSize = (need_pack && cur_kvSplitSize % 2 != 0) ? cur_kvSplitSize + 1 : cur_kvSplitSize;

           for (int64_t qblock_in_group = 0; qblock_in_group < qblocks_in_group; ++qblock_in_group) {
            int64_t q_row_begin = (first_qblock_of_group + qblock_in_group) * qSplitSize;
            int64_t cur_qSplitSize = std::min(qSplitSize, qSize - q_row_begin);
{%- if causal %}
            // Skip calculation as it is fully masked
            if (kv_step_begin >= q_row_begin + cur_qSplitSize + causal_offset) continue;
            // PARTIAL (diagonal) block: the first row already fails to keep the whole
            // block, so some row needs -inf. Anything else here is fully kept and
            // takes the untouched shipped path.
            const int64_t keep_cols_first_row = q_row_begin + causal_offset + 1 - kv_step_begin;
            const bool block_straddles_diagonal = keep_cols_first_row < cur_kvSplitSize;
{%- endif %}
            int64_t accum_row_base = accum_base_of(qblock_in_group, q_row_begin);
            accum_t* block_row_max = running_row_max + accum_row_base;
            accum_t* block_row_sum = running_row_sum + accum_row_base;
            accum_t* output_accum_rows = output_accum + accum_row_base * headSize_v;
            // Slot of THIS Q head's shared packed K/V. All q_heads_per_kv_head Q heads of
            // one KV head read the same slot -- that sharing is the point of the
            // de-duplicated pack. Exact by the alignment invariant.
            const int64_t packed_kv_slot = head_in_block / q_heads_per_kv_head;

            if (!headSize_even && need_pack) {
              {{kernel.kernel_name}}_copy_value_with_pad<scalar_t>(
                  q_data + batch_idx * qStrideB + head_idx * qStrideH + q_row_begin * qStrideM,
                  query_padded_tile,
                  cur_qSplitSize, headSize, cur_qSplitSize, eheadSize, qStrideM);
            }

{{template.codegen_online_softmax(kernel.kernel_name, softmax_ilv, softmax_ilv_pv, softmax_fuse_scale, base2_softmax)}}

            // ---- QK^T ----
            if (qk_blas) {
              at::native::cpublas::brgemm(
                  cur_qSplitSize, cur_kvSplitSize, headSize,
                  qStrideM, cur_kvSplitSize, cur_kvSplitSize, false,
                  q_data + batch_idx * qStrideB + head_idx * qStrideH + q_row_begin * qStrideM,
                  packed_key_block + packed_kv_slot * eheadSize * kvSize + kv_step_begin * eheadSize,
                  qk_data, false);
            } else if (!need_pack) {
              auto k_addr = k_data +
                  (is_broadcast_bs_kv ? batch_idx / q_batch_per_kv_batch : batch_idx) * kStrideB +
                  (is_broadcast_head_kv ? head_idx / q_heads_per_kv_head : head_idx) * kStrideH + kv_step_begin * kStrideN;
              {{kernel.kernel_name}}_kernel_micro_gemm_transpose_b<static_cast<bool>(false)>(
                  q_data + batch_idx * qStrideB + head_idx * qStrideH + q_row_begin * qStrideM,
                  k_addr, qk_data,
                  cur_qSplitSize, cur_kvSplitSize, headSize,
                  qStrideM, kStrideN, cur_kvSplitSize);
            } else {
              const scalar_t* qk_a_ptr = !headSize_even
                  ? query_padded_tile
                  : q_data + batch_idx * qStrideB + head_idx * qStrideH + q_row_begin * qStrideM;
              const scalar_t* qk_b_ptr =
                  packed_key_block + packed_kv_slot * eheadSize * kvSize + kv_step_begin * eheadSize;
              const int64_t qk_lda = headSize_even ? qStrideM : eheadSize;
{%- if amx_gemm %}
              if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
                {{kernel.kernel_name}}_amx_tile_init();
                if ({{kernel.kernel_name}}_amx_gemm_applicable(
                        cur_qSplitSize, cur_kvSplitSize, eheadSize)) {
{%- if softmax_ilv %}
                  // Overlap QK and softmax. The cursor advances ONE zmm per step().
                  // It is called between different tdpbf16ps to achieve fine overlapping
                  {
                    const int64_t row_block_height = {{softmax_ilv_rb}};
                    const int vecs_per_row = (int)(cur_kvSplitSize / 16);
                    {{kernel.kernel_name}}_ExpCursor<scalar_t, accum_t> exp_cursor;
                    exp_cursor.row_max = woven_row_max; exp_cursor.row_sum_out = woven_row_sum;
                    int64_t pending_row_block = -1;
                    for (int64_t row_block_begin = 0; row_block_begin < cur_qSplitSize; row_block_begin += row_block_height) {
                      int pending_row_count = (int)std::min(row_block_height, cur_qSplitSize - pending_row_block);
                      if (pending_row_block >= 0) {
                        exp_cursor.begin(qk_data + pending_row_block * cur_kvSplitSize,
                                 {{kernel.kernel_name}}_conditional_data_ptr(qk_data, qk_reduced_data)
                                     + pending_row_block * cur_ekvSplitSize,
                                 pending_row_count, vecs_per_row, cur_kvSplitSize, cur_ekvSplitSize);
                        {{kernel.kernel_name}}_amx_gemm_rowblock{{'16' if softmax_ilv_rb == 16 else ''}}_woven<false>(
                            reinterpret_cast<const uint16_t*>(qk_a_ptr), qk_lda,
                            reinterpret_cast<const uint16_t*>(qk_b_ptr), cur_kvSplitSize,
                            qk_data, cur_kvSplitSize, row_block_begin, eheadSize, exp_cursor);
                        exp_cursor.drain();
                        exp_cursor.finish(pending_row_count);
                        commit_woven_row_block(pending_row_block, pending_row_count);
                      } else {
                        {{kernel.kernel_name}}_amx_gemm_rowblock{{'16' if softmax_ilv_rb == 16 else ''}}<false>(
                            reinterpret_cast<const uint16_t*>(qk_a_ptr), qk_lda,
                            reinterpret_cast<const uint16_t*>(qk_b_ptr), cur_kvSplitSize,
                            qk_data, cur_kvSplitSize, row_block_begin, eheadSize);
                      }
                      // scale+max for the row-block QK just produced, then MERGE the
                      // carried running max immediately
                      {
                        int row_block_row_count = (int)std::min(row_block_height, cur_qSplitSize - row_block_begin);
{%- if causal %}
                        // Mask this row-block between its QK and its scale+max.
                        if (block_straddles_diagonal) {
                          {{kernel.kernel_name}}_causal_mask_block<accum_t>(
                              qk_data + row_block_begin * cur_kvSplitSize, cur_kvSplitSize,
                              row_block_row_count, cur_kvSplitSize, keep_cols_first_row + row_block_begin);
                        }
{%- endif %}
                        {{kernel.kernel_name}}_scale_max_block<accum_t>(
                            qk_data + row_block_begin * cur_kvSplitSize, cur_kvSplitSize,
                            row_block_row_count, (int)cur_kvSplitSize, scaling_factor, woven_row_max);
                        for (int r = 0; r < row_block_row_count; ++r) {
                          accum_t carried = block_row_max[row_block_begin + r];
                          if (carried > woven_row_max[r]) {
                            woven_row_max[r] = carried;
                          } else if (!output_accum_unwritten && rescale_skip_thresh > accum_t(0) &&
                                     carried != -std::numeric_limits<accum_t>::infinity() &&
                                     woven_row_max[r] - carried <= rescale_skip_thresh) {
                            // Lazy rescale, and it MUST happen here: the woven exp pass
                            // reads woven_row_max, so the decision cannot be deferred to
                            // commit_woven_row_block the way it can in softmax_rows.
                            woven_row_max[r] = carried;
                          }
                        }
                      }
                      pending_row_block = row_block_begin;
                    }
                    // Last row-block: no more QK TMULs to hide its exp pass in.
                    if (pending_row_block >= 0) {
                      int pending_row_count = (int)std::min(row_block_height, cur_qSplitSize - pending_row_block);
{%- if softmax_ilv_pv %}
                      // Hand it to PV. Record only the row-block identity
                      pending_exp_row_block = pending_row_block;
                      pending_exp_row_count = pending_row_count;
{%- else %}
                      exp_cursor.begin(qk_data + pending_row_block * cur_kvSplitSize,
                               {{kernel.kernel_name}}_conditional_data_ptr(qk_data, qk_reduced_data)
                                   + pending_row_block * cur_ekvSplitSize,
                               pending_row_count, vecs_per_row, cur_kvSplitSize, cur_ekvSplitSize);
                      exp_cursor.drain();
                      exp_cursor.finish(pending_row_count);
                      commit_woven_row_block(pending_row_block, pending_row_count);
{%- endif %}
                    }
                    qk_was_woven = true;
                  }
{%- else %}
                  {{kernel.kernel_name}}_amx_gemm<false>(
                      reinterpret_cast<const uint16_t*>(qk_a_ptr), qk_lda,
                      reinterpret_cast<const uint16_t*>(qk_b_ptr), cur_kvSplitSize,
                      qk_data, cur_kvSplitSize, cur_qSplitSize, eheadSize);
{%- endif %}
                } else {
                  at::native::cpublas::brgemm(
                      cur_qSplitSize, cur_kvSplitSize, eheadSize, qk_lda,
                      cur_kvSplitSize, cur_kvSplitSize, false,
                      qk_a_ptr, qk_b_ptr, qk_data, need_pack);
                }
              } else
{%- endif %}
              {
                at::native::cpublas::brgemm(
                    cur_qSplitSize, cur_kvSplitSize, eheadSize, qk_lda,
                    cur_kvSplitSize, cur_kvSplitSize, false,
                    qk_a_ptr, qk_b_ptr, qk_data, need_pack);
              }
            }

{%- if causal %}
            // Mask the diagonal block
            if (block_straddles_diagonal && !qk_was_woven) {
              {{kernel.kernel_name}}_causal_mask_block<accum_t>(
                  qk_data, cur_kvSplitSize, cur_qSplitSize, cur_kvSplitSize, keep_cols_first_row);
            }
{%- endif %}
{%- if not softmax_fuse_scale %}
            {{kernel.kernel_name}}_mul_scale_kernel<accum_t>(qk_data, scaling_factor, cur_qSplitSize*cur_kvSplitSize);
{%- endif %}

{%- if softmax_ilv %}
            // Serial fallback
            if (!qk_was_woven) softmax_rows(0, cur_qSplitSize);
{%- else %}
            softmax_rows(0, cur_qSplitSize);
{%- endif %}

{%- if softmax_ilv_pv %}
            // Check if the pending tail exp ride in PV's TMUL shadow
            bool can_weave_tail_into_pv =
                pending_exp_row_count > 0
                && pending_exp_row_block >= {{softmax_ilv_rb}}
                && need_pack
                && std::is_same_v<scalar_t, at::BFloat16>
                && {{kernel.kernel_name}}_amx_gemm_applicable(
                       cur_qSplitSize, headSize_v, cur_ekvSplitSize);
            // Not weavable -> drain it to avoid affecting the QK weave.
            if (pending_exp_row_count > 0 && !can_weave_tail_into_pv) {
              {{kernel.kernel_name}}_ExpCursor<scalar_t, accum_t> tail_exp_cursor;
              tail_exp_cursor.row_max = woven_row_max; tail_exp_cursor.row_sum_out = woven_row_sum;
              tail_exp_cursor.begin(qk_data + pending_exp_row_block * cur_kvSplitSize,
                       {{kernel.kernel_name}}_conditional_data_ptr(qk_data, qk_reduced_data)
                           + pending_exp_row_block * cur_ekvSplitSize,
                       pending_exp_row_count, (int)(cur_kvSplitSize / 16), cur_kvSplitSize, cur_ekvSplitSize);
              tail_exp_cursor.drain();
              tail_exp_cursor.finish(pending_exp_row_count);
              commit_woven_row_block(pending_exp_row_block, pending_exp_row_count);
              pending_exp_row_count = 0;
            }
{%- endif %}

            // ---- PV ----
            if (!need_pack) {
              auto v_addr = v_data +
                  (is_broadcast_bs_kv ? batch_idx / q_batch_per_kv_batch : batch_idx) * vStrideB +
                  (is_broadcast_head_kv ? head_idx / q_heads_per_kv_head : head_idx) * vStrideH + kv_step_begin * vStrideN;
              if (!std::is_same_v<scalar_t, at::Half>) {
                at::native::cpublas::brgemm(
                    cur_qSplitSize, headSize_v, cur_ekvSplitSize,
                    cur_ekvSplitSize, vStrideN, headSize_v, !output_accum_unwritten,
                    {{kernel.kernel_name}}_conditional_data_ptr(qk_data, qk_reduced_data),
                    v_addr, output_accum_rows, need_pack);
              } else if (!output_accum_unwritten) {
                {{kernel.kernel_name}}_kernel_micro_gemm<static_cast<bool>(true)>(
                    {{kernel.kernel_name}}_conditional_data_ptr(qk_data, qk_reduced_data),
                    v_addr, output_accum_rows, cur_qSplitSize, headSize_v,
                    cur_ekvSplitSize, cur_ekvSplitSize, vStrideN, headSize_v);
              } else {
                {{kernel.kernel_name}}_kernel_micro_gemm<static_cast<bool>(false)>(
                    {{kernel.kernel_name}}_conditional_data_ptr(qk_data, qk_reduced_data),
                    v_addr, output_accum_rows, cur_qSplitSize, headSize_v,
                    cur_ekvSplitSize, cur_ekvSplitSize, vStrideN, headSize_v);
              }
            } else {
              int64_t packed_value_offset = kv_step_begin / kvSplitSize * ekvSplitSize;
              const scalar_t* pv_b_ptr =
                  packed_value_block + packed_kv_slot * padded_kv_token_count * headSize_v + packed_value_offset * headSize_v;
{%- if amx_gemm %}
              if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
                // Reload the tile config configured in AMX QK GEMM
                {{kernel.kernel_name}}_amx_tile_init();
                if ({{kernel.kernel_name}}_amx_gemm_applicable(
                        cur_qSplitSize, headSize_v, cur_ekvSplitSize)) {
{%- if softmax_ilv_pv %}
                  if (can_weave_tail_into_pv) {
                    {{kernel.kernel_name}}_ExpCursor<scalar_t, accum_t> tail_exp_cursor;
                    tail_exp_cursor.row_max = woven_row_max; tail_exp_cursor.row_sum_out = woven_row_sum;
                    tail_exp_cursor.begin(qk_data + pending_exp_row_block * cur_kvSplitSize,
                             {{kernel.kernel_name}}_conditional_data_ptr(qk_data, qk_reduced_data)
                                 + pending_exp_row_block * cur_ekvSplitSize,
                             pending_exp_row_count, (int)(cur_kvSplitSize / 16),
                             cur_kvSplitSize, cur_ekvSplitSize);
                    const int64_t row_block_height = {{softmax_ilv_rb}};
                    const uint16_t* pv_a = reinterpret_cast<const uint16_t*>(qk_reduced_data);
                    const uint16_t* pv_b = reinterpret_cast<const uint16_t*>(pv_b_ptr);

                    if (!output_accum_unwritten) {
                      {{kernel.kernel_name}}_amx_gemm_rowblock{{'16' if softmax_ilv_rb == 16 else ''}}_woven<true>(
                          pv_a, cur_ekvSplitSize, pv_b, headSize_v,
                          output_accum_rows, headSize_v, 0, cur_ekvSplitSize, tail_exp_cursor);
                    } else {
                      {{kernel.kernel_name}}_amx_gemm_rowblock{{'16' if softmax_ilv_rb == 16 else ''}}_woven<false>(
                          pv_a, cur_ekvSplitSize, pv_b, headSize_v,
                          output_accum_rows, headSize_v, 0, cur_ekvSplitSize, tail_exp_cursor);
                    }
                    // Finish the tail rows before their own PV row-block
                    tail_exp_cursor.drain();
                    tail_exp_cursor.finish(pending_exp_row_count);
                    commit_woven_row_block(pending_exp_row_block, pending_exp_row_count);
                    pending_exp_row_count = 0;
                    for (int64_t pv_row_block_begin = row_block_height; pv_row_block_begin < cur_qSplitSize; pv_row_block_begin += row_block_height) {
                      if (!output_accum_unwritten) {
                        {{kernel.kernel_name}}_amx_gemm_rowblock{{'16' if softmax_ilv_rb == 16 else ''}}<true>(
                            pv_a, cur_ekvSplitSize, pv_b, headSize_v,
                            output_accum_rows, headSize_v, pv_row_block_begin, cur_ekvSplitSize);
                      } else {
                        {{kernel.kernel_name}}_amx_gemm_rowblock{{'16' if softmax_ilv_rb == 16 else ''}}<false>(
                            pv_a, cur_ekvSplitSize, pv_b, headSize_v,
                            output_accum_rows, headSize_v, pv_row_block_begin, cur_ekvSplitSize);
                      }
                    }
                  } else
{%- endif %}
                  if (!output_accum_unwritten) {
                    {{kernel.kernel_name}}_amx_gemm<true>(
                        reinterpret_cast<const uint16_t*>(qk_reduced_data), cur_ekvSplitSize,
                        reinterpret_cast<const uint16_t*>(pv_b_ptr), headSize_v,
                        output_accum_rows, headSize_v, cur_qSplitSize, cur_ekvSplitSize);
                  } else {
                    {{kernel.kernel_name}}_amx_gemm<false>(
                        reinterpret_cast<const uint16_t*>(qk_reduced_data), cur_ekvSplitSize,
                        reinterpret_cast<const uint16_t*>(pv_b_ptr), headSize_v,
                        output_accum_rows, headSize_v, cur_qSplitSize, cur_ekvSplitSize);
                  }
                } else {
                  at::native::cpublas::brgemm(
                      cur_qSplitSize, headSize_v, cur_ekvSplitSize,
                      cur_ekvSplitSize, headSize_v, headSize_v, !output_accum_unwritten,
                      qk_reduced_data, pv_b_ptr, output_accum_rows, need_pack);
                }
              } else
{%- endif %}
              {
                at::native::cpublas::brgemm(
                    cur_qSplitSize, headSize_v, cur_ekvSplitSize,
                    cur_ekvSplitSize, headSize_v, headSize_v, !output_accum_unwritten,
                    qk_reduced_data, pv_b_ptr, output_accum_rows, need_pack);
              }
            }
           }
          }
          // ---- Normalize O /= l, single-chunk path ----
          // With one KV chunk this q-group's sweep is complete here
          if (single_chunk) {
            for (int64_t nq = 0; nq < qblocks_in_group; ++nq) {
              int64_t norm_q_row = (first_qblock_of_group + nq) * qSplitSize;
              int64_t norm_rows = std::min(qSplitSize, qSize - norm_q_row);
              int64_t norm_base = accum_base_of(nq, norm_q_row);
              for (int64_t row = 0; row < norm_rows; ++row) {
                accum_t row_sum_final = running_row_sum[norm_base + row];
                // A fully masked-out row has sum 0; emit 0 rather than NaN.
                accum_t sum_reciprocal = row_sum_final == 0 ? static_cast<accum_t>(0)
                                                    : static_cast<accum_t>(1) / row_sum_final;
                at::vec::map<scalar_t>(
                    [sum_reciprocal](Vec x) { return x * Vec(sum_reciprocal); },
                    out_data + batch_idx * oStrideB + head_idx * oStrideH + norm_q_row * oStrideM + row * oStrideM,
                    output_accum + (norm_base + row) * headSize_v,
                    headSize_v);
              }
            }
          }
{%- if not causal_balance %}
          at::native::data_index_step(head_in_block, head_block_size, q_group_idx, qblock_group_count);
{%- endif %}
        }
        at::native::cpublas::brgemm_release(need_pack);
      });
    }

    // ---- Normalize: O /= l  (multi-chunk path only)
    if (!single_chunk)
    at::parallel_for(0, head_block_size * qSlice, 1, [&](int64_t begin, int64_t end) {
      int64_t head_in_block = 0, qblock_idx = 0;
      at::native::data_index_init(begin, head_in_block, head_block_size, qblock_idx, qSlice);
      for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
        int64_t head_instance = head_block_first_instance + head_in_block;
        int64_t batch_idx = head_instance / num_head;
        int64_t head_idx = head_instance % num_head;
        int64_t q_row_begin = qblock_idx * qSplitSize;
        int64_t cur_qSplitSize = std::min(qSplitSize, qSize - q_row_begin);
        int64_t accum_row_base = head_in_block * qSize + q_row_begin;
        for (int64_t row = 0; row < cur_qSplitSize; ++row) {
          accum_t row_sum_final = running_row_sum[accum_row_base + row];
          // A fully masked-out row has sum 0; emit 0 rather than NaN.
          accum_t sum_reciprocal = row_sum_final == 0 ? static_cast<accum_t>(0)
                                              : static_cast<accum_t>(1) / row_sum_final;
          at::vec::map<scalar_t>(
              [sum_reciprocal](Vec x) { return x * Vec(sum_reciprocal); },
              out_data + batch_idx * oStrideB + head_idx * oStrideH + q_row_begin * oStrideM + row * oStrideM,
              output_accum + (accum_row_base + row) * headSize_v,
              headSize_v);
        }
        at::native::data_index_step(head_in_block, head_block_size, qblock_idx, qSlice);
      }
    });
  }
}
"""


class CppCausalAttentionCpuTemplate(CppTemplate):
    """
    Causal CPU attention template: cache-aware and load-balanced among CPU threads.
    Work for plain causal and dense cases.
    """

    def __init__(
        self,
        input_nodes,
        layout: ir.Layout,
        scale,
        kv_block_size,
        q_block_size,
        no_full_kv_block,
        causal_offset=None,
    ) -> None:
        if layout.dtype not in [torch.float, torch.bfloat16, torch.float16]:
            raise AssertionError(f"unsupported layout dtype: {layout.dtype}")
        super().__init__("flex_attention", input_nodes, layout, parallel_num_threads())
        self.scale = scale
        self.causal_offset = causal_offset
        self.kv_block_size = kv_block_size
        self.q_block_size = q_block_size
        self.no_full_kv_block = no_full_kv_block

    def use_base2_softmax(self) -> bool:
        return "CPU_CAPABILITY_AVX512" in cpu_vec_isa.pick_vec_isa().build_macro()

    def use_vnni_pack(self) -> bool:
        return "CPU_CAPABILITY_AVX512" in cpu_vec_isa.pick_vec_isa().build_macro()

    def use_exp2_deg3(self) -> bool:
        return self.use_base2_softmax() and self.input_dtype in (
            torch.bfloat16,
            torch.float16,
        )

    def codegen_amx_gemm(self, kernel_name: str, softmax_ilv, softmax_ilv_rb):
        return self._template_from_string(AMX_GEMM_FUNCTIONS).render(
            dict(
                kernel_name=kernel_name,
                amx_interleave_stores=_amx_interleave_stores(),
                amx_b_nontemporal=_amx_b_nontemporal(),
                softmax_ilv=softmax_ilv,
                softmax_ilv_rb=softmax_ilv_rb,
            )
        )

    def codegen_l3_chunk_sizing(self):
        return self._template_from_string(ANALYTIC_L3_TEMPLATE).render(
            dict(
                l3_bytes=_l3_bytes(),
                l3_fraction=_l3_fraction(),
                jkv_chunk_override=_jkv_chunk_override(),
                nkv_override=_nkv_override(),
            )
        ).strip("\n")

    def codegen_l2_group_sizing(self, causal_balance):
        return self._template_from_string(ANALYTIC_L2_GROUP_TEMPLATE).render(
            dict(
                l2_bytes=_l2_bytes(),
                l2_fraction=_l2_fraction(),
                l2_group_override=_l2_group_override(),
                causal_balance=causal_balance,
            )
        ).strip("\n")

    def codegen_pack_kv(self, kernel_name: str):
        return self._template_from_string(PACK_KV_TEMPLATE).render(
            dict(kernel_name=kernel_name)
        ).strip("\n")

    def codegen_online_softmax(
        self, kernel_name: str, softmax_ilv, softmax_ilv_pv, softmax_fuse_scale,
        base2_softmax,
    ):
        return self._template_from_string(ONLINE_SOFTMAX_TEMPLATE).render(
            dict(
                kernel_name=kernel_name,
                softmax_ilv=softmax_ilv,
                softmax_ilv_pv=softmax_ilv_pv,
                softmax_fuse_scale=softmax_fuse_scale,
                base2_softmax=base2_softmax,
            )
        ).strip("\n")

    def codegen_softmax_fusion(self, kernel_name: str, softmax_ilv, need_mul_scale):
        # TODO: use inductor IR to rewrite those fusions
        return self._template_from_string(CAUSAL_SOFTMAX_FUSIONS).render(
            dict(
                kernel_name=kernel_name,
                base2_softmax=self.use_base2_softmax(),
                exp2_deg3=self.use_exp2_deg3(),
                softmax_ilv=softmax_ilv,
                need_mul_scale=need_mul_scale,
                causal=self.causal_offset is not None,
            )
        )

    def codegen_brgemm_pack_function(self, kernel_name: str):
        # TODO: make them general for common bmm templates
        return self._template_from_string(CAUSAL_BRGEMM_PACK_FUNCTIONS).render(
            dict(kernel_name=kernel_name)
        )

    def codegen_allocate_buffer(self, buffer_name: str, buffer_dtype, buffer_size):
        return self._template_from_string(CAUSAL_ALLOCATE_BUFFER).render(
            dict(
                buffer_name=buffer_name,
                buffer_dtype=buffer_dtype,
                buffer_size=buffer_size,
            )
        )

    def codegen_micro_gemm(self, kernel_name: str):
        from torch._inductor.codegen.cpp_gemm_template import (
            CppTemplateKernel,
            parallel_num_threads,
        )
        from torch._inductor.codegen.cpp_micro_gemm import CppMicroGemmFP32Vec
        from torch._inductor.virtualized import V

        micro_gemm_trans = CppMicroGemmFP32Vec(
            kernel_name + "_kernel_micro_gemm_transpose_b",
            self.input_dtype,
            self.input_dtype,
            self.accumulate_dtype,
            self.accumulate_dtype,
            GemmBlocking(1, 16, 1),
            1,
            True,
            True,
        )

        micro_gemm = CppMicroGemmFP32Vec(
            kernel_name + "_kernel_micro_gemm",
            self.input_dtype,
            self.input_dtype,
            self.accumulate_dtype,
            self.accumulate_dtype,
            GemmBlocking(1, 16, 1),
            1,
            True,
            False,
        )

        with V.set_graph_handler(V.graph):
            kernel = CppTemplateKernel("cpp_micro_gemm", parallel_num_threads())
            code_trans = micro_gemm_trans.codegen_define(kernel)
            code = micro_gemm.codegen_define(kernel)
        return "\n" + code + code_trans

    @staticmethod
    def add_choices(
        choices,
        input_nodes,
        layout,
        scale,
        kv_block_size,
        q_block_size,
        no_full_kv_block,
        causal_offset=None,
    ):
        def preprocessor(input_nodes, layout):
            return input_nodes, layout

        def postprocessor(output):
            return output

        template = DataProcessorTemplateWrapper(
            CppCausalAttentionCpuTemplate,
            preprocessor,
            postprocessor,
            input_nodes=input_nodes,
            layout=layout,
            scale=scale,
            kv_block_size=kv_block_size,
            q_block_size=q_block_size,
            no_full_kv_block=no_full_kv_block,
            causal_offset=causal_offset,
        )
        template.maybe_append_choice(choices)
        return template

    def use_amx_gemm(self) -> bool:
        return (
            _amx_gemm()
            and self.input_dtype == torch.bfloat16
            and torch.cpu._is_amx_tile_supported()
            and "CPU_CAPABILITY_AVX512" in cpu_vec_isa.pick_vec_isa().build_macro()
        )

    def use_softmax_ilv(self) -> bool:
        # Needs the woven cursor (bf16 AMX path) and fuse_scale to enable interleave
        return bool(
            _softmax_ilv()
            and _softmax_fuse_scale()
            and self.use_amx_gemm()
            and self.use_base2_softmax()
        )

    def render(  # type: ignore[override,return]
        self,
        kernel,
        template_buffer_node=None,
        epilogue_nodes=None,
        **kwargs,
    ) -> str:
        if epilogue_nodes is not None and epilogue_nodes != []:
            raise NotImplementedError(
                "Unsupported for `epilogue_nodes` in CppCausalAttentionCpuTemplate."
            )
        causal = self.causal_offset is not None

        query = kernel.permute(self.input_nodes[0], [0, 2, 1, 3])
        key = kernel.permute(self.input_nodes[1], [0, 2, 1, 3])
        value = kernel.permute(self.input_nodes[2], [0, 2, 1, 3])
        self.accumulate_dtype = torch.float
        self.input_dtype = query.layout.dtype

        num_threads = parallel_num_threads()
        if not isinstance(self.output_node, ir.IRNode):
            raise AssertionError(f"expected ir.IRNode, got {type(self.output_node)}")
        buf_out = TensorBox.create(self.output_node)
        if template_buffer_node is not None:
            buf_out = template_buffer_node

        softmax_ilv = self.use_softmax_ilv()
        amx_gemm = self.use_amx_gemm()
        # mul_scale_kernel is reachable only where the scale is a separate pass.
        need_mul_scale = not _softmax_fuse_scale()
        softmax_ilv_pv = _softmax_ilv_pv() if softmax_ilv else 0
        q_size = _static_size(query.get_size()[1])
        kv_size = _static_size(key.get_size()[1])
        qb = _q_split_size() or self.q_block_size
        softmax_ilv_rb = _softmax_ilv_rb_analytic(
            None if q_size is None else min(qb, q_size),
            None if kv_size is None else min(_kv_split_size(), kv_size),
        )

        options = dict(
            query=query,
            key=key,
            value=value,
            kv_num_blocks=self.input_nodes[3],
            kv_indices=self.input_nodes[4],
            full_kv_num_blocks=(
                self.input_nodes[5] if not self.no_full_kv_block else None
            ),
            full_kv_indices=self.input_nodes[6] if not self.no_full_kv_block else None,
            scale=self.scale,
            kvBlockSize=self.kv_block_size,
            qBlockSize=self.q_block_size,
            template=self,
            output=buf_out,
            kernel=kernel,
            num_thread=num_threads,
            causal=causal,
            causal_offset=self.causal_offset if causal else 0,
            kv_hi="group_kv_end" if causal else "chunk_kv_end",
            causal_balance=causal and _causal_balance(),
            kv_split_size=_kv_split_size(),
            q_split_size=_q_split_size(),
            # softmax_ilv needs the scale folded into the max pass (see _softmax_ilv).
            softmax_ilv=softmax_ilv,
            softmax_ilv_rb=softmax_ilv_rb,
            need_mul_scale=need_mul_scale,
            # Only meaningful inside the woven schedule.
            softmax_ilv_pv=softmax_ilv_pv,
            softmax_fuse_scale=_softmax_fuse_scale(),
            base2_softmax=self.use_base2_softmax(),
            amx_gemm=amx_gemm,
            qk_blas=_qk_blas(),
            vnni_pack=self.use_vnni_pack(),
        )
        return self._template_from_string(
            INIT_PARAMS + CAUSAL_ATTENTION_TEMPLATE
        ).render(**options)


# ===========================================================================
# Part 3: route a flex_attention call to the causal attention template
# ===========================================================================
def _cpu_causal_attention_enabled():
    flag = getattr(config.cpp, "cpu_causal_attention", None)
    if flag is not None:
        return flag
    return os.environ.get("TORCHINDUCTOR_CPP_CPU_CAUSAL_ATTENTION", "1") == "1"


class _MaskNotCausal(Exception):
    pass


class _MaskNode:
    # A class, not a tuple: OpsWrapper treats a tuple returned by a handler as
    # multiple outputs and wraps each element separately.
    __slots__ = ("op", "args")

    def __init__(self, op, *args):
        self.op = op
        self.args = args


class _MaskExprRecorder(DefaultHandler):
    """Ops handler that records a lowered mask_mod as a small expression tree.

    Records only the ops a causal compare lowers to; any other op raises, which
    rejects the mask.
    """

    def _default(self, name, args, kwargs):
        raise _MaskNotCausal(name)

    def load(self, name, index):
        return _MaskNode("load", name)

    def constant(self, value, dtype):
        return _MaskNode("constant", value)

    def add(self, a, b):
        return _MaskNode("add", a, b)

    def sub(self, a, b):
        return _MaskNode("sub", a, b)

    def ge(self, a, b):
        return _MaskNode("ge", a, b)

    def le(self, a, b):
        return _MaskNode("le", a, b)

    def where(self, cond, a, b):
        return _MaskNode("where", cond, a, b)


def _index_plus_const(node):
    """(index_name, c) if `node` is q_idx or kv_idx plus an integer constant c,
    (None, c) if it is the constant alone, else None."""
    if isinstance(node, _MaskNode):
        if node.op == "load":
            return (node.args[0], 0) if node.args[0] in ("q_idx", "kv_idx") else None
        if node.op == "constant":
            return _index_plus_const(node.args[0])
        if node.op not in ("add", "sub"):
            return None
        a, b = (_index_plus_const(x) for x in node.args)
        if a is None or b is None:
            return None
        sign = 1 if node.op == "add" else -1
        if b[0] is None:
            return a[0], a[1] + sign * b[1]
        if a[0] is None and sign == 1:
            return b[0], a[1] + b[1]
        return None
    if isinstance(node, int) and not isinstance(node, bool):
        return None, node
    return None


def _causal_offset_of_mask_ir(mask_mod):
    """Offset c if the lowered mask_mod is exactly where(q_idx + c >= kv_idx, score,
    -inf) with an integer c >= 0, else None.

    The selectable port matches the traced FX mask graph in flex_cpu.py; a stock
    flex_cpu.py hands this file only the lowered buffer, so the same forms are read
    off its inner_fn instead:
        q_idx >= kv_idx            kv_idx <= q_idx
        q_idx + c >= kv_idx        kv_idx <= q_idx + c
        q_idx >= kv_idx - c        kv_idx - c <= q_idx
    `q_idx > kv_idx` and the anti-causal `kv_idx >= q_idx` are not matched. A
    negative c is rejected: it leaves rows with no kept column, which the template's
    uninitialised O accumulator cannot represent.
    """
    pw = getattr(mask_mod, "data", None)
    if not isinstance(mask_mod, ir.ComputedBuffer) or not isinstance(pw, ir.Pointwise):
        return None
    try:
        with V.set_ops_handler(_MaskExprRecorder()):
            out = pw.inner_fn(*pw.inner_fn_args())
    except _MaskNotCausal:
        return None
    if isinstance(out, OpsValue):
        out = out.value
    if not (isinstance(out, _MaskNode) and out.op == "where"):
        return None
    cond, kept, dropped = out.args
    if not (isinstance(kept, _MaskNode) and kept.op == "load" and kept.args[0] == "score"):
        return None
    if not (isinstance(dropped, _MaskNode) and dropped.op == "constant"
            and dropped.args[0] == -math.inf):
        return None
    if not (isinstance(cond, _MaskNode) and cond.op in ("ge", "le")):
        return None
    lhs, rhs = cond.args if cond.op == "ge" else cond.args[::-1]
    lhs, rhs = _index_plus_const(lhs), _index_plus_const(rhs)
    if lhs is None or rhs is None or lhs[0] != "q_idx" or rhs[0] != "kv_idx":
        return None
    # q + c_q >= kv + c_kv  <=>  q + (c_q - c_kv) >= kv
    offset = lhs[1] - rhs[1]
    return offset if offset >= 0 else None


def _is_identity_score_ir(score_mod):
    # An identity score_mod lowers to the `score` placeholder itself, not a Pointwise.
    return (
        isinstance(score_mod, ir.ComputedBuffer)
        and isinstance(score_mod.data, ir.InputBuffer)
        and score_mod.data.get_name() == "score"
    )


def _causal_template_route(
    score_mod, mask_mod, has_other_buffer, len_score_other, len_mask_other
):
    """(True, causal_offset) to render the causal template, where causal_offset None
    is the dense case; (False, None) to keep the stock flash template."""
    if not _cpu_causal_attention_enabled():
        return False, None
    # A captured tensor is an extra kernel input that the causal template never reads.
    if has_other_buffer or len_score_other or len_mask_other:
        return False, None
    # SKIP_MASK_SCORE: the caller asserts the mask is all-true and drops both mods.
    if score_mod is None and mask_mod is None:
        return True, None
    if score_mod is None or mask_mod is None or not causal_enabled():
        return False, None
    if not _is_identity_score_ir(score_mod):
        return False, None
    offset = _causal_offset_of_mask_ir(mask_mod)
    return offset is not None, offset

