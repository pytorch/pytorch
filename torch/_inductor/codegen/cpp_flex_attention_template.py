# mypy: allow-untyped-defs
import contextlib
import logging
import re
from typing import List, Optional
from unittest.mock import patch

import sympy

import torch
import torch.utils

from ...utils._ordered_set import OrderedSet
from .. import ir
from ..ir import TensorBox
from ..select_algorithm import DataProcessorTemplateWrapper
from ..utils import parallel_num_threads
from ..virtualized import V
from .cpp_template import CppTemplate


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
          typename std::enable_if_t<std::is_reduced_floating_point_v<scalar_t>, int> = 0>
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
// Transpose a [2, 32] matrix to [32, 2]
// Note: the output leading dimension should be 2,
// that is, the output must be contiguous
static inline void {{kernel_name}}_transpose_pad_2x32_block(
    const uint16_t* src,
    uint16_t* dst,
    int64_t ld_src,
    int krem = 2,
    int nrem = 32) {
#if defined(CPU_CAPABILITY_AVX512)
  __m512i r0, r1;
  __m512i d0, d1;
  // load
  if (nrem < 32) {
    __mmask32 mask_krem_v = (1LL << nrem) - 1;
    r0 = _mm512_maskz_loadu_epi16(mask_krem_v, src);
    // if krem is not 2, pad with zeros
    if (krem == 2) {
      r1 = _mm512_maskz_loadu_epi16(mask_krem_v, src + ld_src);
    } else {
      r1 = _mm512_setzero_si512();
    }
  } else {
    r0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src));
    if (krem == 2) {
      r1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + ld_src));
    } else {
      r1 = _mm512_setzero_si512();
    }
  }
  // transpose
  d0 = _mm512_unpacklo_epi16(r0, r1);
  d1 = _mm512_unpackhi_epi16(r0, r1);
  r0 = _mm512_shuffle_i32x4(d0, d1, 0x88);
  r1 = _mm512_shuffle_i32x4(d0, d1, 0xdd);
  d0 = _mm512_shuffle_i32x4(r0, r1, 0x88);
  d1 = _mm512_shuffle_i32x4(r0, r1, 0xdd);

  // store
  if (nrem < 16) {
    __mmask32 mask_rem_v = (1LL << (nrem * 2)) - 1;
    _mm512_mask_storeu_epi16(dst, mask_rem_v, d0);
  } else if (nrem == 16) {
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst), d0);
  } else if (nrem < 32) {
    __mmask32 mask_rem_v = (1LL << (nrem * 2 - 32)) - 1;
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst), d0);
    _mm512_mask_storeu_epi16(
        reinterpret_cast<__m512i*>(dst + 32), mask_rem_v, d1);
  } else {
    // normal store
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst), d0);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + 32), d1);
  }
#else
TORCH_CHECK(false, "transpose_pad_2x32_block is only supported when avx512 is supported")
#endif
}

// To use AMX to accelerate GEMM,
// reorder the memory format [K, N] -> [K/2, N, 2]
// Note: If K % 2 != 0, pad K implicitly
static inline void {{kernel_name}}_pack_vnni2(
    const uint16_t* src,
    uint16_t* dst,
    int64_t ld_src,
    int64_t K,
    int64_t N) {
#if defined(CPU_CAPABILITY_AVX512)
  int64_t bk = 0;
  int64_t _K = K / 2 * 2;
  int64_t _N = N / 32 * 32;
  for (; bk < _K; bk += 2) {
    int64_t bn = 0;
    for (; bn < _N; bn += 32) {
      {{kernel_name}}_transpose_pad_2x32_block(src + bk * ld_src + bn, dst + bk * N + bn * 2, ld_src);
    }
    int64_t nrem = N - bn;
    if (nrem > 0) {
      {{kernel_name}}_transpose_pad_2x32_block(src + bk * ld_src + bn, dst + bk * N + bn * 2, ld_src, 2, nrem);
    }
  }
  if (K % 2 == 1) {
    int64_t bn = 0;
    for (; bn < _N; bn += 32) {
      {{kernel_name}}_transpose_pad_2x32_block(src + bk * ld_src + bn, dst + bk * N + bn * 2, ld_src, 1);
    }
    int64_t nrem = N - bn;
    if (nrem > 0) {
      {{kernel_name}}_transpose_pad_2x32_block(src + bk * ld_src + bn, dst + bk * N + bn * 2, ld_src, 1, nrem);
    }
  }
#else
TORCH_CHECK(false, "pack_vnni2 is only supported when avx512 is supported")
#endif
}
"""

ALLOCATE_BUFFER = r"""
  int64_t {{buffer_name}}_dtype_itemsize = std::is_same_v<{{buffer_dtype}}, at::BFloat16> ? 2 : 4;
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
{{template.codegen_softmax_fusion(kernel.kernel_name)}}
{{template.codegen_brgemm_pack_function(kernel.kernel_name)}}
{%- set kernel_args = {"query": query, "key": key, "value": value,
                       "kv_num_blocks": kv_num_blocks, "kv_indices": kv_indices, "full_kv_num_blocks": full_kv_num_blocks} %}
{%- set kernel_args = template.update_kernel_args(kernel_args) %}

extern "C"
{{kernel.def_kernel(inputs=kernel_args, outputs={"output": output}, extra_sizevars=template.extra_sizevars)}}
{
  int64_t kvBlockSize = {{kvBlockSize}};
  kvBlockSize = kvBlockSize>{{kernel.size(key, 1)}} ? {{kernel.size(key, 1)}}
                                                    : kvBlockSize;
  int64_t num_thread = {{num_thread}};

  // dtypes of kernel and internal buffers
  using scalar_t = {{kernel.dtype(query)}};
  constexpr bool is_reduced_type = std::is_reduced_floating_point_v<scalar_t>;
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
  auto  kv_indices_data = kv_indices;

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

  // Check total kv block number for kv value.
  int64_t block_num_kv_count = 0;
  bool has_block_indice_zero = true;
  for (int64_t kv_count = 0; kv_count < block_num_kvi; kv_count++) {
    if (*(kv_indices + kv_count) > 0) {
      block_num_kv_count++;
    } else if (*(kv_indices + kv_count) == 0) {
      if (has_block_indice_zero) {
        has_block_indice_zero = false;
        block_num_kv_count++;
      } else {
        break;
      }
    }
  }
  // Check to use kv_indice if total block size is bigger than kv length, e.g.,
  // in PagedAttention case.
  bool use_kv_indice = false;
  if (block_num_kvi != block_num_kv_count && batchSize_k == 1) {
    use_kv_indice = true;
  }
  int64_t kvSize = use_kv_indice ? block_num_kv_count * kvBlockSize
                                 : {{kernel.size(key, 1)}};

  // Split size heuristics tuned for q/k len
  int64_t qSplitSize = 32;
  int64_t kvSplitSize = 512;
  if (qSize >= 768) {
    qSplitSize = 256;
    kvSplitSize = 512;
  } else if (qSize >= 192) {
    qSplitSize = 64;
    kvSplitSize = 512;
  }
  if (kvBlockSize < kvSplitSize) {
    kvSplitSize = kvBlockSize;
  }

  qSplitSize = qSplitSize > qSize ? qSize : qSplitSize;
  kvSplitSize = kvSplitSize > kvSize ? kvSize : kvSplitSize;
  int64_t qSlice = (qSize + qSplitSize - 1) / qSplitSize;
  int64_t kvSlice = (kvSize + kvSplitSize - 1) / kvSplitSize;
  int64_t kvTail = (kvSize - 1) % kvSplitSize + 1;

  bool need_pack = false;
  // Whether pack is needed for BFloat16
  if (std::is_same_v<scalar_t, at::BFloat16>) {
    // check platform ability
    need_pack = at::native::cpublas::could_pack(at::kBFloat16);
  }
  if (need_pack) {
    // When the number of gemm is greater than the number of pack,
    // the pack overhead can be overlaped.
    int64_t thresh_size = 64 ;
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
  {{template.codegen_allocate_buffer("key_reorder_ptr", "scalar_t", "batchSize*num_head*eheadSize*kvSize")}}
  {{template.codegen_allocate_buffer("value_reorder_ptr", "scalar_t", "batchSize*num_head*kv_padding_size*headSize_v")}}
  {{template.codegen_allocate_buffer("transpose_buffer_ptr", "scalar_t", "num_thread*kvSplitSize*headSize")}}
  {{template.codegen_allocate_buffer("query_padding_ptr", "scalar_t", "num_thread*qSplitSize*eheadSize")}}

  // Reorder K, V and transpose K
  at::parallel_for(0, batchSize * num_head * kvSlice, 1, [&](int64_t begin, int64_t end) {
    int ompIdx = at::get_thread_num();
    int64_t i = 0, j = 0, l = 0, n = 0;
    scalar_t* transpose_ptr = need_pack? transpose_buffer_ptr + ompIdx * kvSplitSize * headSize : nullptr;
    at::native::data_index_init(begin, i, batchSize, j, num_head, l, kvSlice);
    for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
      n = l * kvSplitSize;
      int64_t cur_kvSplitSize = std::min(kvSplitSize, kvSize - n);
      auto i_kv = is_broadcast_bs_kv ? i/bs_shards : i;
      auto j_kv = is_broadcast_head_kv ? j/gqa_shards : j;
      auto kv_block_num = n / cur_kvSplitSize;
      auto kv_block_offset = n - kv_block_num * cur_kvSplitSize;
      // getting kv indices by [BS, Head, 1, kv_block_num]
      auto i_kvi = is_broadcast_bs_kvi ? i/bs_shards_kvi : i;
      auto j_kvi = is_broadcast_head_kvi ? j/gqa_shards_kvi : j;
      auto kv_logical_data = kv_indices_data + i_kvi * kviStrideB +
                              j_kvi * kviStrideH + kv_block_num;
      auto k_addr =
            k_data + i_kv * kStrideB + j_kv * kStrideH + n * kStrideN;
      auto v_addr =
            v_data + i_kv * vStrideB + j_kv * vStrideH + n * vStrideN;
      if (use_kv_indice) {
          k_addr =
              k_data + i_kv * kStrideB + j_kv * kStrideH +
              (*kv_logical_data * cur_kvSplitSize + kv_block_offset) * kStrideN;
          v_addr =
              v_data + i_kv * vStrideB + j_kv * vStrideH +
              (*kv_logical_data * cur_kvSplitSize + kv_block_offset) * vStrideN;
      }
      if (need_pack) {
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
        {{kernel.kernel_name}}_pack_vnni2(
          /* src */ reinterpret_cast<const uint16_t*>(transpose_ptr),
          /* dst */ reinterpret_cast<uint16_t*>(key_reorder_ptr + i * num_head * eheadSize * kvSize +
                  j * eheadSize * kvSize + n * eheadSize),
          /* ld_src */ cur_kvSplitSize,
          /* K */ headSize,
          /* N */ cur_kvSplitSize);

        // Pack [cur_kvSplitSize, headSize_v]
        {{kernel.kernel_name}}_pack_vnni2(
          /* src */ reinterpret_cast<const uint16_t*>(v_addr),
          /* dst */ reinterpret_cast<uint16_t*>(value_reorder_ptr +
                  i * num_head * kv_padding_size * headSize_v +
                  j * kv_padding_size * headSize_v + n * headSize_v),
          /* ld_src */ vStrideN,
          /* K */ cur_kvSplitSize,
          /* N */ headSize_v);
      } else {
        using trans_t = std::conditional_t<std::is_same_v<scalar_t, at::BFloat16>, uint16_t, float>;
        at::native::utils::transpose<trans_t>(
          cur_kvSplitSize,
          headSize,
          /* src_ptr */
          reinterpret_cast<const trans_t*>(k_addr),
          /* ld_src */ kStrideN,
          /* dst */ reinterpret_cast<trans_t*>(key_reorder_ptr + i * num_head * eheadSize * kvSize +
                  j * eheadSize * kvSize + n * eheadSize),
          /* ld_dst */ cur_kvSplitSize);
      }
    // Move to the next query
    at::native::data_index_step(i, batchSize, j, num_head, l, kvSlice);
    }
  });

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
      int64_t m = k * qSplitSize;
      int64_t cur_qSplitSize = std::min(qSplitSize, qSize - m);
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
      for (int64_t n = 0; n < kvSize; n += kvSplitSize) {
        int64_t cur_kvSplitSize = std::min(kvSplitSize, kvSize - n);
        int64_t cur_ekvSplitSize = (need_pack && cur_kvSplitSize % 2 != 0) ? cur_kvSplitSize + 1 : cur_kvSplitSize;

        // Calculate scale * q @ k.T
        auto i_kv = is_broadcast_bs_kv ? i/bs_shards : i;
        auto j_kv = is_broadcast_head_kv ? j/gqa_shards : j;
        auto kv_block_num = n / kvBlockSize;
        auto kv_block_offset = n - kv_block_num * kvBlockSize;
        // getting kv indices by [BS, Head, 1, kv_block_num]
        auto i_kvi = is_broadcast_bs_kvi ? i/bs_shards_kvi : i;
        auto j_kvi = is_broadcast_head_kvi ? j/gqa_shards_kvi : j;
        auto kv_logical_data = kv_indices_data + i_kvi * kviStrideB +
                                j_kvi * kviStrideH + kv_block_num;
        if (!need_pack) {
          auto k_addr_t = key_reorder_ptr + i * num_head * eheadSize * kvSize +
                  j * eheadSize * kvSize + n * eheadSize;
          // TODO: use the micro-gemm template instead of brgemm API
          at::native::cpublas::brgemm(
              cur_qSplitSize,
              cur_kvSplitSize,
              eheadSize,
              int64_t(1),
              qStrideM,
              cur_kvSplitSize,
              cur_kvSplitSize,
              false,
              q_data + i * qStrideB + j * qStrideH +
                  m * qStrideM,
              k_addr_t,
              qk_data,
              need_pack);
        } else {
          at::native::cpublas::brgemm(
              cur_qSplitSize,
              cur_kvSplitSize,
              eheadSize,
              int64_t(1),
              headSize_even ? qStrideM : eheadSize,
              cur_kvSplitSize,
              cur_kvSplitSize,
              false,
              !headSize_even
                  ? query_t_padding_ptr
                  : q_data + i * qStrideB + j * qStrideH + m * qStrideM,
              key_reorder_ptr + i * num_head * eheadSize * kvSize +
                  j * eheadSize * kvSize + n * eheadSize,
              qk_data,
              need_pack);
        }

        {{kernel.kernel_name}}_mul_scale_kernel<accum_t>(qk_data, scaling_factor, cur_qSplitSize*cur_kvSplitSize);

{%- if score_mod and mask_mod %}
        // TODO: vectorization optimization for below score and mask codegen functions
        // apply score mod function
        for (int64_t row = 0; row < cur_qSplitSize; ++row) {
          for (int64_t col = 0; col < cur_kvSplitSize; col++) {
            std::vector<int64_t> b_idx = {i};
            std::vector<int64_t> h_idx = {j};
            std::vector<int64_t> q_idx = {m+row};
            int64_t phisical_kv_idx = n+col;
            if (use_kv_indice) {
                phisical_kv_idx= *kv_logical_data * kvBlockSize + col;
            }
            std::vector<int64_t> kv_idx = {phisical_kv_idx};
            accum_t* in_ptr0 = qk_data + row * cur_kvSplitSize + col;
            auto in_ptr1 = b_idx.data();
            auto in_ptr2 = h_idx.data();
            auto in_ptr3 = q_idx.data();
            auto in_ptr4 = kv_idx.data();
            {{ template.generate_other_buffer("score_others", 0, "len_score_other", kernel.args) }}
            accum_t* out_ptr{{score_buf_idx}} = in_ptr0;
            {{ template.modification(score_mod, score_buf_name, score_buf_idx) }}
          }
        }
        // Apply block mask, fill unused with -inf
        for (int64_t row = 0; row < cur_qSplitSize; ++row) {
          for (int64_t col = 0; col < cur_kvSplitSize; col++) {
            std::vector<int64_t> b_idx = {i};
            std::vector<int64_t> h_idx = {j};
            std::vector<int64_t> q_idx = {m+row};
            int64_t phisical_kv_idx = n+col;
            if (use_kv_indice) {
                phisical_kv_idx= *kv_logical_data * kvBlockSize + col;
            }
            std::vector<int64_t> kv_idx = {phisical_kv_idx};
            accum_t* qk_block = qk_data + row * cur_kvSplitSize + col;
            auto in_ptr1 = b_idx.data();
            auto in_ptr2 = h_idx.data();
            auto in_ptr3 = q_idx.data();
            auto in_ptr4 = kv_idx.data();
            {{ template.generate_other_buffer("mask_others", -1, "len_mask_other", kernel.args) }}
            std::vector<int64_t> temp = {0};
            int64_t* out_ptr{{mask_buf_idx}} = temp.data();
            {{ template.modification(mask_mod, mask_buf_name, mask_buf_idx) }}
            *qk_block = *out_ptr{{mask_buf_idx}} != 0
                            ? *qk_block
                            : -std::numeric_limits<accum_t>::infinity();
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
            if (n > 0) {
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
          if (use_kv_indice) {
              v_addr =
                  v_data + i_kv * vStrideB + j_kv * vStrideH +
                  (*kv_logical_data * kvBlockSize + kv_block_offset) * vStrideN;
          }
          at::native::cpublas::brgemm(
                  cur_qSplitSize,
                  headSize_v,
                  cur_ekvSplitSize,
                  int64_t(1),
                  cur_ekvSplitSize,
                  vStrideN,
                  headSize_v,
                  n > 0,
                  {{kernel.kernel_name}}_conditional_data_ptr(qk_data, qk_reduced_data),
                  v_addr,
                  dst_data,
                  need_pack);
        } else {
          int64_t psize = n / kvSplitSize * ekvSplitSize;
          at::native::cpublas::brgemm(
              cur_qSplitSize,
              headSize_v,
              cur_ekvSplitSize,
              int64_t(1),
              cur_ekvSplitSize,
              headSize_v,
              headSize_v,
              n > 0,
              qk_reduced_data,
              value_reorder_ptr +
                  i * num_head * kv_padding_size * headSize_v +
                  j * kv_padding_size * headSize_v + psize * headSize_v,
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
            [sum_reciprocal](Vec x) { return x * Vec(sum_reciprocal); },
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
        has_other_buffer,
        no_full_kv_block,
        fake_buffers,
        len_score_other,
        len_mask_other,
        kernel_input_name_to_buffer,
    ) -> None:
        assert layout.dtype in [torch.float, torch.bfloat16]
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
        self.has_other_buffer = has_other_buffer
        self.no_full_kv_block = no_full_kv_block
        self.other_buffer_input_offset = 1
        if self.no_full_kv_block:
            self.other_buffer_input_offset = 0
        self.fake_buffers = fake_buffers
        self.len_score_other = len_score_other
        self.len_mask_other = len_mask_other
        self.kernel_input_name_to_buffer = kernel_input_name_to_buffer
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
            start_offset = getattr(self, len_attr)

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
        from .cpp import CppKernelProxy, KernelGroup

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

        var_sizes = tuple([])  # type: ignore[var-annotated]  # noqa: C409
        output_index = 0
        var_ranges = {
            sympy_index_symbol_with_prefix(SymT.INDEX, i): sz
            for i, sz in enumerate(var_sizes)
        }

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
        ), "All the buffers in the score and mask subgraph should be in kernel_group.args.input_buffers"

        bodies.append(body)
        var_sizes_list.append((var_sizes, ()))

        cpp_kernel_proxy.codegen_loop_bodies(bodies, var_sizes_list)
        kernel_group.finalize_kernel(cpp_kernel_proxy, [])
        return kernel_group.loops_code.getvalue()

    @staticmethod
    def add_choices(
        choices,
        input_nodes,
        layout,
        scale,
        score_mod,
        mask_mod,
        kv_block_size,
        has_other_buffer,
        no_full_kv_block,
        fake_buffers,
        len_score_other,
        len_mask_other,
        kernel_input_name_to_buffer,
    ):
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
            has_other_buffer=has_other_buffer,
            no_full_kv_block=no_full_kv_block,
            fake_buffers=fake_buffers,
            len_score_other=len_score_other,
            len_mask_other=len_mask_other,
            kernel_input_name_to_buffer=kernel_input_name_to_buffer,
        )
        template.maybe_append_choice(choices)
        return template

    def apply_score_mod(self, score, b, h, q_idx, kv_idx):
        return self.score_mod.graph_module(score, b, h, q_idx, kv_idx).item()

    def render(  # type: ignore[override,return]
        self,
        kernel,
        template_buffer_node: Optional[ir.CppTemplateBuffer] = None,
        epilogue_nodes: Optional[List[ir.IRNode]] = None,
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

        num_threads = parallel_num_threads()
        buf_out = TensorBox.create(self.output_node)
        if template_buffer_node is not None:
            buf_out = template_buffer_node
        options = dict(
            query=query,
            key=key,
            value=value,
            kv_num_blocks=self.input_nodes[3],
            kv_indices=self.input_nodes[4],
            full_kv_num_blocks=self.input_nodes[5]
            if not self.no_full_kv_block
            else None,
            score_mod_other_buffers=self.score_mod_other_buffers,
            mask_mod_other_buffers=self.mask_mod_other_buffers,
            scale=self.scale,
            accumulate_dtype=torch.float,
            query_dtype=query.layout.dtype,
            kvBlockSize=self.kv_block_size,
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
