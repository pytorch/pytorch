# mypy: allow-untyped-defs
import contextlib
import logging
from typing import List, Optional

import torch
import torch.utils

from .. import ir
from ..ir import TensorBox
from ..select_algorithm import DataProcessorTemplateWrapper
from ..utils import parallel_num_threads
from .cpp_template import CppTemplate


log = logging.getLogger(__name__)

ATTENTION_TEMPLATE = r"""
{{template.header().getvalue()}}
#include <ATen/native/CPUBlas.h>

{%- set kernel_args = {"query": query, "key": key, "value": value, "kv_indices": kv_indices} %}
{{kernel.def_kernel(inputs=kernel_args, outputs={"output": output})}}
{
  // kv page size, q and kv split size
  int64_t kv_pagesize = {{kv_pagesize}};
  int64_t q_split_size = {{q_split_size}};
  int64_t kv_split_size = {{kv_pagesize}};

  // dtypes of kernel and internal buffers
  using scalar_t = {{kernel.dtype(query)}};
  constexpr bool is_reduced_type = std::is_reduced_floating_point_v<scalar_t>;
  using accum_t = at::opmath_type<{{kernel.dtype(query)}}>;
  using Vec = at::vec::Vectorized<accum_t>;
  accum_t scaling_factor = {{scale}};

  int64_t batchSize = {{kernel.size(query, 0)}};
  int64_t qSize = {{kernel.size(query, 1)}};
  // real k/v length will be padded based on kv_pagesize
  int64_t kvSize = {{kv_num_blocks}} * kv_pagesize;
  int64_t num_head = {{kernel.size(query, 2)}};
  int64_t headSize = {{kernel.size(query, 3)}};
  int64_t batchSize_k = {{kernel.size(key, 0)}};
  int64_t num_head_k = {{kernel.size(key, 2)}};
  bool is_broadcast_bs_kv = batchSize != batchSize_k;
  bool is_broadcast_head_kv = num_head != num_head_k;
  int64_t gqa_shards = num_head / num_head_k;
  int64_t bs_shards = batchSize / batchSize_k;

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
  int64_t kviStrideB = {{kernel.stride(kv_indices, 0)}};
  int64_t kviStrideH = {{kernel.stride(kv_indices, 1)}};
  int64_t kviStrideQ = {{kernel.stride(kv_indices, 2)}};

  int64_t qSplitSize = q_split_size > qSize ? qSize : q_split_size;
  int64_t kvSplitSize = kv_split_size > kvSize ? kvSize : kv_split_size;
  int64_t qSlice = (qSize + qSplitSize - 1) / qSplitSize;
  int64_t kvSlice = (kvSize + kvSplitSize - 1) / kvSplitSize;
  int64_t kvTail = (kvSize - 1) % kvSplitSize + 1;

  int64_t rHeadSize = headSize;
  int64_t rkvSplitSize = kvSplitSize;
  int64_t rkvTail = kvTail;
  int64_t ekvSplitSize = kvSplitSize;
  int64_t ekvTail = kvTail;

  // allocate per thread temp buf (accumulate type)
  int64_t size_per_thread = qSplitSize * rkvSplitSize + qSplitSize + qSplitSize + qSplitSize * rHeadSize;
  {%- set acc_buf_name = "buf" %}
      {{kernel.define_buffer(acc_buf_name, [num_thread, size_per_thread], dtype=accumulate_dtype)}}
  {%- set acc_reduced_buf_name = "buf_reduced" %}
      {{ kernel.define_buffer(acc_reduced_buf_name, [num_thread, qSplitSize, ekvSplitSize], dtype=query_dtype)}}


  const scalar_t* q_data = query;
  const scalar_t* k_data = key;
  const scalar_t* v_data = value;
  auto  kv_indices_data = kv_indices;

  scalar_t* out_data = output;

  accum_t* buf_data = buf;
  scalar_t* buf_reduced_data = is_reduced_type ? buf_reduced : nullptr;

  at::parallel_for(0, batchSize * num_head * qSlice, 1, [&](int64_t begin, int64_t end) {
     int64_t i = 0, j = 0, k = 0;
     at::native::data_index_init(begin, i, batchSize, j, num_head, k, qSlice);
        int ompIdx = at::get_thread_num();
        accum_t* buf_ptr = buf_data + ompIdx * size_per_thread;
        accum_t* qk_data = buf_ptr;
        accum_t* qk_max_data = qk_data + qSplitSize * rkvSplitSize;
        accum_t* qk_sum_data = qk_max_data + qSplitSize;
        accum_t* dst_data = qk_sum_data + qSplitSize;
        scalar_t* qk_reduced_data = is_reduced_type ? buf_reduced_data + ompIdx * qSplitSize * ekvSplitSize : nullptr;
        scalar_t* query_t_padding_ptr = nullptr;

        for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
          int64_t m = k * qSplitSize;
          int64_t qBlockSize = std::min(qSplitSize, qSize - m);
          // Initialize max and sum
          fill_stub(qk_max_data,
              -std::numeric_limits<accum_t>::infinity(), qBlockSize);
          fill_stub(qk_sum_data,
              static_cast<accum_t>(0), qBlockSize);
          int64_t num_keys = kvSize;

          for (int64_t n = 0; n < num_keys; n += kvSplitSize) {
            int64_t kvBlockSize = std::min(kvSplitSize, kvSize - n);
            int64_t ekvBlockSize = kvBlockSize;
            int64_t rkvBlockSize = kvBlockSize == kvSplitSize ? rkvSplitSize : rkvTail;

            // Calculate scale * q @ k.T
            auto kv_block_num = n / kv_pagesize;
            auto kv_block_offset = n - kv_block_num * kv_pagesize;
            // getting kv indices by [BS, 1, 1, kv_block_num]
            auto kv_logical_data = kv_indices_data + i * kviStrideB  + kv_block_num;
            auto i_kv = is_broadcast_bs_kv ? i/bs_shards : i;
            auto j_kv = is_broadcast_head_kv ? j/gqa_shards : j;

            auto k_addr = k_data + i_kv * kStrideB + j_kv * kStrideH + (*kv_logical_data * kv_pagesize + kv_block_offset)  * kStrideN;

            at::native::cpublas::gemm(
              at::native::TransposeType::Transpose,
              at::native::TransposeType::NoTranspose,
              kvBlockSize,
              qBlockSize,
              headSize,
              static_cast<accum_t>(1),
              k_addr,
              kStrideN,
              q_data + i * qStrideB + j * qStrideH +
                  m * qStrideM,
              qStrideM,
              static_cast<accum_t>(0),
              qk_data,
              kvBlockSize);

            // Update coefficients with Softmax
            accum_t tmp_max = 0, tmp_sum = 0, exp_tmp = 0;
            for (int64_t row = 0; row < qBlockSize; ++row) {
              // apply scaling factor and max per row in fusion
              _mul_reduce_max_fusion_kernel(
                  qk_data + row * rkvBlockSize,
                  scaling_factor,
                  kvBlockSize,
                  qk_data + row * rkvBlockSize,
                  tmp_max);
              tmp_max = qk_max_data[row] > tmp_max ? qk_max_data[row] : tmp_max;
              if (tmp_max == -std::numeric_limits<accum_t>::infinity()) {
                // to avoid `nan = exp2f(-inf - (-inf))`
                fill_stub(conditional_data_ptr(qk_data, qk_reduced_data) + row * ekvBlockSize,
                  static_cast<scalar_t>(0), kvBlockSize);
              } else {
                tmp_sum = tmp_max;
                // qk <- exp(qk - max) and sum per row
                _exp_reduce_sum_fusion_kernel(
                    qk_data + row * rkvBlockSize, kvBlockSize,
                    conditional_data_ptr(qk_data, qk_reduced_data) + row * ekvBlockSize,
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
                    dst_data + row * rHeadSize,
                    dst_data + row * rHeadSize,
                    headSize);
                }
              }
            }
            // Calculate Softmax(q @ k.T) @ v
            auto v_addr = v_data + i_kv * vStrideB + j_kv * vStrideH + (*kv_logical_data * kv_pagesize + kv_block_offset)  * vStrideN;

            at::native::cpublas::gemm(
              at::native::TransposeType::NoTranspose,
              at::native::TransposeType::NoTranspose,
              headSize,
              qBlockSize,
              kvBlockSize,
              static_cast<accum_t>(1),
              v_addr,
              vStrideN,
              conditional_data_ptr(qk_data, qk_reduced_data),
              kvBlockSize,
              n == 0 ? static_cast<accum_t>(0) : static_cast<accum_t>(1),
              dst_data,
              headSize);
          }

          // dst <- dst / sum[row]
          // reorder MHA output with strides
          for (int64_t row = 0; row < qBlockSize; ++row) {
            // Row sums for full masked out rows are 0, we set them to 1
            // in order to avoid NaNs in the output and instead set fully
            // masked out rows to 0
            qk_max_data[row] = qk_max_data[row] == -std::numeric_limits<accum_t>::infinity() ? 0 : qk_max_data[row];
            qk_sum_data[row] = qk_sum_data[row] == 0 ? 1 : qk_sum_data[row];
            accum_t sum_reciprocal = 1 / qk_sum_data[row];
            at::vec::map<scalar_t>(
              [sum_reciprocal](Vec x) { return x * Vec(sum_reciprocal); },
              out_data + i * oStrideB + j * oStrideH + m * oStrideM + row * oStrideM,
              dst_data + row * rHeadSize,
              headSize);
          }
          // Move to the next query
      at::native::data_index_step(i, batchSize, j, num_head, k, qSlice);
    }
    });
}
"""


class CppMHATemplate(CppTemplate):
    def __init__(
        self,
        input_nodes,
        layout: ir.Layout,
        scale,
        score_mod,
        block_mask,
        kv_block_size,
        kv_num_blocks,
    ) -> None:
        assert layout.dtype in [torch.float, torch.float16, torch.bfloat16]
        super().__init__("mha", input_nodes, layout, parallel_num_threads())
        self.scale = scale
        self.score_mod = score_mod
        self.block_mask = block_mask
        self.kv_block_size = kv_block_size
        self.kv_num_blocks = kv_num_blocks

    @staticmethod
    def add_choices(
        choices,
        input_nodes,
        layout,
        scale,
        score_mod,
        block_mask,
        kv_block_size,
        kv_num_blocks,
    ):
        def preprocessor(input_nodes, layout):
            return input_nodes, layout

        def postprocessor(output):
            return output

        template = DataProcessorTemplateWrapper(
            CppMHATemplate,
            preprocessor,
            postprocessor,
            input_nodes=input_nodes,
            layout=layout,
            scale=scale,
            score_mod=score_mod,
            block_mask=block_mask,
            kv_block_size=kv_block_size,
            kv_num_blocks=kv_num_blocks.layout.size[-1],
        )
        template.maybe_append_choice(choices)
        return template

    def apply_score_mod(self, score, b, h, q_idx, kv_idx):
        return self.score_mod.graph_module(score, b, h, q_idx, kv_idx).item()

    def render(  # type: ignore[override,return]
        self,
        kernel,
        template_buffer_node: Optional[ir.CppTemplateBuffer] = None,
        flag_template_buffer_has_other_users: Optional[bool] = None,
        epilogue_nodes: Optional[List[ir.IRNode]] = None,
        **kwargs,
    ) -> str:
        # Query (Batch x Num_heads  x Q_seq_len  x Dim_per_head)
        #     -> (Batch x Q_seq_len  x Num_heads  x Dim_per_head)
        #  Key   (Batch x Num_heads  x KV_seq_len x Dim_per_head)
        #     -> (Batch x KV_seq_len x Num_heads  x Dim_per_head)
        #  Value (Batch x Num_heads  x KV_seq_len x Dim_per_head)
        #     -> (Batch x KV_seq_len x Num_heads  x Dim_per_head)
        query = kernel.permute(self.input_nodes[0], [0, 2, 1, 3])
        key = kernel.permute(self.input_nodes[1], [0, 2, 1, 3])
        value = kernel.permute(self.input_nodes[2], [0, 2, 1, 3])

        q_split_size = 16  # tuning param
        kv_split_size = self.kv_block_size

        qSize = query.layout.size[1]
        kvSize = self.kv_block_size * self.kv_num_blocks
        headSize = query.layout.size[3]
        qSplitSize = min(q_split_size, qSize)
        kvSplitSize = min(kv_split_size, kvSize)
        size_per_thread = (
            qSplitSize * kvSplitSize + qSplitSize + qSplitSize + qSplitSize * headSize
        )
        num_threads = parallel_num_threads()

        buf_out = TensorBox.create(self.output_node)
        if template_buffer_node is not None:
            buf_out = template_buffer_node

        options = dict(
            query=query,
            key=key,
            value=value,
            kv_indices=self.input_nodes[3],
            scale=self.scale,
            size_per_thread=size_per_thread,
            accumulate_dtype=torch.float,
            query_dtype=query.layout.dtype,
            qSplitSize=qSplitSize,
            ekvSplitSize=kvSplitSize,
            q_split_size=q_split_size,
            kv_split_size=kv_split_size,
            kv_pagesize=self.kv_block_size,
            kv_num_blocks=self.kv_num_blocks,
            template=self,
            output=buf_out,
            kernel=kernel,
            num_thread=num_threads,
        )
        with contextlib.ExitStack() as stack:
            return self._template_from_string(ATTENTION_TEMPLATE).render(**options)
