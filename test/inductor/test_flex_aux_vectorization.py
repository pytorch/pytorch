# Owner(s): ["module: inductor"]

from dataclasses import dataclass
from types import SimpleNamespace
from unittest import mock

import sympy

import torch
import torch._inductor.kernel.flex.flex_flash_attention as flex_flash_attention_module
from torch._inductor.kernel.flex.aux_vectorization import (
    AuxLoadVecInfo,
    direct_aux_load_vec_size_and_kind,
    select_mask_mod_vec_size,
    select_score_mod_vec_size,
)
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.testing._internal.inductor_utils import MockGraphHandler


@dataclass
class FakeAuxBuffer:
    size: tuple[int, ...]
    stride: tuple[int, ...]
    offset: int = 0

    def get_size(self):
        return self.size

    def get_stride(self):
        return self.stride

    def get_layout(self):
        return SimpleNamespace(offset=sympy.Integer(self.offset))


def _aux_index_graph():
    graph = torch.fx.Graph()
    return graph, graph.placeholder("q_idx"), graph.placeholder("kv_idx")


@instantiate_parametrized_tests
class TestFlexAuxVectorization(InductorTestCase):
    def test_direct_aux_load_vec_size_selector(self):
        graph, q_idx, kv_idx = _aux_index_graph()
        kv_mod_4 = graph.call_function(torch.ops.aten.remainder.Tensor, (kv_idx, 4))
        kv_stride_mix = graph.call_function(
            torch.ops.aten.sub.Tensor,
            (
                graph.call_function(torch.ops.aten.mul.Tensor, (kv_idx, 2)),
                kv_mod_4,
            ),
        )
        kv_times_2 = graph.call_function(torch.ops.aten.mul.Tensor, (kv_idx, 2))
        kv_floor_div_2 = graph.call_function(
            torch.ops.aten.div.Tensor_mode,
            (kv_idx, 2),
            {"rounding_mode": "floor"},
        )

        buffer = FakeAuxBuffer((128,), (1,))
        with V.set_graph_handler(MockGraphHandler()):
            self.assertEqual(
                direct_aux_load_vec_size_and_kind([kv_idx], buffer, q_idx, kv_idx),
                AuxLoadVecInfo.contiguous(8),
            )
            self.assertEqual(
                direct_aux_load_vec_size_and_kind([kv_mod_4], buffer, q_idx, kv_idx),
                AuxLoadVecInfo.contiguous(4),
            )
            self.assertEqual(
                direct_aux_load_vec_size_and_kind([q_idx], buffer, q_idx, kv_idx),
                AuxLoadVecInfo.lane_uniform(),
            )
            self.assertEqual(
                direct_aux_load_vec_size_and_kind(
                    [kv_stride_mix], buffer, q_idx, kv_idx
                ),
                AuxLoadVecInfo.gather(),
            )
            self.assertEqual(
                direct_aux_load_vec_size_and_kind([kv_times_2], buffer, q_idx, kv_idx),
                AuxLoadVecInfo.gather(),
            )
            self.assertEqual(
                direct_aux_load_vec_size_and_kind(
                    [kv_floor_div_2], buffer, q_idx, kv_idx
                ),
                AuxLoadVecInfo.gather(),
            )
            self.assertEqual(
                direct_aux_load_vec_size_and_kind(
                    [kv_idx, q_idx],
                    FakeAuxBuffer((128, 128), (128, 1)),
                    q_idx,
                    kv_idx,
                ),
                AuxLoadVecInfo.gather(),
            )

    def test_direct_aux_load_vec_size_requires_contiguous_aligned_vector_dim(self):
        graph, q_idx, kv_idx = _aux_index_graph()
        with V.set_graph_handler(MockGraphHandler()):
            self.assertEqual(
                direct_aux_load_vec_size_and_kind(
                    [q_idx, kv_idx],
                    FakeAuxBuffer((128, 128), (128, 1)),
                    q_idx,
                    kv_idx,
                ),
                AuxLoadVecInfo.contiguous(8),
            )
            self.assertEqual(
                direct_aux_load_vec_size_and_kind(
                    [q_idx, kv_idx],
                    FakeAuxBuffer((128, 48), (48, 1)),
                    q_idx,
                    kv_idx,
                    max_vec_size=32,
                ),
                AuxLoadVecInfo.contiguous(16),
            )
            self.assertEqual(
                direct_aux_load_vec_size_and_kind(
                    [q_idx, kv_idx],
                    FakeAuxBuffer((128, 128), (1, 128)),
                    q_idx,
                    kv_idx,
                ),
                AuxLoadVecInfo.gather(),
            )
            self.assertEqual(
                direct_aux_load_vec_size_and_kind(
                    [q_idx, kv_idx],
                    FakeAuxBuffer((128, 128), (9, 1)),
                    q_idx,
                    kv_idx,
                ),
                AuxLoadVecInfo.gather(),
            )
            self.assertEqual(
                direct_aux_load_vec_size_and_kind(
                    [kv_idx], FakeAuxBuffer((128,), (1,), offset=1), q_idx, kv_idx
                ),
                AuxLoadVecInfo.gather(),
            )

    @parametrize(
        "case",
        [
            "direct_qkv",
            "chained_rank4_batch_head_qkv",
            "kv_in_prefix",
            "rank1_kv",
            "lane_uniform_aux",
            "mixed_gather",
            "gather_only",
        ],
        name_fn=lambda case: case,
    )
    def test_mask_mod_vec_size_selector_invariants(self, case):
        graph = torch.fx.Graph()
        b = graph.placeholder("b")
        h = graph.placeholder("h")
        q_idx = graph.placeholder("q_idx")
        kv_idx = graph.placeholder("kv_idx")

        def load(buffer, indices):
            return graph.call_function(torch.ops.aten.index.Tensor, (buffer, indices))

        def floordiv(index, divisor):
            return graph.call_function(
                torch.ops.aten.div.Tensor_mode,
                (index, divisor),
                {"rounding_mode": "floor"},
            )

        expected = 32
        match case:
            case "direct_qkv":
                mask_bias = graph.placeholder("mask_bias")
                output = load(mask_bias, [q_idx, kv_idx])
                buffers = [FakeAuxBuffer((128, 128), (128, 1))]
            case "chained_rank4_batch_head_qkv":
                bias = graph.placeholder("bias")
                batch_slice = load(bias, [b])
                head_slice = load(batch_slice, [h])
                row = load(head_slice, [q_idx])
                output = load(row, [kv_idx])
                buffers = [FakeAuxBuffer((2, 4, 128, 128), (65536, 16384, 128, 1))]
            case "kv_in_prefix":
                bias = graph.placeholder("bias")
                direct_load = load(bias, [b, kv_idx, q_idx])
                kv_slice = load(bias, [b, kv_idx])
                output = (direct_load, load(kv_slice, [q_idx]))
                buffers = [FakeAuxBuffer((2, 128, 128), (16384, 128, 1))]
                expected = None
            case "rank1_kv":
                mask_bias = graph.placeholder("mask_bias")
                output = load(mask_bias, [kv_idx])
                buffers = [FakeAuxBuffer((128,), (1,))]
                expected = None
            case "lane_uniform_aux":
                mask_bias = graph.placeholder("mask_bias")
                output = load(mask_bias, [h])
                buffers = [FakeAuxBuffer((4,), (1,))]
            case "mixed_gather":
                mask_bias = graph.placeholder("mask_bias")
                block_keep = graph.placeholder("block_keep")
                output = (
                    load(mask_bias, [q_idx, kv_idx]),
                    load(block_keep, [floordiv(q_idx, 128), floordiv(kv_idx, 128)]),
                )
                buffers = [
                    FakeAuxBuffer((128, 128), (128, 1)),
                    FakeAuxBuffer((1, 1), (1, 1)),
                ]
            case "gather_only":
                block_keep = graph.placeholder("block_keep")
                output = load(block_keep, [floordiv(q_idx, 128), floordiv(kv_idx, 128)])
                buffers = [FakeAuxBuffer((1, 1), (1, 1))]
                expected = None
            case _:
                raise AssertionError(case)

        graph.output(output)
        graph_module = torch.fx.GraphModule({}, graph)
        with V.set_graph_handler(MockGraphHandler()):
            self.assertEqual(
                select_mask_mod_vec_size(
                    has_mask_mod=True,
                    has_mask_aux_tensors=True,
                    supports_mask_mod_vec=True,
                    graph_module=graph_module,
                    other_buffers=buffers,
                ),
                expected,
            )

    def test_score_mod_vec_size_selector_allows_mixed_gather(self):
        graph = torch.fx.Graph()
        graph.placeholder("score")
        graph.placeholder("b")
        graph.placeholder("h")
        q_idx = graph.placeholder("q_idx")
        kv_idx = graph.placeholder("kv_idx")
        bias = graph.placeholder("bias")
        block_keep = graph.placeholder("block_keep")
        direct_load = graph.call_function(
            torch.ops.aten.index.Tensor, (bias, [q_idx, kv_idx])
        )
        block_idx = graph.call_function(
            torch.ops.aten.div.Tensor_mode,
            (kv_idx, 128),
            {"rounding_mode": "floor"},
        )
        gather_load = graph.call_function(
            torch.ops.aten.index.Tensor, (block_keep, [block_idx])
        )
        graph.output((direct_load, gather_load))
        graph_module = torch.fx.GraphModule({}, graph)
        with V.set_graph_handler(MockGraphHandler()):
            self.assertEqual(
                select_score_mod_vec_size(
                    has_score_mod=True,
                    has_aux_tensors=True,
                    is_sm100_or_later=True,
                    graph_module=graph_module,
                    other_buffers=[
                        FakeAuxBuffer((128, 128), (128, 1)),
                        FakeAuxBuffer((1,), (1,)),
                    ],
                ),
                8,
            )

    @parametrize("chained", [False, True], name_fn=lambda chained: str(chained))
    def test_score_mod_vec_size_selector_rejects_score_placeholder_index(self, chained):
        graph = torch.fx.Graph()
        score = graph.placeholder("score")
        graph.placeholder("b")
        graph.placeholder("h")
        graph.placeholder("q_idx")
        kv_idx = graph.placeholder("kv_idx")
        bias = graph.placeholder("bias")
        if chained:
            row = graph.call_function(torch.ops.aten.index.Tensor, (bias, [score]))
            load = graph.call_function(torch.ops.aten.index.Tensor, (row, [kv_idx]))
        else:
            load = graph.call_function(
                torch.ops.aten.index.Tensor, (bias, [score, kv_idx])
            )
        graph.output(load)
        graph_module = torch.fx.GraphModule({}, graph)
        with V.set_graph_handler(MockGraphHandler()):
            self.assertEqual(
                select_score_mod_vec_size(
                    has_score_mod=True,
                    has_aux_tensors=True,
                    is_sm100_or_later=True,
                    graph_module=graph_module,
                    other_buffers=[FakeAuxBuffer((128, 128), (128, 1))],
                ),
                1,
            )

    @parametrize(
        "cuda_major", [9, 10, 11, 12], name_fn=lambda cuda_major: str(cuda_major)
    )
    def test_mask_mod_vec_config_supports_only_sm100_path(self, cuda_major):
        expected_config = (
            flex_flash_attention_module.FlexFlashConfig(mask_mod_vec_size=32)
            if cuda_major in (10, 11)
            else flex_flash_attention_module.FlexFlashConfig()
        )
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(
                torch.cuda, "get_device_capability", return_value=(cuda_major, 0)
            ),
        ):
            self.assertEqual(
                flex_flash_attention_module.get_flex_flash_fwd_configs(
                    False, False, has_mask_mod=True
                ),
                [expected_config],
            )

    @torch._inductor.config.patch(
        {"max_autotune": True, "test_configs.max_flex_configs": None}
    )
    def test_mask_mod_vec_config_combines_with_max_autotune_score_sizes(self):
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(
                torch.cuda, "get_device_capability", return_value=(10, 0)
            ),
        ):
            self.assertEqual(
                flex_flash_attention_module.get_flex_flash_fwd_configs(
                    True, False, has_mask_mod=True
                ),
                [
                    flex_flash_attention_module.FlexFlashConfig(
                        score_mod_vec_size=vec_size,
                        mask_mod_vec_size=32,
                    )
                    for vec_size in (1, 2, 4, 8, 16, 32, 64, 128)
                ],
            )


if __name__ == "__main__":
    run_tests()
