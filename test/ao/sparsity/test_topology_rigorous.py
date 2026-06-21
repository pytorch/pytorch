# Owner(s): ["module: sparse"]
# ruff: noqa: S101

import importlib.util
import io
import os
import pathlib
import sys
import unittest

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


try:
    from torch.testing._internal.common_utils import raise_on_run_directly
except (ImportError, ModuleNotFoundError):
    raise_on_run_directly = None


try:
    from torch.ao.pruning._experimental.topology import (
        LyapunovSpectralGuard,
        SpectralPrototypeEmbedding,
        TopologicalSpectralAttention,
        TopologyGatedLowRankLinear,
    )
except ModuleNotFoundError:
    if os.environ.get("PYTORCH_TEST_TOPOLOGY_SOURCE_FALLBACK") != "1":
        raise
    _ROOT = pathlib.Path(__file__).resolve().parents[3]
    _TOPOLOGY = _ROOT / "torch" / "ao" / "pruning" / "_experimental" / "topology"
    _SPEC = importlib.util.spec_from_file_location(
        "topology_under_test",
        _TOPOLOGY / "__init__.py",
        submodule_search_locations=[str(_TOPOLOGY)],
    )
    _MODULE = importlib.util.module_from_spec(_SPEC)
    sys.modules["topology_under_test"] = _MODULE
    assert _SPEC.loader is not None
    _SPEC.loader.exec_module(_MODULE)
    LyapunovSpectralGuard = _MODULE.LyapunovSpectralGuard
    SpectralPrototypeEmbedding = _MODULE.SpectralPrototypeEmbedding
    TopologicalSpectralAttention = _MODULE.TopologicalSpectralAttention
    TopologyGatedLowRankLinear = _MODULE.TopologyGatedLowRankLinear


def _rank1_linear(in_features=8, out_features=6, bias=True, device="cpu"):
    linear = torch.nn.Linear(in_features, out_features, bias=bias, device=device)
    with torch.no_grad():
        u = torch.linspace(1.0, 2.0, out_features, device=device).unsqueeze(1)
        v = torch.linspace(-0.5, 0.5, in_features, device=device).unsqueeze(0)
        linear.weight.copy_(u @ v)
        if bias:
            linear.bias.copy_(torch.linspace(-0.1, 0.1, out_features, device=device))
    return linear


def _community_embedding(device="cpu"):
    embedding = torch.nn.Embedding(8, 3, device=device)
    with torch.no_grad():
        embedding.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.5, 0.0],
                    [-1.0, 0.5, 0.0],
                    [-1.0, 0.5, 0.0],
                    [-1.0, 0.5, 0.0],
                ],
                device=device,
            )
        )
    return embedding


class TestTopologyGatedLowRankLinearRigorous(unittest.TestCase):
    def test_constructor_validation_and_counts(self):
        with pytest.raises(ValueError, match="energy_threshold"):
            TopologyGatedLowRankLinear(2, 2, energy_threshold=0.0)
        with pytest.raises(ValueError, match="max_rank"):
            TopologyGatedLowRankLinear(2, 2, max_rank=0)
        with pytest.raises(ValueError, match="min_compression_ratio"):
            TopologyGatedLowRankLinear(2, 2, min_compression_ratio=1.0)
        module = TopologyGatedLowRankLinear(3, 4, bias=True)
        assert module.dense_parameter_count() == 16
        assert module.compressed_parameter_count(rank=1) == 11
        assert module.compressed_parameter_count() == 25
        assert "is_compressed=False" in module.extra_repr()
        no_bias = TopologyGatedLowRankLinear(3, 4, bias=False)
        assert no_bias.dense_parameter_count() == 12
        no_bias.reset_parameters()

    def test_factory_forward_fallback_and_tiny_edge_case(self):
        dense = torch.nn.Linear(1, 1)
        module = TopologyGatedLowRankLinear.from_linear(dense)
        x = torch.empty(0, 1)
        torch.testing.assert_close(module(x), dense(x))
        assert not module.try_compress()
        torch.testing.assert_close(module(torch.ones(2, 1)), dense(torch.ones(2, 1)))

    def test_compressed_forward_matches_rank1_dense_and_is_deterministic(self):
        torch.manual_seed(123)
        dense = _rank1_linear()
        module = TopologyGatedLowRankLinear.from_linear(
            dense,
            energy_threshold=0.999,
            max_rank=1,
            min_compression_ratio=1.2,
        )
        x = torch.randn(4, 8)
        expected = dense(x)
        assert module.try_compress()
        assert module.try_compress()
        torch.testing.assert_close(module(x), expected, atol=1e-5, rtol=1e-5)
        assert module.compressed_parameter_count() == 20
        assert "is_compressed=True" in module.extra_repr()

        torch.manual_seed(123)
        dense_again = _rank1_linear()
        module_again = TopologyGatedLowRankLinear.from_linear(
            dense_again,
            energy_threshold=0.999,
            min_compression_ratio=1.2,
        )
        assert module_again.try_compress()
        torch.testing.assert_close(module(x), module_again(x), atol=0, rtol=0)

        full_rank = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            full_rank.weight.copy_(torch.eye(4))
        clipped = TopologyGatedLowRankLinear.from_linear(
            full_rank,
            energy_threshold=0.99,
            max_rank=1,
            min_compression_ratio=1.1,
        )
        assert not clipped.try_compress()

    def test_gradients_flow_and_training_loss_decreases_after_compression(self):
        torch.manual_seed(0)
        teacher = _rank1_linear()
        module = TopologyGatedLowRankLinear.from_linear(teacher, energy_threshold=0.999, min_compression_ratio=1.2)
        assert module.try_compress()
        x = torch.randn(32, 8)
        target = teacher(x).detach() + 0.1
        opt = torch.optim.SGD(module.parameters(), lr=0.2)
        first_loss = None
        for _ in range(10):
            opt.zero_grad()
            loss = F.mse_loss(module(x), target)
            if first_loss is None:
                first_loss = loss.detach()
            loss.backward()
            assert module.low_rank_left.grad is not None
            assert module.low_rank_right.grad is not None
            opt.step()
        assert loss < first_loss

    def test_optimizer_state_is_rewritten_when_compressing_after_warmup(self):
        torch.manual_seed(0)
        teacher = _rank1_linear()
        module = TopologyGatedLowRankLinear.from_linear(teacher, energy_threshold=0.999, min_compression_ratio=1.2)
        opt = torch.optim.Adam(module.parameters(), lr=0.01)
        x = torch.randn(8, 8)
        F.mse_loss(module(x), teacher(x).detach()).backward()
        opt.step()
        old_weight = module.weight
        assert old_weight in opt.state

        assert module.try_compress(optimizer=opt)
        assert old_weight not in opt.state
        assert all(param is not old_weight for group in opt.param_groups for param in group["params"])
        assert any(param is module.low_rank_left for group in opt.param_groups for param in group["params"])
        assert any(param is module.low_rank_right for group in opt.param_groups for param in group["params"])

    def test_state_dict_and_torch_save_round_trip_after_compression(self):
        dense = _rank1_linear()
        module = TopologyGatedLowRankLinear.from_linear(dense, energy_threshold=0.999, min_compression_ratio=1.2)
        assert module.try_compress()
        x = torch.randn(3, 8)
        expected = module(x)

        loaded = TopologyGatedLowRankLinear(8, 6)
        loaded.load_state_dict(module.state_dict())
        assert loaded.is_compressed
        torch.testing.assert_close(loaded(x), expected)

        buffer = io.BytesIO()
        torch.save(module.state_dict(), buffer)
        buffer.seek(0)
        round_tripped_state = torch.load(buffer)
        round_tripped = TopologyGatedLowRankLinear(8, 6)
        round_tripped.load_state_dict(round_tripped_state)
        torch.testing.assert_close(round_tripped(x), expected)

    def test_internal_edge_paths_and_dense_reload_after_compression(self):
        dense = _rank1_linear(bias=False)
        module = TopologyGatedLowRankLinear.from_linear(dense, energy_threshold=0.999, min_compression_ratio=1.2)
        assert module._target_rank(torch.empty(0)) is None
        assert module._target_rank(torch.zeros(2)) is None
        assert module.try_compress()
        compressed_state = module.state_dict()
        module.low_rank_left = None
        with pytest.raises(RuntimeError, match="factors are missing"):
            module(torch.randn(2, 8))

        dense_state = TopologyGatedLowRankLinear.from_linear(dense).state_dict()
        module.load_state_dict(dense_state)
        assert not module.is_compressed
        x = torch.randn(2, 8)
        torch.testing.assert_close(module(x), dense(x))

        wrong_shape = TopologyGatedLowRankLinear(2, 2, bias=False)
        with pytest.raises(RuntimeError, match="size mismatch for low-rank checkpoint"):
            wrong_shape.load_state_dict(compressed_state)

    def test_half_precision_compression_and_state_load_device_dtype(self):
        dense = _rank1_linear().half()
        module = TopologyGatedLowRankLinear.from_linear(dense, energy_threshold=0.999, min_compression_ratio=1.2)
        assert module.try_compress()
        assert module.low_rank_left.dtype == torch.float16

        if torch.cuda.is_available():
            target = TopologyGatedLowRankLinear(8, 6, device="cuda", dtype=torch.float16)
            target.load_state_dict({name: value.cpu() for name, value in module.state_dict().items()})
            assert target.low_rank_left.device.type == "cuda"
            assert target.low_rank_left.dtype == torch.float16

    def test_checkpoint_compile_and_amp_compatibility(self):
        dense = _rank1_linear()
        module = TopologyGatedLowRankLinear.from_linear(dense, energy_threshold=0.999, min_compression_ratio=1.2)
        assert module.try_compress()
        x = torch.randn(2, 8, requires_grad=True)
        out = checkpoint(module, x, use_reentrant=False)
        out.sum().backward()
        assert x.grad is not None

        compiled = torch.compile(module, backend="eager")
        torch.testing.assert_close(compiled(x.detach()), module(x.detach()))

        if torch.cuda.is_available():
            cuda_module = module.cuda()
            cuda_x = x.detach().cuda()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                amp_out = cuda_module(cuda_x)
            assert amp_out.dtype == torch.float16

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA memory check requires a GPU")
    @pytest.mark.skipif(os.environ.get("PYTORCH_TEST_TOPOLOGY_MEMORY") != "1", reason="allocator-sensitive memory check is opt-in")
    def test_gpu_peak_memory_is_lower_than_dense_baseline(self):
        device = "cuda"
        torch.cuda.empty_cache()
        dense = _rank1_linear(2048, 2048, device=device)
        x = torch.randn(16, 2048, device=device, requires_grad=True)
        torch.cuda.reset_peak_memory_stats()
        dense(x).sum().backward()
        dense_peak = torch.cuda.max_memory_allocated()
        del dense, x
        torch.cuda.empty_cache()

        dense_for_compression = _rank1_linear(2048, 2048, device=device)
        compressed = TopologyGatedLowRankLinear.from_linear(
            dense_for_compression,
            energy_threshold=0.999,
            min_compression_ratio=10.0,
        )
        assert compressed.try_compress()
        del dense_for_compression
        torch.cuda.empty_cache()
        x = torch.randn(16, 2048, device=device, requires_grad=True)
        torch.cuda.reset_peak_memory_stats()
        compressed(x).sum().backward()
        compressed_peak = torch.cuda.max_memory_allocated()
        assert compressed_peak < dense_peak


class TestSpectralPrototypeEmbeddingRigorous(unittest.TestCase):
    def test_constructor_validation_observation_and_empty_graph_fallback(self):
        with pytest.raises(ValueError, match="num_prototypes"):
            SpectralPrototypeEmbedding(4, 2, num_prototypes=0)
        with pytest.raises(ValueError, match="min_observations"):
            SpectralPrototypeEmbedding(4, 2, num_prototypes=2, min_observations=0)
        with pytest.raises(ValueError, match="window_size"):
            SpectralPrototypeEmbedding(4, 2, num_prototypes=2, window_size=0)
        module = SpectralPrototypeEmbedding(1000, 4, num_prototypes=4, min_observations=2)
        module.observe_tokens(torch.empty(0, dtype=torch.long))
        module.observe_tokens(torch.full((2,), -1, dtype=torch.long))
        assert not module.try_compress()
        assert "cooccurrence" not in module.state_dict()
        assert "edge_index" not in module.state_dict()
        assert "edge_weight" not in module.state_dict()
        assert "token_counts" not in module.state_dict()
        padded = SpectralPrototypeEmbedding(4, 2, num_prototypes=2, padding_idx=0, min_observations=1)
        assert torch.count_nonzero(padded.weight[0]) == 0
        padded.observe_tokens(torch.tensor([[0, 1, 0]]))
        assert padded.token_counts[0] == 0
        assert padded._spectral_assignments() is None

        options = torch.nn.Embedding(4, 2, max_norm=1.5, scale_grad_by_freq=True, sparse=True)
        copied = SpectralPrototypeEmbedding.from_embedding(options, num_prototypes=2)
        assert copied.max_norm == options.max_norm
        assert copied.scale_grad_by_freq == options.scale_grad_by_freq
        assert copied.sparse == options.sparse

    def test_compression_forward_numerics_and_fallback(self):
        embedding = _community_embedding()
        module = SpectralPrototypeEmbedding.from_embedding(embedding, num_prototypes=2, min_observations=4)
        tokens = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
        module.observe_tokens(tokens)
        assert module.try_compress()
        assert module.try_compress()
        assert module.is_compressed
        torch.testing.assert_close(module(tokens), embedding(tokens), atol=1e-6, rtol=1e-6)

        fallback = SpectralPrototypeEmbedding.from_embedding(embedding, num_prototypes=2, min_observations=100)
        assert not fallback.try_compress()
        torch.testing.assert_close(fallback(tokens), embedding(tokens))

    def test_compressed_embedding_preserves_padding_and_index_errors(self):
        embedding = _community_embedding()
        embedding.padding_idx = 0
        with torch.no_grad():
            embedding.weight[0].zero_()
        module = SpectralPrototypeEmbedding.from_embedding(embedding, num_prototypes=3, min_observations=4)
        tokens = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
        module.observe_tokens(tokens)
        assert module.try_compress()
        assert module.edge_index.numel() == 0
        out = module(tokens)
        torch.testing.assert_close(out[0, 0], torch.zeros(3))
        out.sum().backward()
        torch.testing.assert_close(module.prototype_weight.grad[0], torch.zeros(3))
        with pytest.raises(IndexError):
            module(torch.tensor([-1]))
        with pytest.raises(IndexError):
            module(torch.tensor([8]))

    def test_embedding_optimizer_state_is_rewritten_when_compressing_after_warmup(self):
        embedding = _community_embedding()
        module = SpectralPrototypeEmbedding.from_embedding(embedding, num_prototypes=2, min_observations=4)
        module.observe_tokens(torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]))
        opt = torch.optim.Adam(module.parameters(), lr=0.01)
        tokens = torch.tensor([0, 1, 4, 5])
        F.mse_loss(module(tokens), embedding(tokens).detach()).backward()
        opt.step()
        old_weight = module.weight
        assert old_weight in opt.state

        assert module.try_compress(optimizer=opt)
        assert old_weight not in opt.state
        assert all(param is not old_weight for group in opt.param_groups for param in group["params"])
        assert any(param is module.prototype_weight for group in opt.param_groups for param in group["params"])

    def test_padding_single_token_rows_runtime_error_and_dense_reload(self):
        embedding = torch.nn.Embedding(4, 2, padding_idx=0)
        module = SpectralPrototypeEmbedding.from_embedding(embedding, num_prototypes=2, min_observations=2)
        module.observe_tokens(torch.tensor([[1], [2]]))
        assert not module.try_compress()

        compressed = SpectralPrototypeEmbedding.from_embedding(_community_embedding(), num_prototypes=2, min_observations=4)
        compressed.observe_tokens(torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]))
        assert compressed.try_compress()
        compressed.prototype_weight = None
        with pytest.raises(RuntimeError, match="prototype weights are missing"):
            compressed(torch.tensor([0]))

        dense_state = SpectralPrototypeEmbedding.from_embedding(_community_embedding(), num_prototypes=2).state_dict()
        compressed.load_state_dict(dense_state)
        assert not compressed.is_compressed

    def test_gradients_training_determinism_and_serialization(self):
        torch.manual_seed(7)
        embedding = _community_embedding()
        module = SpectralPrototypeEmbedding.from_embedding(embedding, num_prototypes=2, min_observations=4)
        module.observe_tokens(torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]))
        assert module.try_compress()
        tokens = torch.tensor([0, 1, 4, 5])
        target = torch.zeros(4, 3)
        opt = torch.optim.SGD(module.parameters(), lr=0.2)
        first_loss = None
        for _ in range(8):
            opt.zero_grad()
            loss = F.mse_loss(module(tokens), target)
            if first_loss is None:
                first_loss = loss.detach()
            loss.backward()
            assert module.prototype_weight.grad is not None
            opt.step()
        assert loss < first_loss

        torch.manual_seed(7)
        module2 = SpectralPrototypeEmbedding.from_embedding(_community_embedding(), num_prototypes=2, min_observations=4)
        module2.observe_tokens(torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]))
        assert module2.try_compress()
        torch.testing.assert_close(module2(tokens), _community_embedding()(tokens))

        loaded = SpectralPrototypeEmbedding(8, 3, num_prototypes=2)
        loaded.load_state_dict(module.state_dict())
        assert loaded.is_compressed
        torch.testing.assert_close(loaded(tokens), module(tokens))

        buffer = io.BytesIO()
        torch.save(module.state_dict(), buffer)
        buffer.seek(0)
        round_tripped_state = torch.load(buffer)
        round_tripped = SpectralPrototypeEmbedding(8, 3, num_prototypes=2)
        round_tripped.load_state_dict(round_tripped_state)
        torch.testing.assert_close(round_tripped(tokens), module(tokens))

    def test_checkpoint_compile_amp_and_zero_length_inputs(self):
        module = SpectralPrototypeEmbedding.from_embedding(_community_embedding(), num_prototypes=2, min_observations=4)
        module.observe_tokens(torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]))
        assert module.try_compress()
        tokens = torch.tensor([], dtype=torch.long)
        assert module(tokens).shape == (0, 3)

        tokens = torch.tensor([0, 4])
        mapped = module.token_to_prototype[tokens]
        out = checkpoint(lambda weight: F.embedding(mapped, weight).sum(), module.prototype_weight, use_reentrant=False)
        out.backward()
        assert module.prototype_weight.grad is not None

        compiled = torch.compile(module, backend="eager")
        torch.testing.assert_close(compiled(torch.tensor([0, 4])), module(torch.tensor([0, 4])))

        if torch.cuda.is_available():
            cuda_module = module.cuda()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = cuda_module(torch.tensor([0, 4], device="cuda"))
            assert out.dtype == torch.float32


class TestTopologicalSpectralAttentionRigorous(unittest.TestCase):
    def test_constructor_validation_and_dense_fallbacks(self):
        with pytest.raises(ValueError, match="min_seq_len"):
            TopologicalSpectralAttention(min_seq_len=0)
        with pytest.raises(ValueError, match="num_clusters"):
            TopologicalSpectralAttention(num_clusters=0)
        with pytest.raises(ValueError, match="spectral_gap"):
            TopologicalSpectralAttention(spectral_gap_threshold=-1.0)
        module = TopologicalSpectralAttention(min_seq_len=8, num_clusters=2)
        q = torch.randn(1, 2, 0, 4)
        out = module(q, q, q)
        assert out.shape == q.shape
        q = torch.randn(1, 2, 4, 4)
        mask = torch.zeros(4, 4)
        torch.testing.assert_close(
            module(q, q, q, attn_mask=mask),
            F.scaled_dot_product_attention(q, q, q, attn_mask=mask, dropout_p=0.0),
        )
        no_structure = TopologicalSpectralAttention(min_seq_len=1, num_clusters=2)
        flat = torch.zeros(1, 1, 4, 4)
        assert no_structure._cluster_assignments(flat) is None
        too_many_clusters = TopologicalSpectralAttention(min_seq_len=1, num_clusters=4)
        assert too_many_clusters._cluster_assignments(torch.randn(1, 1, 4, 4)) is None
        no_gap = TopologicalSpectralAttention(min_seq_len=1, num_clusters=2, spectral_gap_threshold=100.0)
        assert no_gap._cluster_assignments(torch.randn(1, 1, 5, 4)) is None
        causal = TopologicalSpectralAttention(min_seq_len=1, num_clusters=2, is_causal=True)
        torch.testing.assert_close(
            causal(q, q, q),
            F.scaled_dot_product_attention(q, q, q, dropout_p=0.0, is_causal=True),
        )

    def test_eval_disables_attention_dropout(self):
        module = TopologicalSpectralAttention(min_seq_len=8, dropout_p=0.9).eval()
        q = torch.randn(1, 2, 4, 4)
        torch.testing.assert_close(module(q, q, q), F.scaled_dot_product_attention(q, q, q, dropout_p=0.0))

    def test_cluster_size_bias_matches_dense_for_identical_members(self):
        module = TopologicalSpectralAttention(min_seq_len=1, num_clusters=2).eval()
        query = torch.tensor([[[[1.0, 0.0]]]])
        key = torch.tensor([[[[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]]])
        value = torch.tensor([[[[2.0, 0.0], [2.0, 0.0], [-4.0, 0.0]]]])
        assignments = torch.tensor([0, 0, 1])
        torch.testing.assert_close(
            module._clustered_attention(query, key, value, assignments, None),
            F.scaled_dot_product_attention(query, key, value, dropout_p=0.0),
        )

    def test_clustered_attention_empty_representatives_falls_back(self):
        module = TopologicalSpectralAttention(min_seq_len=1, num_clusters=2)
        q = torch.randn(1, 1, 3, 4)
        assignments = torch.full((3,), 9, dtype=torch.long)
        torch.testing.assert_close(
            module._clustered_attention(q, q, q, assignments, None),
            F.scaled_dot_product_attention(q, q, q, dropout_p=0.0),
        )

    def test_clustered_path_shape_gradients_determinism_checkpoint_and_compile(self):
        torch.manual_seed(9)
        module = TopologicalSpectralAttention(min_seq_len=4, num_clusters=2, spectral_gap_threshold=0.05)
        base = torch.tensor([[2.0, 0, 0, 0], [2.1, 0, 0, 0], [-2.0, 0, 0, 0], [-2.1, 0, 0, 0]])
        q = base.view(1, 1, 4, 4).requires_grad_()
        k = q.clone().detach().requires_grad_()
        v = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4).requires_grad_()
        assert module._cluster_assignments(k) is not None
        out = module(q, k, v)
        assert out.shape == v.shape
        out.sum().backward()
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None

        torch.manual_seed(9)
        module2 = TopologicalSpectralAttention(min_seq_len=4, num_clusters=2, spectral_gap_threshold=0.05)
        torch.testing.assert_close(module2(q.detach(), k.detach(), v.detach()), out.detach())

        q2 = q.detach().requires_grad_()
        checkpoint_out = checkpoint(lambda x: module(x, x, v.detach()), q2, use_reentrant=False)
        assert checkpoint_out.shape == v.shape

        compile_module = TopologicalSpectralAttention(min_seq_len=8, num_clusters=2)
        compiled = torch.compile(compile_module, backend="eager", fullgraph=True)
        torch.testing.assert_close(
            compiled(q.detach(), k.detach(), v.detach()),
            F.scaled_dot_product_attention(q.detach(), k.detach(), v.detach(), dropout_p=0.0),
        )

    def test_amp_dense_path_on_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA AMP requires a GPU")
        module = TopologicalSpectralAttention(min_seq_len=8, num_clusters=2).cuda()
        q = torch.randn(1, 2, 4, 8, device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = module(q, q, q)
        assert out.dtype == torch.float16


class TestLyapunovSpectralGuardRigorous(unittest.TestCase):
    def test_constructor_validation_observe_accept_reject_and_relative_math(self):
        with pytest.raises(ValueError, match="warmup_steps"):
            LyapunovSpectralGuard(warmup_steps=0)
        with pytest.raises(ValueError, match="max_loss"):
            LyapunovSpectralGuard(max_loss_relative_increase=-0.1)
        with pytest.raises(ValueError, match="max_spectral"):
            LyapunovSpectralGuard(max_spectral_relative_increase=-0.1)
        with pytest.raises(ValueError, match="history_size"):
            LyapunovSpectralGuard(history_size=0)
        with pytest.raises(ValueError, match="history_size"):
            LyapunovSpectralGuard(warmup_steps=4, history_size=3)
        assert LyapunovSpectralGuard._relative_increase(0.0, 1.0) > 0
        assert not LyapunovSpectralGuard(warmup_steps=2).accept(loss=1.0, spectral_energy=1.0)

        guard = LyapunovSpectralGuard(warmup_steps=2, max_loss_relative_increase=0.1, max_spectral_relative_increase=0.1)
        assert not guard.observe(loss=1.0, spectral_energy=2.0)
        assert guard.observe(loss=torch.tensor(1.0), spectral_energy=torch.tensor(2.0))
        assert guard.accept(loss=1.05, spectral_energy=2.1)
        assert guard.reject(loss=1.5, spectral_energy=2.1)

    def test_snapshot_rollback_and_state_dict_round_trip(self):
        module = torch.nn.Linear(3, 2)
        snapshot = LyapunovSpectralGuard.snapshot_module(module)
        with torch.no_grad():
            module.weight.add_(3.0)
        LyapunovSpectralGuard.rollback_module(module, snapshot)
        torch.testing.assert_close(module.weight, snapshot["weight"])

        guard = LyapunovSpectralGuard(warmup_steps=1)
        guard.observe(loss=1.0, spectral_energy=1.0)
        buffer = io.BytesIO()
        torch.save(guard.state_dict(), buffer)
        buffer.seek(0)
        state = torch.load(buffer)
        loaded = LyapunovSpectralGuard(warmup_steps=1)
        loaded.load_state_dict(state)
        assert loaded.is_ready
        assert loaded.accept(loss=1.0, spectral_energy=1.0)


if __name__ == "__main__":
    if raise_on_run_directly is not None:
        raise_on_run_directly("test/test_ao_sparsity.py")
    unittest.main()
