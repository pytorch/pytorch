# Owner(s): ["module: inductor"]

import functools
import io
import os
import unittest
from unittest import mock

import torch
from torch._dynamo.testing import CompileCounterWithBackend
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.kernel.flex import flex_helion_attention
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import fresh_cache, run_fw_bw_and_get_code
from torch.nn.attention.flex_attention import (
    AuxRequest,
    create_block_mask,
    flex_attention,
    FlexKernelOptions,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
)


HAS_HELION_SM100 = (
    flex_helion_attention.ensure_helion_available()
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] == 10
)
HELION: FlexKernelOptions = {"BACKEND": "HELION"}


def _causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def _softcap(score, b, h, q_idx, kv_idx):
    return 30.0 * torch.tanh(score / 30.0)


def _tensors(b, hq, hkv, m, n, d, dtype=torch.bfloat16, requires_grad=True):
    q = torch.randn(b, hq, m, d, device="cuda", dtype=dtype)
    k = torch.randn(b, hkv, n, d, device="cuda", dtype=dtype)
    v = torch.randn(b, hkv, n, d, device="cuda", dtype=dtype)
    return tuple(t.requires_grad_(requires_grad) for t in (q, k, v))


def _case(name):
    """(q, k, v, score_mod, block_mask) of a test case."""
    if name == "dense":
        return (*_tensors(2, 4, 4, 256, 256, 64), None, None)
    if name == "causal":
        mask = create_block_mask(_causal, None, None, 256, 256, device="cuda")
        return (*_tensors(2, 4, 4, 256, 256, 128), None, mask)
    if name == "softcap_causal":
        mask = create_block_mask(_causal, None, None, 256, 256, device="cuda")
        return (*_tensors(2, 4, 4, 256, 256, 128), _softcap, mask)
    if name in ("alibi_document", "bias_document"):
        bias = torch.randn(2, 4, 256, 256, device="cuda")
        slopes = torch.rand(4, device="cuda")
        doc = torch.repeat_interleave(torch.arange(4, device="cuda"), 64)

        def alibi(score, b, h, q_idx, kv_idx):
            return score + slopes[h] * (kv_idx - q_idx)

        def biased(score, b, h, q_idx, kv_idx):
            return score + bias[b, h, q_idx, kv_idx]

        def document_causal(b, h, q_idx, kv_idx):
            return (doc[q_idx] == doc[kv_idx]) & (q_idx >= kv_idx)

        mask = create_block_mask(document_causal, None, None, 256, 256, device="cuda")
        score_mod = alibi if name == "alibi_document" else biased
        return (*_tensors(2, 4, 4, 256, 256, 64), score_mod, mask)
    if name == "gqa":
        mask = create_block_mask(_causal, None, None, 256, 256, device="cuda")
        return (*_tensors(2, 8, 2, 256, 256, 128), None, mask)
    if name == "rect":

        def offset_causal(b, h, q_idx, kv_idx):
            return kv_idx <= q_idx + 256

        mask = create_block_mask(offset_causal, None, None, 256, 512, device="cuda")
        return (*_tensors(2, 4, 4, 256, 512, 64), None, mask)
    raise AssertionError(name)


CASES = [
    "dense",
    "causal",
    "softcap_causal",
    "alibi_document",
    "bias_document",
    "gqa",
    "rect",
]


@unittest.skipUnless(HAS_HELION_SM100, "requires helion, its CuTe DSL and sm100")
@instantiate_parametrized_tests
class TestFlexHelion(InductorTestCase):
    hw_classification = HardwareClassification.CUDA

    def setUp(self):
        super().setUp()
        torch.manual_seed(0)
        counters.clear()

    def _check_against_triton(self, q, k, v, score_mod=None, block_mask=None):
        """Check BACKEND="HELION" against an fp32 reference: within twice
        BACKEND="TRITON"'s error plus the reference's input-dtype rounding.
        With grad-requiring inputs the loss reads both the output and the
        logsumexp, so the backward also folds in the logsumexp's gradient."""
        backward = q.requires_grad
        b, hq, m, _ = q.shape
        grad_out = torch.randn(b, hq, m, v.size(-1), device="cuda")
        grad_lse = torch.randn(b, hq, m, device="cuda")
        compiled = torch.compile(flex_attention, fullgraph=True)

        def run(fn, q, k, v, kernel_options=None):
            out, aux = fn(
                q,
                k,
                v,
                score_mod=score_mod,
                block_mask=block_mask,
                enable_gqa=q.size(1) != k.size(1),
                return_aux=AuxRequest(lse=True),
                kernel_options=kernel_options,
            )
            if not backward:
                return out.float(), aux.lse
            loss = (out * grad_out).sum() + (aux.lse * grad_lse).sum()
            grads = torch.autograd.grad(loss, (q, k, v))
            return out.float(), aux.lse, *(g.float() for g in grads)

        fp32 = [t.detach().float().requires_grad_(backward) for t in (q, k, v)]
        ref = run(flex_attention, *fp32)
        triton = run(compiled, q, k, v, {"BACKEND": "TRITON"})
        helion = run(compiled, q, k, v, HELION)
        names = ("out", "lse", "dq", "dk", "dv")
        for name, r, t, h in zip(names, ref, triton, helion):
            self.assertEqual(h.shape, r.shape)
            self.assertTrue(h.isfinite().all(), name)
            rounding = (r.to(q.dtype).float() - r).abs().max().item()
            triton_err = (t - r).abs().max().item()
            helion_err = (h - r).abs().max().item()
            self.assertLessEqual(
                helion_err,
                2 * triton_err + rounding,
                f"{name}: HELION error {helion_err:.2e} vs TRITON {triton_err:.2e}",
            )

    @parametrize("case", CASES)
    def test_forward(self, case):
        q, k, v, score_mod, block_mask = _case(case)
        q, k, v = (t.detach() for t in (q, k, v))
        self._check_against_triton(q, k, v, score_mod, block_mask)

    @parametrize("case", CASES)
    def test_backward(self, case):
        self._check_against_triton(*_case(case))

    def test_float16(self):
        mask = create_block_mask(_causal, None, None, 256, 256, device="cuda")
        q, k, v = _tensors(2, 4, 4, 256, 256, 64, dtype=torch.float16)
        self._check_against_triton(q, k, v, _softcap, mask)

    def test_lowers_to_helion_kernels(self):
        q, k, v, score_mod, block_mask = _case("softcap_causal")
        compiled = torch.compile(
            functools.partial(flex_attention, kernel_options=HELION), fullgraph=True
        )
        _, (fwd_code, bwd_code) = run_fw_bw_and_get_code(
            lambda: compiled(q, k, v, score_mod=score_mod, block_mask=block_mask)
        )
        self.assertIn("_helion_flex_attention_forward", fwd_code)
        self.assertNotIn("triton_tem_fused", fwd_code)
        self.assertIn("_helion_flex_attention_backward", bwd_code)
        self.assertNotIn("triton_tem_fused", bwd_code)

    def test_recompiles_per_shape(self):
        # The Helion kernels are compiled for static shapes: under
        # dynamic=True the lowering guards the graph on its sizes (including
        # the captured tensor's, which the mask_mod reads as a symbolic int),
        # so a new batch size recompiles where BACKEND="TRITON" reuses the
        # graph.
        doc = torch.repeat_interleave(torch.arange(4, device="cuda"), 64)

        def document_causal(b, h, q_idx, kv_idx):
            return (doc[q_idx] == doc[kv_idx]) & (q_idx >= kv_idx)

        mask = create_block_mask(document_causal, None, None, 256, 256, device="cuda")
        compile_counts = {}
        for backend in ("TRITON", "HELION"):
            compile_counts[backend] = CompileCounterWithBackend("inductor")
            compiled = torch.compile(
                functools.partial(flex_attention, kernel_options={"BACKEND": backend}),
                backend=compile_counts[backend],
                dynamic=True,
            )
            # Batch sizes that no other size equals, so duck sizing adds no
            # guards.
            for b in (3, 5):
                q, k, v = _tensors(b, 4, 4, 256, 256, 64, requires_grad=False)
                ref = flex_attention(q, k, v, block_mask=mask)
                out = compiled(q, k, v, block_mask=mask)
                torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
        self.assertEqual(compile_counts["TRITON"].frame_count, 1)
        self.assertEqual(compile_counts["HELION"].frame_count, 2)

    def test_pretuned_configs_apply(self):
        # Each pretuned config is valid for the installed Helion (a stale one
        # would silently fall back to Helion's default config).
        from torch._inductor.kernel.flex import flex_helion_kernels

        pretuned = flex_helion_attention._PRETUNED_SM100
        self.assertTrue(pretuned)
        mask = create_block_mask(_causal, None, None, 256, 256, device="cuda")
        for d in sorted({d for _, d, _ in pretuned}):
            torch._dynamo.reset()
            q, k, v = _tensors(2, 4, 4, 256, 256, d)
            with mock.patch.object(
                flex_helion_kernels, "get_kernel", wraps=flex_helion_kernels.get_kernel
            ) as get_kernel:
                out = torch.compile(flex_attention)(
                    q, k, v, block_mask=mask, kernel_options=HELION
                )
                out.sum().backward()
            for direction in ("forward", "backward"):
                config = pretuned.get((direction, d, d))
                if config is not None:
                    self.assertIn(
                        mock.call(direction, config=config),
                        get_kernel.call_args_list,
                        f"pretuned {direction} config for head dim {d} was not used",
                    )

    def test_stale_pretuned_config_falls_back(self):
        from torch._inductor.kernel.flex import flex_helion_kernels

        q, k, v = _tensors(2, 4, 4, 256, 256, 64, requires_grad=False)
        stale = {("forward", 64, 64): {"not_a_helion_config_key": 1}}
        with (
            mock.patch.dict(flex_helion_attention._PRETUNED_SM100, stale),
            mock.patch.object(
                flex_helion_kernels, "get_kernel", wraps=flex_helion_kernels.get_kernel
            ) as get_kernel,
        ):
            out = torch.compile(flex_attention)(q, k, v, kernel_options=HELION)
        self.assertTrue(out.isfinite().all())
        for call in get_kernel.call_args_list:
            self.assertIsNone(call.kwargs.get("config"))

    def test_autotune_opt_in(self):
        # HELION_AUTOTUNE_EFFORT autotunes on real inputs, with dense (every
        # block full) BlockMask tables, and lowers the config it finds.
        from helion.runtime.kernel import Kernel

        from torch._inductor.kernel.flex import flex_helion_kernels

        mask = create_block_mask(_causal, None, None, 256, 256, device="cuda")
        q, k, v = _tensors(2, 4, 4, 256, 256, 64, requires_grad=False)
        found = {}

        def autotune(kernel, args, force):
            self.assertFalse(force)
            self.assertEqual(kernel.settings.autotune_effort, "quick")
            names = list(kernel.signature.parameters)
            table = dict(zip(names, args))
            self.assertNotIsInstance(table["query"], torch._subclasses.FakeTensor)
            ncols = table["block_mask_full_kv_indices"].size(-1)
            self.assertTrue((table["block_mask_kv_num_blocks"] == 0).all())
            self.assertTrue((table["block_mask_full_kv_num_blocks"] == ncols).all())
            found["config"] = kernel.bind(args).config_spec.default_config()
            return found["config"]

        with (
            mock.patch.object(Kernel, "autotune", autospec=True) as kernel_autotune,
            mock.patch.object(
                flex_helion_kernels, "get_kernel", wraps=flex_helion_kernels.get_kernel
            ) as get_kernel,
        ):
            kernel_autotune.side_effect = autotune
            out = torch.compile(flex_attention)(
                q,
                k,
                v,
                block_mask=mask,
                kernel_options={**HELION, "HELION_AUTOTUNE_EFFORT": "quick"},
            )
        self.assertTrue(out.isfinite().all())
        self.assertEqual(kernel_autotune.call_count, 1)
        get_kernel.assert_called_with("forward", config=dict(found["config"].config))

    def test_mods_survive_pickling(self):
        # The Helion autotuner's benchmark subprocess loads the kernel args
        # with torch.load, which re-traces each GraphModule mod: the hl.load
        # calls of the prepared mods must survive that.
        q, k, v, score_mod, block_mask = _case("bias_document")
        prepare_mod = flex_helion_attention._prepare_mod
        prepared = []

        def record(*args):
            prepared.append(prepare_mod(*args))
            return prepared[-1]

        with mock.patch.object(flex_helion_attention, "_prepare_mod", record):
            torch.compile(flex_attention)(
                q.detach(),
                k.detach(),
                v.detach(),
                score_mod=score_mod,
                block_mask=block_mask,
                kernel_options=HELION,
            )
        self.assertEqual(len(prepared), 2)
        for gm in prepared:
            buffer = io.BytesIO()
            torch.save(gm, buffer)
            buffer.seek(0)
            loaded = torch.load(buffer, weights_only=False)
            self.assertEqual(
                [(n.op, n.target) for n in loaded.graph.nodes],
                [(n.op, n.target) for n in gm.graph.nodes],
            )

    def test_rejects_invalid_autotune_effort(self):
        q, k, v = _tensors(2, 4, 4, 256, 256, 64, requires_grad=False)
        with self.assertRaisesRegex(Exception, "Invalid HELION_AUTOTUNE_EFFORT 'max'"):
            torch.compile(flex_attention)(
                q, k, v, kernel_options={**HELION, "HELION_AUTOTUNE_EFFORT": "max"}
            )

    def test_rejects_max_scores(self):
        q, k, v = _tensors(2, 4, 4, 256, 256, 64, requires_grad=False)
        with self.assertRaisesRegex(
            NotImplementedError,
            r"Returning max scores is not supported with BACKEND='HELION'",
        ):
            flex_attention(
                q, k, v, return_aux=AuxRequest(max_scores=True), kernel_options=HELION
            )

    def test_rejects_float32(self):
        q, k, v = _tensors(2, 4, 4, 256, 256, 64, torch.float32, requires_grad=False)
        with self.assertRaisesRegex(Exception, "float16 or bfloat16"):
            torch.compile(flex_attention)(q, k, v, kernel_options=HELION)

    def test_rejects_unsupported_block_size(self):
        q, k, v = _tensors(2, 4, 4, 256, 256, 64, requires_grad=False)
        mask = create_block_mask(
            _causal, None, None, 256, 256, device="cuda", BLOCK_SIZE=64
        )
        with self.assertRaisesRegex(
            Exception, "does not fit Helion's fused CuTe flash kernel"
        ):
            torch.compile(flex_attention)(
                q, k, v, block_mask=mask, kernel_options=HELION
            )

    def test_rejects_captured_tensor_grad(self):
        q, k, v = _tensors(2, 4, 4, 256, 256, 64)
        slopes = torch.rand(4, device="cuda", requires_grad=True)

        def alibi(score, b, h, q_idx, kv_idx):
            return score + slopes[h] * (kv_idx - q_idx)

        with self.assertRaisesRegex(
            Exception, "gradients of tensors captured by score_mod"
        ):
            out = torch.compile(flex_attention)(
                q, k, v, score_mod=alibi, kernel_options=HELION
            )
            out.sum().backward()

    @config.patch({"fx_graph_cache": True, "fx_graph_remote_cache": False})
    def test_fx_graph_cache_keys_on_helion(self):
        # Helion's source and settings shape the generated code but are not in
        # the graph: a change to them must not reuse a cached graph.
        q, k, v = _tensors(2, 4, 4, 256, 256, 64, requires_grad=False)

        def compile_and_run():
            torch._dynamo.reset()
            torch.compile(flex_attention)(q, k, v, kernel_options=HELION)

        with fresh_cache():
            compile_and_run()
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            compile_and_run()
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
            with mock.patch.dict(os.environ, {"HELION_FLEX_TEST_SETTING": "1"}):
                compile_and_run()
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
            with mock.patch.object(
                flex_helion_attention, "_helion_source_hash", return_value="edited"
            ):
                compile_and_run()
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 3)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
