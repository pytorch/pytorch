# Owner(s): ["module: inductor"]
import functools
import itertools
import math

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import HAS_CPU


def checkpoint_wrapper(fn):
    def inner(*args):
        return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=True)

    return inner


@config.patch({"freezing": True})
@config.patch({"onednn_graph": True})
class TestSDPAPatternRewriterTemplate(TestCase):
    use_static_shapes = True

    def _clone_inputs(self, inputs):
        def clone(x):
            if not isinstance(x, torch.Tensor):
                return x
            return x.clone()

        return [clone(x) for x in inputs]

    def _check_common(
        self,
        dot_prod_attention,
        args1=None,
        atol=1e-5,
        fusion_name="sdpa",
        has_dropout=False,
        override_check_equal=False,
        dtype=torch.float,
        uses_causal_mask=False,
        rtol=1.3e-6,
        batch_size=4,
    ):
        if args1 is None:
            tensor_shape = (batch_size, 2, 16, 32)
            args1 = [
                torch.randn(tensor_shape, device=self.device, dtype=dtype),
                torch.randn(tensor_shape, device=self.device, dtype=dtype),
                torch.randn(tensor_shape, device=self.device, dtype=dtype),
            ]
            if uses_causal_mask:
                args1.append(
                    torch.full(
                        (),
                        math.sqrt(args1[0].size(1)),
                        dtype=dtype,
                        device="cpu",
                    )
                )
                args1.append(
                    torch.full((), -3.4028234663852886e38, dtype=dtype, device="cpu")
                )
                if batch_size == 1:
                    args1.append(torch.ones(1, 1, 16, 16).to(torch.bool))
                else:
                    args1.append(torch.ones(1, 1, 2, 2).to(torch.bool))
        else:
            args1 = list(args1)
        args2 = self._clone_inputs(args1)

        for x in itertools.chain(args1[:], args2[:]):
            if isinstance(x, torch.Tensor) and x.is_floating_point():
                x.requires_grad = False

        if not self.use_static_shapes:
            torch._dynamo.mark_dynamic(args2[0], 0)
            torch._dynamo.mark_dynamic(args2[1], 0)
            torch._dynamo.mark_dynamic(args2[2], 0)

        dropout_arg = [False] if has_dropout else []
        torch.manual_seed(1234)
        result1 = dot_prod_attention(*(args1 + dropout_arg))

        counters.clear()
        torch.manual_seed(1234)
        result2, source_code = run_and_get_code(
            torch.compile(dot_prod_attention, fullgraph=True),
            *(args2 + dropout_arg),
        )
        source_code = "\n".join(source_code)
        self.assertGreaterEqual(counters["inductor"][fusion_name], 1)
        # some tests configured with very low dropout where we still want to check equality
        # if not has_dropout or override_check_equal:
        #    self.assertEqual(result1, result2, atol=atol, rtol=1.3e-6)

    def _test_bert_uint8_sdpa(self, amp_bf16=False):
        # uint8 sdpa
        import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
        from torch._export import capture_pre_autograd_graph
        from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
        from torch.ao.quantization.quantizer.x86_inductor_quantizer import (
            X86InductorQuantizer,
        )

        class SelfAttnLikeModule(torch.nn.Module):
            def __init__(
                self,
                input_dim,
                num_attention_heads=None,
                attention_head_size=None,
            ) -> None:
                super().__init__()
                self.input_dim = input_dim
                self.q_proj = torch.nn.Linear(input_dim, input_dim, bias=False)
                self.k_proj = torch.nn.Linear(input_dim, input_dim, bias=False)
                self.v_proj = torch.nn.Linear(input_dim, input_dim, bias=False)
                self.softmax = torch.nn.Softmax(dim=-1)
                assert num_attention_heads is not None
                assert attention_head_size is not None
                self.num_attention_heads = num_attention_heads
                self.attention_head_size = attention_head_size
                self.all_head_size = self.num_attention_heads * self.attention_head_size
                self.dense = torch.nn.Linear(self.all_head_size, self.all_head_size)
                self.dropout = torch.nn.Dropout(0)

            def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
                new_x_shape = x.size()[:-1] + (
                    self.num_attention_heads,
                    self.attention_head_size,
                )
                x = x.view(new_x_shape)
                return x.permute([0, 2, 1, 3])

            def forward(self, x, mask):
                q = self.q_proj(x)
                k = self.k_proj(x)
                v = self.v_proj(x)
                q = self.transpose_for_scores(q)
                k = self.transpose_for_scores(k)
                v = self.transpose_for_scores(v)
                scores = torch.matmul(q, k.transpose(-1, -2)) / (self.input_dim**0.5)
                scores = scores + mask
                attention = self.softmax(scores)
                attention = self.dropout(attention)
                context_layer = torch.matmul(attention, v)
                context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
                context_layer = context_layer.view(
                    context_layer.size()[:-2] + (self.all_head_size,)
                )
                return self.dense(context_layer)

        def _generate_qdq_quantized_model(mod, inputs, quantizer):
            export_model = capture_pre_autograd_graph(mod, inputs)
            prepare_model = prepare_pt2e(export_model, quantizer)
            prepare_model(*inputs)
            convert_model = convert_pt2e(prepare_model)
            torch.ao.quantization.move_exported_model_to_eval(convert_model)
            return convert_model

        with torch.no_grad():
            mod = SelfAttnLikeModule(
                input_dim=64 * 16,
                num_attention_heads=16,
                attention_head_size=64,
            ).eval()
            inputs = [
                torch.randn((2, 384, 64 * 16), device=self.device),
                torch.randn((2, 1, 1, 384), device=self.device),
            ]

            quantizer = X86InductorQuantizer()
            quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
            quantizer._set_aten_operator_qconfig(
                torch.ops.aten.matmul.default, quantizer.global_config
            )

            convert_model = _generate_qdq_quantized_model(mod, inputs, quantizer)
            with torch.cpu.amp.autocast(enabled=amp_bf16):
                self._check_common(
                    convert_model,
                    args1=inputs,
                    atol=3e-3,
                    fusion_name="onednn_graph_bert_int8_mha_fusion",
                )


if HAS_CPU:

    class SDPAPatternRewriterCpuTests(TestSDPAPatternRewriterTemplate):
        device = "cpu"
        test_onednn_graph_bert_int8_rewriter = (
            TestSDPAPatternRewriterTemplate._test_bert_uint8_sdpa
        )
        if torch.cpu._does_cpu_support_avx512bf16():
            test_onednn_graph_bert_int8_bf16_rewriter = functools.partialmethod(
                TestSDPAPatternRewriterTemplate._test_bert_uint8_sdpa, amp_bf16=True
            )


if __name__ == "__main__":
    if IS_LINUX and torch._C._has_onednn_graph and torch.cpu._does_cpu_support_vnni():
        run_tests()
