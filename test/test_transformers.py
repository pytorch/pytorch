# Owner(s): ["module: nn"]

import torch
import unittest

from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import TEST_FAIRSEQ, parametrize, instantiate_parametrized_tests
from torch.testing._internal.common_cuda import TEST_CUDA

if TEST_FAIRSEQ:
    import fairseq.models.transformer as fairseq_transformer

class TestTransformers(NNTestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    @parametrize("use_torchscript", [True, False])
    @parametrize("with_no_grad", [True, False])
    @parametrize("training", [True, False])
    def test_transformerencoder_fastpath_torchscript(self, use_torchscript, with_no_grad, training):
        """
        Test TransformerEncoder does not crash
        """
        model = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=2, nhead=2, dim_feedforward=8, batch_first=True),
            num_layers=2,
            enable_nested_tensor=True
        )

        if training:
            model = model.train()
        else:
            model = model.eval()

        if use_torchscript:
            model = torch.jit.script(model)

        x = torch.Tensor([[[1, 2], [3, 4]]]).to(torch.float)
        mask = torch.Tensor([[0, 1]]).to(torch.bool)

        if with_no_grad:
            with torch.no_grad():
                model(x, src_key_padding_mask=mask)
        else:
            model(x, src_key_padding_mask=mask)

    @unittest.skipIf(not TEST_FAIRSEQ, "numpy not found")
    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    def test_decoder_only_layer(self):
        DEFAULT_PADDING_IDX = 0

        class FairseqDecoder(torch.nn.Module):
            def __init__(
                self,
                embed_dim,
                attention_heads,
                ffn_embed_dim,
                num_layers,
                embedding_layer,  # torch.nn.Embedding. Must have a padding_idx field
                dropout=0,
                normalize_before=False,
                torch_encoder=None,  # torch encoder that you can map weights from
                activation="relu",
            ):
                super().__init__()

                cfg = fairseq_transformer.TransformerConfig()
                cfg.decoder.embed_dim = embed_dim
                cfg.decoder.output_dim = embed_dim
                cfg.decoder.attention_heads = attention_heads
                cfg.decoder.ffn_embed_dim = ffn_embed_dim
                cfg.dropout = dropout
                cfg.decoder.normalize_before = normalize_before
                cfg.decoder.layers = num_layers
                # make embedding behavior same as other encoders
                cfg.no_token_positional_embeddings = True
                cfg.no_scale_embedding = True
                cfg.activation_fn = activation

                dictionary = {}  # TODO: verify what this is

                self.decoder = fairseq_transformer.TransformerDecoder(
                    cfg,
                    dictionary,
                    embedding_layer,
                    no_encoder_attn=True,
                    output_projection=None,
                )

                if torch_encoder is not None:
                    self.decoder = torch_to_fairseq(torch_encoder, self.decoder)
                self.decoder = self.decoder.eval().cuda().half()

            def forward(
                self,
                tokens,
                src_lengths=None,
                with_triangle_mask=False,
                incremental_state=None,
            ):
                return self.decoder(
                    prev_output_tokens=tokens,
                    encoder_out=None,
                    incremental_state=incremental_state,
                    features_only=True,
                    full_context_alignment=not with_triangle_mask,
                    alignment_layer=None,
                    alignment_heads=None,
                    src_lengths=src_lengths,
                    return_all_hiddens=False,
                )[0]

        class BetterDecoder(torch.nn.Module):
            """
            Only incremental decoder for now
            """

            def __init__(self, transformer, embedding, pad_idx):
                super().__init__()
                self.transformer = transformer
                self.embedding = embedding
                self.padding_idx = pad_idx

            def forward(
                self,
                x,
                src_mask=None,
                include_padding_mask=True,
                incr_key_lst=None,
                incr_value_lst=None,
                is_incremental_decoding=False,
            ):
                padding_mask = None
                if not x.is_nested and include_padding_mask:
                    padding_mask = x.eq(self.padding_idx)
                if(is_incremental_decoding):
                    x = x[:, -1:]  # only take the last token
                x = self.embedding(x)

                one_encoder_layer = self.transformer.layers[0]
                self_attn = one_encoder_layer.self_attn
                embed_dim = self_attn.embed_dim
                num_heads = self_attn.num_heads

                use_gelu = (
                    one_encoder_layer.activation_relu_or_gelu == 2
                )  # see torch/nn/modules/activation attention impl. 1 == relu, 2 == gelu
                assert (
                    one_encoder_layer.activation_relu_or_gelu != 0
                )  # 0 == not relu or gelu

                norm_first = one_encoder_layer.norm_first


                # TODO: make this a bit less janky. but for now we initialize with an empty tensor.
                if(not is_incremental_decoding):
                    assert len(incr_key_lst) == 0 or incr_key_lst[0] is None
                    assert len(incr_value_lst) == 0 or incr_value_lst[0] is None
                while len(incr_key_lst) <= len(self.transformer.layers):
                    if(is_incremental_decoding):
                        incr_key_lst.append(torch.Tensor([]).cuda().half())
                        incr_value_lst.append(torch.Tensor([]).cuda().half())
                    else:
                        incr_key_lst.append(None)
                        incr_value_lst.append(None)

                for i, layer in enumerate(self.transformer.layers):
                    incr_key = incr_key_lst[i]
                    incr_value = incr_value_lst[i]

                    x, incr_key, incr_value = torch._transformer_decoder_only_layer_fwd(
                        src=x,
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        qkv_weight=layer.self_attn.in_proj_weight,
                        qkv_bias=layer.self_attn.in_proj_bias,
                        proj_weight=layer.self_attn.out_proj.weight,
                        proj_bias=layer.self_attn.out_proj.bias,
                        use_gelu=use_gelu,
                        norm_first=norm_first,
                        # TODO: layer_norm_eps hardcoded to be same as nn.TransformerEncoder default.
                        # fix by pulling from self_attn.norm1
                        eps=1e-5,
                        norm_weight_1=layer.norm1.weight,
                        norm_bias_1=layer.norm1.bias,
                        norm_weight_2=layer.norm2.weight,
                        norm_bias_2=layer.norm2.bias,
                        ffn_weight_1=layer.linear1.weight,
                        ffn_bias_1=layer.linear1.bias,
                        ffn_weight_2=layer.linear2.weight,
                        ffn_bias_2=layer.linear2.bias,
                        mask=src_mask,
                        incr_key=incr_key,  # altered in place
                        incr_value=incr_value,
                    )

                    # not in-place
                    if(not is_incremental_decoding):
                        incr_key = None
                        incr_value = None
                    incr_key_lst[i] = incr_key
                    incr_value_lst[i] = incr_value

                return x, incr_key_lst, incr_value_lst

        def torch_to_fairseq(torch_encoder, fairseq_encoder):
            for src_layer, dst_layer in zip(torch_encoder.layers, fairseq_encoder.layers):
                w_q, w_k, w_v = src_layer.self_attn.in_proj_weight.chunk(3, dim=0)
                b_q, b_k, b_v = src_layer.self_attn.in_proj_bias.chunk(3, dim=0)

                dst_layer.self_attn.q_proj.weight = torch.nn.Parameter(w_q)
                dst_layer.self_attn.q_proj.bias = torch.nn.Parameter(b_q)
                dst_layer.self_attn.k_proj.weight = torch.nn.Parameter(w_k)
                dst_layer.self_attn.k_proj.bias = torch.nn.Parameter(b_k)
                dst_layer.self_attn.v_proj.weight = torch.nn.Parameter(w_v)
                dst_layer.self_attn.v_proj.bias = torch.nn.Parameter(b_v)

                dst_layer.self_attn.out_proj.weight = src_layer.self_attn.out_proj.weight
                dst_layer.self_attn.out_proj.bias = src_layer.self_attn.out_proj.bias

                dst_layer.fc1.weight = src_layer.linear1.weight
                dst_layer.fc1.bias = src_layer.linear1.bias

                # fairseq may use fusedlayernorm from nvidia apex - diff properties
                dst_layer.self_attn_layer_norm.load_state_dict(src_layer.norm1.state_dict())

                dst_layer.fc2.weight = src_layer.linear2.weight
                dst_layer.fc2.bias = src_layer.linear2.bias

                dst_layer.final_layer_norm.load_state_dict(src_layer.norm2.state_dict())

            return fairseq_encoder

        def set_weights_deterministic(model):
            for idx, p in enumerate(model.parameters()):
                x = p.data
                sz = x.view(-1).size(0)
                shape = x.shape
                x = torch.cos(torch.arange(0, sz).float().view(shape))
                p.data.copy_(x)

        D = 4  # d_model
        H = 2  # nhead
        FD = 16  # dim_feedforward
        V = 100  # vocab size
        L = 2  # num layers

        embedding_layer = torch.nn.Embedding(V, D, DEFAULT_PADDING_IDX)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=D,
            nhead=H,
            dim_feedforward=FD,
            batch_first=True,
            activation="gelu",
        )
        transformer = torch.nn.TransformerEncoder(
            layer,
            num_layers=L,
        ).eval().cuda().half()

        set_weights_deterministic(embedding_layer)
        set_weights_deterministic(transformer)

        better_decoder = (
            BetterDecoder(transformer, embedding_layer, DEFAULT_PADDING_IDX)
            .eval()
            .cuda()
            .half()
        )
        fairseq_decoder = (
            FairseqDecoder(
                D,
                H,
                FD,
                L,
                embedding_layer,
                dropout=0,
                normalize_before=False,
                torch_encoder=transformer,
                activation="gelu",
            )
            .eval()
            .cuda()
            .half()
        )

        tokens = torch.Tensor([
            [5, 6, 7, 8],
            [9, 10, 11, 12]
        ]).to(torch.int).cuda()
        lengths_tensor = torch.Tensor([2, 2]).to(torch.int).cuda()
        # bs = 2, seqlen = 4
        bs, seqlen = tokens.shape

        upper_triangle = torch.zeros(seqlen, seqlen)
        upper_triangle.fill_(-100000000)
        upper_triangle = torch.triu(upper_triangle, 1)
        upper_triangle = upper_triangle.cuda().half()
        upper_triangle_expanded = upper_triangle.unsqueeze(0).unsqueeze(0)
        upper_triangle_expanded = upper_triangle_expanded.expand(
            bs, H, -1, -1
        )

        # test forced decoding
        with torch.no_grad():
            result, _, _ = better_decoder(
                tokens,
                src_mask=upper_triangle_expanded,
                include_padding_mask=False,
                incr_key_lst=[],
                incr_value_lst=[],
                is_incremental_decoding=False,
            )
        ref_output = fairseq_decoder(tokens, lengths_tensor, with_triangle_mask=True)

        self.assertEqual(result.shape, ref_output.shape)
        torch.testing.assert_close(result, ref_output, atol=1e-3, rtol=1e-2)

        # test incremental decoding
        bs, seqlen = tokens.shape

        incr_state = {}
        ref_outputs = [fairseq_decoder(
            tokens[:, :i],
            src_lengths=None,
            with_triangle_mask=False,
            incremental_state=incr_state,
        ) for i in range(1, seqlen + 1)]
        ref_output = torch.stack(ref_outputs)

        incr_key_lst = []
        incr_value_lst = []
        results = []
        for i in range(1, seqlen + 1):
            res, incr_key_lst, incr_value_lst = better_decoder(
                tokens[:, :i],
                src_mask=None,
                include_padding_mask=False,
                incr_key_lst=incr_key_lst,
                incr_value_lst=incr_value_lst,
                is_incremental_decoding=True,
            )
            results.append(res)
        result = torch.stack(results)

        self.assertEqual(result.shape, ref_output.shape)
        torch.testing.assert_close(result, ref_output, atol=1e-3, rtol=1e-2)

instantiate_parametrized_tests(TestTransformers)
