# Owner(s): ["module: nn"]

import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest

from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import TEST_FAIRSEQ, run_tests, parametrize, instantiate_parametrized_tests
from torch.testing._internal.common_cuda import TEST_CUDA

if TEST_FAIRSEQ:
    import fairseq.models.transformer as fairseq_transformer

@contextlib.contextmanager
def set_default_dtype(dtype):
    saved_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(saved_dtype)

class TestTransformers(NNTestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    device_list = ['cpu']  # TODO: is there a way to do parametrize for this?
    if TEST_CUDA:
        device_list.append('cuda')

    @unittest.skip("4D mask not supported yet - activate when 4D mask supported")
    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")  # TODO: make this work for both cuda and cpu
    def test_self_attn_TxT_attn_mask(self):
        embed_dim = 16
        num_heads = 4
        batch_size = 10
        tgt_len = 16

        query = torch.rand(batch_size, tgt_len, embed_dim, device="cuda")  # [N, T, D]
        attn_mask = torch.randint(0, 2, (tgt_len, tgt_len)).cuda().float()  # [T, T]
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))

        attn_mask_4d = attn_mask.expand(batch_size, num_heads, tgt_len, tgt_len)

        mta_model = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).cuda()
        mta_model.eval()

        # Generate 3D results
        with torch.inference_mode():
            output_mask_4d = mta_model(query, query, query, attn_mask=attn_mask_4d)[0]
            output_mask_4d = output_mask_4d.transpose(0, 1)  # [N, T, D]

            output_mask_TxT = mta_model(query, query, query, attn_mask=attn_mask)[0]
            output_mask_TxT = output_mask_TxT.transpose(0, 1)  # [N, T, D]

            self.assertEqual(output_mask_4d, output_mask_TxT)

    @parametrize("device", device_list)
    def test_transformerencoderlayer_src_mask(self, device):
        batch_size = 2
        seqlen = 4
        d_model = 8
        nhead = 8
        dim_feedforward = 32

        model = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True).to(device)
        src = torch.rand(batch_size, seqlen, d_model).to(device)  # bs, seqlen, d_model
        src_mask = torch.zeros(seqlen, seqlen).to(torch.bool).to(device)

        model(src, src_mask=src_mask)
        model.eval()
        with torch.no_grad():
            model(src, src_mask=src_mask)

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
            cm = torch.no_grad()
        else:
            cm = contextlib.nullcontext()
        with cm:
            model(x, src_key_padding_mask=mask)

    @parametrize("with_no_grad", [True, False])
    @parametrize("training", [True, False])
    @parametrize("enable_nested_tensor", [False])
    @parametrize("device", device_list)
    def test_transformerencoder_square_input(self, with_no_grad, training, enable_nested_tensor, device):
        """
        Test for edge cases when input of shape (batch size, sequence length, embedding dimension) has
        batch size == sequence length
        """
        model = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=4, nhead=2, dim_feedforward=16, dropout=0.0, batch_first=True),
            num_layers=2,
            enable_nested_tensor=enable_nested_tensor
        ).to(device)

        with torch.no_grad():
            # set constant weights of the model
            for idx, p in enumerate(model.parameters()):
                x = p.data
                sz = x.view(-1).size(0)
                shape = x.shape
                x = torch.cos(torch.arange(0, sz).float().view(shape))
                p.data.copy_(x)

        if training:
            model = model.train()
        else:
            model = model.eval()
        x = torch.arange(0, 16).reshape(2, 2, 4).to(torch.float).to(device)
        src_mask = torch.Tensor([[0, 1], [0, 0]]).to(torch.bool).to(device)

        if with_no_grad:
            cm = torch.no_grad()
        else:
            cm = contextlib.nullcontext()
        with cm:
            result = model(x, mask=src_mask)

        ref_output = torch.Tensor([[[2.420306205749512, 0.017629241570830, -0.607857942581177, -0.085519507527351],
                                    [2.420306205749512, 0.017629241570830, -0.607857942581177, -0.085519507527351]],
                                   [[2.419836044311523, 0.017548924311996, -0.608187675476074, -0.085347734391689],
                                    [2.419836044311523, 0.017548924311996, -0.608187675476074, -0.085347734391689]]]
                                  ).to(device)
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

    @parametrize("batch_first", [True, False])
    @parametrize("training", [True, False])
    @parametrize("enable_nested_tensor", [True, False])
    @parametrize("device", device_list)
    def test_transformerencoder(self, batch_first, training, enable_nested_tensor, device):
        def get_a_test_layer(activation, batch_first=False):
            d_model = 4
            nhead = 2
            dim_feedforward = 16
            dropout = 0.0

            layer = nn.TransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=batch_first,
            ).to(device)

            with torch.no_grad():
                # set constant weights of the model
                for idx, p in enumerate(layer.parameters()):
                    x = p.data
                    sz = x.view(-1).size(0)
                    shape = x.shape
                    x = torch.cos(torch.arange(0, sz).float().view(shape))
                    p.data.copy_(x)

            return layer

        # this is a deterministic test for TransformerEncoder
        activation = F.relu

        def _test(batch_first, training, enable_nested_tensor):
            def perm_fn(x):
                return x.transpose(1, 0) if batch_first else x

            encoder_layer = get_a_test_layer(activation=activation,
                                             batch_first=batch_first)

            model = nn.TransformerEncoder(encoder_layer, 1).to(device)
            if not training:
                model = model.eval()

            # deterministic input
            encoder_input = perm_fn(torch.tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
                                                   [0.5387, 0.1655, 0.3565, 0.0471]],
                                                  [[0.8335, 0.2799, 0.5031, 0.2947],
                                                   [0.1402, 0.0318, 0.7636, 0.1346]],
                                                  [[0.6333, 0.9344, 0.1376, 0.9938],
                                                   [0.8924, 0.2872, 0.6692, 0.2944]],
                                                  [[0.9897, 0.6915, 0.3154, 0.1733],
                                                   [0.8645, 0.3513, 0.3064, 0.0767]],
                                                  [[0.8117, 0.2366, 0.4838, 0.7881],
                                                   [0.3718, 0.4945, 0.9511, 0.0864]]]
                                                 )).to(device)
            result = model(encoder_input)
            ref_output = perm_fn(torch.tensor([[[2.428589, 0.020835, -0.602055, -0.085249],
                                                [2.427987, 0.021213, -0.602496, -0.084103]],
                                               [[2.424689, 0.019155, -0.604793, -0.085672],
                                                [2.413863, 0.022211, -0.612486, -0.072490]],
                                               [[2.433774, 0.021598, -0.598343, -0.087548],
                                                [2.425104, 0.019748, -0.604515, -0.084839]],
                                               [[2.436185, 0.022682, -0.596625, -0.087261],
                                                [2.433556, 0.021891, -0.598509, -0.086832]],
                                               [[2.416246, 0.017512, -0.610712, -0.082961],
                                                [2.422901, 0.024187, -0.606178, -0.074929]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            # all 0 src_mask
            src_mask = torch.zeros([5, 5]).to(device) == 1
            result = model(encoder_input, mask=src_mask)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            # all 0
            mask = torch.zeros([2, 5]).to(device) == 1
            result = model(encoder_input, src_key_padding_mask=mask)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            mask[0, 1] = 1
            mask[1, 3] = 1
            mask[1, 4] = 1
            # If mask is not left aligned
            # We disable nested tensor
            model.enable_nested_tensor = enable_nested_tensor
            result = model(encoder_input, src_key_padding_mask=mask)
            ref_output = perm_fn(torch.tensor([[[2.429026, 0.020793, -0.601741, -0.085642],
                                                [2.428811, 0.021445, -0.601912, -0.084252]],
                                               [[2.425009, 0.019155, -0.604566, -0.085899],
                                                [2.415408, 0.02249, -0.611415, -0.073]],
                                               [[2.434199, 0.021682, -0.598039, -0.087699],
                                                [2.42598, 0.019941, -0.603896, -0.085091]],
                                               [[2.436457, 0.022736, -0.59643, -0.08736],
                                                [2.434021, 0.022093, -0.598179, -0.08679]],
                                               [[2.416531, 0.017498, -0.610513, -0.083181],
                                                [2.4242, 0.024653, -0.605266, -0.074959]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            # test case 2, multiple layers no norm
            model = nn.TransformerEncoder(encoder_layer, 2, enable_nested_tensor=enable_nested_tensor).to(device)
            if not training:
                model = model.eval()
            result = model(encoder_input, src_key_padding_mask=mask)
            ref_output = perm_fn(torch.tensor([[[2.419051, 0.017446, -0.608738, -0.085003],
                                                [2.419102, 0.017452, -0.608703, -0.085026]],
                                               [[2.419043, 0.017445, -0.608744, -0.084999],
                                                [2.419052, 0.017446, -0.608738, -0.085004]],
                                               [[2.419067, 0.017448, -0.608727, -0.085010],
                                                [2.419098, 0.017452, -0.608706, -0.085024]],
                                               [[2.419072, 0.017449, -0.608724, -0.085012],
                                                [2.419119, 0.017455, -0.608691, -0.085034]],
                                               [[2.419019, 0.017442, -0.608761, -0.084989],
                                                [2.419075, 0.017449, -0.608722, -0.085014]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            model = nn.TransformerEncoder(encoder_layer, 6, enable_nested_tensor=enable_nested_tensor).to(device)
            if not training:
                model = model.eval()
            result = model(encoder_input, src_key_padding_mask=mask)
            ref_output = perm_fn(torch.tensor([[[2.419101, 0.017453, -0.608703, -0.085025],
                                                [2.419101, 0.017453, -0.608704, -0.085025]],
                                               [[2.419101, 0.017453, -0.608703, -0.085025],
                                                [2.419101, 0.017453, -0.608704, -0.085025]],
                                               [[2.419101, 0.017453, -0.608703, -0.085025],
                                                [2.419101, 0.017453, -0.608704, -0.085025]],
                                               [[2.419101, 0.017453, -0.608703, -0.085025],
                                                [2.419101, 0.017453, -0.608704, -0.085025]],
                                               [[2.419101, 0.017453, -0.608703, -0.085025],
                                                [2.419101, 0.017453, -0.608704, -0.085025]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            # test case 3, multiple layers with norm
            # d_model = 4
            norm = nn.LayerNorm(4)
            model = nn.TransformerEncoder(encoder_layer, 2, norm=norm, enable_nested_tensor=enable_nested_tensor).to(device)
            if not training:
                model = model.eval()
            result = model(encoder_input, src_key_padding_mask=mask)
            ref_output = perm_fn(torch.tensor([[[1.695949, -0.357635, -0.893077, -0.445238],
                                                [1.695955, -0.357639, -0.893050, -0.445266]],
                                               [[1.695948, -0.357634, -0.893082, -0.445233],
                                                [1.695950, -0.357635, -0.893077, -0.445238]],
                                               [[1.695951, -0.357636, -0.893069, -0.445246],
                                                [1.695955, -0.357639, -0.893052, -0.445264]],
                                               [[1.695952, -0.357636, -0.893066, -0.445249],
                                                [1.695957, -0.357641, -0.893041, -0.445276]],
                                               [[1.695946, -0.357632, -0.893095, -0.445220],
                                                [1.695952, -0.357637, -0.893065, -0.445251]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            model = nn.TransformerEncoder(encoder_layer, 6, norm=norm, enable_nested_tensor=enable_nested_tensor).to(device)
            if not training:
                model = model.eval()
            result = model(encoder_input, src_key_padding_mask=mask)
            ref_output = perm_fn(torch.tensor([[[1.695955, -0.357639, -0.893051, -0.445265],
                                                [1.695955, -0.357639, -0.893051, -0.445265]],
                                               [[1.695955, -0.357639, -0.893051, -0.445265],
                                                [1.695955, -0.357639, -0.893051, -0.445265]],
                                               [[1.695955, -0.357639, -0.893051, -0.445265],
                                                [1.695955, -0.357639, -0.893051, -0.445265]],
                                               [[1.695955, -0.357639, -0.893051, -0.445265],
                                                [1.695955, -0.357639, -0.893051, -0.445265]],
                                               [[1.695955, -0.357639, -0.893051, -0.445265],
                                                [1.695955, -0.357639, -0.893051, -0.445265]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

        # TODO: remove set default dtype to double by making ref_output more precise.
        # Added because this test was copied from test_nn.py, which has default
        # dtype double. If default dtype is float, tests will say tensors not close because
        # ref output precision too low
        with set_default_dtype(torch.double):
            if training:
                cm = contextlib.nullcontext()
            else:
                cm = torch.no_grad()  # transformer fast path requires no grad
            with cm:
                _test(batch_first, training, enable_nested_tensor)

    @unittest.skipIf(not TEST_FAIRSEQ, "Fairseq not found")
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

if __name__ == '__main__':
    run_tests()
