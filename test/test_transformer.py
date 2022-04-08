import nativetransformers_modules as transformer
import torch
import torch.nn as nn
from torch.testing._internal.common_device_type import (
    dtypes,
    dtypesIfCUDA,
    instantiate_device_type_tests,
    skipMeta,
)
from torch.testing._internal.common_utils import TestCase


class TestTransformerDeviceType(TestCase):
    @dtypes(torch.float, torch.double)
    @dtypesIfCUDA(torch.float, torch.half)
    @skipMeta
    def test_ffn(self, device, dtype):
        B = 10
        N = 12
        M = 14
        K = 16
        L = 18
        # 3d linear of cuda float32 with tf32 has numeric issue
        torch.backends.cuda.matmul.allow_tf32 = False
        # B * N * M
        x = torch.randn(B, N, M, dtype=dtype, device=device)
        # K * M
        w1 = torch.randn(K, M, dtype=dtype, device=device)
        # L * K
        w2 = torch.randn(L, K, dtype=dtype, device=device)
        # K
        b1 = torch.randn(K, dtype=dtype, device=device)
        # L
        b2 = torch.randn(L, dtype=dtype, device=device)

        linear1 = torch.nn.Linear(M, K, device=device, dtype=dtype)
        linear2 = torch.nn.Linear(K, L, device=device, dtype=dtype)
        linear1.weight = torch.nn.Parameter(w1)
        linear2.weight = torch.nn.Parameter(w2)
        linear1.bias = torch.nn.Parameter(b1)
        linear2.bias = torch.nn.Parameter(b2)
        res = linear2(torch.relu(linear1(x)))
        res_pt = torch.ops.nativetransformers._ffn(x, w1, b1, w2, b2)
        self.assertEqual(res_pt, res)
        torch.backends.cuda.matmul.allow_tf32 = True

    @dtypes(torch.float, torch.double)
    @skipMeta
    def test_ffn_module(self, device, dtype):
        N = 16
        num_heads = 4
        d_model = 256
        dim_feed_forward = 512
        # N * num_heads * d_model
        x = torch.randn(N, num_heads, d_model, dtype=dtype, device=device)
        ffn = transformer.FeedForwardNetwork(
            d_model=d_model,
            dim_feed_forward=dim_feed_forward,
            activation="relu",
            dtype=dtype,
            device=device,
        )

        w1 = ffn.weight1
        w2 = ffn.weight2
        b1 = ffn.bias1
        b2 = ffn.bias2

        res_pt = torch.nn.functional.linear(
            torch.relu(torch.nn.functional.linear(x, w1, b1)),
            w2,
            b2,
        )
        res_native = ffn(x)
        self.assertEqual(res_pt, res_native)

    @dtypesIfCUDA(torch.float, torch.half)
    @dtypes(torch.float)
    @skipMeta
    def test_native_multihead_attention_module(self, device, dtype):
        embed_dim = 36
        num_heads = 6
        sl = 20
        bs = 16

        x = torch.randn(sl, bs, embed_dim, device=device, dtype=dtype) * 10
        qkv = torch.nn.Linear(embed_dim, 3 * embed_dim, device=device, dtype=dtype)
        proj = torch.nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype)

        pt = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, device=device, dtype=dtype
        )
        pt.in_proj_weight = qkv.weight
        pt.in_proj_bias = qkv.bias
        pt.out_proj.weight = proj.weight
        pt.out_proj.bias = proj.bias

        npt = transformer.NativeMultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, device=device, dtype=dtype
        )

        npt.in_proj_weight = qkv.weight
        npt.in_proj_bias = qkv.bias
        npt.out_proj.weight = proj.weight
        npt.out_proj.bias = proj.bias

        ypt = pt(x, x, x, average_attn_weights=True)
        ypt = ypt[0]
        ynpt = npt(x, x, x)

        # For fp16 cuda the accuracy is lower
        torch.testing.assert_close(ypt, ynpt, atol=0.1, rtol=0.1)

    @dtypesIfCUDA(torch.float, torch.half)
    @dtypes(torch.float)
    def test_bettertransformerencoderlayer(self, device, dtype):
        # this is a deterministic test for BetterTransformerEncoderLayer
        d_model = 4
        nhead = 2
        dim_feedforward = 16
        dropout = 0.0
        bsz = 2

        def _test(training, batch_first=True):
            def perm_fn(x):
                return x.transpose(1, 0) if batch_first else x

            model = transformer.BetterTransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout, batch_first=batch_first
            )
            if not training:
                model = model.eval()

            # set constant weights of the model
            for idx, p in enumerate(model.parameters()):
                x = p.data
                sz = x.view(-1).size(0)
                shape = x.shape
                x = torch.cos(torch.arange(0, sz).float().view(shape))
                p.data.copy_(x)

            # deterministic input
            encoder_input = torch.tensor([[[20.0, 30.0, 40.0, 50.0]]])
            result = model(encoder_input)
            ref_output = torch.tensor([[[2.258703, 0.127985, -0.697881, 0.170862]]])
            self.assertEqual(result.shape, ref_output.shape)
            torch.testing.assert_close(result, ref_output)

            # deterministic input
            encoder_input = perm_fn(
                torch.tensor([[[1.0, 2.0, 3.0, 4.0]], [[5.0, 6.0, 7.0, 8.0]]])
            )
            result = model(encoder_input)
            ref_output = perm_fn(
                torch.tensor(
                    [
                        [[2.272644, 0.119035, -0.691669, 0.153486]],
                        [[2.272644, 0.119035, -0.691669, 0.153486]],
                    ]
                )
            )
            self.assertEqual(result.shape, ref_output.shape)
            torch.testing.assert_close(result, ref_output)

            # deterministic input
            encoder_input = perm_fn(
                torch.tensor(
                    [
                        [
                            [0.7462, 0.6653, 0.5679, 0.4891],
                            [0.5387, 0.1655, 0.3565, 0.0471],
                        ],
                        [
                            [0.8335, 0.2799, 0.5031, 0.2947],
                            [0.1402, 0.0318, 0.7636, 0.1346],
                        ],
                        [
                            [0.6333, 0.9344, 0.1376, 0.9938],
                            [0.8924, 0.2872, 0.6692, 0.2944],
                        ],
                        [
                            [0.9897, 0.6915, 0.3154, 0.1733],
                            [0.8645, 0.3513, 0.3064, 0.0767],
                        ],
                        [
                            [0.8117, 0.2366, 0.4838, 0.7881],
                            [0.3718, 0.4945, 0.9511, 0.0864],
                        ],
                    ]
                )
            )
            result = model(encoder_input)
            ref_output = perm_fn(
                torch.tensor(
                    [
                        [
                            [2.428589, 0.020835, -0.602055, -0.085249],
                            [2.427987, 0.021213, -0.602496, -0.084103],
                        ],
                        [
                            [2.424689, 0.019155, -0.604793, -0.085672],
                            [2.413863, 0.022211, -0.612486, -0.072490],
                        ],
                        [
                            [2.433774, 0.021598, -0.598343, -0.087548],
                            [2.425104, 0.019748, -0.604515, -0.084839],
                        ],
                        [
                            [2.436185, 0.022682, -0.596625, -0.087261],
                            [2.433556, 0.021891, -0.598509, -0.086832],
                        ],
                        [
                            [2.416246, 0.017512, -0.610712, -0.082961],
                            [2.422901, 0.024187, -0.606178, -0.074929],
                        ],
                    ]
                )
            )
            self.assertEqual(result.shape, ref_output.shape)
            torch.testing.assert_close(result, ref_output)

            # padding
            encoder_input[0][4] = torch.zeros_like(encoder_input[0][4])
            mask = torch.zeros(encoder_input.shape[:-1], dtype=torch.bool)
            mask[0][4] = True

            nt = torch.nested_tensor([encoder_input[0][:-1], encoder_input[1]])
            # Mask left in to make it easier to regenerate reference output.
            # result = model(encoder_input, src_key_padding_mask=mask)
            result = model(nt)
            ref_output = torch.tensor(
                [
                    [
                        [2.4268184, 0.02042419, -0.603311, -0.08476824],
                        [2.423306, 0.01889652, -0.6057701, -0.08519465],
                        [2.431538, 0.02078694, -0.5999354, -0.08746159],
                        [2.4348664, 0.02212971, -0.5975677, -0.08733892],
                        [2.423133, 0.02097577, -0.60594773, -0.08113337],
                    ],
                    [
                        [2.4279876, 0.02121329, -0.60249615, -0.08410317],
                        [2.4138637, 0.02221113, -0.6124869, -0.07249016],
                        [2.4251041, 0.01974815, -0.6045152, -0.08483928],
                        [2.4335563, 0.0218913, -0.59850943, -0.08683228],
                        [2.4229012, 0.02418739, -0.6061784, -0.07492948],
                    ],
                ]
            )
            result = resut.to_padded_tensor(0)
            ref_output[0][4] = torch.zeros_like(
                ref_output[0][4], device=device, dtype=dtype
            )
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output)

        _test(training=False)
        # Not currently supported because the relevant NestedTensor ops aren't registered.
        # _test(training=True)

    def _test_random_bettertransformerencoderlayer_impl(self, device, dtype, use_gelu, use_nested):
        # this is a random test for BetterTransformerEncoderLayer
        batch = 10
        d_model = 8
        nhead = 4
        seq_length = 16
        dim_feedforward = 16
        dropout = 0.0
        batch_first = True
        encoder_input = torch.randn(
            batch, seq_length, d_model, device=device, dtype=dtype
        )
        better_encoder_input = encoder_input
        if use_nested:
            better_encoder_input = torch.nested_tensor(encoder_input.unbind())

        model = torch.nn.modules.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            batch_first=batch_first,
            activation="gelu" if use_gelu else "relu",
            device=device,
            dtype=dtype,
        )
        for p in list(model.parameters()):
            x = p.data
            x = torch.randn(x.shape, device=device, dtype=dtype)
            p.data.copy_(x)

        # FIXME: THIS IS GONE NOW
        better_model = transformer.convert_encoder_from(
            model,
            d_model,
            nhead,
            dim_feedforward,
        )
        better_model = better_model.to(device=device, dtype=dtype).eval()

        result_better = better_model(better_encoder_input)
        result = model(encoder_input)

        if use_nested:
            result_better = torch.stack(result_better.unbind())
        # These are the default tolerances for float32 test; without
        # setting them, the float64 NestedTensor tests fail due to
        # "near misses" in a small number of locations.
        torch.testing.assert_close(result_better, result, rtol=1.3e-6, atol=1e-5)

    @dtypes(torch.float, torch.double)
    @dtypesIfCUDA(torch.float)
    @skipMeta
    def test_random_bettertransformerencoderlayer_relu(self, device, dtype):
        for use_nested in (False, True):
            self._test_random_bettertransformerencoderlayer_impl(
                device, dtype, use_gelu=False, use_nested=use_nested
            )

    @dtypes(torch.float, torch.double)
    @dtypesIfCUDA(torch.float)
    @skipMeta
    def test_random_bettertransformerencoderlayer_gelu(self, device, dtype):
        for use_nested in (False, True):
            self._test_random_bettertransformerencoderlayer_impl(
                device, dtype, use_gelu=True, use_nested=use_nested
            )


instantiate_device_type_tests(TestTransformerDeviceType, globals())
