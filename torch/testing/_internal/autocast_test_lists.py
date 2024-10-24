

import collections

import torch
from torch.testing._internal.common_utils import TEST_WITH_ROCM
from torch.testing._internal.common_utils import TestCase


class AutocastTestLists:
    def _rnn_cell_args(self, n, num_chunks, is_lstm, dev, dtype):
        input = (torch.randn((n, n), device=dev, dtype=torch.float32),)

        hx = ((torch.randn((n, n), device=dev, dtype=torch.float32),
               torch.randn((n, n), device=dev, dtype=torch.float32)) if is_lstm else
              torch.randn((n, n), device=dev, dtype=torch.float32),)

        weights = (torch.randn((num_chunks * n, n), device=dev, dtype=torch.float32),  # weight_ih
                   torch.randn((num_chunks * n, n), device=dev, dtype=torch.float32),  # weight_hh
                   torch.randn((num_chunks * n), device=dev, dtype=torch.float32),  # bias_ih
                   torch.randn((num_chunks * n), device=dev, dtype=torch.float32))  # bias_hh

        # returns args as a tuple
        return input + hx + weights

    # Supplies ops and arguments for test_autocast_* in test/test_cuda.py
    def __init__(self, dev):
        super().__init__()
        n = 8
        # Utility arguments, created as one-element tuples
        pointwise0_fp16 = (torch.randn(n, dtype=torch.float16, device=dev),)
        pointwise1_fp16 = (torch.randn(n, dtype=torch.float16, device=dev),)
        pointwise2_fp16 = (torch.randn(n, dtype=torch.float16, device=dev),)
        mat0_fp16 = (torch.randn((n, n), dtype=torch.float16, device=dev),)
        mat1_fp16 = (torch.randn((n, n), dtype=torch.float16, device=dev),)
        mat2_fp16 = (torch.randn((n, n), dtype=torch.float16, device=dev),)

        dimsets = ((n, n, n), (n, n, n, n), (n, n, n, n, n))
        conv_args_fp32 = [(torch.randn(dimset, dtype=torch.float32, device=dev),
                           torch.randn(dimset, dtype=torch.float32, device=dev))
                          for dimset in dimsets]
        bias_fp32 = (torch.randn((n,), dtype=torch.float32, device=dev),)
        element0_fp32 = (torch.randn(1, dtype=torch.float32, device=dev),)
        pointwise0_fp32 = (torch.randn(n, dtype=torch.float32, device=dev),)
        pointwise1_fp32 = (torch.randn(n, dtype=torch.float32, device=dev),)
        mat0_fp32 = (torch.randn((n, n), dtype=torch.float32, device=dev),)
        mat1_fp32 = (torch.randn((n, n), dtype=torch.float32, device=dev),)
        mat2_fp32 = (torch.randn((n, n), dtype=torch.float32, device=dev),)
        mat3_fp32 = (torch.randn((n, n), dtype=torch.float32, device=dev),)

        # The lists below organize ops that autocast needs to test.
        # self.list_name corresponds to test_autocast_list_name in test/test_cuda.py.
        # Each op is associated with a tuple of valid arguments.
        # In addition, cudnn conv ops are not supported on ROCm and hence will
        # be skipped by passing TEST_WITH_ROCM flag to those ops in self.torch_fp16 list.

        # Some ops implement built-in type promotion.  These don't need autocasting,
        # but autocasting relies on their promotion, so we include tests to double-check.
        self.torch_expect_builtin_promote = [
            ("eq", pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("ge", pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("gt", pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("le", pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("lt", pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("ne", pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("add", pointwise0_fp32 + pointwise1_fp16, torch.float32),
            ("div", pointwise0_fp32 + pointwise1_fp16, torch.float32),
            ("mul", pointwise0_fp32 + pointwise1_fp16, torch.float32),
            ("cat", (pointwise0_fp16 + pointwise1_fp32,), torch.float32),
            ("equal", pointwise0_fp32 + pointwise1_fp16, torch.float32),
            ("stack", (pointwise0_fp16 + pointwise1_fp32,), torch.float32),
        ]
        self.methods_expect_builtin_promote = [
            ("__eq__", pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("__ge__", pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("__gt__", pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("__le__", pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("__lt__", pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("__ne__", pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("__add__", pointwise0_fp32 + pointwise1_fp16, torch.float32),
            ("__div__", pointwise0_fp32 + pointwise1_fp16, torch.float32),
            ("__mul__", pointwise0_fp32 + pointwise1_fp16, torch.float32),
        ]

        # The remaining lists organize ops that autocast treats explicitly.
        self.torch_fp16 = [
            # deprecated _convolution
            ("_convolution", conv_args_fp32[1] + bias_fp32 + ((1, 1), (0, 0), (1, 1), False,
                                                              (0, 0), 1, False, True, True)),
            # the current  _convolution
            ("_convolution", conv_args_fp32[1] + bias_fp32 + ((1, 1), (0, 0), (1, 1), False,
                                                              (0, 0), 1, False, True, True, True)),
            ("conv1d", conv_args_fp32[0]),
            ("conv2d", conv_args_fp32[1]),
            ("conv3d", conv_args_fp32[2]),
            ("conv_tbc", conv_args_fp32[0] + bias_fp32),
            ("conv_transpose1d", conv_args_fp32[0]),
            ("conv_transpose2d", conv_args_fp32[1]),
            ("conv_transpose3d", conv_args_fp32[2]),
            ("convolution", conv_args_fp32[1] + bias_fp32 + ((1, 1), (0, 0), (1, 1), False, (0, 0), 1)),
            ("cudnn_convolution", conv_args_fp32[1] + ((0, 0), (1, 1), (1, 1), 1, False, True, True), TEST_WITH_ROCM),
            ("cudnn_convolution_transpose", conv_args_fp32[1] + ((0, 0), (0, 0), (1, 1),
                                                                 (1, 1), 1, False, True, True), TEST_WITH_ROCM),
            ("prelu", pointwise0_fp32 + element0_fp32),
            ("addmm", mat1_fp32 + mat2_fp32 + mat3_fp32),
            ("addmv", pointwise0_fp32 + mat2_fp32 + pointwise1_fp32),
            ("addr", mat0_fp32 + pointwise0_fp32 + pointwise1_fp32),
            ("matmul", mat0_fp32 + mat1_fp32),
            ("einsum", "bkhd,bqhd->bqkh", mat0_fp32 + mat1_fp32),
            ("mm", mat0_fp32 + mat1_fp32),
            ("mv", mat0_fp32 + pointwise0_fp32),
            ("chain_matmul", mat0_fp32 + mat1_fp32 + mat2_fp32),
            ("addbmm", mat0_fp32 + (torch.randn((n, n, n), device=dev, dtype=torch.float32),
                                    torch.randn((n, n, n), device=dev, dtype=torch.float32))),
            ("baddbmm", (torch.randn((n, n, n), device=dev, dtype=torch.float32),
                         torch.randn((n, n, n), device=dev, dtype=torch.float32),
                         torch.randn((n, n, n), device=dev, dtype=torch.float32))),
            ("bmm", (torch.randn((n, n, n), device=dev, dtype=torch.float32),
                     torch.randn((n, n, n), device=dev, dtype=torch.float32))),
            # _thnn_fused_lstm_cell and _thnn_fused_gru_cell are not Python-exposed as far as I can tell.
            # ("_thnn_fused_lstm_cell", mat0_fp32 + mat1_fp32 + mat2_fp32 + pointwise0_fp32 + pointwise1_fp32),
            # ("_thnn_fused_gru_cell", mat0_fp32 + mat1_fp32 + mat2_fp32 + pointwise0_fp32 + pointwise1_fp32),
            ("lstm_cell", self._rnn_cell_args(n, num_chunks=4, is_lstm=True, dev=dev, dtype=torch.float32)),
            ("gru_cell", self._rnn_cell_args(n, num_chunks=3, is_lstm=False, dev=dev, dtype=torch.float32)),
            ("rnn_tanh_cell", self._rnn_cell_args(n, num_chunks=1, is_lstm=False, dev=dev, dtype=torch.float32)),
            ("rnn_relu_cell", self._rnn_cell_args(n, num_chunks=1, is_lstm=False, dev=dev, dtype=torch.float32)),
        ]
        self.torch_fp32 = [
            ("acos", (pointwise0_fp16[0].clamp(-.9, 0.9),)),
            ("asin", (pointwise0_fp16[0].clamp(-.9, 0.9),)),
            ("cosh", pointwise0_fp16),
            ("erfinv", (pointwise0_fp16[0].clamp(-.9, .9),)),
            ("exp", pointwise0_fp16),
            ("expm1", pointwise0_fp16),
            ("log", (pointwise0_fp16[0].clamp(0.1, 100.0),)),
            ("log10", (pointwise0_fp16[0].clamp(0.1, 100.0),)),
            ("log2", (pointwise0_fp16[0].clamp(0.1, 100.0),)),
            ("log1p", (pointwise0_fp16[0].clamp(-0.9, 100.0),)),
            ("reciprocal", pointwise0_fp16),
            ("rsqrt", (pointwise0_fp16[0].clamp(0.0, 100.0),)),
            ("sinh", pointwise0_fp16),
            ("tan", (pointwise0_fp16[0].clamp(-3.1 / 2, 3.1 / 2),)),
            ("pow", ((pointwise0_fp16[0] + 1.).clamp(0.0, 100.0),) + pointwise1_fp16),
            ("pow", ((pointwise0_fp16[0] + 1.).clamp(0.0, 100.0),) + (1.7,)),
            # ("pow", (1.7,) + pointwise0_fp16), # This variant has a backend, but is not documented in the API.
            ("softmax", pointwise0_fp16 + (0,)),
            ("log_softmax", pointwise0_fp16 + (0,)),
            ("layer_norm", pointwise0_fp16 + ((pointwise0_fp16[0].numel(),),)),
            ("group_norm", mat0_fp16 + (1,)),
            ("norm", pointwise0_fp16),
            ("norm", pointwise0_fp16, {"dim": 0}),
            # these need magma
            # ("norm", mat0_fp16, {"p": "nuc"}),
            # ("norm", mat0_fp16, {"p": "nuc", "dim": 0}),
            ("norm", pointwise0_fp16, {"p": 1}),
            ("norm", pointwise0_fp16, {"p": 1, "dim": 0}),
            ("cosine_similarity", mat0_fp16 + mat1_fp16),
            ("poisson_nll_loss", mat0_fp16 + mat1_fp16 + (True, False, 1.e-8, torch.nn._reduction.get_enum('mean'))),
            ("cosine_embedding_loss", (torch.tensor([[1, 2, 3]], device=dev, dtype=torch.float16),
                                       torch.tensor([[1, 3, 4]], device=dev, dtype=torch.float16),
                                       torch.tensor([1], device=dev, dtype=torch.int))),
            ("hinge_embedding_loss", mat0_fp16 + (torch.ones(n, device=dev, dtype=torch.int),)),
            ("kl_div", mat0_fp16 + (torch.rand((n, n), device=dev, dtype=torch.float16),)),
            ("margin_ranking_loss", mat0_fp16 + mat1_fp16 + (torch.ones((n,), device=dev, dtype=torch.float16),)),
            ("triplet_margin_loss", mat0_fp16 + mat1_fp16 + mat2_fp16),
            ("binary_cross_entropy_with_logits", mat0_fp16 + (torch.rand((n, n), device=dev, dtype=torch.float16),)),
            ("cumprod", pointwise0_fp16 + (0,)),
            ("cumsum", pointwise0_fp16 + (0,)),
            ("dist", pointwise0_fp16 + pointwise1_fp16),
            ("pdist", mat0_fp16),
            ("cdist", mat0_fp16 + mat1_fp16),
            ("prod", pointwise0_fp16),
            ("prod", pointwise0_fp16 + (0,)),
            ("renorm", mat0_fp16 + (2, 0, 1.0)),
            ("sum", pointwise0_fp16),
            ("sum", mat0_fp16 + (1,)),
            ("logsumexp", mat0_fp16 + (1,)),
        ]
        self.torch_need_autocast_promote = [
            ("addcdiv", pointwise0_fp32 + pointwise1_fp16 + (pointwise2_fp16[0].clamp(0.1, 100),)),
            ("addcmul", pointwise0_fp32 + pointwise1_fp16 + pointwise2_fp16),
            ("atan2", pointwise0_fp32 + (pointwise1_fp16[0].clamp(0.1, 100),)),
            ("bilinear", (torch.randn((1, 2), dtype=torch.float16, device=dev),
                          torch.randn((1, 2), dtype=torch.float32, device=dev),
                          torch.randn((1, 2, 2), dtype=torch.float16, device=dev),
                          torch.randn((1,), dtype=torch.float32, device=dev))),
            ("cross", (torch.randn(3, dtype=torch.float32, device=dev),
                       torch.randn(3, dtype=torch.float16, device=dev))),
            ("dot", pointwise0_fp16 + pointwise1_fp32),
            ("vdot", pointwise0_fp16 + pointwise1_fp32),
            ("grid_sampler", (torch.randn((2, 3, 33, 22), dtype=torch.float16, device=dev),
                              torch.randn((2, 22, 11, 2), dtype=torch.float32, device=dev),
                              0, 0, False)),
            ("index_put", pointwise0_fp32 + ((torch.tensor([1], device=dev, dtype=torch.long),),
                                             torch.randn(1, device=dev, dtype=torch.float16))),
            ("index_put", pointwise0_fp16 + ((torch.tensor([1], device=dev, dtype=torch.long),),
                                             torch.randn(1, device=dev, dtype=torch.float32))),
            ("tensordot", (torch.randn((2, 2, 2), dtype=torch.float32, device=dev),
                           torch.randn((2, 2, 2), dtype=torch.float16, device=dev))),
            ("scatter_add", (torch.zeros(2, 2, 2, dtype=torch.float32, device=dev),
                             0,
                             torch.randint(0, 2, (2, 2, 2), device=dev),
                             torch.randn((2, 2, 2), dtype=torch.float16, device=dev))),
            ("scatter_add", (torch.zeros(2, 2, 2, dtype=torch.float16, device=dev),
                             0,
                             torch.randint(0, 2, (2, 2, 2), device=dev),
                             torch.randn((2, 2, 2), dtype=torch.float32, device=dev))),
        ]
        self.nn_fp16 = [
            ("linear", mat0_fp32 + mat1_fp32 + mat2_fp32),
        ]
        self.nn_fp32 = [
            ("softplus", pointwise0_fp16),
            ("nll_loss", (torch.rand((n, n), device=dev, dtype=torch.float),
                          torch.zeros((n,), device=dev, dtype=torch.long))),
            ("nll_loss2d", (torch.rand((n, n, n, n), device=dev, dtype=torch.half),
                            torch.zeros((n, n, n), device=dev, dtype=torch.long))),
            ("l1_loss", mat0_fp16 + mat1_fp16),
            ("smooth_l1_loss", mat0_fp16 + mat1_fp16),
            ("mse_loss", mat0_fp16 + mat1_fp16),
            ("multilabel_margin_loss", mat0_fp16 + (torch.ones((n, n), device=dev, dtype=torch.long),)),
            ("soft_margin_loss", mat0_fp16 + (torch.ones((n, n), device=dev, dtype=torch.long),)),
            ("multi_margin_loss", mat0_fp16 + (torch.ones((n,), device=dev, dtype=torch.long),)),
        ]
        self.linalg_fp16 = [
            ("linalg_vecdot", mat0_fp32 + mat0_fp32),
            ("linalg_multi_dot", (mat0_fp32 + mat1_fp32 + mat2_fp32,)),
        ]
        self.methods_fp16 = [
            ("__matmul__", mat0_fp32 + mat1_fp32)
        ]
        self.methods_fp32 = [
            ("__pow__", (torch.rand(n, device=dev, dtype=torch.float16), 1.5)),
        ]
        self.banned = [
            ("binary_cross_entropy", (torch.rand((n, n), device=dev, dtype=torch.float32),
                                      torch.rand((n, n), device=dev, dtype=torch.float32)), torch._C._nn),
        ]


class AutocastCPUTestLists:
    # Supplies ops and arguments for test_autocast_* in test/test_cpu.py
    def __init__(self, dev):
        super().__init__()
        n = 8
        # Utility arguments, created as one-element tuples
        pointwise0_bf16 = (torch.randn(n, dtype=torch.bfloat16, device=dev),)
        pointwise1_bf16 = (torch.randn(n, dtype=torch.bfloat16, device=dev),)
        mat0_bf16 = (torch.randn((n, n), dtype=torch.bfloat16, device=dev),)
        mat1_bf16 = (torch.randn((n, n), dtype=torch.bfloat16, device=dev),)
        mat2_bf16 = (torch.randn((n, n), dtype=torch.bfloat16, device=dev),)

        pointwise0_fp16 = (torch.randn(n, dtype=torch.float16, device=dev),)
        pointwise1_fp16 = (torch.randn(n, dtype=torch.float16, device=dev),)

        dummy_dimsets = ((n,), (n, n), (n, n, n), (n, n, n, n), (n, n, n, n, n))

        dummy_bf16 = [(torch.randn(dimset, dtype=torch.bfloat16, device=dev),)
                      for dimset in dummy_dimsets]

        dimsets = ((n, n, n), (n, n, n, n), (n, n, n, n, n))
        conv_args_fp32 = [(torch.randn(dimset, dtype=torch.float32, device=dev),
                           torch.randn(dimset, dtype=torch.float32, device=dev))
                          for dimset in dimsets]

        element0_fp32 = (torch.randn(1, dtype=torch.float32, device=dev),)
        pointwise0_fp32 = (torch.randn(n, dtype=torch.float32, device=dev),)
        pointwise1_fp32 = (torch.randn(n, dtype=torch.float32, device=dev),)
        mat0_fp32 = (torch.randn((n, n), dtype=torch.float32, device=dev),)
        mat1_fp32 = (torch.randn((n, n), dtype=torch.float32, device=dev),)
        mat2_fp32 = (torch.randn((n, n), dtype=torch.float32, device=dev),)
        mat3_fp32 = (torch.randn((n, n), dtype=torch.float32, device=dev),)

        dummy_fp32 = [  # noqa: F841
            (torch.randn(dimset, dtype=torch.float32, device=dev),)
            for dimset in dummy_dimsets
        ]
        # The lists below organize ops that autocast needs to test.
        # self.list_name corresponds to test_autocast_list_name in test/test_cpu.py.
        # Each op is associated with a tuple of valid arguments.

        # Some ops implement built-in type promotion.  These don't need autocasting,
        # but autocasting relies on their promotion, so we include tests to double-check.
        self.torch_expect_builtin_promote = [
            ("eq", pointwise0_fp32 + pointwise1_bf16, pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("ge", pointwise0_fp32 + pointwise1_bf16, pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("gt", pointwise0_fp32 + pointwise1_bf16, pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("le", pointwise0_fp32 + pointwise1_bf16, pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("lt", pointwise0_fp32 + pointwise1_bf16, pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("ne", pointwise0_fp32 + pointwise1_bf16, pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("add", pointwise0_fp32 + pointwise1_bf16, pointwise0_fp32 + pointwise1_fp16, torch.float32),
            ("div", pointwise0_fp32 + pointwise1_bf16, pointwise0_fp32 + pointwise1_fp16, torch.float32),
            ("mul", pointwise0_fp32 + pointwise1_bf16, pointwise0_fp32 + pointwise1_fp16, torch.float32),
        ]

        self.methods_expect_builtin_promote = [
            ("__eq__", pointwise0_fp32 + pointwise1_bf16, pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("__ge__", pointwise0_fp32 + pointwise1_bf16, pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("__gt__", pointwise0_fp32 + pointwise1_bf16, pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("__le__", pointwise0_fp32 + pointwise1_bf16, pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("__lt__", pointwise0_fp32 + pointwise1_bf16, pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("__ne__", pointwise0_fp32 + pointwise1_bf16, pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("__add__", pointwise0_fp32 + pointwise1_bf16, pointwise0_fp32 + pointwise1_fp16, torch.float32),
            ("__div__", pointwise0_fp32 + pointwise1_bf16, pointwise0_fp32 + pointwise1_fp16, torch.float32),
            ("__mul__", pointwise0_fp32 + pointwise1_bf16, pointwise0_fp32 + pointwise1_fp16, torch.float32),
        ]
        # The remaining lists organize ops that autocast treats explicitly.
        self.torch_16 = [
            ("conv1d", conv_args_fp32[0]),
            ("conv2d", conv_args_fp32[1]),
            ("conv3d", conv_args_fp32[2]),
            ("bmm", (torch.randn((n, n, n), device=dev, dtype=torch.float32),
                     torch.randn((n, n, n), device=dev, dtype=torch.float32))),
            ("mm", mat0_fp32 + mat1_fp32),
            ("matmul", mat0_fp32 + mat1_fp32),
            ("baddbmm", (torch.randn((n, n, n), device=dev, dtype=torch.float32),
                         torch.randn((n, n, n), device=dev, dtype=torch.float32),
                         torch.randn((n, n, n), device=dev, dtype=torch.float32))),
            ("addmm", mat1_fp32 + mat2_fp32 + mat3_fp32),
            ("_addmm_activation", mat1_fp32 + mat2_fp32 + mat3_fp32, {"beta": 1, "alpha": 1, "use_gelu": True}),
            ("addbmm", mat0_fp32 + (torch.randn((n, n, n), device=dev, dtype=torch.float32),
                                    torch.randn((n, n, n), device=dev, dtype=torch.float32))),
            ("conv_tbc", (torch.randn((10, 7, 3), device=dev, dtype=torch.float32),
                          torch.randn((5, 3, 5), device=dev, dtype=torch.float32),
                          torch.randn(5, device=dev, dtype=torch.float32),
                          0)),
            ("conv_transpose1d", conv_args_fp32[0]),
            ("conv_transpose2d", conv_args_fp32[1]),
            ("conv_transpose3d", conv_args_fp32[2]),
            ("prelu", pointwise0_fp32 + element0_fp32),
            ("_native_multi_head_attention", (torch.randn((n, n, n), device=dev, dtype=torch.float32),
                                              torch.randn((n, n, n), device=dev, dtype=torch.float32),
                                              torch.randn((n, n, n), device=dev, dtype=torch.float32),
                                              n, 4, torch.randn((3 * n, n), device=dev, dtype=torch.float32),
                                              torch.randn((3 * n), device=dev, dtype=torch.float32),
                                              torch.randn((n, n), device=dev, dtype=torch.float32),
                                              torch.randn((n), device=dev, dtype=torch.float32))),
        ]
        self.torch_fp32 = [
            ("poisson_nll_loss", mat0_bf16 + mat1_bf16 + (True, False, 1.e-8, torch.nn._reduction.get_enum('mean'))),
            ("cosine_embedding_loss", (torch.tensor([[1, 2, 3]], device=dev, dtype=torch.bfloat16),
                                       torch.tensor([[1, 3, 4]], device=dev, dtype=torch.bfloat16),
                                       torch.tensor([1], device=dev, dtype=torch.int))),
            ("hinge_embedding_loss", mat0_bf16 + (torch.ones(n, device=dev, dtype=torch.int),)),
            ("margin_ranking_loss", mat0_bf16 + mat1_bf16 + (torch.ones((n,), device=dev, dtype=torch.bfloat16),)),
            ("triplet_margin_loss", mat0_bf16 + mat1_bf16 + mat2_bf16),
            ("binary_cross_entropy_with_logits", mat0_bf16 + (torch.rand((n, n), device=dev, dtype=torch.bfloat16),)),
        ]
        self.nn_16 = [
            ("linear", mat0_fp32 + mat1_fp32, {}),
        ]
        self.nn_fp32 = [
            ("avg_pool3d", dummy_bf16[3], {"kernel_size": (3, 3, 3), "stride": (1, 1, 1)}),
            ("binary_cross_entropy", (torch.rand((n, n), device=dev, dtype=torch.bfloat16),) +
                                     (torch.rand((n, n), device=dev, dtype=torch.bfloat16),)),
            ("reflection_pad1d", dummy_bf16[2], {"padding": (3, 3)}),
            ("nll_loss", (torch.rand((n, n), device=dev, dtype=torch.bfloat16),
                          torch.zeros((n,), device=dev, dtype=torch.long))),
            ("nll_loss2d", (torch.rand((n, n, n, n), device=dev, dtype=torch.bfloat16),
                            torch.zeros((n, n, n), device=dev, dtype=torch.long))),
            ("l1_loss", mat0_bf16 + mat1_bf16),
            ("smooth_l1_loss", mat0_bf16 + mat1_bf16),
            ("mse_loss", mat0_bf16 + mat1_bf16),
            ("multilabel_margin_loss", mat0_bf16 + (torch.ones((n, n), device=dev, dtype=torch.long),)),
            ("soft_margin_loss", mat0_bf16 + (torch.ones((n, n), device=dev, dtype=torch.long),)),
            ("multi_margin_loss", mat0_bf16 + (torch.ones((n,), device=dev, dtype=torch.long),)),
            ("huber_loss", mat0_bf16 + mat1_bf16),
        ]
        self.torch_need_autocast_promote = [
            ("cat", (pointwise0_bf16 + pointwise1_fp32,), (pointwise0_fp16 + pointwise1_fp32,)),
            ("stack", (pointwise0_bf16 + pointwise1_fp32,), (pointwise0_fp16 + pointwise1_fp32,)),
        ]


class TestAutocast(TestCase):
    def args_maybe_kwargs(self, op_with_args):
        if len(op_with_args) == 2:
            return op_with_args[0], op_with_args[1], {}
        else:
            return op_with_args[0], op_with_args[1], op_with_args[2]

    def _run_autocast_outofplace(
        self,
        op,
        args,
        run_as_type,
        device,
        out_type=None,
        module=torch,
        add_kwargs=None,
        amp_dtype=torch.bfloat16,
    ):
        # helper to cast args
        def cast(val, to_type):
            if isinstance(val, torch.Tensor):
                return val.to(to_type) if val.is_floating_point() else val
            elif isinstance(val, collections.abc.Iterable):
                return type(val)(cast(v, to_type) for v in val)
            else:
                return val

        if add_kwargs is None:
            add_kwargs = {}

        self.assertFalse(torch.is_autocast_enabled(device_type=device))
        with torch.amp.autocast(device_type=device, dtype=amp_dtype):
            self.assertTrue(torch.is_autocast_enabled(device_type=device))

            out_type = out_type if out_type is not None else run_as_type
            output = output_method = None

            # Try module.* variant, if requested:
            if module is not None and hasattr(module, op):
                output = getattr(module, op)(*args, **add_kwargs)
                if isinstance(output, torch.Tensor):
                    self.assertTrue(
                        out_type == output.dtype,
                        f"autocast for torch.{op} produced {output.dtype}, should produce {out_type}",
                    )
            # Try Tensor.* variant:
            if hasattr(torch.Tensor, op):
                output_method = getattr(args[0], op)(*args[1:], **add_kwargs)
                if isinstance(output_method, torch.Tensor):
                    self.assertTrue(
                        out_type == output_method.dtype,
                        f"autocast for torch.{op} produced {output_method.dtype}, should produce torch.{out_type}",
                    )

            self.assertTrue(
                (output is not None) or (output_method is not None),
                f"{op} not found as an attribute on either Tensor or the requested module {module}",
            )

            # Accounts for ops that return Tensors, iterables, and other non-Tensors.
            # For example, lstm_cell returns a tuple and equal returns bool.
            def compare(first, second):
                if isinstance(first, torch.Tensor):
                    return torch.equal(first, second)
                elif isinstance(first, collections.abc.Iterable):
                    return all(compare(f, s) for f, s in zip(first, second))
                else:
                    return first == second

            # If both torch.* and Tensor.* variants were found, check outputs are identical
            if (output is not None) and (output_method is not None):
                self.assertTrue(type(output) == type(output_method))
                comparison = compare(output, output_method)
                self.assertTrue(
                    comparison, f"torch.{op} result did not match Tensor.{op} result"
                )

            # Compare numerics to Python-side "autocasting" that (we expect) does the same thing
            # as the C++-side autocasting, and should be bitwise accurate.
            output_to_compare = output if output is not None else output_method
            with torch.amp.autocast(device_type=device, enabled=False):
                self.assertFalse(
                    torch.is_autocast_enabled(device_type=device)
                )

                if module is not None and hasattr(module, op):
                    control = getattr(module, op)(
                        *cast(args, run_as_type), **add_kwargs
                    )
                else:
                    control = getattr(args[0].to(run_as_type), op)(
                        *cast(args[1:], run_as_type), **add_kwargs
                    )
                self.assertTrue(type(output_to_compare) == type(control))
                comparison = compare(output_to_compare, control)
                self.assertTrue(comparison, f"torch.{op} result did not match control")
            self.assertTrue(torch.is_autocast_enabled(device_type=device))
        self.assertFalse(torch.is_autocast_enabled(device_type=device))
