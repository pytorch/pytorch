import torch
from torch.testing._internal.common_utils import TEST_WITH_ROCM


class AutocastTestLists(object):
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
        ]
        self.methods_expect_builtin_promote = [
            ("__eq__", pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("__ge__", pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("__gt__", pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("__le__", pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("__lt__", pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("__ne__", pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("__add__", pointwise0_fp32 + pointwise1_fp16, torch.float32),
            ("__truediv__", pointwise0_fp32 + pointwise1_fp16, torch.float32),
            ("__mul__", pointwise0_fp32 + pointwise1_fp16, torch.float32),
        ]

        # The remaining lists organize ops that autocast treats explicitly.
        self.torch_fp16 = [
            ("_convolution", conv_args_fp32[1] + bias_fp32 + ((1, 1), (0, 0), (1, 1), False,
                                                              (0, 0), 1, False, True, True)),
            ("_convolution_nogroup", conv_args_fp32[1] + bias_fp32 + ((1, 1), (0, 0), (1, 1), False, (0, 0))),
            ("conv1d", conv_args_fp32[0]),
            ("conv2d", conv_args_fp32[1]),
            ("conv3d", conv_args_fp32[2]),
            ("conv_tbc", conv_args_fp32[0] + bias_fp32),
            ("conv_transpose1d", conv_args_fp32[0]),
            ("conv_transpose2d", conv_args_fp32[1]),
            ("conv_transpose3d", conv_args_fp32[2]),
            ("convolution", conv_args_fp32[1] + bias_fp32 + ((1, 1), (0, 0), (1, 1), False, (0, 0), 1)),
            # cudnn_convolutions with bias
            ("cudnn_convolution", conv_args_fp32[1] + bias_fp32 + ((0, 0), (1, 1), (1, 1), 1, False, True), TEST_WITH_ROCM),
            ("cudnn_convolution_transpose", conv_args_fp32[1] + bias_fp32 + ((0, 0), (0, 0), (1, 1),
                                                                             (1, 1), 1, False, True), TEST_WITH_ROCM),
            # cudnn_convolutions with no bias
            ("cudnn_convolution", conv_args_fp32[1] + ((0, 0), (1, 1), (1, 1), 1, False, True), TEST_WITH_ROCM),
            ("cudnn_convolution_transpose", conv_args_fp32[1] + ((0, 0), (0, 0), (1, 1), (1, 1), 1, False, True), TEST_WITH_ROCM),
            ("prelu", pointwise0_fp32 + element0_fp32),
            ("addmm", mat1_fp32 + mat2_fp32 + mat3_fp32),
            ("addmv", pointwise0_fp32 + mat2_fp32 + pointwise1_fp32),
            ("addr", mat0_fp32 + pointwise0_fp32 + pointwise1_fp32),
            ("matmul", mat0_fp32 + mat1_fp32),
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
            ("poisson_nll_loss", mat0_fp16 + mat1_fp16 + (True, False, 1.e-8,
                                                          torch.nn.functional._Reduction.get_enum('mean'))),
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
        ]
        self.torch_need_autocast_promote = [
            ("addcdiv", pointwise0_fp32 + pointwise1_fp16 + (pointwise2_fp16[0].clamp(0.1, 100),)),
            ("addcmul", pointwise0_fp32 + pointwise1_fp16 + pointwise2_fp16),
            ("atan2", pointwise0_fp32 + (pointwise1_fp16[0].clamp(0.1, 100),)),
            ("cross", (torch.randn(3, dtype=torch.float32, device=dev),
                       torch.randn(3, dtype=torch.float16, device=dev))),
            ("bilinear", (torch.randn((1, 2), dtype=torch.float16, device=dev),
                          torch.randn((1, 2), dtype=torch.float32, device=dev),
                          torch.randn((1, 2, 2), dtype=torch.float16, device=dev),
                          torch.randn((1,), dtype=torch.float32, device=dev))),
            ("dot", pointwise0_fp16 + pointwise1_fp32),
            ("tensordot", (torch.randn((2, 2, 2), dtype=torch.float32, device=dev),
                           torch.randn((2, 2, 2), dtype=torch.float16, device=dev))),
            ("equal", pointwise0_fp32 + pointwise1_fp16),
            ("cat", (pointwise0_fp16 + pointwise1_fp32,)),
            ("stack", (pointwise0_fp16 + pointwise1_fp32,)),
        ]
        self.nn_fp16 = [
            ("linear", mat0_fp32 + mat1_fp32 + mat2_fp32),
        ]
        self.nn_fp32 = [
            ("softplus", pointwise0_fp16),
            ("gelu", pointwise0_fp16),
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
