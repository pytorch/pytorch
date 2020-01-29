import torch

# test_cuda.py's setUp creates an instance of this class to supply the ops Amp needs to test.
class AmpLists(object):
    def __init__(self):
        super(AmpLists, self).__init__()
        # Prepare some utility args.
        conv_args_fp16 = [(torch.randn((8, 8, *(8,) * dims), dtype=torch.float16, device="cuda"),
                     torch.randn((8, *(8,) * (dims + 1)), dtype=torch.float16, device="cuda"))
                     for dims in (1, 2, 3)]
        bias_fp16 = (torch.randn((8,), dtype=torch.float16, device="cuda"),)
        pointwise0_fp16 = (torch.randn(8, dtype=torch.float16, device="cuda"),)
        pointwise1_fp16 = (torch.randn(8, dtype=torch.float16, device="cuda"),)
        pointwise2_fp16 = (torch.randn(8, dtype=torch.float16, device="cuda"),)
        pointwise3_fp16 = (torch.randn(8, dtype=torch.float16, device="cuda"),)
        element0_fp16 = (torch.randn(1, dtype=torch.float16, device="cuda"),)
        mat0_fp16 = (torch.randn((8, 8), dtype=torch.float16, device="cuda"),)
        mat1_fp16 = (torch.randn((8, 8), dtype=torch.float16, device="cuda"),)
        mat2_fp16 = (torch.randn((8, 8), dtype=torch.float16, device="cuda"),)
        mat3_fp16 = (torch.randn((8, 8), dtype=torch.float16, device="cuda"),)
        mat4_fp16 = (torch.randn((8, 8), dtype=torch.float16, device="cuda"),)

        conv_args_fp32 = [(torch.randn((8, 8, *(8,) * dims), dtype=torch.float32, device="cuda"),
                     torch.randn((8, *(8,) * (dims + 1)), dtype=torch.float32, device="cuda"))
                     for dims in (1, 2, 3)]
        bias_fp32 = (torch.randn((8,), dtype=torch.float32, device="cuda"),)
        pointwise0_fp32 = (torch.randn(8, dtype=torch.float32, device="cuda"),)
        pointwise1_fp32 = (torch.randn(8, dtype=torch.float32, device="cuda"),)
        pointwise2_fp32 = (torch.randn(8, dtype=torch.float32, device="cuda"),)
        pointwise3_fp32 = (torch.randn(8, dtype=torch.float32, device="cuda"),)
        element0_fp32 = (torch.randn(1, dtype=torch.float32, device="cuda"),)
        mat0_fp32 = (torch.randn((8, 8), dtype=torch.float32, device="cuda"),)
        mat1_fp32 = (torch.randn((8, 8), dtype=torch.float32, device="cuda"),)
        mat2_fp32 = (torch.randn((8, 8), dtype=torch.float32, device="cuda"),)
        mat3_fp32 = (torch.randn((8, 8), dtype=torch.float32, device="cuda"),)
        mat4_fp32 = (torch.randn((8, 8), dtype=torch.float32, device="cuda"),)

        # The lists below organize the different kinds of ops Amp needs to handle.
        # To assist tests, each op is associated with a tuple of valid arguments.
        # Most of the lists are empty, but writing them all out makes the classification
        # clear(er).  If I only wrote the non-empty ones, it would look like a total mess of
        # special cases (and it still is), but there's a method to the madness and seeing
        # the full classification gives you a sense why each case exists.
        #
        # Only non-empty lists are given a test in test_cuda.py.
        self.torch_fp16 = [
            ("_convolution", conv_args_fp32[1] + bias_fp32 + ((1, 1), (0, 0), (1, 1), False, (0, 0), 1, False, False, True)),
            ("_convolution_nogroup", conv_args_fp32[1] + bias_fp32 + ((1, 1), (0, 0), (1, 1), False, (0, 0))),
            ("conv1d", conv_args_fp32[0]),
            ("conv2d", conv_args_fp32[1]),
            ("conv3d", conv_args_fp32[2]),
            ("conv_tbc", conv_args_fp32[0] + bias_fp32),
            ("conv_transpose1d", conv_args_fp32[0]),
            ("conv_transpose2d", conv_args_fp32[1]),
            ("conv_transpose3d", conv_args_fp32[2]),
            ("convolution", conv_args_fp32[1] + bias_fp32 + ((1, 1), (0, 0), (1, 1), False, (0, 0), 1)),
            ("cudnn_convolution", conv_args_fp32[1] + bias_fp32 + ((0, 0), (1, 1), (1, 1), 1, False, False)),
            ("cudnn_convolution_transpose", conv_args_fp32[1] + bias_fp32 + ((0, 0), (0, 0), (1, 1), (1, 1), 1, False, False)),
            # versions with no bias
            ("cudnn_convolution", conv_args_fp32[1] + ((0, 0), (1, 1), (1, 1), 1, False, False)),
            ("cudnn_convolution_transpose", conv_args_fp32[1] + ((0, 0), (0, 0), (1, 1), (1, 1), 1, False, False)),
            ("prelu", pointwise0_fp32 + element0_fp32),
            ("addmm", mat1_fp32 + mat2_fp32 + mat3_fp32),
            ("addmv", pointwise0_fp32 + mat2_fp32 + pointwise1_fp32),
            ("addr", mat0_fp32 + pointwise0_fp32 + pointwise1_fp32),
            ("matmul", mat0_fp32 + mat1_fp32),
            ("mm", mat0_fp32 + mat1_fp32),
            ("mv", mat0_fp32 + pointwise0_fp32),
            ("chain_matmul", mat0_fp32 + mat1_fp32 + mat2_fp32),
            ("addbmm", mat0_fp32 + (torch.randn((8,8,8), device="cuda", dtype=torch.float32),
                                    torch.randn((8,8,8), device="cuda", dtype=torch.float32))),
            ("baddbmm", (torch.randn((8,8,8), device="cuda", dtype=torch.float32),
                         torch.randn((8,8,8), device="cuda", dtype=torch.float32),
                         torch.randn((8,8,8), device="cuda", dtype=torch.float32))),
            ("bmm", (torch.randn((8,8,8), device="cuda", dtype=torch.float32),
                     torch.randn((8,8,8), device="cuda", dtype=torch.float32))),
        ]
        # self.torch_fp16_inplace = []
        # self.torch_fp16_user_supplied_out = []
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
            ("tan", (pointwise0_fp16[0].clamp(-3.1/2, 3.1/2),)),
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
            ("cosine_embedding_loss", (torch.tensor([[1,2,3]], device="cuda", dtype=torch.float16),
                                       torch.tensor([[1,3,4]], device="cuda", dtype=torch.float16),
                                       torch.tensor([1], device="cuda", dtype=torch.int))),
            ("hinge_embedding_loss", mat0_fp16 +(torch.ones((8,), device="cuda", dtype=torch.int),)),
            ("kl_div", mat0_fp16 + (torch.rand((8,8), device="cuda", dtype=torch.float16),)),
            ("margin_ranking_loss", mat0_fp16 + mat1_fp16 + (torch.ones((8,), device="cuda", dtype=torch.float16),)),
            ("triplet_margin_loss", mat0_fp16 + mat1_fp16 + mat2_fp16),
            ("binary_cross_entropy_with_logits", mat0_fp16 + (torch.rand((8,8), device="cuda", dtype=torch.float16),)),
            ("cumprod", pointwise0_fp16 + (0,)),
            ("cumsum", pointwise0_fp16 + (0,)),
            ("dist", pointwise0_fp16 + pointwise1_fp16),
            ("pdist", mat0_fp16),
            ("cdist", mat0_fp16 + mat1_fp16),
            ("prod", pointwise0_fp16),
            ("prod", pointwise0_fp16 + (0,)),
        ]
        # self.torch_fp32_inplace = []
        # self.torch_fp32_user_supplied_out = []
        # self.torch_fp32 = []
        # self.torch_fp32_inplace = []
        # self.torch_fp32_user_supplied_out = []
        self.torch_need_autocast_promote = [
            ("addcdiv", pointwise0_fp32 + pointwise1_fp16 + (pointwise2_fp16[0].clamp(0.1, 100),)),
            ("addcmul", pointwise0_fp32 + pointwise1_fp16 + pointwise2_fp16),
            ("atan2", pointwise0_fp32 + (pointwise1_fp16[0].clamp(0.1, 100),)),
            ("cross", (torch.randn(3, dtype=torch.float32, device="cuda"),
                       torch.randn(3, dtype=torch.float16, device="cuda"))),
            ("bilinear", (torch.randn((1,2), dtype=torch.float16, device="cuda"),
                          torch.randn((1,2), dtype=torch.float32, device="cuda"),
                          torch.randn((1,2,2), dtype=torch.float16, device="cuda"),
                          torch.randn((1,), dtype=torch.float32, device="cuda"))),
            ("dot", pointwise0_fp16 + pointwise1_fp32),
            ("tensordot", (torch.randn((2,2,2), dtype=torch.float32, device="cuda"),
                           torch.randn((2,2,2), dtype=torch.float16, device="cuda"))),
            ("equal", pointwise0_fp32 + pointwise1_fp16),
            ("cat", (pointwise0_fp16 + pointwise1_fp32,)),
            ("stack", (pointwise0_fp16 + pointwise1_fp32,)),
        ]
        # self.torch_passthrough_user_supplied_out = []
        # self.torch_firstarg_inplace = []
        # self.torch_firstarg_user_supplied_out = []
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
        # self.torch_expect_builtin_promote_inplace = []
        # self.torch_expect_builtin_promote_user_supplied_out = []
        # self.torch_need_autocast_sequence_cast_ops = [
        #     ("cat", mat1_fp16 + mat1_fp32 + mat1_fp16),
        #     ("stack", mat1_fp16 + mat1_fp32 + mat1_fp16)]

        self.nn_fp16 = [
            ("linear", mat0_fp32 + mat1_fp32 + mat2_fp32),
        ]
        # self.nn_fp16_inplace = []
        # self.nn_fp16_user_supplied_out = []
        self.nn_fp32 = [
            ("softplus", pointwise0_fp16),
            ("gelu", pointwise0_fp16),
            ("nll_loss", (torch.rand((8,8), device="cuda", dtype=torch.float),
                          torch.zeros((8,), device="cuda", dtype=torch.long))),
            ("nll_loss2d", (torch.rand((8,8,8,8), device="cuda", dtype=torch.half),
                            torch.zeros((8,8,8), device="cuda", dtype=torch.long))),
            ("l1_loss", mat0_fp16 + mat1_fp16),
            ("smooth_l1_loss", mat0_fp16 + mat1_fp16),
            ("mse_loss", mat0_fp16 + mat1_fp16),
            ("multilabel_margin_loss", mat0_fp16 + (torch.ones((8,8), device="cuda", dtype=torch.long),)),
            ("soft_margin_loss", mat0_fp16 + (torch.ones((8,8), device="cuda", dtype=torch.long),)),
            ("multi_margin_loss", mat0_fp16 + (torch.ones((8,), device="cuda", dtype=torch.long),)),
        ]
        # self.nn_fp32_inplace = []
        # self.nn_fp32_user_supplied_out = []
        # self.nn_fp32 = []
        # self.nn_fp32_inplace = []
        # self.nn_fp32_user_supplied_out = []
        # self.nn_need_autocast_promote = []
        # self.nn_firstarg_inplace = []
        # self.nn_firstarg_user_supplied_out = []
        # self.nn_expect_builtin_promote = []
        # self.nn_expect_builtin_promote_inplace = []
        # self.nn_expect_builtin_promote_user_supplied_out = []

        # self.tensor_only_fp16 = []
        # self.tensor_only_fp16_inplace = []
        # self.tensor_only_fp16_user_supplied_out = []
        # self.tensor_only_fp32 = []
        # self.tensor_only_fp32_inplace = []
        # self.tensor_only_fp32_user_supplied_out = []
        # self.tensor_only_fp32 = []
        # self.tensor_only_fp32_inplace = []
        # self.tensor_only_fp32_user_supplied_out = []
        # self.tensor_only_need_autocast_promote = []
        # self.tensor_only_firstarg_inplace = []
        # self.tensor_only_firstarg_user_supplied_out = []
        # self.tensor_only_expect_builtin_promote = []
        # self.tensor_only_expect_builtin_promote_inplace = []
        # self.tensor_only_expect_builtin_promote_user_supplied_out = []

        self.banned = [
          ("binary_cross_entropy", (torch.rand((8,8), device="cuda", dtype=torch.float32),
                                    torch.rand((8,8), device="cuda", dtype=torch.float32)), torch._C._nn),
        ]
