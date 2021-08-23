import timeit
import torch
import torch.nn.functional as F
from numpy import median
from functools import partial, reduce
from operator import mul

torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._debug_set_fusion_group_inlining(False)
torch.set_num_threads(1)

def test_depthwise_conv():
    def conv_2_1_16(input_image_arg, weight_arg, bias_arg):
        with torch.no_grad():
            return F.conv2d(input_image_arg, weight_arg, bias=bias_arg, stride=2, padding=1, dilation=1, groups=16)

    def conv_2_1_72(input_image_arg, weight_arg, bias_arg):
        with torch.no_grad():
            return F.conv2d(input_image_arg, weight_arg, bias=bias_arg, stride=2, padding=1, dilation=1, groups=72)

    def conv_1_1_88(input_image_arg, weight_arg, bias_arg):
        with torch.no_grad():
            return F.conv2d(input_image_arg, weight_arg, bias=bias_arg, stride=1, padding=1, dilation=1, groups=88)

    def conv_2_2_96(input_image_arg, weight_arg, bias_arg):
        with torch.no_grad():
            return F.conv2d(input_image_arg, weight_arg, bias=bias_arg, stride=2, padding=2, dilation=1, groups=96)

    def conv_1_2_240(input_image_arg, weight_arg, bias_arg):
        with torch.no_grad():
            return F.conv2d(input_image_arg, weight_arg, bias=bias_arg, stride=1, padding=2, dilation=1, groups=240)

    def conv_1_2_120(input_image_arg, weight_arg, bias_arg):
        with torch.no_grad():
            return F.conv2d(input_image_arg, weight_arg, bias=bias_arg, stride=1, padding=2, dilation=1, groups=120)

    def conv_1_2_144(input_image_arg, weight_arg, bias_arg):
        with torch.no_grad():
            return F.conv2d(input_image_arg, weight_arg, bias=bias_arg, stride=1, padding=2, dilation=1, groups=144)

    def conv_2_2_288(input_image_arg, weight_arg, bias_arg):
        with torch.no_grad():
            return F.conv2d(input_image_arg, weight_arg, bias=bias_arg, stride=2, padding=2, dilation=1, groups=288)

    def conv_1_2_576(input_image_arg, weight_arg, bias_arg):
        with torch.no_grad():
            return F.conv2d(input_image_arg, weight_arg, bias=bias_arg, stride=1, padding=2, dilation=1, groups=576)

    conv2d_shapes = [
        [(16, 16, (3, 3), (2, 2), (1, 1), (1, 1), 16, False, 'zeros'),(1, 16, 112, 112), conv_2_1_16],
        [(72, 72, (3, 3), (2, 2), (1, 1), (1, 1), 72, False, 'zeros'),(1, 72, 56, 56), conv_2_1_72],
        [(88, 88, (3, 3), (1, 1), (1, 1), (1, 1), 88, False, 'zeros'),(1, 88, 28, 28), conv_1_1_88],
        [(96, 96, (5, 5), (2, 2), (2, 2), (1, 1), 96, False, 'zeros'),(1, 96, 28, 28), conv_2_2_96],
        [(240, 240, (5, 5), (1, 1), (2, 2), (1, 1), 240, False, 'zeros'),(1, 240, 14, 14), conv_1_2_240],
        [(120, 120, (5, 5), (1, 1), (2, 2), (1, 1), 120, False, 'zeros'),(1, 120, 14, 14), conv_1_2_120],
        [(144, 144, (5, 5), (1, 1), (2, 2), (1, 1), 144, False, 'zeros'),(1, 144, 14, 14), conv_1_2_144],
        [(288, 288, (5, 5), (2, 2), (2, 2), (1, 1), 288, False, 'zeros'),(1, 288, 14, 14), conv_2_2_288],
        [(576, 576, (5, 5), (1, 1), (2, 2), (1, 1), 576, False, 'zeros'),(1, 576, 7, 7), conv_1_2_576],
      ]

    print("{:50s} {:20s} {:>10s} {:>10s} {:>10s}".format("op", "shape", "eager gflops", "nnc gflops", "speedup"))
    for init_args, input_shape, conv_op in conv2d_shapes:
        myconv = torch.nn.Conv2d(*init_args)
        input_image = torch.rand(*input_shape)
        weight = myconv.weight
        bias = torch.zeros(init_args[1])

        def myconv_op(input):
            with torch.no_grad():
                return myconv(input)

        scripted = torch.jit.script(conv_op)

        # Warmup.
        warmup_iters = 2
        for _ in range(warmup_iters):
            conv_op(input_image, weight, bias)
            scripted(input_image, weight, bias)

        graph=torch.jit.last_executed_optimized_graph()

        # Validate result.
        myconv_out = myconv_op(input_image)
        torch.testing.assert_allclose(myconv_out, scripted(input_image, weight, bias))

        weight_size = list(weight.size())
        out_size = list(myconv_out.size())
        product = partial(reduce, mul)
        gflops = 2 * product(out_size + weight_size) / weight_size[0] / 1000000000.0

        # Benchmark.
        bench_iters, repeat = 10000, 50
        teager = median(timeit.repeat(lambda: myconv_op(input_image), number=bench_iters, repeat=repeat))
        tjit = median(timeit.repeat(lambda: scripted(input_image, weight, bias), number=bench_iters, repeat=repeat))
        eager_gflops = gflops * bench_iters / teager
        jit_gflops = gflops * bench_iters / tjit
        print(f"conv2d ({init_args}, {input_shape}) {eager_gflops:10.3f} {jit_gflops:10.3f} {jit_gflops/eager_gflops:10.2f}")

test_depthwise_conv()
