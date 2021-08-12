import timeit
import torch
import torch.nn.functional as F

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

    conv2d_shapes = [
       # [(16, 16, (3, 3), (2, 2), (1, 1), (1, 1), 16, False, 'zeros'),(1, 16, 112, 112), conv_2_1_16],
       # [(72, 72, (3, 3), (2, 2), (1, 1), (1, 1), 72, False, 'zeros'),(1, 72, 56, 56), conv_2_1_72],
       # [(88, 88, (3, 3), (1, 1), (1, 1), (1, 1), 88, False, 'zeros'),(1, 88, 28, 28), conv_1_1_88],
        [(96, 96, (5, 5), (2, 2), (2, 2), (1, 1), 96, False, 'zeros'),(1, 96, 28, 28), conv_2_2_96],
        [(240, 240, (5, 5), (1, 1), (2, 2), (1, 1), 240, False, 'zeros'),(1, 240, 14, 14), conv_1_2_240],
        [(120, 120, (5, 5), (1, 1), (2, 2), (1, 1), 120, False, 'zeros'),(1, 120, 14, 14), conv_1_2_120],
        [(144, 144, (5, 5), (1, 1), (2, 2), (1, 1), 144, False, 'zeros'),(1, 144, 14, 14), conv_1_2_144],
        [(288, 288, (5, 5), (2, 2), (2, 2), (1, 1), 288, False, 'zeros'),(1, 288, 14, 14), conv_2_2_288]
      ]

    print("{:20s} {:20s} {:>10s} {:>10s} {:>10s}".format("op", "shape", "eager", "nnc", "speedup"))
    for init_args, input_shape, conv_op in conv2d_shapes:
        myconv = torch.nn.Conv2d(*init_args)
        input_image = torch.rand(*input_shape)
        weight = myconv.weight
        bias = torch.zeros(init_args[1])

        scripted = torch.jit.script(conv_op)

        # Warmup.
        warmup_iters = 2
        for _ in range(warmup_iters):
            conv_op(input_image, weight, bias)
            scripted(input_image, weight, bias)

        #graph=torch.jit.last_executed_optimized_graph()
        #print(graph)

        # Validate result.
        torch.testing.assert_allclose(myconv(input_image), scripted(input_image, weight, bias))

        # Benchmark.
        bench_iters = 1
        teager = timeit.timeit(stmt="conv_op(input_image, weight, bias)", globals=locals(), number=bench_iters)
        tjit = timeit.timeit(stmt="scripted(input_image, weight, bias)", globals=locals(), number=bench_iters)
        print(f"{conv_op.__name__:20s} ({init_args}, {input_shape}) {teager:10.3f} {tjit:10.3f} {teager/tjit:10.2f}")
test_depthwise_conv()
