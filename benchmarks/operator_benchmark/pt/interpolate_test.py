import operator_benchmark as op_bench
import torch

"""Microbenchmarks for interpolate operator."""


class InterpolateBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, input_size, output_size, channels_last=False):

        input_image = torch.randint(0, 256, size=input_size, dtype=torch.float, device='cpu',
                                    requires_grad=self.auto_set())
        if channels_last:
            input_image = input_image.contiguous(memory_format=torch.channels_last)

        ndim_to_mode = {
            3: 'linear',
            4: 'bilinear',
            5: 'trilinear',
        }

        self.inputs = {
            "input_image": input_image,
            "output_size": output_size,
            "mode": ndim_to_mode[input_image.ndim],
        }

        self.set_module_name("interpolate")

    def forward(self, input_image, output_size, mode):
        return torch.nn.functional.interpolate(input_image, size=output_size, mode=mode,
                                               align_corners=False)


config_short = op_bench.config_list(
    attr_names=["input_size", "output_size"],
    attrs=[
        [(1, 3, 60, 40), (24, 24)],
        [(1, 3, 600, 400), (240, 240)],
        [(1, 3, 320, 320), (256, 256)],
    ],
    cross_product_configs={
        'channels_last': [True, False],
    },
    tags=["short"],
)


config_long = op_bench.config_list(
    attr_names=["input_size", "output_size"],
    attrs=[
        [(1, 3, 320, 320), (512, 512)],
        [(1, 3, 500, 500), (256, 256)],
        [(1, 3, 500, 500), (800, 800)],

        [(2, 128, 64, 46), (128, 128)],
    ],
    cross_product_configs={
        'channels_last': [True, False],
    },
    tags=["long"],
)


config_not_4d = op_bench.config_list(
    # no channels_last as it's only valid for 4D tensors
    attr_names=["input_size", "output_size"],
    attrs=[
        [(1, 3, 16, 320, 320), (8, 256, 256)],
        [(1, 3, 16, 320, 320), (32, 512, 512)],

        [(4, 512, 320), (256,)],
        [(4, 512, 320), (512,)],
    ],
    tags=["long"],
)


for config in (config_short, config_long, config_not_4d):
    op_bench.generate_pt_test(config, InterpolateBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
