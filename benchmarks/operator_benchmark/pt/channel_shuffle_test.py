import operator_benchmark as op_bench

import torch


"""Microbenchmarks for channel_shuffle operator."""


# Configs for PT channel_shuffle operator
channel_shuffle_long_configs = op_bench.cross_product_configs(
    batch_size=[4, 8],
    channels_per_group=[32, 64],
    height=[32, 64],
    width=[32, 64],
    groups=[4, 8],
    channel_last=[True, False],
    tags=["long"],
)


channel_shuffle_short_configs = op_bench.config_list(
    attr_names=["batch_size", "channels_per_group", "height", "width", "groups"],
    attrs=[
        [2, 16, 16, 16, 2],
        [2, 32, 32, 32, 2],
        [4, 32, 32, 32, 4],
        [4, 64, 64, 64, 4],
        [8, 64, 64, 64, 8],
        [16, 64, 64, 64, 16],
    ],
    cross_product_configs={
        "channel_last": [True, False],
    },
    tags=["short"],
)


class ChannelSHuffleBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, batch_size, channels_per_group, height, width, groups, channel_last):
        channels = channels_per_group * groups
        data_shape = (batch_size, channels, height, width)
        input_data = torch.rand(data_shape)
        if channel_last:
            input_data = input_data.contiguous(memory_format=torch.channels_last)
        self.inputs = {"input_data": input_data, "groups": groups}
        self.set_module_name("channel_shuffle")

    def forward(self, input_data, groups: int):
        return torch.channel_shuffle(input_data, groups)


op_bench.generate_pt_test(
    channel_shuffle_short_configs + channel_shuffle_long_configs,
    ChannelSHuffleBenchmark,
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
