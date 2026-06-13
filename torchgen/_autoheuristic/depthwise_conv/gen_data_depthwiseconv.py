import argparse
import itertools

import pandas as pd

import torch
import torch.nn as nn


class BenchmarkRunnerDepthwiseConv:
    def __init__(self):
        self.batch_sizes = [1, 2, 4, 8, 16, 32, 64]
        self.in_channels = [32 * 2**i for i in range(6)]
        self.heights = [112 // (2**i) for i in range(5)] + [32, 64, 128]
        self.strides = [1, 2]
        self.kernel_sizes = [1, 3, 5]

        self.nb_warmup_iters = 50
        self.nb_iters = 100

        self.columns = [
            "sm",
            "bs",
            "ch",
            "w",
            "filter",
            "stride",
            "time_fwd",
            "time_bwd",
            "time_all",
            "time_fwd_cudnn",
            "time_bwd_cudnn",
            "time_all_cudnn",
            "cudnn_speedup_fwd",
            "cudnn_speedup_bwd",
            "cudnn_speedup_all",
        ]

        # things used for event based timing
        self.rrr = torch.empty(512, 1024, 1024, device="cuda", dtype=torch.float32)
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

        self.parser = argparse.ArgumentParser()
        self.add_base_arguments()
        self.args = None

        major, minor = torch.cuda.get_device_capability()
        self.sm = major * 10 + minor

    def add_base_arguments(self):
        self.parser.add_argument(
            "--device",
            type=str,
            default="",
            help="Label for device being benchmarked",
        )

    def parse_args(self):
        return self.parser.parse_args()

    def run_benchmark(self, batch_size, c, h, s, k):
        w = h
        # Note: cuDNN depthwise conv only supports FP16
        x = torch.randn(
            batch_size, c, h, w, device="cuda", dtype=torch.half, requires_grad=True
        )

        pad = k // 2
        conv = (
            nn.Conv2d(
                in_channels=c,
                out_channels=c,
                kernel_size=k,
                stride=s,
                padding=pad,
                groups=c,
                bias=False,
            )
            .half()
            .to("cuda")
        )

        print(
            "Testing [N, C, H, W]=[{}, {}, {}, {}], kH/kW={}, stride={}, pad={}, cudnn={}".format(
                *x.size(), k, s, pad, torch.backends.cudnn.force
            )
        )

        # Perform some dummy iterations to warmup cudnn.benchmark
        for _ in range(self.nb_warmup_iters):
            output = conv(x)

        # Perform warumup for backwards
        g0 = torch.rand_like(output)
        for _ in range(self.nb_warmup_iters):
            output = conv(x)
            output.backward(g0)

        # Add some super long kernel to make it not cpu bound
        for _ in range(150):
            self.rrr.random_()

        # Profile forward pass
        self.start_event.record()
        for _ in range(self.nb_iters):
            output = conv(x)

        self.end_event.record()
        torch.cuda.synchronize()
        fwd_time = self.start_event.elapsed_time(self.end_event) / self.nb_iters

        # Profile backward pass
        for _ in range(150):
            self.rrr.random_()
        self.start_event.record()
        for _ in range(self.nb_iters):
            output = conv(x)
            x.grad = None
            conv.weight.grad = None
            output.backward(g0)
        self.end_event.record()
        torch.cuda.synchronize()
        all_time = self.start_event.elapsed_time(self.end_event) / self.nb_iters
        bwd_time = all_time - fwd_time
        return fwd_time, bwd_time, all_time

    def run(self):
        self.args = self.parse_args()
        if self.args.device == "":
            self.args.device = torch.cuda.get_device_name().replace(" ", "-")

        results = pd.DataFrame()

        for batch_size, c, h, s, k in itertools.product(
            self.batch_sizes,
            self.in_channels,
            self.heights,
            self.strides,
            self.kernel_sizes,
        ):
            torch.backends.cudnn.depthwise_kernel = "native"
            fwd_time, bwd_time, all_time = self.run_benchmark(batch_size, c, h, s, k)

            torch.backends.cudnn.depthwise_kernel = "cudnn"
            fwd_time_cudnn, bwd_time_cudnn, all_time_cudnn = self.run_benchmark(
                batch_size, c, h, s, k
            )

            cudnn_speedup_fwd = fwd_time / fwd_time_cudnn
            cudnn_speedup_bwd = bwd_time / bwd_time_cudnn
            cudnn_speedup_all = all_time / all_time_cudnn

            tmp_df = pd.DataFrame(
                [
                    [
                        self.sm,
                        batch_size,
                        c,
                        h,
                        k,
                        s,
                        fwd_time,
                        bwd_time,
                        all_time,
                        fwd_time_cudnn,
                        bwd_time_cudnn,
                        all_time_cudnn,
                        cudnn_speedup_fwd,
                        cudnn_speedup_bwd,
                        cudnn_speedup_all,
                    ]
                ],
                columns=self.columns,
            )
            results = pd.concat([results, tmp_df])

        results.to_csv(f"data_depthwiseconv_{self.args.device}.csv", index=False)


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    runner = BenchmarkRunnerDepthwiseConv()
    runner.run()
