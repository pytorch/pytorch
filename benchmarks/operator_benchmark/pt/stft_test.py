import operator_benchmark as op_bench

import torch


"""Microbenchmarks for the stft and istft operators.

On ROCm these run on hipFFT by default and on rocFFT when
TORCH_ROCM_PREFER_ROCFFT=1 is set, so run the benchmark once per setting to
compare the two backends. On a runtime whose rocFFT can JIT them, stft also
gathers and windows its frames inside the transform and istft applies its
synthesis window there; set TORCH_ROCM_DISABLE_ROCFFT_CALLBACKS=1 to measure
those separately.
"""


# n_fft and hop_length are paired rather than crossed: istft requires
# hop_length <= win_length, which a cross product would violate.
stft_short_configs = op_bench.config_list(
    attr_names=["batch", "length", "n_fft", "hop_length"],
    attrs=[
        [1, 16000, 400, 160],
        [8, 16000, 512, 128],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
        "dtype": [torch.float32],
    },
    tags=["short"],
)

# Window sizes typical of 16 kHz speech and 44.1 kHz music pipelines.
stft_long_configs = op_bench.config_list(
    attr_names=["batch", "length", "n_fft", "hop_length"],
    attrs=[
        [8, 160000, 400, 160],
        [8, 160000, 512, 128],
        [4, 1323000, 1024, 256],
        [4, 1323000, 2048, 512],
        [64, 80000, 512, 128],
    ],
    cross_product_configs={
        "device": ["cuda"],
        "dtype": [torch.float32, torch.float64],
    },
    tags=["long"],
)


class StftBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, batch, length, n_fft, hop_length, device, dtype):
        self.inputs = {
            "signal": torch.rand(batch, length, device=device, dtype=dtype),
            "window": torch.hann_window(n_fft, device=device, dtype=dtype),
            "n_fft": n_fft,
            "hop_length": hop_length,
        }
        self.set_module_name("stft")

    def forward(self, signal, window, n_fft: int, hop_length: int):
        return torch.stft(
            signal,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            center=True,
            return_complex=True,
        )


class IstftBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, batch, length, n_fft, hop_length, device, dtype):
        signal = torch.rand(batch, length, device=device, dtype=dtype)
        window = torch.hann_window(n_fft, device=device, dtype=dtype)
        self.inputs = {
            "spectrogram": torch.stft(
                signal,
                n_fft=n_fft,
                hop_length=hop_length,
                window=window,
                center=True,
                return_complex=True,
            ),
            "window": window,
            "n_fft": n_fft,
            "hop_length": hop_length,
            "length": length,
        }
        self.set_module_name("istft")

    def forward(self, spectrogram, window, n_fft: int, hop_length: int, length: int):
        return torch.istft(
            spectrogram,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            center=True,
            length=length,
        )


op_bench.generate_pt_test(stft_short_configs + stft_long_configs, StftBenchmark)
op_bench.generate_pt_test(stft_short_configs + stft_long_configs, IstftBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
