"""Microbenchmarks for the torch.fft module"""
from argparse import ArgumentParser
import math
from collections import namedtuple
from collections.abc import Iterable

import torch
import torch.fft
from torch.utils import benchmark
from torch.utils.benchmark import FuzzedParameter, FuzzedTensor


MIN_DIM_SIZE = 16
MAX_DIM_SIZE = 16 * 1024

def power_range(upper_bound, base):
    return (base ** i for i in range(int(math.log(upper_bound, base)) + 1))

# List of regular numbers from MIN_DIM_SIZE to MAX_DIM_SIZE
# These numbers factorize into multiples of prime factors 2, 3, and 5 only
# and are usually the fastest in FFT implementations.
REGULAR_SIZES = []
for i in power_range(MAX_DIM_SIZE, 2):
    for j in power_range(MAX_DIM_SIZE // i, 3):
        ij = i * j
        for k in power_range(MAX_DIM_SIZE // ij, 5):
            ijk = ij * k
            if ijk > MIN_DIM_SIZE:
                REGULAR_SIZES.append(ijk)
REGULAR_SIZES.sort()

class SpectralOpFuzzer(benchmark.Fuzzer):
    def __init__(self, *, seed, dtype, cuda):

        super().__init__(
            parameters=[
                # Dimensionality of x. (e.g. 1D, 2D, or 3D.)
                FuzzedParameter("ndim", distribution={1: 0.3, 2: 0.4, 3: 0.3}, strict=True),

                # Shapes for `x`.
                [
                    FuzzedParameter(
                        name=f"k{i}",
                        distribution={size: 1. / len(REGULAR_SIZES) for size in REGULAR_SIZES}
                    ) for i in range(3)
                ],

                # Steps for `x`. (Benchmarks strided memory access.)
                [
                    FuzzedParameter(
                        name=f"step_{i}",
                        distribution={1: 0.8, 2: 0.06, 4: 0.06, 8: 0.04, 16: 0.04},
                    ) for i in range(3)
                ],
            ],
            tensors=[
                FuzzedTensor(
                    name="x",
                    size=("k0", "k1", "k2"),
                    steps=("step_0", "step_1", "step_2"),
                    probability_contiguous=0.75,
                    min_elements=4 * 1024,
                    max_elements=32 * 1024 ** 2,
                    max_allocation_bytes=2 * 1024**3,  # 2 GB
                    dim_parameter="ndim",
                    dtype=dtype,
                    cuda=cuda,
                ),
            ],
            seed=seed,
        )


def _dim_options(ndim):
    if ndim == 1:
        return [None]
    elif ndim == 2:
        return [0, 1, None]
    elif ndim == 3:
        return [0, 1, 2, (0, 1), (0, 2), None]
    raise ValueError(f"Expected ndim in range 1-3, got {ndim}")


def run_benchmark(name: str, function: object, dtype: torch.dtype, seed: int, device: str, samples: int):
    cuda = device == 'cuda'
    spectral_fuzzer = SpectralOpFuzzer(seed=seed, dtype=dtype, cuda=cuda)
    results = []
    for tensors, tensor_params, params in spectral_fuzzer.take(samples):
        shape = [params['k0'], params['k1'], params['k2']][:params['ndim']]
        str_shape = ' x '.join(["{:<4}".format(s) for s in shape])
        sub_label=f"{str_shape} {'' if tensor_params['x']['is_contiguous'] else '(discontiguous)'}"
        for dim in _dim_options(params['ndim']):
            # dim = _select_dim(params)
            for nthreads in (1, 4, 16) if not cuda else (1,):
                measurement = benchmark.Timer(
                    stmt='func(x, dim=dim)',
                    globals={'func': function, 'x': tensors['x'], 'dim': dim},
                    label=f"{name}_{device}",
                    sub_label=sub_label,
                    description=f"dim={dim}",
                    num_threads=nthreads,
                ).blocked_autorange(min_run_time=1)
                measurement.metadata = {
                    'name': name,
                    'device': device,
                    'dim': dim,
                    'shape': shape,
                }
                measurement.metadata.update(tensor_params['x'])
                results.append(measurement)
    return results


Benchmark = namedtuple('Benchmark', ['name', 'function', 'dtype'])
BENCHMARKS = [
    Benchmark('fft_real', torch.fft.fftn, torch.float32),
    Benchmark('fft_complex', torch.fft.fftn, torch.complex64),
    Benchmark('ifft', torch.fft.ifftn, torch.complex64),
    Benchmark('rfft', torch.fft.rfftn, torch.float32),
    Benchmark('irfft', torch.fft.irfftn, torch.complex64),
]
BENCHMARK_MAP = {b.name: b for b in BENCHMARKS}
BENCHMARK_NAMES = [b.name for b in BENCHMARKS]
DEVICE_NAMES = ['cpu', 'cuda']

def _output_csv(file):
    file.write('benchmark,device,num_threads,numel,shape,contiguous,dim,mean (us),median (us),iqr (us)\n')
    for measurement in results:
        metadata = measurement.metadata
        device, dim, shape, name, numel, contiguous = (
            metadata['device'], metadata['dim'], metadata['shape'],
            metadata['name'], metadata['numel'], metadata['is_contiguous'])

        if isinstance(dim, Iterable):
            dim_str = '-'.join(str(d) for d in dim)
        else:
            dim_str = str(dim)
            shape_str = 'x'.join(str(s) for s in shape)

        print(name, device, measurement.task_spec.num_threads, numel, shape_str, contiguous, dim_str,
              measurement.mean * 1e6, measurement.median * 1e6, measurement.iqr * 1e6,
              sep=',', file=file)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=str, choices=DEVICE_NAMES, nargs='+', default=DEVICE_NAMES)
    parser.add_argument('--bench', type=str, choices=BENCHMARK_NAMES, nargs='+', default=BENCHMARK_NAMES)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('-o', '--output', type=str)
    args = parser.parse_args()

    num_benchmarks = len(args.device) * len(args.bench)
    i = 0
    results = []
    for device in args.device:
        for bench in (BENCHMARK_MAP[b] for b in args.bench):
            results += run_benchmark(
                name=bench.name, function=bench.function, dtype=bench.dtype,
                seed=args.seed, device=device, samples=args.samples)
            i += 1
            print(f'Completed {bench.name} benchmark on {device} ({i} of {num_benchmarks})')

    if args.output is not None:
        with open(args.output, 'w') as f:
            _output_csv(f)
    else:
        compare = benchmark.Compare(results)
        compare.trim_significant_figures()
        compare.colorize()
        compare.print()
