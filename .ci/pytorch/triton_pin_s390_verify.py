"""Verify the original VJP using full native CPU kernels and unchanged tolerances."""

import argparse
import hashlib
import inspect
import itertools
import json
import os
from pathlib import Path
import platform
import re
import shlex
import sys

import pytest
import torch
from torch.testing._internal.common_utils import TestCase
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'test/test-reports/triton-pin-s390-verify'
DEBUG = ROOT / 'test/debug/triton-pin-s390-verify'
NODE = 'functorch/test_ops.py::TestOperatorsDeviceCPU::test_vjp_grid_sampler_2d_cpu_'


def extract_backward():
    text = (ROOT / 'aten/src/ATen/native/GridSampler.cpp').read_text()
    match = re.search(r'std::tuple<Tensor,\s*Tensor>\s+grid_sampler_2d_backward_cpu\(', text)
    assert match
    start = text.index('{', match.start())
    depth = 0
    for end in range(start, len(text)):
        depth += (text[end] == '{') - (text[end] == '}')
        if not depth:
            return text[match.start():end + 1]
    raise RuntimeError('CPU wrapper body was not found')


def install_kernel(label, source, flags, openmp):
    source = source.replace('#define TORCH_ASSERT_NO_OPERATORS',
        '#include <ATen/Functions.h>\n#include <ATen/native/GridSamplerUtils.h>\n'
        '#include <ATen/ops/_grid_sampler_2d_cpu_fallback_backward_native.h>\n#include <torch/library.h>')
    for name in ['grid_sampler_2d_cpu_kernel', 'grid_sampler_2d_backward_cpu_kernel']:
        source = source.replace(f'REGISTER_DISPATCH({name}, &{name}_impl)', '')
    wrapper = extract_backward().replace('grid_sampler_2d_backward_cpu(', f'pin_backward_{label}(')
    wrapper = re.sub(r'grid_sampler_2d_backward_cpu_kernel\(\s*kCPU,\s*',
        'decltype(&grid_sampler_2d_backward_cpu_kernel_impl) volatile pin_fn = '
        '&grid_sampler_2d_backward_cpu_kernel_impl;\n  pin_fn(', wrapper, count=1)
    assert 'pin_fn(' in wrapper
    source += '\nnamespace at::native {\n' + wrapper + '\n}\n'
    source += '\nTORCH_LIBRARY_IMPL(aten, CPU, m) {\n'
    source += f'  m.impl("grid_sampler_2d_backward", &at::native::pin_backward_{label});\n}}\n'
    folder = DEBUG / label
    folder.mkdir(parents=True, exist_ok=True)
    cpp = folder / 'kernel.cpp'
    cpp.write_text(source)
    load(name='triton_pin_s390_verify_' + label, sources=[str(cpp)],
         extra_cflags=flags, extra_ldflags=['-fopenmp'] if openmp else [],
         build_directory=str(folder), is_python_module=False, verbose=True)


def original_test(label, dtype):
    return int(pytest.main(['-q', '--reruns=0', '--tb=short',
                           f'--junitxml={REPORT / (label + ".xml")}', NODE + dtype]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-repro', type=Path)
    parser.add_argument('--native-wheel-only', action='store_true')
    args = parser.parse_args()
    native = platform.machine() == 's390x'
    if args.native_wheel_only:
        assert native and not torch.cuda.is_available()
        assert args.local_repro is None
        assert torch.version.git_version == 'aaa1b73d62015d15004355001092982aa1bf53bf'
        REPORT.mkdir(parents=True, exist_ok=True)
        os.chdir(ROOT / 'test')
        sys.path.insert(0, str(ROOT / 'test'))
        result32 = original_test('full-native-wheel-original-float32', 'float32')
        assert result32 == 0, result32
        torch.manual_seed(1234)
        double_input = torch.randn(1, 1, 3, 4, dtype=torch.float64, requires_grad=True)
        double_grid = (torch.rand(1, 2, 2, 2, dtype=torch.float64) * 1.6 - 0.8).requires_grad_()
        double_gradcheck = torch.autograd.gradcheck(
            lambda a, b: torch.nn.functional.grid_sample(
                a, b, mode='bicubic', padding_mode='zeros', align_corners=False),
            (double_input, double_grid))
        assert double_gradcheck
        result = {
            'mode': 'full-native-wheel', 'machine': platform.machine(),
            'torch_sha': torch.version.git_version,
            'source_sha256': hashlib.sha256(
                (ROOT / 'aten/src/ATen/native/cpu/GridSamplerKernel.cpp').read_bytes()
            ).hexdigest(),
            'original_float32_exit': result32, 'float64_gradcheck': double_gradcheck,
            'configuration': torch.__config__.show(),
        }
        (REPORT / 'full-native-wheel-results.json').write_text(json.dumps(result, indent=2))
        print('FULL_NATIVE_S390_WHEEL_VERIFICATION', json.dumps(result), flush=True)
        return
    assert torch.version.git_version == '914c15e40b4d5818fb0e85fe7b94d77b5f00c332'
    if native:
        assert not torch.cuda.is_available() and args.local_repro is None
        os.environ['CXX'] = '/opt/rh/gcc-toolset-14/root/usr/bin/c++'
    os.environ['MAX_JOBS'] = '2'
    REPORT.mkdir(parents=True, exist_ok=True)
    DEBUG.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT / 'test')
    sys.path.insert(0, str(ROOT / 'test'))
    configuration = torch.__config__.show()
    match = re.search(r'\bCXX_FLAGS=(.*?)(?=, [A-Z][A-Z0-9_]*=|\n|$)', configuration)
    assert match
    flags = shlex.split(match.group(1)) + ['-O3', '-std=c++20']
    openmp = 'USE_OPENMP=ON' in configuration
    if openmp:
        flags.append('-fopenmp')
    flags += (['-march=z15', '-mvx', '-mzvector', '-DCPU_CAPABILITY=ZVECTOR', '-DCPU_CAPABILITY_ZVECTOR']
              if native else ['-mavx2', '-mfma', '-mf16c', '-DCPU_CAPABILITY=AVX2', '-DCPU_CAPABILITY_AVX2'])
    captures = []
    original_equal = TestCase.assertEqual

    def capture(self, *values, **kwargs):
        try:
            return original_equal(self, *values, **kwargs)
        except AssertionError:
            frame = inspect.currentframe().f_back
            while frame:
                local = frame.f_locals
                if frame.f_code.co_name == '_test' and 'noncontig_result_vjps' in local:
                    names = ['primals', 'cotangents', 'noncontig_primals', 'noncontig_cotangents']
                    data = {name: local[name] for name in names}
                    data['sample_args'] = local['sample'].args
                    captures.append(data)
                    break
                frame = frame.f_back
            raise

    TestCase.assertEqual = capture
    try:
        baseline_exit = original_test('wheel-original', 'float32')
    finally:
        TestCase.assertEqual = original_equal
    if native:
        assert baseline_exit == 1 and captures, (baseline_exit, captures)
        data = captures[-1]
    else:
        assert args.local_repro, 'Local verification requires the captured native tensors'
        data = torch.load(args.local_repro, weights_only=False)
    torch.save(data, DEBUG / 'layout-repro.pt')
    inp, grid = data['primals']
    ni, ng = data['noncontig_primals']
    cotangent, nc = data['cotangents'], data['noncontig_cotangents']
    assert data['sample_args'][1:] == (2, 0, False)
    layouts = list(itertools.product([inp, ni], [grid, ng], [cotangent, nc]))

    def gradients():
        return [torch.ops.aten.grid_sampler_2d_backward(cv, iv, gv, 2, 0, False, [True, True])
                for iv, gv, cv in layouts]

    wheel = gradients()
    original = (ROOT / '.ci/pytorch/triton_pin_s390_original.cpp').read_text()
    assert hashlib.sha256(original.encode()).hexdigest() == '15ce2cd6f0e5eb3d3ba0aaefc42f11ec0d50da3fa419b3b705f478d280c91b58'
    install_kernel('original', original, flags, openmp)
    rebuilt = gradients()
    assert all(torch.equal(a, b) for left, right in zip(wheel, rebuilt) for a, b in zip(left, right)), 'Unchanged native kernel differs from the wheel'
    fixed = (ROOT / 'aten/src/ATen/native/cpu/GridSamplerKernel.cpp').read_text()
    install_kernel('fixed', fixed, flags, openmp)
    outputs = gradients()
    layout_difference = [max((a[i] - outputs[0][i]).abs().max().item() for a in outputs) for i in (0, 1)]
    assert layout_difference[1] == 0.0, layout_difference
    result32 = original_test('fixed-original-float32', 'float32')
    assert result32 == 0, result32
    torch.manual_seed(1234)
    double_input = torch.randn(1, 1, 3, 4, dtype=torch.float64, requires_grad=True)
    double_grid = (torch.rand(1, 2, 2, 2, dtype=torch.float64) * 1.6 - 0.8).requires_grad_()
    double_gradcheck = torch.autograd.gradcheck(
        lambda a, b: torch.nn.functional.grid_sample(
            a, b, mode='bicubic', padding_mode='zeros', align_corners=False),
        (double_input, double_grid))
    assert double_gradcheck
    result = {'machine': platform.machine(), 'torch_sha': torch.version.git_version,
              'source_sha256': hashlib.sha256(fixed.encode()).hexdigest(),
              'baseline_exit': baseline_exit, 'unchanged_kernel_matches_wheel': True,
              'layout_difference': layout_difference,
              'fixed_original_float32_exit': result32, 'float64_gradcheck': double_gradcheck,
              'compiler_flags': flags}
    (REPORT / 'results.json').write_text(json.dumps(result, indent=2))
    print('S390_ORIGINAL_KERNEL_FIX_VERIFICATION', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
