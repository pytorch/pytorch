"""
TEMPORARY BENCHMARK SCRIPT FOR UNARY MATH OPERATORS
"""

import torch
import time
import subprocess

devices = [(torch.device(type='cpu'), 1000)]

cpu = subprocess.check_output("lscpu | grep 'Model name'", shell=True).decode().split(':', 1)[1].strip()
print('CPU:', cpu)

if torch.has_cuda:
    gpu = [line.split(':', 1)[1].strip() for line in subprocess.check_output(
        "((lspci | grep VGA) || echo 'VGA: none')",
        shell=True).decode().splitlines()]
    gpu_str = '; '.join(set(gpu))
    print('GPU[x%s]: %s' % (len(gpu), gpu_str))
    devices.append((torch.device(type='cuda'), 100000))

print('torch: version=%s, has_mkl=%s, has_mkldnn=%s' % (torch.__version__, torch.has_mkl, torch.has_mkldnn))

ops = [
    ('abs', ()),
    ('neg', ()),
    ('reciprocal', ()),
    ('frac', ()),
    ('digamma', ()),
    ('lgamma', ()),
    ('erfinv', ()),
    # ('fill', ()),
    ('clone', ()),
    # ('contiguous', ()),
    ('clamp', (0.2, 0.8)),
    ('sign', ()),
    ('nonzero', ()),
    ('acos', ()),
    ('asin', ()),
    ('atan', ()),
    ('ceil', ()),
    ('cos', ()),
    ('cosh', ()),
    ('erf', ()),
    ('erfc', ()),
    ('exp', ()),
    ('expm1', ()),
    ('floor', ()),
    ('log', ()),
    ('log10', ()),
    ('log1p', ()),
    ('log2', ()),
    ('round', ()),
    ('rsqrt', ()),
    ('sigmoid', ()),
    ('sign', ()),
    ('sin', ()),
    ('sinh', ()),
    ('sqrt', ()),
    ('tan', ()),
    ('tanh', ()),
    ('trunc', ()),
]
ops.sort()

# atan2, div, fmod, lerp, add, mul, mvlgamma, pow, reminder


def cut(line, width=150):
    if len(line) > width:
        return line[:width - 20] + ' ... ' + line[-(20):]
    return line


for device, NITER in devices:
    x = torch.rand(1024, 1024, device=device)

    data = []
    print('Benchmarking: ', end='')
    exp_elapsed_sec = None
    for op, args in ops:
        print(op, end=' ', flush=True)
        s = time.time()
        for i in range(NITER):
            getattr(x, op)(*args)
        elapsed_sec = ((time.time() - s) / NITER)
        data.append((elapsed_sec, op))
        if op == 'exp':
            exp_elapsed_sec = elapsed_sec
    data.sort(reverse=True)
    print('[DONE]')

    print('device=%s, shape=%s, NITER=%s' % (device, x.shape, NITER))

    row1 = []
    row2 = []
    row3 = []
    row4 = []
    for elapsed_sec, op in data:
        row1.append('{:^9}'.format(op))
        row2.append('{:9.3f}'.format(elapsed_sec * 1000))
        row3.append('{:9.3f}'.format(1024 ** 2 / elapsed_sec / 1e9))
        row4.append('{:9.1f}'.format(elapsed_sec / exp_elapsed_sec))


    print('--------------------:', cut('-' * 300))
    print('operation           :', cut('|'.join(row1)))
    print('--------------------:', cut('-' * 300))
    print('time per iter (ms)  :', cut('|'.join(row2)))
    print('gops/s              :', cut('|'.join(row3)))
    print('slowness rel. to exp:', cut('|'.join(row4)))
    print('--------------------:', cut('-' * 300))
