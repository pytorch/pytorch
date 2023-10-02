import torch


def create_blocked_tensor(B, M, N, blocksize, sparsity, dtype, device):
    assert sparsity <= 1.0 and sparsity >= 0.0, "sparsity should be a value between 0 and 1"
    assert M % blocksize[0] == 0
    assert N % blocksize[1] == 0
    shape = (B, M // blocksize[0], N // blocksize[1])[int(B == 0):]
    A = torch.bernoulli(torch.full(shape, 1 - sparsity, dtype=dtype, device=device))
    A = torch.repeat_interleave(A, blocksize[0], dim=-2)
    A = torch.repeat_interleave(A, blocksize[1], dim=-1)
    return A


def _test_worker(test_func):
    import triton

    ms, ms_min, ms_max = triton.testing.do_bench(test_func, warmup=500, rep=100, fast_flush=False)

    tflops = 2 * m * k * n * 1e-12 / (ms * 1e-3)
    return ms, tflops


def test_dense_dense_mm(x, y):

    def test_func(x=x.to_dense(), y=y):
        return torch.matmul(x, y)

    return _test_worker(test_func)


def test_torch_matmul(x, y):

    def test_func(x=x, y=y):
        return torch.matmul(x, y)

    return _test_worker(test_func)


def test_bsr_dense_mm(x, y):
    from torch.sparse._triton_ops import bsr_dense_mm

    def test_func(x=x, y=y):
        return bsr_dense_mm(x, y)

    return _test_worker(test_func)


def test_bsr_scatter_mm2(x, y):
    from torch.sparse._triton_ops import bsr_scatter_mm, bsr_scatter_mm_indices_data

    indices_data = dict()
    indices_data.update(tasks2=bsr_scatter_mm_indices_data(x, y)['tasks2'])

    def test_func(x=x, y=y):
        return bsr_scatter_mm(x, y, indices_data=indices_data)

    return _test_worker(test_func)


def test_bsr_scatter_mm5(x, y, SPLIT_N=1):
    from torch.sparse._triton_ops import bsr_scatter_mm, bsr_scatter_mm_indices_data

    indices_data = dict()
    indices_data.update(tasks5=bsr_scatter_mm_indices_data(x, y, SPLIT_N=SPLIT_N)['tasks5'])

    def test_func(x=x, y=y):
        return bsr_scatter_mm(x, y, indices_data=indices_data)

    return _test_worker(test_func)

test_bsr_scatter_mm5.variants = [dict(SPLIT_N=1), dict(SPLIT_N=2), dict(SPLIT_N=4), dict(SPLIT_N=8),
                                 dict(SPLIT_N=16), dict(SPLIT_N=32), dict(SPLIT_N=64), dict(SPLIT_N=128)]

if __name__ == "__main__":
    import argparse
    import sys
    from torch.testing import make_tensor
    import triton
    torch.manual_seed(0)

    parser = argparse.ArgumentParser(description="SpTritonOps")

    parser.add_argument("--ops", default="dense_dense_mm,bsr_dense_mm,bsr_scatter_mm2,bsr_scatter_mm5", type=str)
    parser.add_argument("--b", default="0", type=int)
    parser.add_argument("--m", default="1024", type=int)
    parser.add_argument("--k", default="1024", type=int)
    parser.add_argument("--n", default="1024", type=int)
    parser.add_argument("--bm", default="16", type=int)
    parser.add_argument("--bk", default="16", type=int)
    parser.add_argument("--sparsity", "--sparsity", default="0.5", type=float)
    parser.add_argument("--dtype", default="float16", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--repeat", default="1", type=int)
    parser.add_argument("--outfile", default="stdout", type=str)


    args = parser.parse_args()

    if args.outfile == "stdout":
        outfile = sys.stdout
    elif args.outfile == "stderr":
        outfile = sys.stderr
    else:
        outfile = open(args.outfile, "a")

    ops = args.ops.split(',')

    b = args.b
    m = args.m
    n = args.n
    k = args.k
    blocksize = (args.bm, args.bk)
    sparsity = args.sparsity
    dtype = getattr(torch, args.dtype)
    device = args.device

    x = create_blocked_tensor(b, m, k, blocksize, sparsity, dtype, device).to_sparse_bsr(blocksize)
    y = make_tensor(k, n, dtype=dtype, device=device)

    for op in ops:
        test_func = globals()['test_' + op]
        variants = getattr(test_func, 'variants', [dict()])
        for variant in variants:
            variant_str = ','.join(f'{k}={v}' for k, v in variant.items())
            time_ms_lst = []
            performance_tflops_lst = []
            for r in range(args.repeat):
                try:
                    time_ms, performance_tflops = test_func(x, y, **variant)
                except triton.compiler.OutOfResources as msg:
                    print(f'op={op}[{variant_str}]({bsr_size},{k}x{n}) dtype={args.dtype} {sparsity=}(nnz={x._nnz()})'
                          f' blocksize={args.bm}x{args.bk}'
                          f' OutOfResources', file=outfile)
                    continue
                except Exception as msg:
                    print(f'op={op}[{variant_str}]({bsr_size},{k}x{n}) dtype={args.dtype} {sparsity=}(nnz={x._nnz()})'
                          f' blocksize={args.bm}x{args.bk}'
                          f' {msg}', file=outfile)
                    continue
                time_ms_lst.append(time_ms)
                performance_tflops_lst.append(performance_tflops)
                bsr_size = f'{b}x{m}x{k}' if b > 0 else f'{k}x{n}'

                print(f'op={op}[{variant_str}]({bsr_size},{k}x{n}) dtype={args.dtype} {sparsity=}(nnz={x._nnz()})'
                      f' blocksize={args.bm}x{args.bk}'
                      f' time={time_ms:.3f} ms performance={performance_tflops:.3f} TFLOPS', file=outfile)

            if args.repeat > 1:
                avg_time_ms = sum(time_ms_lst) / len(time_ms_lst)
                avg_performance_tflops = sum(performance_tflops_lst) / len(performance_tflops_lst)
                print(f'op={op}[{variant_str}]({bsr_size},{k}x{n}) dtype={args.dtype} {sparsity=}(nnz={x._nnz()})'
                      f' blocksize={args.bm}x{args.bk}'
                      f' time={time_ms:.3f} ms performance={performance_tflops:.3f} TFLOPS [AVERAGE]', file=outfile)
