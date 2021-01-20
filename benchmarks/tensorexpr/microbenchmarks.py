import torch
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

class kernel_arena_scope(object):
    def __enter__(self):
        self.scope = torch._C.te.KernelScope()

    def __exit__(self, typ, val, traceback):
        self.scope = None

unary_ops = [
    ("sin", torch.sin),
    ("cos", torch.cos),
    ("tan", torch.tan),
    ("asin", torch.asin),
    ("acos", torch.acos),
    ("atan", torch.atan),
    ("sinh", torch.sinh),
    ("cosh", torch.cosh),
    ("tanh", torch.tanh),
    ("sigmoid", torch.sigmoid),
    ("exp", torch.exp),
    ("expm1", torch.expm1),
    ("expm1", torch.expm1),
    ("abs", torch.abs),
    ("log", torch.log),
    ("fast_log", torch.log),
    ("log2", torch.log2),
    ("log10", torch.log10),
    ("log1p", torch.log1p),
    ("erf", torch.erf),
    ("erfc", torch.erfc),
    ("sqrt", torch.sqrt),
    ("rsqrt", torch.rsqrt),
    ("ceil", torch.ceil),
    ("floor", torch.floor),
    ("round", torch.round),
    ("trunc", torch.trunc),
    ("lgamma", torch.lgamma),
    # ("frac", torch.frac), # seems unimplemented
    # ("isnan", torch.isnan), # no out variant
]

def gen_unary_nnc_fun(nnc_name):
    def nnc_fun(A, B):
        def compute(i, j):
            return getattr(A.load([i, j]), nnc_name)()
        return compute
    return nnc_fun

def gen_unary_torch_fun(torch_op):
    def torch_fun(a, b, out):
        def fun():
            torch_op(a, out=out)
        return fun
    return torch_fun


def gen_binary_nnc_fun(fn):
    def nnc_fun(A, B):
        def compute(i, j):
            return fn(A.load([i, j]), B.load([i, j]))
        return compute
    return nnc_fun

def gen_binary_torch_fun(fn):
    def pt_fun(a, b, out):
        def fun():
            return fn(a, b, out=out)
        return fun
    return pt_fun

binary_ops = [
    ('add', (lambda a, b: a + b), torch.add),
    ('mul', (lambda a, b: a * b), torch.mul),
    ('sub', (lambda a, b: a - b), torch.sub),
    ('div', (lambda a, b: a / b), torch.div),
    ('eq', (lambda a, b: a == b), torch.eq),
    ('gt', (lambda a, b: a > b), torch.gt),
    ('lt', (lambda a, b: a < b), torch.lt),
    ('gte', (lambda a, b: a >= b), torch.greater_equal),
    ('lte', (lambda a, b: a <= b), torch.less_equal),
    # ('neq', (lambda a, b: a != b), None)), # no one-op equivalent
    # ('&', (lambda a, b: a & b), torch.bitwise_and), # requires more work to test
]


def nnc_relu(i, j):
    return torch._C.te.ifThenElse(A.load([i, j]) < torch._C.te.ExprHandle.float(0), torch._C.te.ExprHandle.float(0), A.load([i, j]))

custom_ops = [
    ('relu', nnc_relu, lambda a, b, c: c.copy_(a).relu_()),
    # ('nnc_mul_relu', nnc_mul_relu, pt_mul_relu)
    # ('manual_sigmoid', nnc_manual_sigmoid, lambda a, b, c: torch.sigmoid(a, out=c))
]

def gen_custom_nnc_fun(fn):
    def nnc_fun(A, B):
        return fn
    return nnc_fun

def gen_custom_torch_fun(fn):
    def pt_fun(a, b, out):
        def fun():
            return fn(a, b, out)
        return fun
    return pt_fun

names = []
nnc_fns = []
pt_fns = []

for nnc_name, pt_op in unary_ops:
    names.append(nnc_name)
    nnc_fns.append(gen_unary_nnc_fun(nnc_name))
    pt_fns.append(gen_unary_torch_fun(pt_op))

for name, lmbda, pt_fn in binary_ops:
    names.append(name)
    nnc_fns.append(gen_binary_nnc_fun(lmbda))
    pt_fns.append(gen_binary_torch_fun(pt_fn))

for name, lmbda, pt_fn in custom_ops:
    names.append(name)
    nnc_fns.append(gen_custom_nnc_fun(lmbda))
    pt_fns.append(gen_custom_torch_fun(pt_fn))

benchmarks = list(zip(names, nnc_fns, pt_fns))

def run_benchmarks(benchmarks):
    df = pd.DataFrame(columns=['name', 'N', 'M', 'nnc_time', 'torch_time', 'ratio'])
    sizes = [1, 4, 16, 64, 256, 1024]
    with torch.no_grad():
        for name, nnc_fun, torch_fun in benchmarks:
            for N in sizes:
                for M in sizes:
                    iters = int(1e5 / (N + M))
                    with kernel_arena_scope():
                        dtype = torch._C.te.Dtype.Float

                        dM = torch._C.te.ExprHandle.int(M)
                        dN = torch._C.te.ExprHandle.int(N)

                        dims_MN = [dM, dN]
                        A = torch._C.te.Placeholder('A', dtype, [dM, dN])
                        B = torch._C.te.Placeholder('B', dtype, [dM, dN])


                        dim_args = [torch._C.te.DimArg(*args) for args in [(dM, 'm'), (dN, 'n')]]

                        compute = nnc_fun(A, B)
                        X = torch._C.te.Compute('X', dim_args, compute)
                        loopnest = torch._C.te.LoopNest([X])
                        loopnest.prepare_for_codegen()
                        stmt = torch._C.te.simplify(loopnest.root_stmt())
                        cg = torch._C.te.construct_codegen('llvm', stmt, [torch._C.te.BufferArg(x) for x in [A, B, X]])

                        tA = torch.rand(M, N).clamp(0.01, 0.99)
                        tB = torch.rand(M, N).clamp(0.01, 0.99)
                        tX = torch.empty(M, N)
                        tR = torch.empty(M, N)


                        # warmup
                        for _ in range(10):
                            cg.call([tA, tB, tX])
                        start = time.time()
                        for it in range(iters):
                            cg.call([tA, tB, tX])
                        time1 = time.time() - start


                        fn = torch_fun(tA, tB, tR)
                        # warmup
                        for _ in range(10):
                            fn()
                        start = time.time()
                        for it in range(iters):
                            fn()
                        time2 = time.time() - start


                        df = df.append({'name': name, 'N': N, 'M': M, 'nnc_time': time1, 'torch_time': time2, 'ratio': time2 / time1}, ignore_index=True)
                        print(name, N, M)
                        print(time2 / time1, time1, time2)
                        print()

                        def check_correctness(a, b):
                            diff = abs(a.sum() - b.sum())
                            if not (diff < 2.0):
                                if a.bool().sum() == b.bool().sum():
                                    return
                                print(name, diff)
                                assert(diff < 2.0)
                        check_correctness(tX, tR)
    return df

def dump_plot(df):
    keys = []
    vals = []
    indexed = df[df['N'] == df['M']]
    for index, row in indexed.iterrows():
        keys.append(row['name'])
        vals.append(row['ratio'])


    keys = keys[::len(sizes)]
    sns.set(rc={'figure.figsize' : (5.0, 20.0)})

    cmap = sns.diverging_palette(10, 120, n=9, as_cmap=True)
    np_vals = np.array([vals]).reshape(-1, len(sizes))
    g = sns.heatmap(np_vals, annot=True, cmap=cmap, center=1.0, yticklabels=True)
    plt.yticks(rotation=0)
    plt.title('PyTorch performance divided by NNC performance (single core)')
    plt.xlabel('Size of NxN matrix')
    plt.ylabel('Operation')
    g.set_yticklabels(keys)
    g.set_xticklabels([1, 16, 64, 256, 1024])

    plt.savefig('nnc.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs NNC microbenchmarks')
    parser.add_argument('--multi_threaded', action='store_true', help='Run with more than one thread')
    args = parser.parse_args()
    if not args.multi_threaded:
        torch.set_num_threads(1)

    df = run_benchmarks(benchmarks)
    dump_plot(df)
