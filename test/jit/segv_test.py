import torch

class kernel_arena_scope(object):
    def __enter__(self):
        self.scope = torch._C._te.KernelScope()

    def __exit__(self, typ, val, traceback):
        self.scope = None

with kernel_arena_scope():
    f32 = torch._C._te.Dtype.Float
    i32 = torch._C._te.Dtype.Int

    X, Y, M, N = 64, 72, 56, 56
    Xte, Yte, Mte, Nte = [torch._C._te.ExprHandle.int(x) for x in [X, Y, M, N]]
    A = torch._C._te.Placeholder('A', f32, [Xte, Yte, Mte, Nte])

    dim_args = [torch._C._te.DimArg(*args) for args in [(Xte, 'n'), (Yte, 'c'), (Mte, 'h'), (Nte, 'w')]]

    def compute(x, y, m, n):
        return A.load([x, y, m, n])
    B = torch._C._te.Compute('B', dim_args, compute)
    loopnest = torch._C._te.LoopNest([B])
    loopnest.simplify()

    n, c, h, w = loopnest.get_loops_for(B)
    xtail = loopnest.tile(c, h, 16, 16)
    print("loopnest:", loopnest)
    ho = loopnest.get_loop_at(c, [0])
    print("ho:", ho)
    htail = loopnest.get_loop_at(c, [1])
    print("htail:", htail)
