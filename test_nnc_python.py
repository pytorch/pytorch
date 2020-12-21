import torch

class kernel_arena_scope(object):
    def __enter__(self):
        self.scope = torch._C.KernelScope()

    def __exit__(self, typ, val, traceback):
        self.scope = None

with kernel_arena_scope():
    dtype = torch._C.Dtype.Float
    dims = [torch._C.ExprHandle.int(i) for i in [64, 32]]

    A = torch._C.Placeholder('A', dtype, dims)
    B = torch._C.Placeholder('B', dtype, dims)

    dim_args = [torch._C.DimArg(*args) for args in [(dims[0], 'i'), (dims[1], 'j')]]
    def compute(i, j):
        return A.load([i, j]) + B.load([i, j]) + i + j
    X = torch._C.Compute('X', dim_args, compute)
    def compute(i, j):
        return X.load([i, j]) * X.load([i, j])
    Y = torch._C.Compute('Y', dim_args, compute)

    loopnest = torch._C.LoopNest([X, Y])
    print(loopnest)

    loopnest.compute_inline(loopnest.get_loop_body_for(X))
    stmt = torch._C.simplify_ir(loopnest.root_stmt())
    print(stmt)

    loops_x = loopnest.get_loops_for(X)
    (o, i, t) = loopnest.split_with_tail(loops_x[0], 11)
    stmt = torch._C.simplify_ir(loopnest.root_stmt())
    print(stmt)

    loopnest.vectorize_inner_loops()
    stmt = torch._C.simplify_ir(loopnest.root_stmt())
    print(stmt)

    tA = torch.ones(64, 32) * 5
    tB = torch.ones(64, 32) * 6
    tX = torch.empty(64, 32)
    tY = torch.empty(64, 32)

    torch._C.execute_llvm(stmt, [tA, tB, tX, tY], [torch._C.BufferArg(x) for x in [A, B, X, Y]])
    torch._C.execute_ireval(stmt, [tA, tB, tX, tY], [torch._C.BufferArg(x) for x in [A, B, X, Y]])

    print(tX)
    print(tY)

print('fin')
