import torch

class kernel_arena_scope(object):
    def __enter__(self):
        self.orig_arena = torch._C._te_enter_new_kernel_scope()

    def __exit__(self, typ, val, traceback):
        torch._C._te_exit_kernel_scope(self.orig_arena)


with kernel_arena_scope():
    dtype = torch._C.Dtype.Float
    dims = [torch._C.ExprHandle.int(i) for i in [64, 32]]

    A = torch._C.Placeholder('A', dtype, dims)
    B = torch._C.Placeholder('B', dtype, dims)

    dim_args = [torch._C.DimArg(*args) for args in [(dims[0], 'i'), (dims[1], 'j')]]
    def compute(i, j):
        return A.load([i, j]) + B.load([i, j])
    X = torch._C.Compute('X', dim_args, compute)

    loopnest = torch._C.LoopNest([X])
    print(loopnest)
    loopnest.vectorize_inner_loops()
    print(loopnest)
    stmt = torch._C.simplify_ir(loopnest.root_stmt())
    print(loopnest)

    tA = torch.ones(64, 32) * 5
    tB = torch.ones(64, 32) * 6
    tX = torch.empty(64, 32)

    torch._C.execute_llvm(stmt, [tA, tB, tX], [torch._C.BufferArg(x) for x in [A, B, X]])

    print(tX)
