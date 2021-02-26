import torch

class kernel_arena_scope(object):
    def __enter__(self):
        self.scope = torch._C._te.KernelScope()

    def __exit__(self, typ, val, traceback):
        self.scope = None

with kernel_arena_scope():
    def get_dim_args(dims):
        dim_args = []
        for dim in dims:
            dim_args.append(torch._C._te.DimArg(dim, 'i' + str(len(dim_args))))
        return dim_args

    dtype = torch._C._te.Dtype.Float

    ZERO = torch._C._te.ExprHandle.int(0)
    ONE = torch._C._te.ExprHandle.int(1)
    FOUR = torch._C._te.ExprHandle.int(4)
    A = torch._C._te.BufHandle('A', [ONE, FOUR], dtype)
    B = torch._C._te.BufHandle('B', [FOUR, ONE], dtype)
    C = torch._C._te.BufHandle('C', [ONE, ONE], dtype)

    s = torch._C._te.ExternalCall(C, "nnc_aten_matmul", [A, B], [])

    loopnest = torch._C._te.LoopNest(s, [C])
    print('Original loopnest stmt:\n', loopnest)
    loopnest.prepare_for_codegen()
    print('Stmt after prepare_for_codegen:\n', loopnest)
    codegen = torch._C._te.construct_codegen('llvm', s, [torch._C._te.BufferArg(x) for x in [A, B, C]])

    a_buf = torch.ones(1, 4)
    b_buf = torch.ones(4, 1)
    c_buf = torch.empty(1, 1)
    codegen.call([a_buf, b_buf, c_buf])
    print(c_buf)
