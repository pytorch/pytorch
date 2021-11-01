# Tensor Expression: python API
#
# This tutorial explains the python API of TE that includes: 1) TE IR, 2) TE
# loop transformations, and 3) TE codegen.

import torch
import torch._C._te as te
from numpy import median
import timeit

LLVM_ENABLED = torch._C._llvm_enabled()

# 1. End-to-End example: addition of two tensors

# We first create two BufHandles in TE which correspond to the two input
# tensors.

# The data (physical memory) of a tensor is represented as a buffer
# in TE, which has dimensions, a data type and a name. BufHandle is a pointer
# to a buffer that defines load/store operations on it. Here, A points to a
# buffer with dimentions [32], 'float' data type and name 'A'; B points to a
# buffer with dimentions [32], 'float' data type and name 'B'.
A = te.BufHandle("A", [32], torch.float32)
B = te.BufHandle("B", [32], torch.float32)

# We then define the addition operation of two BufHandles: 'add'. It takes the
# axis argument 'i' and returns the value of the ouput buffer on 'i', i.e.,
# A[i] + B[i]. A[i], B[i] correspond to 'A.load([i])', 'B.load([i])' in TE
# respectively.
def add(i):
    return A.load([i]) + B.load([i])

# Next, we define the output tensor using TE 'Compute' - it specifies the name,
# dimensions of the output tensor, and how to compute its values, i.e., the
# 'add' operation we constructed above.
C = te.Compute("C", [32], add)

# At this point, we have constructed the IR stmt that computes the tensor C.
# Use the following code to print its stmt.
print("The stmt to compute tensor C:")
print(C.stmt())

# The stmt to compute tensor C:
# for (int v = 0; v < 32; v++) {
#   C[v] = (A[v]) + (B[v]);
# }

# Lastly, we construct a loopnest for C to prepare for codegen. Namely, we
# transform the stmt to 1) expand reduce operations, and 2) flatten buffer
# dimemsions. In this example, there are no reduces or multi-dimentional
# buffers, thus the stmt remaining the same.
loopnest = te.LoopNest([C])
loopnest.prepare_for_codegen()

# Print out the transformed stmt.
print("The stmt before codegen:")
print(loopnest.root_stmt())

# The stmt before codegen:
# {
#   for (int v = 0; v < 32; v++) {
#     C[v] = (A[v]) + (B[v]);
#   }
# }

# Optional: simplify the stmt.
# TODO: Should we add 'simplify' to 'prepare_for_codegen'? This seems to be a
# common practise we do before codegen.
stmt = te.simplify(loopnest.root_stmt())

# Now we can get to codegen. Here, we use the IR Evaluator as the backend to
# lower the stmt to assembly code. Besides the stmt, we also need to specify
# the inputs and output data for codegen.
cg = te.construct_codegen("ir_eval", stmt, [A, B, C])

# We have generated native code in TE to add two tensors. Let's try it out!
# Prepare the input/output tensors, and call TE kernel. The result is saved in
# the output tensor `tC`.
tA = torch.randn(32, dtype=torch.float32)
tB = torch.randn(32, dtype=torch.float32)
tC = torch.empty(32, dtype=torch.float32)
cg.call_raw([tA.data_ptr(), tB.data_ptr(), tC.data_ptr()])

# We can check the correctness of the results as follows.
torch.testing.assert_close(tA + tB, tC)

# Hoory, now you have learned how to create and run TE kernel!


# 2. Construct your own TE IR

# TE constants
c = te.ExprHandle.int(5)
f = te.ExprHandle.float(5.2)
print("\nTE constants:")
print(c, f)

# TE integer variables
m = te.VarHandle("m", torch.int32)
n = te.VarHandle("n", torch.int32)
print("\nTE integer variables:")
print(m, n)

# TE arithmeic expression
# no needs to convert 5 to a TE constant before the addition; it will
# be converted to the same type of m when performing the addition operation
mc = m + 5
mn = m + n
e = te.sin(mc * mn)
print("\nTE arithmetic expression:")
print(e)

# TE buffer variables
A = te.BufHandle("A", [te.ExprHandle.int(32), te.ExprHandle.int(32)], torch.float32)
B = te.BufHandle("B", [te.ExprHandle.int(32), te.ExprHandle.int(32)], torch.float32)
print("\nTE buffer variables:")
print(A, B)

# TE load expression
i = te.VarHandle("i", torch.int32)
j = te.VarHandle("j", torch.int32)
print("\nTE load expression:")
print(A.load([i, j]))

# TE store statement
C = te.BufHandle("C", [te.ExprHandle.int(32), te.ExprHandle.int(32)], torch.float32)
store = C.store([i, j], A.load([i, j]) + B.load([i, j]))
print("\nTE store statement:")
print(store)

# TE block statement
block = te.Block([store])
print("\nTE block statement:")
print(block)

# TE conditional statement
cond = te.Cond.make(i > j, block, None)
print("\nTE cond statement:")
print(cond)

# TE for statement
forst = te.For.make(j, te.ExprHandle.int(0), te.ExprHandle.int(32), block)
forst = te.For.make(i, te.ExprHandle.int(0), te.ExprHandle.int(32), forst)
print("\nTE for statement:")
print(forst)

# See more TE IRs: torch/csrc/jit/tensorexpr/IRSpecification.md

# 3. Use TE loop transformations to optimize the computation
loopnest = te.LoopNest(te.Block([forst]), [C])
print("\nTE loopnest:")
print(loopnest.root_stmt())

# obtain the loop handles
# 1) LoopNest::root_stmt() returns the Block stmt we constructed above; 2)
# Block::stmts() returns the list of stmts in the block, i.e., [`forst`]; 3) here we
# use `forst_i` to point to the stmt `forst`.
forst_i = loopnest.root_stmt().stmts()[0]
print("\nTE loopnest for-i stmt:")
print(forst_i)
# LoopNest::get_loop_at(root, indicies): returns the For stmt indexed by
# 'indices' in the 'root' For stmt.  Here 'indices' indicates the path to the
# returned loop from 'root' in AST. `forst_j` is directly embeded in the body
# of `forst_i` as the only stmt, i.e., the path to `forst_j` from `forst_i`
# is [0]. We use `get_loop_at` to obtain the handle of `forst_j`.
forst_j = loopnest.get_loop_at(forst_i, [0])
print("\nTE loopnest for-j stmt:")
print(forst_j)

# tile(x, y, x_factor, y_factor): it takes a 2d domain (x, y) and splits the
# domain into small rectangular blocks each with shape (x_factor, y_factor).
# The traversal over the domain turns into an outer iteration over the blocks
# and an inner traversal over all points in the block.
loopnest.tile(forst_i, forst_j, 8, 8)
loopnest.simplify()
print("\nTE loopnest after tiling:")
print(loopnest.root_stmt())

# vectorize_inner_loops(): find the inner-most loops and vectorize them.
# Note: this only works for the LLVM backend currently, when no reductions are
# involved.
loopnest.vectorize_inner_loops()
loopnest.simplify()
print("\nTE loopnest after vectorization:")
print(loopnest.root_stmt())

# TODO: add more loop transformations

# prepare for codegen
loopnest.prepare_for_codegen()
stmt = loopnest.root_stmt()
print("\nTE loopnest after codegen preparation:")
print(loopnest.root_stmt())

# 4. LLVM backend in TE codegen
if LLVM_ENABLED:
    cg = te.construct_codegen("llvm", stmt, [A, B, C])

    # print assembly code
    asm = cg.get_code_text("asm")
    print("\nassembly code:")
    print(asm)

    # Prepare the input/output tensors, and call TE kernel. The result is saved in
    # the output tensor `tC`.
    tA = torch.randn(32, 32, dtype=torch.float32)
    tB = torch.randn(32, 32, dtype=torch.float32)
    tC = torch.empty(32, 32, dtype=torch.float32)
    cg.call_raw([tA.data_ptr(), tB.data_ptr(), tC.data_ptr()])

    # Check correctness
    torch.testing.assert_close(tA + tB, tC)

    # Measure performance
    repeat, times = 1000, 50
    time_te = median(timeit.repeat(lambda: cg.call_raw([tA.data_ptr(), tB.data_ptr(), tC.data_ptr()]), number=times, repeat=repeat))
    time_eager = median(timeit.repeat(lambda: tA + tB, number=times, repeat=repeat))
    speedup = time_eager / time_te
    print(f"eager: {time_eager*1000:5.3f}us, tensorexpr: {time_te*1000:5.3f}us, speedup: {speedup:4.2f}")
