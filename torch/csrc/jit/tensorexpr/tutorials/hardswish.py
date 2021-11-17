import torch
import numpy as np
from numpy import median
import timeit

def hardswish():
    def my_hardswish(x):
        return x*(x+3)/6

    device, size = 'cpu', (4, 4)
    x = torch.rand(size, dtype=torch.float, device=device)
    result = my_hardswish(x)

    #torch._C._jit_override_can_fuse_on_cpu(True)
    jit_hardswish = torch.jit.script(my_hardswish, (x))

    for i in range(2):
        jit_hardswish(x)

    graph=torch.jit.last_executed_optimized_graph()

    # print graph IR of my_hardswish
    print("Graph IR:\n", graph)

    # create NNC kernel for my_hardswish
    node =  graph.findNode("prim::TensorExprGroup", True)
    fusion_graph = node.g('Subgraph')
    kernel = torch._C._te.TensorExprKernel(fusion_graph)

    # print NNC IR of my_hardswish
    stmt = kernel.get_codegen_stmt()
    print("\nNNC IR:\n", stmt)

    # print lower level IR (LLVM IR) of my_hardswish
    code = kernel.get_code_text()
    print("\nLLVM code:\n", code)

    # check correctness
    _, compiler_result = kernel.run((x, x))
    np.testing.assert_allclose(compiler_result.cpu().numpy(), result.cpu().numpy(), atol=2e-3)

    # measure performance
    repeat, times = 1000, 50
    time_ori = median(timeit.repeat(lambda: my_hardswish(x), number=times, repeat=repeat))
    time_opt = median(timeit.repeat(lambda: kernel.run((x, x)), number=times, repeat=repeat))
    speedup = time_ori / time_opt
    print(f"original: {time_ori*1000:5.3f}us, compiler: {time_opt*1000:5.3f}us, speedup: {speedup:4.2f}")

hardswish()
