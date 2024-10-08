import torch
torch.set_default_device('cuda')
import functools

def make_dynamic(func):
    graph = None
    initialized = False

    # What does functools.wraps do?
    @functools.wraps(func)
    def wrapper(*args):
        # TODO: Change to using pytrees
        assert all(isinstance(arg, torch.Tensor) for arg in args), "All arguments must be tensors"

        nonlocal graph, initialized

        if not initialized:
            # First time: create and capture graphs with all allocations as nullptr
            g1 = torch.cuda.CUDAGraph()
            g1.enable_debug_mode() # because we need the graph_t to stick around, not just the graphexec_t
            with torch.cuda.graph(g1, sentinel_allocations_mode=1):
                out1_static = func(*[torch.empty_like(arg) for arg in args])
            # Second time, do it again where all allocations are allocIdx+1
            g2 = torch.cuda.CUDAGraph()
            g2.enable_debug_mode()
            with torch.cuda.graph(g2, sentinel_allocations_mode=2):
                out2_static = func(*[torch.empty_like(arg) for arg in args])
            print("lets just make sure torch still works:", torch.arange(10).sum())
            # g1.debug_dump("g1_leif.dot")
            # g2.debug_dump("g2_leif.dot")
            # assert False
            g1.compare_with_recapture(g2) # Investigate the graphs and determine which allocation is which
            graph = g1
            initialized = True

        # assert False

        # Replay the graph with the actual arguments
        graph.replay_dynamic(list(args))

    return wrapper


def myFunc(a, out):
    torch.mul(a, 3, out=out)
    # out = a * 3

myFuncWrapped = make_dynamic(myFunc)

for i in range(10):
    a = torch.ones(8)
    out = torch.empty_like(a)
    myFunc(a, out)

    out_wrapped = out.clone()
    myFuncWrapped(a, out_wrapped)
    print("out actually is", out_wrapped)
    print(out == out_wrapped)
    assert torch.all(out == out_wrapped)
