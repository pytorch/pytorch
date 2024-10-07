import torch
torch.set_default_device('cuda')
import functools

def make_dynamic(func):
    graph = None
    initialized = False

    @functools.wraps(func)
    def wrapper(*args):
        assert all(isinstance(arg, torch.Tensor) for arg in args), "All arguments must be tensors"

        nonlocal graph, initialized

        if not initialized:
            # First time: create and capture graphs with all allocations as nullptr
            g1 = torch.cuda.CUDAGraph()
            g1.enable_debug_mode() # because we need the graph_t to stick around, not just the graphexec_t
            with torch.cuda.graph(g1, sentinel_allocations_mode=1):
                func(*[torch.empty_like(arg) for arg in args])
            # Second time, do it again where all allocations are allocIdx+1
            g2 = torch.cuda.CUDAGraph()
            g2.enable_debug_mode()
            with torch.cuda.graph(g2, sentinel_allocations_mode=2):
                func(*[torch.empty_like(arg) for arg in args])
            print("lets just make sure torch still works:", torch.arange(10).sum())
            g1.compare_with_recapture(g2) # Investigate the graphs and determine which allocation is which
            graph = g1
            initialized = True

        # Replay the graph with the actual arguments
        graph.replay_dynamic(list(args))

    return wrapper


def myFunc(a, b, c):
    print("myFunc is running") # Can't actually print a,b,c here, because when it's run in capture mode, their data_ptrs are nullptr
    a += b.sum() * c
    temp = torch.ones_like(c) # we can even allocate :)
    temp += 1
    a += temp
    #a[:] = c

myFuncWrapped = make_dynamic(myFunc)

for i in range(10):
    a = torch.ones(8)
    b = torch.full((3,), i)
    c = torch.arange(8)
    print("in", a, b, c)

    a1 = a.clone()
    myFunc(a1, b, c)
    print("out should be", a1)

    a2 = a.clone()
    myFuncWrapped(a2, b, c)
    print("out actually is", a2)
    print(a1 == a2)