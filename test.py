import torch
torch.set_default_device('cuda')

def make_dynamic(func):
    graph = None
    def wrapper(*args):
        nonlocal graph
        if graph is None:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, dynamic_graph=True):
                func(*args)
            graph.become_dynamic(args)
        graph.replay_dynamic(args)
    return wrapper

@make_dynamic
def myFuncSimple(a):
    print("myFuncSimple is running")
    a += 1

test = torch.zeros(8)

for i in range(10):
    myFuncSimple(test)
assert torch.equal(test, torch.full((8,), 10))

for i in range(10):
    test = test.clone() # force the tensor to a new memory location
    myFuncSimple(test)
assert torch.equal(test, torch.full((8,), 20))


def myFunc(a, b, c):
    print("myFunc is running")

    # check that simple kernels work:
    a += b.sum() * c
    temp = torch.ones_like(c)
    for i in range(100):
        # demonstrate that it can handle weird allocator stuff
        # (these lines will reuse memory a bunch)
        temp = temp.clone() + 1 / temp.clone() + torch.ones_like(c)
    temp += 1
    a += temp
    a[3:] += 2
    a[4] += 1000
    a[5] += b[1]
    a[6] = b[1]
    a[7:8] = 123

    # check that GPU_LAMBDA works:
    c = torch.clamp(c, min=2, max=5)
    a += c

    # check that cudaMemcpyAsync device-to-device works
    torch.sort(a, out=(a, torch.empty_like(a).to(dtype=torch.int64)))
    a[4] += a[5].clone().clone().clone() # this line does three cudaMemcpyAsyncs

    # torch very rarely uses cudaMemsetAsync
    # to trigger that case, we need to call top-k and get it to use its multiblock kernel
    # this is an easy way to do that:
    temp = torch.empty(100000, device=a.device)
    temp[:] = a[0] - 1000
    a[0] = torch.topk(temp, 1).values

myFuncWrapped = make_dynamic(myFunc)

ensure_ptrs_change = []
for i in range(10):
    a = torch.ones(8)
    b = torch.full((3,), i)
    c = torch.arange(8)
    print("inputs:", a, b, c, a.data_ptr(), b.data_ptr(), c.data_ptr())
    ensure_ptrs_change.append((a, b, c)) # don't let them get GC'd and possibly reused

    a1 = a.clone()
    myFunc(a1, b, c)
    print("out should be", a1)

    a2 = a.clone()
    myFuncWrapped(a2, b, c)
    print("out actually is", a2)
    assert torch.equal(a1, a2)
