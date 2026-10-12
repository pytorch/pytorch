# Throwaway probe (not for landing): localize the rx7900 failure of
# test_autograd.py::TestAutogradDeviceTypeCUDA::test_graph_save_on_cpu_cuda
# ("number of specified values in sparse COO tensors does not match: 1 != 2").
#
# Every iteration uses different data, so a copy that never lands cannot be
# hidden by a reused allocator block that still holds the previous values.
import os
import sys

import torch


def d(t):
    return f"idx={t._indices().tolist()} val={t._values().tolist()} coal={t.is_coalesced()} nnz={t._nnz()}"


def h2d(n):
    # non_blocking H2D from pageable memory, read by a kernel before any sync.
    # kernel_bad only: the copy landed after the kernel ran. late_bad: it never landed.
    kernel_bad = late_bad = printed = 0
    for it in range(n):
        want = ([[it + 1, it + 1]], [it + 1.0, it + 1.0])
        gi = torch.tensor(want[0]).to("cuda", non_blocking=True)
        gv = torch.tensor(want[1]).to("cuda", non_blocking=True)
        ki, kv = gi + 0, gv + 0
        seen = (ki.tolist(), kv.tolist())
        late = (gi.tolist(), gv.tolist())
        kernel_bad += seen != want
        late_bad += late != want
        if seen != want and printed < 5:
            printed += 1
            print(f"FAIL h2d it={it} want={want} kernel_saw={seen} after_sync={late}", flush=True)
    return {"kernel_bad": kernel_bad, "late_bad": late_bad}


def save_on_cpu(n, sync):
    fails = {True: 0, False: 0}
    printed = 0
    for it in range(n):
        k, v = 1 + it % 3, it + 1.0
        for pin in (True, False):
            x = torch.sparse_coo_tensor(
                torch.tensor([[k, k]], device="cuda"),
                torch.tensor([v, v], device="cuda"),
                requires_grad=True,
            )
            hooks = torch.autograd.graph.save_on_cpu(pin)
            if sync:
                unpack = hooks.unpack_hook

                def unpack_sync(packed, unpack=unpack):
                    t = unpack(packed)
                    torch.cuda.synchronize()
                    return t

                hooks.unpack_hook = unpack_sync
            branch_grads = []
            with hooks:
                y = x * x
                y.grad_fn.register_hook(lambda gi, go: branch_grads.extend(d(g) for g in gi if g is not None))
                saved = d(y.grad_fn._saved_self)
                y.to_dense().sum().backward()
            raw = d(x.grad)
            g = x.grad.coalesce()
            if g._indices().tolist() != [[k]] or g._values().tolist() != [4 * v]:
                fails[pin] += 1
                if printed < 5:
                    printed += 1
                    print(f"FAIL it={it} pin={pin} want idx=[[{k}]] val=[{4 * v}]\n  grad {raw}\n  saved_self {saved}")
                    print(f"  mul_backward_grad_inputs {branch_grads}", flush=True)
    return {"pin_true": fails[True], "pin_false": fails[False]}


if __name__ == "__main__":
    mode, n = sys.argv[1], int(sys.argv[2])
    names = ("AMD_SERIALIZE_COPY", "AMD_SERIALIZE_KERNEL", "HSA_CO_DMACOPY_SIZE")
    env = {k: os.environ[k] for k in names if k in os.environ}
    props = torch.cuda.get_device_properties(0)
    if mode == "h2d":
        result = h2d(n)
    else:
        result = save_on_cpu(n, sync=mode == "save_on_cpu_sync")
    print(f"PROBE mode={mode} n={n} env={env} gpu={props.name!r} uuid={props.uuid} result={result}", flush=True)
