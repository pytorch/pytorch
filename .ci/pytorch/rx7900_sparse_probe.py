# Throwaway probe (not for landing): localize the rx7900 failure of
# test_autograd.py::TestAutogradDeviceTypeCUDA::test_graph_save_on_cpu_cuda
# ("number of specified values in sparse COO tensors does not match: 1 != 2").
import sys

import torch


def d(t):
    return f"idx={t._indices().tolist()} val={t._values().tolist()} coal={t.is_coalesced()} nnz={t._nnz()}"


def autograd_case(n, use_save_on_cpu):
    fails = {True: 0, False: 0}
    printed = 0
    for it in range(n):
        for pin in (True, False):
            x = torch.sparse_coo_tensor(
                torch.tensor([[1, 1]], device="cuda").long(),
                torch.tensor([1.0, 1.0], device="cuda"),
                requires_grad=True,
            )
            branch_grads = []
            ctx = torch.autograd.graph.save_on_cpu(pin) if use_save_on_cpu else torch.enable_grad()
            with ctx:
                y = x * x
                y.grad_fn.register_hook(lambda gi, go: branch_grads.extend(d(g) for g in gi if g is not None))
                saved = d(y.grad_fn._saved_self)
                y.to_dense().sum().backward()
            g = x.grad
            raw = d(g)
            e, a = g.coalesce(), (2 * x).coalesce()
            if e._nnz() != a._nnz() or not torch.equal(e._indices(), a._indices()):
                fails[pin] += 1
                if printed < 10:
                    printed += 1
                    print(f"FAIL it={it} pin={pin}\n  grad {raw}\n  coalesced {d(e)}\n  saved_self {saved}\n  mul_backward_grad_inputs {branch_grads}", flush=True)
    return fails


def kernel_checks(n):
    cat_bad = h2d_bad = coal_bad = 0
    i = torch.tensor([[1, 1]], dtype=torch.long)
    for _ in range(n):
        ig = i.to("cuda", non_blocking=True)
        if ig.tolist() != [[1, 1]]:
            h2d_bad += 1
        c = torch.cat([ig, ig], 1)
        if c.tolist() != [[1, 1, 1, 1]]:
            cat_bad += 1
        s = torch.sparse_coo_tensor(c, torch.ones(4, device="cuda"), (2,)).coalesce()
        if s._nnz() != 1 or s._indices().tolist() != [[1]]:
            coal_bad += 1
    return {"h2d_pageable_nonblocking": h2d_bad, "cat": cat_bad, "coalesce": coal_bad}


if __name__ == "__main__":
    n = int(sys.argv[1])
    print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).multi_processor_count, flush=True)
    print("save_on_cpu fails by pin_memory:", autograd_case(n, True), flush=True)
    print("no save_on_cpu fails:", autograd_case(n, False), flush=True)
    print("standalone kernel mismatches:", kernel_checks(n), flush=True)
