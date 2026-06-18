# Owner(s): ["module: inductor"]
import torch
import torch.utils._pytree as pytree
from torch._inductor import compile_to_python as inductor_compile_to_python
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.utils import stateless
from torch.testing._internal.common_utils import run_tests, TestCase


def _capture(m, x):
    """Capture m(x) as a flat-input ATen graph (params+buffers then x lifted to
    inputs), mirroring how torch.precompile feeds standalone graphs to a backend.

    NOTE: tracing runs m(x) once, which mutates m's buffers for stateful modules;
    callers should capture from a throwaway model and run on a fresh one.
    """
    pnames = [n for n, _ in m.named_parameters()]
    bnames = [n for n, _ in m.named_buffers()]
    pb = [p for _, p in m.named_parameters()] + [b for _, b in m.named_buffers()]
    k = len(pnames)

    def flat_fn(flat):
        params = dict(zip(pnames, flat[:k]))
        buffers = dict(zip(bnames, flat[k : k + len(bnames)]))
        with stateless._reparametrize_module(
            m, {**params, **buffers}, tie_weights=True
        ):
            out = m(flat[-1])
        leaves, _ = pytree.tree_flatten(out)
        return leaves

    with torch.enable_grad():
        gm = make_fx(flat_fn)(pb + [x])
    return gm


def _flat_inputs(m, x):
    return (
        [p for _, p in m.named_parameters()] + [b for _, b in m.named_buffers()] + [x]
    )


def _exec(src):
    ns = {"__name__": "_compiled"}
    exec(compile(src, "<compiled>", "exec"), ns)
    return ns["call"]


class TestInductorCompileToPython(TestCase):
    # torch._inductor.compile_to_python returns the INNER call only (no epilogue);
    # for a dense graph that is the whole computation, run under no_grad.
    def test_inner_call_dense_matches_eager(self):
        torch.manual_seed(0)
        m = torch.nn.Sequential(
            torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)
        ).eval()
        x = torch.randn(5, 4)
        gm = _capture(m, x)

        inner_src, cache = inductor_compile_to_python(gm, _flat_inputs(m, x))
        self.assertIsInstance(inner_src, str)
        self.assertIsNotNone(cache)  # non-mutating graph is serializable

        call = _exec(inner_src)
        with torch.no_grad():
            out = call(_flat_inputs(m, x))
        self.assertEqual(out[0], m(x))


if __name__ == "__main__":
    run_tests()
