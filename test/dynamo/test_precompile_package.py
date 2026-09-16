# Owner(s): ["module: dynamo"]


import torch._inductor.test_case
from torch.compiler._precompile_types import PrecompileSummary


class TestPrecompilePackage(torch._inductor.test_case.TestCase):
    def test_summary_is_incomplete_without_a_backend_graph(self):
        def summary(**kw):
            base = dict(
                frames=1,
                resume_functions=0,
                guarded_codes=1,
                backend_graphs=1,
                bypassed=(),
            )
            base.update(kw)
            return PrecompileSummary(**base)

        self.assertTrue(summary().complete)
        self.assertFalse(summary(backend_graphs=0).complete)
        self.assertFalse(summary(guarded_codes=0).complete)
        self.assertFalse(summary(capture_errors=("boom",)).complete)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
