# Owner(s): ["module: inductor"]
"""torchTLX template-selection tests.

These assert the contract between this repo and FBTriton: torchTLX templates
live in FBTriton (triton.language.extra.tlx.inductor), and Inductor proposes
them through torch/_inductor/heuristics/template/tlx.py. A change on either
side can break the pairing, and most of the failure modes are silent -- the
registry import is wrapped in `except ImportError: pass`, and an unselectable
template merely logs and falls back.

Run with an FBTriton install:

    python tools/torchtlx/bringup.py switch fbtriton --from-source <checkout>
    python tools/torchtlx/bringup.py test

Everything here skips when the active Triton has no TLX Inductor registry, so
the file is inert on upstream Triton.
"""

import importlib
import io
import logging
import unittest

import torch
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import parametrize
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


TLX_REGISTRY = "triton.language.extra.tlx.inductor.registry"


def _has_tlx() -> bool:
    try:
        importlib.import_module(TLX_REGISTRY)
    except Exception:
        return False
    return True


HAS_TLX = _has_tlx()
IS_ROCM = torch.version.hip is not None


@unittest.skipUnless(HAS_GPU, "requires GPU")
@unittest.skipUnless(HAS_TLX, f"requires an FBTriton providing {TLX_REGISTRY}")
class TestTorchTLXTemplates(TestCase):
    def _compile_capture(self, fn, *args):
        """Compile fn, returning (result, records from the heuristic registry)."""
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        log = logging.getLogger("torch._inductor.heuristics.registry")
        log.addHandler(handler)
        prev = log.level
        log.setLevel(logging.DEBUG)
        try:
            torch._dynamo.reset()
            out = torch.compile(fn, fullgraph=True)(*args)
            torch.cuda.synchronize()
        finally:
            log.removeHandler(handler)
            log.setLevel(prev)
        return out, buf.getvalue()

    @parametrize("op", ["mm", "addmm", "bmm"])
    @config.patch({"triton.tlx_mode": "allow"})
    def test_no_off_arch_template_proposed(self, op):
        """Every proposed TLX template must have a heuristic on this device.

        registry.py gates registration by arch, so proposing the other arch's
        template yields "No template heuristic found ... Using fallback": the
        result stays correct, but the template can never be selected and the
        error fires on every compile. Regression test for P2462423082, where
        plain mm proposed the Blackwell WS template on gfx950.
        """
        if op == "mm":
            a = torch.randn(256, 256, device=GPU_TYPE, dtype=torch.float16)
            b = torch.randn(256, 256, device=GPU_TYPE, dtype=torch.float16)
            fn, args = (lambda x, y: x @ y), (a, b)
        elif op == "addmm":
            bias = torch.randn(256, device=GPU_TYPE, dtype=torch.float16)
            a = torch.randn(256, 256, device=GPU_TYPE, dtype=torch.float16)
            b = torch.randn(256, 256, device=GPU_TYPE, dtype=torch.float16)
            fn, args = torch.addmm, (bias, a, b)
        else:
            a = torch.randn(4, 128, 128, device=GPU_TYPE, dtype=torch.float16)
            b = torch.randn(4, 128, 128, device=GPU_TYPE, dtype=torch.float16)
            fn, args = torch.bmm, (a, b)

        out, log_text = self._compile_capture(fn, *args)
        self.assertEqual(out, fn(*args), atol=1e-2, rtol=1e-2)

        misses = [
            line.split("template_name=")[1].split(",")[0]
            for line in log_text.splitlines()
            if "No template heuristic found" in line and "template_name=" in line
        ]
        self.assertEqual(
            misses,
            [],
            f"{op}: TLX proposed template(s) with no heuristic on this device: "
            f"{sorted(set(misses))}. Proposal in FBTriton's append_tlx must match "
            "the register= conditions in its registry.py.",
        )

    def test_amd_templates_still_proposed(self):
        """The arch gate must silence the wrong templates, not all of them."""
        if not IS_ROCM:
            self.skipTest("AMD template coverage")
        from torch._inductor.kernel.bmm import bmm_template
        from torch._inductor.kernel.mm import mm_template
        from triton.language.extra.tlx.inductor.mm_templates import append_tlx

        def uids(templates):
            return {getattr(t, "uid", None) for t in templates}

        addmm = uids(append_tlx([mm_template], "addmm"))
        self.assertTrue(
            any(u and "tlx_amd_addmm" in u for u in addmm),
            f"AMD addmm warp-pipe template no longer proposed: {sorted(addmm)}",
        )
        bmm = uids(append_tlx([bmm_template], "bmm"))
        self.assertTrue(
            any(u and "tlx_amd_bmm" in u for u in bmm),
            f"AMD bmm warp-pipe template no longer proposed: {sorted(bmm)}",
        )
        # No AMD TLX template exists for plain mm; the Blackwell one must not
        # stand in for it.
        mm = uids(append_tlx([mm_template], "mm"))
        self.assertNotIn("tlx_blackwell_gemm_ws", mm)

    def test_registry_is_importable_and_registers_templates(self):
        """The pairing itself: tlx.py swallows ImportError, so assert it here."""
        from torch._inductor.template_heuristics.registry import (
            _TEMPLATE_HEURISTIC_REGISTRY,
        )

        importlib.import_module(TLX_REGISTRY)
        tlx_uids = {
            key[0]
            for key in _TEMPLATE_HEURISTIC_REGISTRY
            if isinstance(key, tuple) and key and str(key[0]).startswith("triton::tlx")
        }
        self.assertTrue(
            tlx_uids, "importing the TLX registry registered no TLX heuristics"
        )


from torch.testing._internal.common_utils import instantiate_parametrized_tests


instantiate_parametrized_tests(TestTorchTLXTemplates)


if __name__ == "__main__":
    run_tests()
