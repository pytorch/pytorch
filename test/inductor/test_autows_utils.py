# Owner(s): ["module: inductor"]
import os
import subprocess
import sys

from torch._inductor.test_case import run_tests, TestCase


def _use_meta_ws(env_val: str | None) -> tuple[bool, bool]:
    # USE_META_WS is evaluated at import, so vary the env in a fresh interpreter.
    env = os.environ.copy()
    if env_val is None:
        env.pop("TRITON_USE_META_WS", None)
    else:
        env["TRITON_USE_META_WS"] = env_val
    code = (
        "import torch._inductor.heuristics.template.triton as m;"
        "from torch._inductor.autows_utils import has_meta_ws;"
        "print(int(m.USE_META_WS), int(has_meta_ws()))"
    )
    out = subprocess.check_output([sys.executable, "-c", code], env=env)
    use_meta_ws, available = (bool(int(x)) for x in out.split())
    return use_meta_ws, available


class AutoWSUtilsTests(TestCase):
    def test_use_meta_ws_env_gating(self) -> None:
        for env_val, enabled in ((None, False), ("0", False), ("1", True)):
            with self.subTest(env=env_val):
                use_meta_ws, available = _use_meta_ws(env_val)
                self.assertEqual(use_meta_ws, available and enabled)


if __name__ == "__main__":
    run_tests()
