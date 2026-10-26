# Owner(s): ["module: ci"]
from pathlib import Path
from runpy import run_path

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


ok_changed_file = run_path(
    str(
        Path(__file__).resolve().parents[1]
        / ".github/actions/reuse-old-whl/reuse_old_whl.py"
    )
)["ok_changed_file"]


@instantiate_parametrized_tests
class TestReuseOldWhl(TestCase):
    @parametrize(
        "path,expected",
        [
            ("torch/_native/ops/topk/aot.py", False),
            ("torch/_vendor/quack/rmsnorm.py", False),
            ("torch/nn/modules/linear.py", True),
        ],
    )
    def test_python_sources(self, path: str, expected: bool) -> None:
        self.assertEqual(ok_changed_file(path), expected)


if __name__ == "__main__":
    run_tests()
