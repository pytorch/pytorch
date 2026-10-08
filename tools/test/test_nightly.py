import contextlib
import importlib.util
import io
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "nightly.py"
SPEC = importlib.util.spec_from_file_location("nightly", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise AssertionError("unable to load tools/nightly.py")
nightly = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nightly)


class TestRocmSourceSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.sources = {
            name: nightly.PipSource(
                name=name,
                index_url=f"https://download.pytorch.org/whl/nightly/{name}",
                supported_platforms={"Linux"},
                accelerator="rocm",
            )
            for name in ("rocm-10.0", "rocm-10.1", "rocm-preview")
        }

    def test_explicit_preview(self) -> None:
        source = nightly._select_accelerator_source(
            "ROCm", "preview", platform="Linux", sources=self.sources
        )
        self.assertEqual(source.name, "rocm-preview")

    def test_default_excludes_named_channels(self) -> None:
        source = nightly._select_accelerator_source(
            "ROCm", None, platform="Linux", sources=self.sources
        )
        self.assertEqual(source.name, "rocm-10.1")

    def test_invalid_request_lists_versions_and_channels(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"Available version\(s\): 10\.0, 10\.1, preview",
        ):
            nightly._select_accelerator_source(
                "ROCm", "missing", platform="Linux", sources=self.sources
            )

    def test_help_describes_preview_channel(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit):
            nightly.make_parser().parse_args(["checkout", "--help"])
        help_text = output.getvalue()
        self.assertIn("ROCm version or named channel", help_text)
        self.assertIn("preview (defaults to the latest numeric version", help_text)


if __name__ == "__main__":
    unittest.main()
