# Owner(s): ["module: inductor"]

import tempfile
import zipfile
from pathlib import Path

import torch
from torch._inductor.package import load_package
from torch.testing._internal.common_utils import run_tests, TestCase


MARKER = b"AOTI_ZIPSLIP_CANARY\n"
_TRAVERSAL = r"path traversal \('\.\.'\)"


def _prefixed_escape(canary: Path, *, backslash: bool) -> str:
    # Stay under the model-directory prefix so the old string check selects
    # the member, then climb out of the loader temp directory.
    climb = ("..\\" * 20) if backslash else ("../" * 20)
    tail = canary.relative_to("/").as_posix()
    return f"archive/data/aotinductor/model/{climb}{tail}"


class TestAOTIPackageZipSlip(TestCase):
    def _canary_root(self) -> tempfile.TemporaryDirectory:
        return tempfile.TemporaryDirectory(prefix="aoti_zipslip_")

    def _write_pkg(self, root: Path, members: dict[str, bytes]) -> Path:
        pkg = root / "pkg.pt2"
        with zipfile.ZipFile(pkg, "w") as zf:
            for name, payload in members.items():
                zf.writestr(name, payload)
        return pkg

    def _assert_canary_absent(self, canary: Path) -> None:
        self.assertFalse(canary.exists(), f"extracted outside temp dir: {canary}")

    def test_in_tree_member_reaches_native_load(self) -> None:
        with self._canary_root() as root_name:
            root = Path(root_name)
            canary = root / "canary.txt"
            pkg = self._write_pkg(
                root,
                {
                    "archive/data/aotinductor/model/model.so": b"\x7fELFFAKE",
                    "archive/data/aotinductor/model/model_metadata.json": (
                        b'{"AOTI_DEVICE_KEY": "cpu"}\n'
                    ),
                },
            )
            with self.assertRaisesRegex(RuntimeError, "file too short"):
                torch._C._aoti.AOTIModelPackageLoader(str(pkg), "model")
            self._assert_canary_absent(canary)

    def test_model_member_dotdot_is_not_extracted(self) -> None:
        self._assert_traversal_rejected(backslash=False)

    def test_model_member_backslash_dotdot_is_not_extracted(self) -> None:
        self._assert_traversal_rejected(backslash=True)

    def _assert_traversal_rejected(self, *, backslash: bool) -> None:
        with self._canary_root() as root_name:
            root = Path(root_name)
            canary = root / "canary.txt"
            escape = _prefixed_escape(canary, backslash=backslash)
            pkg = self._write_pkg(
                root,
                {
                    escape: MARKER,
                    "archive/data/aotinductor/model/model.so": b"\x7fELFFAKE",
                },
            )
            with self.assertRaisesRegex(RuntimeError, _TRAVERSAL):
                torch._C._aoti.AOTIModelPackageLoader(str(pkg), "model")
            self._assert_canary_absent(canary)
            with self.assertRaisesRegex(RuntimeError, _TRAVERSAL):
                load_package(str(pkg), "model")
            self._assert_canary_absent(canary)

    def test_metadata_member_dotdot_is_not_extracted(self) -> None:
        with self._canary_root() as root_name:
            root = Path(root_name)
            canary = root / "wrapper_metadata.json"
            escape = _prefixed_escape(canary, backslash=False)
            pkg = self._write_pkg(
                root,
                {
                    escape: b'{"k":"v"}\n',
                    "archive/data/aotinductor/model/model.so": b"\x7fELFFAKE",
                },
            )
            with self.assertRaisesRegex(RuntimeError, _TRAVERSAL):
                torch._C._aoti.AOTIModelPackageLoader.load_metadata_from_package(
                    str(pkg), "model"
                )
            self._assert_canary_absent(canary)


if __name__ == "__main__":
    run_tests()
