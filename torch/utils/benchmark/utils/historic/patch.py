"""Utility to patch benchmark utils into earlier versions of PyTorch.

In order to reduce conflicts files are copied to a mirrored location. e.g.
    `from torch.utils_backport.benchmark import Timer`
rather than:
    `from torch.utils.benchmark import Timer`

Caveats:
    1) C++ only works on PyTorch 1.0 and above. The API changes around the major
       version bump are too significant to easily bridge. Python was tested back
       to PyTorch 0.4.
    2) Imports are rewritten to use the modified path.


NOTE:
    This is brittle. The files transfered and particular rewrites were based on
    manual inspection + trial and error, and this file is not covered by CI.
    However, to borrow a wry observation from a teammate: "<The nice thing
    about back testing is that if it breaks you can always fix it later.>"
"""
import functools
import os
import re
import shutil
from typing import Iterable, Sequence, Tuple


TORCH_ROOT = functools.reduce(
    lambda s, _: os.path.split(s)[0], range(5), os.path.abspath(__file__))

BACKPORT_NAME = "utils_backport"
PATTERNS = (
    # Point imports at the new namespace.
    ("torch.utils", f"torch.{BACKPORT_NAME}"),
    ("torch._appdirs", f"torch.{BACKPORT_NAME}._appdirs"),

    # Attributes which are not in old versions.
    ("torch.has_cuda", "getattr(torch, 'has_cuda', False)"),
    ("torch.version.hip", "None"),
    ('getattr\(torch._C, f"_PYBIND11_{name}"\)',
     'getattr(torch._C, f"_PYBIND11_{name}", None)'),
    ('getattr\(torch._C, f"_PYBIND11_{pname}"\)',
     'getattr(torch._C, f"_PYBIND11_{pname}", None)'),
    ("torch._C._GLIBCXX_USE_CXX11_ABI",
     'getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", 0)'),
    ('"TBB" in torch.__config__.parallel_info\(\)', 'False'),

    # There has been a some reorganization of libtorch, so we have to make sure
    # the right shared objects get linked in.
    ("extra_ldflags.append\('-lc10'\)",
     "extra_ldflags.extend(['-lc10'] + (['-lcaffe2'] if os.path.exists(os.path.join(TORCH_LIB_PATH, 'libcaffe2.so')) else []))"),
    ("extra_ldflags.append\('-ltorch_cpu'\)",
     "extra_ldflags.extend(['-ltorch_cpu'] if os.path.exists(os.path.join(TORCH_LIB_PATH, 'libtorch_cpu.so')) else [])"),

    # Silence warnings. (This doesn't come with a warranty anyway...)
    ("warnings.warn(ABI_INCOMPATIBILITY_WARNING.format(compiler))",
     "return True  # Silence warning when back testing."),
    ("WRONG_COMPILER_WARNING.format", "''.format"),
)


def clean_backport(destination_install: str) -> None:
    destination_root = os.path.join(destination_install, BACKPORT_NAME)
    if os.path.exists(destination_root):
        shutil.rmtree(destination_root)


def backport(destination_install: str) -> None:
    assert os.path.split(TORCH_ROOT)[1] == "torch"
    if os.path.split(destination_install)[1] != "torch":
        raise ValueError(
            f"`{destination_install}` does not appear to be the root of a "
            "PyTorch installation.")

    destination_root = os.path.join(destination_install, BACKPORT_NAME)
    clean_backport(destination_install)
    os.makedirs(destination_root)

    source_root = os.path.join(TORCH_ROOT, "utils")
    copy_targets: Sequence[Tuple[str, Iterable[str], Iterable[str]]] = (
        list(os.walk(os.path.join(source_root, "benchmark"))) +
        [(source_root, [], ["cpp_extension.py", "_cpp_extension_versioner.py", "file_baton.py"])]
    )

    # _appdirs isn't part of `torch.utils`, and we don't want to trample the
    # "actual" one so we move it to torch.utils_backport.
    shutil.copy(
        os.path.join(TORCH_ROOT, "_appdirs.py"),
        os.path.join(destination_root, "_appdirs.py")
    )

    # We don't really care about HIP, it just has to be copied for imports.
    shutil.copytree(
        os.path.join(source_root, "hipify"),
        os.path.join(destination_root, "hipify")
    )

    for d, _, files in copy_targets:
        if os.path.split(d)[1] == "__pycache__":
            continue

        for fname in files:
            src = os.path.join(d, fname)
            with open(src, "rt") as f:
                contents = f.read()

            for old, new in PATTERNS:
                contents = re.sub(old, new, contents)

            dest = re.sub(f"^{source_root}", destination_root, src)
            os.makedirs(os.path.split(dest)[0], exist_ok=True)
            with open(dest, "wt") as f:
                f.write(contents)
