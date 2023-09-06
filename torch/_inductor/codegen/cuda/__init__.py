import logging
import os
import sys
import tempfile

from typing import List

import torch
from ...config import cuda as inductor_cuda_config

log = logging.getLogger(__name__)

HAS_CUTLASS = True

_CUTLASS_PY_FULL_PATH = os.path.join(
    inductor_cuda_config.cutlass_dir, "tools/library/scripts"
)
_TMP_CUTLASS_PY_FULL_PATH = os.path.abspath(
    os.path.join(tempfile.gettempdir(), "torch_cutlass_script")
)


def _rename_cutlass_import(content: str, cutlass_modules: List[str]) -> str:
    for cutlass_module in cutlass_modules:
        content = content.replace(
            f"from {cutlass_module} import ", f"from cutlass_{cutlass_module} import "
        )
    return content


def _gen_cutlass_file(file_name: str, cutlass_modules: List[str]) -> None:
    orig_full_path = os.path.abspath(os.path.join(_CUTLASS_PY_FULL_PATH, file_name))
    text = ""
    with open(orig_full_path) as f:
        text = f.read()
    text = _rename_cutlass_import(text, cutlass_modules)
    dst_full_path = os.path.abspath(
        os.path.join(
            _TMP_CUTLASS_PY_FULL_PATH,
            f"cutlass_{file_name}" if file_name != "__init__.py" else file_name,
        )
    )
    with open(dst_full_path, "w") as f:
        f.write(text)


# Copy CUTLASS python scripts to a temp dir and add the temp dir to Python search path.
# This is a temporary hack to avoid CUTLASS module naming conflicts.
# TODO(ipiszy): remove this hack when CUTLASS solves Python scripts packaging structure issues.
try:
    if os.path.isdir(_CUTLASS_PY_FULL_PATH):
        cutlass_file_names = [
            file_name
            for file_name in os.listdir(_CUTLASS_PY_FULL_PATH)
            if file_name.endswith(".py")
        ]
        cutlass_module_names = [file_name[:-3] for file_name in cutlass_file_names]
        if not os.path.isdir(_TMP_CUTLASS_PY_FULL_PATH):
            os.mkdir(_TMP_CUTLASS_PY_FULL_PATH)
        for file_name in cutlass_file_names:
            _gen_cutlass_file(file_name, cutlass_module_names)
        sys.path.append(_TMP_CUTLASS_PY_FULL_PATH)

        import cutlass_generator  # type: ignore[import]
        import cutlass_library  # type: ignore[import]
        import cutlass_manifest  # type: ignore[import]

except ImportError as e:
    HAS_CUTLASS = False
    log.warning(
        "Failed to import CUTLASS packages: %s, ignoring the CUTLASS backend.", str(e)
    )
