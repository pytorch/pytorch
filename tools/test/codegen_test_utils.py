from __future__ import annotations

import glob
import os
import tempfile
from pathlib import Path

from torchgen.gen import gen_source_files, get_grouped_native_functions, ParsedYaml
from torchgen.model import DispatchKey, NativeFunctionsGroup
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import FileManager


TEMPLATES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "aten/src/ATen/templates",
)


def gen_register_dispatch_key(
    parsed: ParsedYaml,
    dispatch_key: DispatchKey,
    *,
    per_operator_headers: bool,
    native_aot_manifests: dict | None = None,
) -> str:
    """Every Register<dispatch_key> shard gen_source_files writes, concatenated."""
    grouped = get_grouped_native_functions(parsed.native_functions)
    with tempfile.TemporaryDirectory() as d:
        fm = FileManager(install_dir=d, template_dir=TEMPLATES, dry_run=False)
        gen_source_files(
            native_functions=parsed.native_functions,
            grouped_native_functions=grouped,
            structured_native_functions=[
                g for g in grouped if isinstance(g, NativeFunctionsGroup)
            ],
            view_groups=[],
            selector=SelectiveBuilder.get_nop_selector(),
            static_dispatch_idx=[],
            backend_indices=parsed.backend_indices,
            aoti_fm=fm,
            core_fm=fm,
            cpu_vec_fm=fm,
            cpu_fm=fm,
            device_fms={"cuda": fm},
            dispatch_keys=[dispatch_key],
            functions_keys={dispatch_key},
            rocm=False,
            force_schema_registration=False,
            per_operator_headers=per_operator_headers,
            skip_dispatcher_op_registration=False,
            update_aoti_c_shim=False,
            aoti_backends=set(),
            extend_aoti_c_shim=False,
            native_aot_manifests=native_aot_manifests,
        )
        shards = sorted(glob.glob(os.path.join(d, f"Register{dispatch_key}_*.cpp")))
        if not shards:
            raise AssertionError(
                f"no Register{dispatch_key} shards in {sorted(os.listdir(d))}"
            )
        return "\n".join(Path(s).read_text() for s in shards)
