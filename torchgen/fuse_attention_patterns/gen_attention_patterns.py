#!/usr/bin/env python3
import os
import shutil
from collections import defaultdict
from pathlib import Path

import torch._inductor

from torch._inductor.fx_passes.fuse_attention import _get_sfdp_patterns
from torch._inductor.pattern_matcher import (
    _TargetExpr,
    gen_pattern,
    PatternExpr,
    PatternPrettyPrinter,
)

auto_generated_msg = """# This is an auto-generated file. Please do not modify it by hand.
# To re-generate, run:
# cd ~/pytorch && python
# torchgen/fuse_attention_patterns/gen_attention_patterns.py
"""


def get_file_template() -> str:
    file_template = f"""# noqa: F401, E501
{auto_generated_msg}
import torch
import torch._inductor

aten = torch.ops.aten
prims = torch.ops.prims

"""
    pattern_matcher_imports = []
    for name in dir(torch._inductor.pattern_matcher):
        attr = getattr(torch._inductor.pattern_matcher, name)
        if isinstance(attr, type) and issubclass(attr, (PatternExpr, _TargetExpr)):
            pattern_matcher_imports.append(name)

    formatted_imports = ",\n   ".join(pattern_matcher_imports)
    formatted_imports = (
        f"from torch._inductor.pattern_matcher import (\n   {formatted_imports},\n)\n"
    )
    return f"{file_template}{formatted_imports}"


def get_central_index_epilogue() -> str:
    epilogue = """
def get_serialized_pattern(key):
    import torch._inductor  # noqa: F401
    from torch._inductor import config
    if config.fallback_random:
        return None

    # TODO - could add more validation that the same set of decomps used when
    # tracing SDPA are also used in current context. softmax, dropout, etc
    # decomp use is stable so not an issue in practice.
    return central_index.get(key)
"""
    return epilogue


def clean_directory(dir: Path) -> None:
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def serialize_functions() -> None:
    file_path = Path.cwd() / "torch" / "_inductor" / "fx_passes" / "serialized_patterns"
    if not file_path.exists():
        raise Exception(
            "Could not find serialized patterns directory. Make sure you are at Pytorch root directory"
        )

    clean_directory(file_path)

    with open(file_path / "__init__.py", "w"):
        pass

    central_index = {}
    file_to_keys = defaultdict(list)
    seen_patterns = set()

    file_template = get_file_template()
    for i, (
        key,
        kwargs,
    ) in enumerate(_get_sfdp_patterns()):
        pattern_name = kwargs["search_fn"].__name__
        gen_kwargs = {
            key: kwargs[key]
            for key in ("search_fn", "example_inputs", "trace_fn", "scalar_workaround")
        }

        # temporary to batch adding new patterns
        if i >= 26:
            continue

        from torch._functorch import config as functorch_config

        with functorch_config.patch(functionalize_rng_ops=False):
            pattern = gen_pattern(**gen_kwargs)

        serialized_pattern = PatternPrettyPrinter.run(pattern, output_name=key)
        if pattern_name not in seen_patterns:
            write_mode = "w"
            seen_patterns.add(pattern_name)
        else:
            write_mode = "a"

        with open(file_path / f"{pattern_name}.py", write_mode) as f:
            if write_mode == "w":
                f.write(file_template)
            f.write(serialized_pattern)
            f.write("\n")

        central_index[f"{key}"] = f"{pattern_name}.py"

        file_to_keys[pattern_name].append(f"{key}")

    with open(file_path / "central_index.py", "w") as f:
        f.write(auto_generated_msg)
        for pattern_name, keys in file_to_keys.items():
            f.write(f"from .{pattern_name} import ({', '.join(keys)})\n")

        f.write("\ncentral_index = {\n")
        for k in central_index.keys():
            f.write(f"    '{k}': {k},\n")
        f.write("}\n\n")

        f.write(get_central_index_epilogue())


if __name__ == "__main__":
    with torch._subclasses.FakeTensorMode():
        serialize_functions()
