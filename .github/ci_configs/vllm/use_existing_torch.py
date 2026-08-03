# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import glob
import os
import re
import sys


# Only strip targeted libraries when checking prefix
TORCH_LIB_PREFIXES = (
    # requirements/*.txt/in
    "torch=",
    "torchvision=",
    "torchaudio=",
    # pyproject.toml
    '"torch =',
    '"torchvision =',
    '"torchaudio =',
)

# Match lines where the package name is exactly torch/torchvision/torchaudio,
# not a substring of another package (e.g. terratorch, open_clip_torch).
_TORCH_PKG_RE = re.compile(
    r"""^\s*['"]?\s*(?:torchvision|torchaudio|torch)\s*(?:[=<>!;\[,\]'"@~#(]|$)""",
    re.IGNORECASE,
)


def main(argv):
    parser = argparse.ArgumentParser(
        description="Strip torch lib requirements to use installed version."
    )
    parser.add_argument(
        "--prefix",
        action="store_true",
        help="Strip prefix matches only (default: False)",
    )
    args = parser.parse_args(argv)

    for file in (
        *glob.glob("requirements/**/*.txt", recursive=True),
        *glob.glob("requirements/**/*.in", recursive=True),
        "pyproject.toml",
    ):
        if not os.path.exists(file):
            continue
        with open(file) as f:
            lines = f.readlines()
        if "torch" in "".join(lines).lower():
            with open(file, "w") as f:
                for line in lines:
                    if (
                        args.prefix
                        and not line.lower().strip().startswith(TORCH_LIB_PREFIXES)
                        or not args.prefix
                        and not _TORCH_PKG_RE.match(line)
                    ):
                        f.write(line)
                    else:
                        print(f">>> removed from {file}:", line.strip())


if __name__ == "__main__":
    main(sys.argv[1:])
