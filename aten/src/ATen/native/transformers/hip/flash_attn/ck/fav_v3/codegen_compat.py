"""Stdlib reimplementation of third_party/aiter/hsa/codegen.py for Python 3.15+.

aiter's codegen.py imports pandas/numpy solely to read a handful of small CSV
config files and emit a C++ header (asm_<module>_configs.hpp). numpy and pandas
publish no CPython 3.15 wheels yet (and pandas hard-requires numpy), so installing
them into the codegen venv forces a from-source build that fails on 3.15. PyTorch
itself treats numpy as optional (USE_NUMPY auto-disables; numpy is not in the
wheel's install_requires), so this one codegen step was the only place in the
ROCm build that hard-required a pandas/numpy install.

This script reproduces codegen.py's output byte-for-byte using only the standard
library. The C++ template strings below are copied verbatim from codegen.py.

The equivalence rests on one detail of codegen.py: every emitted cell value goes
through ``f"{int(x):>4}" if str(x).replace(".", "", 1).isdigit() else f'"{x}"'``,
i.e. it is stringified and re-parsed. So a raw CSV string ("192") formats
identically to pandas' np.int64(192). Only the generated struct field type, which
codegen.py derives from ``isinstance(combine_df.iloc[0][col], (int, float,
np.integer))``, depends on pandas' per-column dtype; we reproduce that with
explicit numeric inference over the column's values.

Remove this script and revert fav_v3/CMakeLists.txt to the venv path once
numpy/pandas ship CPython 3.15 wheels, or once the aiter submodule drops the
pandas/numpy dependency. See https://github.com/pytorch/pytorch/issues/184900.

Usage: codegen_compat.py --hsa-dir <aiter/hsa> -m <module> -o <output_dir>
       (reads AITER_GPU_ARCHS from the environment, like codegen.py).
"""

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict


def _is_number(value: str) -> bool:
    # pandas read_csv infers a numeric (int64/float64) dtype only when every
    # value in the column parses as a number; otherwise the column is object
    # (str). codegen.py's isinstance(..., (int, float, np.integer)) check
    # observes that dtype via iloc[0], so a column is "int" iff all values are
    # numeric. float("inf"/"nan") never appear in these CSVs.
    try:
        float(value)
        return True
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="codegen_compat",
        description="pandas-free reimplementation of aiter hsa/codegen.py",
    )
    parser.add_argument("-m", "--module", required=True)
    parser.add_argument("-o", "--output_dir", default="aiter/jit/build")
    parser.add_argument(
        "--hsa-dir",
        required=True,
        help="aiter hsa directory (the dir codegen.py lives in, holding the "
        "per-arch subdirectories)",
    )
    args = parser.parse_args()

    this_dir = os.path.abspath(args.hsa_dir)
    archs = os.environ["AITER_GPU_ARCHS"].split(";")
    archs_supported = [
        os.path.basename(os.path.normpath(path)) for path in glob.glob(f"{this_dir}/*/")
    ]

    content = """// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include <unordered_map>

"""

    # Group CSVs by config name across all per-arch subdirectories. Mirrors
    # codegen.py's glob order so the emitted ordering is identical.
    csv_groups = defaultdict(list)
    for arch in archs_supported:
        for el in glob.glob(
            f"{this_dir}/{arch}/{args.module}/**/*.csv", recursive=True
        ):
            cfgname = os.path.basename(el).split(".")[0]
            csv_groups[cfgname].append({"file_path": el, "arch": arch})

    cfgs = []
    have_get_header = False
    for cfgname, file_info_list in csv_groups.items():
        # Concatenate rows across the group's files, tagging each row with its
        # source arch (codegen.py's df["arch"] = arch + pd.concat). All files in
        # a group share the same schema, so the column union is just the CSV
        # header plus "arch".
        columns = None
        combined_rows = []
        single_file = None
        arch = None
        for file_info in file_info_list:
            single_file = file_info["file_path"]
            arch = file_info["arch"]
            with open(single_file, newline="") as f:
                reader = csv.reader(f)
                file_rows = list(reader)
            headers_list = file_rows[0] if file_rows else []
            required_columns = {"knl_name", "co_name"}
            if not required_columns.issubset(headers_list):
                missing = required_columns - set(headers_list)
                print(
                    f"ERROR: Invalid assembly CSV format -- {single_file}. "
                    f"Missing required columns: {', '.join(missing)}"
                )
                sys.exit(1)
            if columns is None:
                columns = [*headers_list, "arch"]
            for raw in file_rows[1:]:
                if not raw:  # pandas read_csv drops blank lines (skip_blank_lines)
                    continue
                row = dict(zip(headers_list, raw))
                row["arch"] = arch
                combined_rows.append(row)

        if file_info_list:
            relpath = os.path.relpath(
                os.path.dirname(single_file), f"{this_dir}/{arch}"
            )
            if not have_get_header:
                headers_list = columns
                required_columns = {"knl_name", "co_name", "arch"}
                other_columns = [
                    col for col in headers_list if col not in required_columns
                ]
                other_columns_comma = ", ".join(other_columns)
                col_types = {
                    col: "int"
                    if combined_rows and all(_is_number(r[col]) for r in combined_rows)
                    else "std::string"
                    for col in other_columns
                }
                other_columns_cpp_def = "\n".join(
                    f"    {col_types[col]} {col};" for col in other_columns
                )
                content += f"""
#define ADD_CFG({other_columns_comma}, arch, path, knl_name, co_name)         \\
    {{                                         \\
        arch knl_name, {{ knl_name, path co_name, arch, {other_columns_comma} }}         \\
    }}

struct {args.module}Config
{{
    std::string knl_name;
    std::string co_name;
    std::string arch;
{other_columns_cpp_def}
}};

using CFG = std::unordered_map<std::string, {args.module}Config>;

"""
                have_get_header = True
            cfg = [
                "ADD_CFG("
                + ", ".join(
                    (
                        f"{int(row[col]):>4}"
                        if str(row[col]).replace(".", "", 1).isdigit()
                        else f'"{row[col]}"'
                    )
                    for col in other_columns
                )
                + f', "{row["arch"]}", "{relpath}/", "{row["knl_name"]}", "{row["co_name"]}"),'
                for row in combined_rows
                if row["arch"] in archs
            ]
            cfg_txt = "\n    ".join(cfg) + "\n"

            txt = f"""static CFG cfg_{cfgname} = {{
    {cfg_txt}}};"""
            cfgs.append(txt)

    content += "\n".join(cfgs) + "\n"

    with open(f"{args.output_dir}/asm_{args.module}_configs.hpp", "w") as f:
        f.write(content)


if __name__ == "__main__":
    main()
