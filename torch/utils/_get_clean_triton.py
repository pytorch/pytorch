# mypy: allow-untyped-defs
import argparse
import os
import re
from pathlib import Path
from typing import Dict, List


def remove_triton_function_declaration(source_code: str) -> str:
    remove_head = re.sub(r"(\n.+\s\'\'\'\n)", "\n", source_code)
    remove_tail = re.sub(r"(\'\'\'\,.+)", "\n", remove_head)
    return remove_tail


def remove_async_compile(source_code: str) -> str:
    remove_top_level = str.replace(source_code, "async_compile = AsyncCompile()", "")
    remove_compile = str.replace(remove_top_level, "async_compile.wait(globals())", "")
    remove_del = str.replace(remove_compile, "del async_compile", "")
    return remove_del


def rename_kernels(source_code: str) -> str:
    pattern = r"(\w+)\s*=\s*async_compile\.triton\('triton_',\s"
    triton_kernel_decl = "def triton_"
    matches = [
        (match.end(), match.group(1))
        for match in re.finditer(pattern, source_code, re.DOTALL)
    ]

    # Starting from the last match to avoid issues with shifting indices after replacements
    for end_index, captured_string in reversed(matches):
        # Find the index of the next "B" after the current match
        index_of_B = source_code.find(triton_kernel_decl, end_index)
        if index_of_B != -1:
            # Replace the triton_kernel_decl with the captured string
            source_code = (
                source_code[:index_of_B]
                + f"def {captured_string}"
                + source_code[index_of_B + len(triton_kernel_decl) :]
            )
        else:
            # If triton_kernel_decl is not found after the current match, continue to the next
            continue

    return source_code


def merge_params(original_params: List[str], new_params: List[str]) -> List[str]:
    assert len(new_params) >= len(original_params)
    for idx in range(len(new_params)):
        if new_params[idx] == "T":
            new_params[idx] = original_params[idx]
    return new_params


def add_launch_params(original: str, kernel_to_params: Dict[str, str]) -> str:
    # Regex to match the function call in the original string
    pattern = r"(\w+)\.run\((.*), grid=(.*\)), [^)]*\)"

    def replace(match) -> str:
        # Extract parts from the regex match
        func_name = match.group(1)
        params = match.group(2)
        grid = match.group(3)
        new_params = kernel_to_params[func_name]
        new_params = merge_params(params.split(", "), new_params.split(", "))

        # Format the new function call
        new_string = f"{func_name}[{grid}]({', '.join(new_params)})"
        return new_string

    transformed = re.sub(pattern, replace, original)

    remove_inductor_wrappers = re.sub(
        r"@triton_heuristics[^@]*@triton.jit",
        r"@triton.jit",
        transformed,
        flags=re.DOTALL,
    )

    return remove_inductor_wrappers


def process_file(input_filename: str, output_filename: str) -> str:
    with open(input_filename) as file:
        source_code = file.read()

    transformed_code = source_code
    if "def triton_(" in source_code:
        raise RuntimeError(
            "Need to run original Pytorch code generating kernels with TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1"
        )
    # transformed_code = rename_kernels(transformed_code)
    transformed_code = remove_triton_function_declaration(transformed_code)
    transformed_code = remove_async_compile(transformed_code)

    launch_params_filename = f"{input_filename}.launch_params"
    if not os.path.exists(launch_params_filename):
        raise RuntimeError(
            f"Missing {launch_params_filename}. Run `TORCHINDUCTOR_DUMP_LAUNCH_PARAMS=1 python {input_filename} first."
        )

    with open(launch_params_filename) as f:
        launch_params_meta = f.readlines()

    split_params = [i.split("|") for i in launch_params_meta]
    strip_params = [[a.strip(), b.strip()] for a, b in split_params]
    kernel_to_args: Dict[str, str] = dict(strip_params)
    transformed_code = add_launch_params(transformed_code, kernel_to_args)

    with open(output_filename, "w") as file:
        file.write(transformed_code)
    return transformed_code


def get_clean_triton(
    input_path: Path, output_path: Path = Path("triton_only_repro.py")
):
    """Run experiments and output results to file

    Args:
        input_path (Optional[Path]): Path to inductor generated output codede
        output_path (Optional[Path]): Path to write out the new python file
    """
    return process_file(str(input_path), str(output_path))


if __name__ == "__main__":
    """Sample usage:
    # Running sweep
    python inputcode.py
    """
    parser = argparse.ArgumentParser(
        description="Clean Inductor generated code to remove Inductor dependencies"
    )

    # Add the arguments
    parser.add_argument(
        "input_path", type=Path, help="Path to inductor generated output code"
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("triton_only_repro.py"),
        help="Path to write out the clean triton output",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with parsed arguments
    result = get_clean_triton(args.input_path, args.output_path)
