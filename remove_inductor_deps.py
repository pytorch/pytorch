from typing import Optional

from jsonargparse import CLI
from pathlib import Path
import re
import os
import sys


def remove_triton_function_declaration(source_code: str):
    remove_head = re.sub(r'(\n.+\s\'\'\'\n)', '\n', source_code)
    remove_tail = re.sub(r'(\'\'\'\,.+)', '\n', remove_head)
    return remove_tail


def remove_async_compile(source_code: str):
    remove_top_level = str.replace(source_code, 'async_compile = AsyncCompile()', '')
    remove_compile = str.replace(remove_top_level, 'async_compile.wait(globals())', '')
    remove_del = str.replace(remove_compile, 'del async_compile', '')
    return remove_del

    return text

def rename_kernels(source_code: str):
    pattern = r"(\w+)\s*=\s*async_compile\.triton\('triton_',\s"
    triton_kernel_decl = "def triton_"
    matches = [(match.end(), match.group(1)) for match in re.finditer(pattern, source_code, re.DOTALL)]

    # Starting from the last match to avoid issues with shifting indices after replacements
    for end_index, captured_string in reversed(matches):
        # Find the index of the next "B" after the current match
        index_of_B = source_code.find(triton_kernel_decl, end_index)
        if index_of_B != -1:
            # Replace the triton_kernel_decl with the captured string
            source_code = source_code[:index_of_B] + f"def {captured_string}" + source_code[index_of_B + len(triton_kernel_decl):]
        else:
            # If triton_kernel_decl is not found after the current match, continue to the next
            continue

    return source_code

def run_script_with_env_vars(script, env_vars):
    # Backup the original environment variables
    original_env = {key: os.environ.get(key) for key in env_vars}
    original_argv = sys.argv.copy()
    sys.argv = [original_argv[0]]


    # Set the new environment variables
    os.environ.update(env_vars)
    local_context = {"__name__": "__main__"}  # Copy the current global context
    kernel_launch_metadata = {}
    local_context.update({'triton_kernel_launch_metadata': kernel_launch_metadata})

    try:
        # Execute the script in the current global and local context
        exec(script, local_context)
    finally:
        sys.argv = original_argv
        # Restore the original environment variables
        for key, value in original_env.items():
            if value is None:
                del os.environ[key]
            else:
                os.environ[key] = value

def merge_params(original_params, new_params):
    assert len(new_params) >= len(original_params)
    for idx in range(len(new_params)):
        if new_params[idx] == 'T':
            new_params[idx] = original_params[idx]
    print(original_params, new_params)
    return new_params

def transform_string(original, launch_params):
    # Regex to match the function call in the original string
    pattern = r"(\w+)\.run\(([^)]*), grid=(.*\)), [^)]*\)"

    def replace(match):
        # Extract parts from the regex match
        func_name = match.group(1)
        params = match.group(2)
        grid = match.group(3)
        new_params = launch_params[func_name]
        new_params = merge_params(params.split(', '), new_params.split(', '))

        # Format the new function call
        new_string = f"{func_name}[{grid}]({', '.join(new_params)})"
        return new_string
    transformed = re.sub(pattern, replace, original)

    remove_inductor_wrappers = re.sub(r'@triton_heuristics.*@triton.jit', r'@triton.jit', transformed, flags=re.DOTALL)

    return remove_inductor_wrappers

def process_file(input_filename, output_filename):
    with open(input_filename, 'r') as file:
        source_code = file.read()


    transformed_code = source_code
    if "def triton_(" in source_code:
        raise RuntimeError("Need to run original Pytorch code generating kernels with TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1")
    # transformed_code = rename_kernels(transformed_code)
    transformed_code = remove_triton_function_declaration(transformed_code)
    transformed_code = remove_async_compile(transformed_code)

    launch_params_filename = f"{input_filename}.launch_params"
    if not os.path.exists(launch_params_filename):
        raise RuntimeError(f"Missing {launch_params_filename}. Need to run {input_filename} with ENV_VAR first!")

    with open(launch_params_filename, 'r') as f:
        launch_params_meta = f.readlines()

    launch_params_meta = [i.split('|') for i in launch_params_meta]
    launch_params_meta = [[a.strip(), b.strip()] for a, b in launch_params_meta]
    kernel_to_args = {a: b for a, b in launch_params_meta}
    transformed_code = transform_string(transformed_code, kernel_to_args)
    # run_script_with_env_vars(source_code, {'TORCHINDUCTOR_COMPILE_THREADS': "1", 'TORCHINDUCTOR_CONVERT_TO_TRITON': "1"})

    with open(output_filename, 'w') as file:
        file.write(transformed_code)

def main(input_path: Path, output_path: Path="triton_only_repro.py"):
    """Run experiments and output results to file

    Args:
        input_path (Optional[Path]): Path to inductor generated output codede
        output_path (Optional[Path]): Path to write out the new python file
    """
    process_file(input_path, output_path)

if __name__ == "__main__":
    """Sample usage:
    # Running sweep
    python inputcode.py out.py
    """
    CLI(main)
