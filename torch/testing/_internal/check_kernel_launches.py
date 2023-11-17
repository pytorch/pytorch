import os
import re
import sys
from typing import List

__all__ = [
    "check_code_for_cuda_kernel_launches",
    "check_cuda_kernel_launches",
]

# FILES TO EXCLUDE (match is done with suffix using `endswith`)
# You wouldn't drive without a seatbelt, though, so why would you
# launch a kernel without some safety? Use this as a quick workaround
# for a problem with the checker, fix the checker, then de-exclude
# the files in question.
exclude_files: List[str] = []

# Without using a C++ AST we can't 100% detect kernel launches, so we
# model them as having the pattern "<<<parameters>>>(arguments);"
# We then require that `C10_CUDA_KERNEL_LAUNCH_CHECK` be
# the next statement.
#
# We model the next statement as ending at the next `}` or `;`.
# If we see `}` then a clause ended (bad) if we see a semi-colon then
# we expect the launch check just before it.
#
# Since the kernel launch can include lambda statements, it's important
# to find the correct end-paren of the kernel launch. Doing this with
# pure regex requires recursive regex, which aren't part of the Python
# standard library. To avoid an additional dependency, we build a prefix
# regex that finds the start of a kernel launch, use a paren-matching
# algorithm to find the end of the launch, and then another regex to
# determine if a launch check is present.

# Finds potential starts of kernel launches
kernel_launch_start = re.compile(
    r"^.*<<<[^>]+>>>\s*\(", flags=re.MULTILINE
)

# This pattern should start at the character after the final paren of the
# kernel launch. It returns a match if the launch check is not the next statement
has_check = re.compile(
    r"\s*;(?![^;}]*C10_CUDA_KERNEL_LAUNCH_CHECK\(\);)", flags=re.MULTILINE
)

def find_matching_paren(s: str, startpos: int) -> int:
    """Given a string "prefix (unknown number of characters) suffix"
    and the position of the first `(` returns the index of the character
    1 past the `)`, accounting for paren nesting
    """
    opening = 0
    for i, c in enumerate(s[startpos:]):
        if c == '(':
            opening += 1
        elif c == ')':
            opening -= 1
            if opening == 0:
                return startpos + i + 1

    raise IndexError("Closing parens not found!")


def should_exclude_file(filename) -> bool:
    for exclude_suffix in exclude_files:
        if filename.endswith(exclude_suffix):
            return True
    return False


def check_code_for_cuda_kernel_launches(code, filename=None):
    """Checks code for CUDA kernel launches without cuda error checks.

    Args:
        filename - Filename of file containing the code. Used only for display
                   purposes, so you can put anything here.
        code     - The code to check

    Returns:
        The number of unsafe kernel launches in the code
    """
    if filename is None:
        filename = "##Python Function Call##"

    # We break the code apart and put it back together to add
    # helpful line numberings for identifying problem areas
    code = enumerate(code.split("\n"))                             # Split by line breaks
    code = [f"{lineno}: {linecode}" for lineno, linecode in code]  # Number the lines
    code = '\n'.join(code)                                         # Put it back together

    num_launches_without_checks = 0
    for m in kernel_launch_start.finditer(code):
        end_paren = find_matching_paren(code, m.end() - 1)
        if has_check.match(code, end_paren):
            num_launches_without_checks += 1
            context = code[m.start():end_paren + 1]
            print(f"Missing C10_CUDA_KERNEL_LAUNCH_CHECK in '{filename}'. Context:\n{context}", file=sys.stderr)

    return num_launches_without_checks


def check_file(filename):
    """Checks a file for CUDA kernel launches without cuda error checks

    Args:
        filename - File to check

    Returns:
        The number of unsafe kernel launches in the file
    """
    if not (filename.endswith((".cu", ".cuh"))):
        return 0
    if should_exclude_file(filename):
        return 0
    with open(filename) as fo:
        contents = fo.read()
        unsafeCount = check_code_for_cuda_kernel_launches(contents, filename)
    return unsafeCount


def check_cuda_kernel_launches():
    """Checks all pytorch code for CUDA kernel launches without cuda error checks

    Returns:
        The number of unsafe kernel launches in the codebase
    """
    torch_dir = os.path.dirname(os.path.realpath(__file__))
    torch_dir = os.path.dirname(torch_dir)  # Go up to parent torch
    torch_dir = os.path.dirname(torch_dir)  # Go up to parent caffe2

    kernels_without_checks = 0
    files_without_checks = []
    for root, dirnames, filenames in os.walk(torch_dir):
        # `$BASE/build` and `$BASE/torch/include` are generated
        # so we don't want to flag their contents
        if root == os.path.join(torch_dir, "build") or root == os.path.join(torch_dir, "torch/include"):
            # Curtail search by modifying dirnames and filenames in place
            # Yes, this is the way to do this, see `help(os.walk)`
            dirnames[:] = []
            continue

        for x in filenames:
            filename = os.path.join(root, x)
            file_result = check_file(filename)
            if file_result > 0:
                kernels_without_checks += file_result
                files_without_checks.append(filename)

    if kernels_without_checks > 0:
        count_str = f"Found {kernels_without_checks} instances in " \
                    f"{len(files_without_checks)} files where kernel " \
                    "launches didn't have checks."
        print(count_str, file=sys.stderr)
        print("Files without checks:", file=sys.stderr)
        for x in files_without_checks:
            print(f"\t{x}", file=sys.stderr)
        print(count_str, file=sys.stderr)

    return kernels_without_checks


if __name__ == "__main__":
    unsafe_launches = check_cuda_kernel_launches()
    sys.exit(0 if unsafe_launches == 0 else 1)
