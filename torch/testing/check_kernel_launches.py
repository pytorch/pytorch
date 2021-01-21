import os
import re
import sys


# Regular expression identifies a kernel launch indicator by
# finding something approximating the pattern ">>>(arguments);"
# It then requires that `C10_CUDA_KERNEL_LAUNCH_CHECK` be
# the next command.
# It allows a single backslash `\` between the end of the launch
# command and the beginning of the kernel check. This handles
# cases where the kernel launch is in a multiline preprocessor
# definition.
#
# There are various ways this can fail:
# * If the semicolon is in a string for some reason
# * If there's a triply-nested template
# But this should be sufficient to detect and fix most problem
# instances and can be refined before the test is made binding
kernel_launch_regex = re.compile(r"""
    ^.*>>>        # Identifies kernel launch
    \s*           # Maybe some whitespace (includes newlines)
    \([^;]+\);    # And then arguments in parens and semi-colon
    (?!           # Negative lookahead: we trigger if we don't find the launch guard
        \s*                                  # Maybe some whitespace (includes newlines)
        \\?                                  # 0 or 1 backslashes (for launches in preprocessor macros)
        \s*                                  # Maybe some whitespace (includes newlines)
        (?:[0-9]+: )?                        # Detects and ignores a line numbering, if present
        \s*                                  # Maybe some whitespace (includes newlines)
        C10_CUDA_KERNEL_LAUNCH_CHECK\(\);  # Kernel launch guard!
    )             # End negative lookahead
""", flags=re.MULTILINE | re.VERBOSE)

# This identifies kernel launch checks that aren't paired with kernels
unmatched_kernel_launch_regex = re.compile(r"""
    C10_CUDA_KERNEL_LAUNCH_CHECK\(\); # Desired kernel launch guard
    (?:                               # Non capturing group
        (?!>>>).                      # Negative lookahead for ">>>" followed by a character
    )*?                               # Repeat the group 0-Inf times as few times as possible
    C10_CUDA_KERNEL_LAUNCH_CHECK\(\); # Kernel launch guard without a kernel
""", flags=re.MULTILINE | re.VERBOSE | re.DOTALL)

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

    results = kernel_launch_regex.findall(code)               # Search for bad launches
    for r in results:
        print(f"Missing C10_CUDA_KERNEL_LAUNCH_CHECK in '{filename}'. Context:\n{r}", file=sys.stderr)
    return len(results)


def check_code_for_cuda_kernel_launch_checks_without_launches(code, filename=None):
    """Checks code for CUDA kernel launch checks without kernel launches.
       (That is, excess launch checks.)

    Args:
        filename - Filename of file containing the code. Used only for display
                   purposes, so you can put anything here.
        code     - The code to check

    Returns:
        The number of unmathced launch checks in the code
    """
    if filename is None:
        filename = "##Python Function Call##"

    # We break the code apart and put it back together to add
    # helpful line numberings for identifying problem areas
    code = enumerate(code.split("\n"))                             # Split by line breaks
    code = [f"{lineno}: {linecode}" for lineno, linecode in code]  # Number the lines
    code = '\n'.join(code)                                         # Put it back together

    results = unmatched_kernel_launch_regex.findall(code)          # Search for unmatched checks
    for r in results:
        print(f"Extraneous C10_CUDA_KERNEL_LAUNCH_CHECK in '{filename}'. Context:\n{r}", file=sys.stderr)
    return len(results)

def check_file(filename):
    """Checks a file for CUDA kernel launches without cuda error checks

    Args:
        filename - File to check

    Returns:
        A tuple: (# of unsafe kernel launches, # of extra launch checks)
    """
    if not (filename.endswith(".cu") or filename.endswith(".cuh")):
        return 0, 0
    contents = open(filename, "r").read()
    missing_checks = check_code_for_cuda_kernel_launches(contents, filename)
    extra_checks = check_code_for_cuda_kernel_launch_checks_without_launches(contents, filename)
    return missing_checks, extra_checks

def check_cuda_kernel_launches():
    """Checks all pytorch code for CUDA kernel launches without cuda error checks

    Returns:
        A tuple: (# of unsafe kernel launches, # of extra launch checks)
    """
    torch_dir = os.path.dirname(os.path.realpath(__file__))
    torch_dir = os.path.dirname(torch_dir)  # Go up to parent torch
    torch_dir = os.path.dirname(torch_dir)  # Go up to parent caffe2

    kernels_without_checks = 0
    extra_launch_checks = 0
    files_without_checks = []
    files_with_extra_checks = []
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
            this_missing_checks, this_extra_checks = check_file(filename)
            kernels_without_checks += this_missing_checks
            extra_launch_checks += this_extra_checks
            if this_missing_checks > 0:
                files_without_checks.append(filename)
            if this_extra_checks > 0:
                files_with_extra_checks.append(filename)

    if kernels_without_checks > 0:
        count_str = f"Found {kernels_without_checks} instances in " \
                    f"{len(files_without_checks)} files where kernel " \
                    "launches didn't have checks."
        print(count_str, file=sys.stderr)
        print("Files without checks:", file=sys.stderr)
        for x in files_without_checks:
            print(f"\t{x}", file=sys.stderr)
        print(count_str, file=sys.stderr)

    if extra_launch_checks > 0:
        count_str = f"Found {extra_launch_checks} instances in " \
                    f"{len(files_with_extra_checks)} files with extraneous " \
                    "kernel launch checks."
        print(count_str, file=sys.stderr)
        print("Files with extraneous kernel launch checks:", file=sys.stderr)
        for x in files_with_extra_checks:
            print(f"\t{x}", file=sys.stderr)
        print(count_str, file=sys.stderr)

    return kernels_without_checks, extra_launch_checks


if __name__ == "__main__":
    unsafe_launches = check_cuda_kernel_launches()
    sys.exit(0)
