"""Find functions using CUDA which don't have device guards.

Note that doing this correctly requires using either LLVM or GCC to build
an AST, but these are heavy dependencies, so in practice we are approximating
the structure of the language and hoping that developers don't do anything
too convoluted.

The strategy here is to read in CUDA files, remove comments, separate out
the functions, then check the functions for CUDA, then determine if
functions that use CUDA have device guards.
"""
import os
import re
import sys

class MissingEndBracket(Exception):
    pass

# https://stackoverflow.com/a/241506
def comment_remover(text):
    """Removes all comments from the code"""
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " "  # note: a space and not an empty string
        else:
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)


def find_matching_bracket(char_idx, code):
    """Find matching bracket to the one indicated by `char_idx`"""
    assert code[char_idx] == "{"

    # Simple state machine. Each opening bracket increments depth
    # and each closing bracket decrements. Note that this isn't
    # super safe, but is being used after comments have been striped
    # on code without many textual strings, so it should be generally
    # safe
    depth = 0
    for i, char in enumerate(code[char_idx:]):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return char_idx + i

    raise MissingEndBracket("No matching bracket found!")


def strip_namespaces(code):
    """Eliminates opening and closing namespace brackets, which would
    otherwise look like top-level brackets and therefore be
    potentially interpreted as functions
    """
    while True:
        namespace = re.search(r"namespace[\sA-Za-z0-9]*{", code)
        if not namespace:
            break
        ebi = find_matching_bracket(namespace.end() - 1, code)  # End bracket index
        code = code[0:namespace.start()] + code[namespace.end():ebi] + code[ebi + 1:]
    return code


def find_chunks_within_outermost_brackets(code):
    """Yield chunks of outermost brackets from the code, these
    chunks are interpreted as probably being functions
    """

    closing_bracket = 0
    while True:
        opening_bracket = code.find("{", closing_bracket)       # Start of function code
        prev_opening = code.rfind("}", 0, opening_bracket) + 1  # Include function name
        if opening_bracket == -1:
            break
        closing_bracket = find_matching_bracket(opening_bracket, code) + 1
        yield code[prev_opening:closing_bracket]


def is_probably_a_function(code):
    """Determines if the chunk of code is probably a function
    eliminates CUDA kernel code
    """
    invalid_function_indicators = [
        "__global__", "__device__", "struct", "class "
    ]
    for invalid_indicator in invalid_function_indicators:
        if invalid_indicator in code:
            return False
    return True


def contains_cuda_calls_that_should_be_guarded(code):
    """Checks through a list of funcitons and other signs
    that would indicate CUDA is being used in this function
    """
    return re.search("\bcuda[A-Z]", code) or (">>>" in code)


def has_guard(code):
    """Determines whether the code contains a CUDA device guard"""
    return "OptionalCUDAGuard" in code


def check_code_for_unguarded_cuda(code, filename=None):
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

    # Try to clean up the syntax to reduce false positives and provide
    # more helpful messages
    code = comment_remover(code)                                      # Remove the comments
    try:
        code = strip_namespaces(code)
    except MissingEndBracket as meb:
        print(f"Missing end bracket in '{filename}'")
        return 0

    # Preprocess the code to provide better context
    code = code.split("\n")                                           # Split by line breaks
    code = [x for x in code if not re.match(r"^\s*[0-9]+: #", x)]     # Strip preprocessing
    code = [x for x in code if "namespace" not in x]                  # Eliminate namespaces
    code = [x for x in code if len(x[x.find(":") + 1:].strip()) > 0]  # Eliminate blank lines
    code = '\n'.join(code)                                            # Put it back together

    # Look through each function of the file to see if it's guarded
    missing_guards = 0
    try:
        for func in find_chunks_within_outermost_brackets(code):
            if not is_probably_a_function(func):
                continue
            # print(f"~~~~~~~~~~~~~\n{filename}\n{func}\n~~~~~~~~~~~~~~~~~")
            # continue
            if contains_cuda_calls_that_should_be_guarded(func) and not has_guard(func):
                func_name = func[0:func.find("{") + 1]
                print(f"\nMissing CUDA device_guard in '{filename}'. Context:\n{func_name}", file=sys.stderr)
                missing_guards += 1
    except MissingEndBracket as err:
        print(f"Failed to find an end bracket in '{filename}'. Skipping.'")
        return 0

    return missing_guards


def check_file(filename):
    """Checks a file for CUDA kernel launches without cuda error checks

    Args:
        filename - File to check

    Returns:
        The number of unsafe kernel launches in the file
    """
    if not (filename.endswith(".cu") or filename.endswith(".cuh")):
        return 0
    contents = open(filename, "r").read()
    return check_code_for_unguarded_cuda(contents, filename)


def check_cuda_guards():
    """Checks all pytorch code for CUDA code without device guards

    Returns:
        The number of functions without device guards
    """
    torch_dir = os.path.dirname(os.path.realpath(__file__))
    torch_dir = os.path.dirname(torch_dir)  # Go up to parent torch
    torch_dir = os.path.dirname(torch_dir)  # Go up to parent caffe2

    kernels_without_guards = 0
    files_without_guards = []
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
                kernels_without_guards += file_result
                files_without_guards.append(filename)

    if kernels_without_guards > 0:
        count_str = f"Found {kernels_without_guards} instances in " \
                    f"{len(files_without_guards)} files where CUDA " \
                    "was used without device guards."
        print(count_str, file=sys.stderr)
        print("Files without guards:", file=sys.stderr)
        for x in files_without_guards:
            print(f"\t{x}", file=sys.stderr)
        print(count_str, file=sys.stderr)

    return kernels_without_guards


if __name__ == "__main__":
    unsafe_launches = check_cuda_guards()
    sys.exit(0)
