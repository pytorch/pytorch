#!/usr/bin/env python
""" The Python Hipify script.
##
# Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.
#               2017-2018 Advanced Micro Devices, Inc. and
#                         Facebook Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""

from __future__ import absolute_import, division, print_function
import argparse
import fnmatch
import re
import shutil
import sys
import os
import json

from enum import Enum
from pyHIPIFY import constants
from pyHIPIFY.cuda_to_hip_mappings import CUDA_TO_HIP_MAPPINGS
from pyHIPIFY.cuda_to_hip_mappings import MATH_TRANSPILATIONS

# Hardcode the PyTorch template map
"""This dictionary provides the mapping from PyTorch kernel template types
to their actual types."""
PYTORCH_TEMPLATE_MAP = {"Dtype": "scalar_t", "T": "scalar_t"}
CAFFE2_TEMPLATE_MAP = {}


class InputError(Exception):
    # Exception raised for errors in the input.

    def __init__(self, message):
        super(InputError, self).__init__(message)
        self.message = message

    def __str__(self):
        return "{}: {}".format("Input error", self.message)


def openf(filename, mode):
    if sys.version_info[0] == 3:
        return open(filename, mode, errors='ignore')
    else:
        return open(filename, mode)


# Color coding for printing
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class disablefuncmode(Enum):
    """ How to disable functions
    REMOVE - Remove the function entirely (includes the signature).
        e.g.
        FROM:
            ```ret_type function(arg_type1 arg1, ..., ){
                ...
                ...
                ...
            }```

        TO:
            ```
            ```

    STUB - Stub the function and return an empty object based off the type.
        e.g.
        FROM:
            ```ret_type function(arg_type1 arg1, ..., ){
                ...
                ...
                ...
            }```

        TO:
            ```ret_type function(arg_type1 arg1, ..., ){
                ret_type obj;
                return obj;
            }```


    HCC_MACRO - Add !defined(__HIP_PLATFORM_HCC__) preprocessors around the function.
        This macro is defined by HIP if the compiler used is hcc.
        e.g.
        FROM:
            ```ret_type function(arg_type1 arg1, ..., ){
                ...
                ...
                ...
            }```

        TO:
            ```#if !defined(__HIP_PLATFORM_HCC__)
                    ret_type function(arg_type1 arg1, ..., ){
                    ...
                    ...
                    ...
                }
               #endif
            ```


    DEVICE_MACRO - Add !defined(__HIP_DEVICE_COMPILE__) preprocessors around the function.
        This macro is defined by HIP if either hcc or nvcc are used in the device path.
        e.g.
        FROM:
            ```ret_type function(arg_type1 arg1, ..., ){
                ...
                ...
                ...
            }```

        TO:
            ```#if !defined(__HIP_DEVICE_COMPILE__)
                    ret_type function(arg_type1 arg1, ..., ){
                    ...
                    ...
                    ...
                }
               #endif
            ```


    EXCEPTION - Stub the function and throw an exception at runtime.
        e.g.
        FROM:
            ```ret_type function(arg_type1 arg1, ..., ){
                ...
                ...
                ...
            }```

        TO:
            ```ret_type function(arg_type1 arg1, ..., ){
                throw std::runtime_error("The function function is not implemented.")
            }```


    ASSERT - Stub the function and throw an assert(0).
        e.g.
        FROM:
            ```ret_type function(arg_type1 arg1, ..., ){
                ...
                ...
                ...
            }```

        TO:
            ```ret_type function(arg_type1 arg1, ..., ){
                assert(0);
            }```


    EMPTYBODY - Stub the function and keep an empty body.
        e.g.
        FROM:
            ```ret_type function(arg_type1 arg1, ..., ){
                ...
                ...
                ...
            }```

        TO:
            ```ret_type function(arg_type1 arg1, ..., ){
                ;
            }```



    """
    REMOVE = 0
    STUB = 1
    HCC_MACRO = 2
    DEVICE_MACRO = 3
    EXCEPTION = 4
    ASSERT = 5
    EMPTYBODY = 6


def update_progress_bar(total, progress):
    """Displays and updates a console progress bar."""
    barLength, status = 20, ""
    progress = float(progress) / float(total)
    if progress >= 1.:
        progress, status = 1, "\r\n"

    # Number of blocks to display. Used to visualize progress.
    block = int(round(barLength * progress))
    text = "\r[{}] {:.0f}% {}".format(
        "#" * block + "-" * (barLength - block), round(progress * 100, 0),
        status)

    # Send the progress to stdout.
    sys.stderr.write(text)

    # Send the buffered text to stdout!
    sys.stderr.flush()


def matched_files_iter(root_path, includes=('*',), ignores=(), extensions=(), hipify_caffe2=False):
    def _fnmatch(filepath, patterns):
        return any(fnmatch.fnmatch(filepath, pattern) for pattern in patterns)

    def match_extensions(filename):
        """Helper method to see if filename ends with certain extension"""
        return os.path.splitext(filename)[1] in extensions

    for (dirpath, _, filenames) in os.walk(root_path, topdown=True):
        for fn in filenames:
            filepath = os.path.join(dirpath, fn)
            rel_filepath = os.path.relpath(filepath, root_path)
            if _fnmatch(rel_filepath, includes) and (not _fnmatch(rel_filepath, ignores)) and match_extensions(fn):
                if hipify_caffe2 and not is_caffe2_gpu_file(filepath):
                    continue

                yield filepath


def preprocess(
        all_files,
        show_detailed=False,
        show_progress=True,
        hipify_caffe2=False,
        hip_suffix='cc',
        extensions_to_hip_suffix=()):
    """
    Call preprocessor on selected files.

    Arguments)
        show_detailed - Show a detailed summary of the transpilation process.
    """

    # Compute the total number of files to be traversed.
    total_count = len(all_files)
    finished_count = 0

    # Preprocessing statistics.
    stats = {"unsupported_calls": [], "kernel_launches": []}

    for filepath in all_files:
        preprocessor(filepath, stats, hipify_caffe2, hip_suffix, extensions_to_hip_suffix)
        # Update the progress
        if show_progress:
            print(filepath)
            update_progress_bar(total_count, finished_count)
            finished_count += 1

    print(bcolors.OKGREEN + "Successfully preprocessed all matching files." + bcolors.ENDC)

    # Show detailed summary
    if show_detailed:
        compute_stats(stats)


def compute_stats(stats):
    unsupported_calls = {cuda_call for (cuda_call, _filepath) in stats["unsupported_calls"]}

    # Print the number of unsupported calls
    print("Total number of unsupported CUDA function calls: {0:d}".format(len(unsupported_calls)))

    # Print the list of unsupported calls
    print(", ".join(unsupported_calls))

    # Print the number of kernel launches
    print("\nTotal number of replaced kernel launches: {0:d}".format(len(stats["kernel_launches"])))


def add_dim3(kernel_string, cuda_kernel):
    '''adds dim3() to the second and third arguments in the kernel launch'''
    count = 0
    closure = 0
    kernel_string = kernel_string.replace("<<<", "").replace(">>>", "")
    arg_locs = [{} for _ in range(2)]
    arg_locs[count]['start'] = 0
    for ind, c in enumerate(kernel_string):
        if count > 1:
            break
        if c == "(":
            closure += 1
        elif c == ")":
            closure -= 1
        elif (c == "," or ind == len(kernel_string) - 1) and closure == 0:
            arg_locs[count]['end'] = ind + (c != ",")
            count += 1
            if count < 2:
                arg_locs[count]['start'] = ind + 1

    first_arg_raw = kernel_string[arg_locs[0]['start']:arg_locs[0]['end'] + 1]
    second_arg_raw = kernel_string[arg_locs[1]['start']:arg_locs[1]['end']]

    first_arg_clean = kernel_string[arg_locs[0]['start']:arg_locs[0]['end']].replace("\n", "").strip(" ")
    second_arg_clean = kernel_string[arg_locs[1]['start']:arg_locs[1]['end']].replace("\n", "").strip(" ")

    first_arg_dim3 = "dim3({})".format(first_arg_clean)
    second_arg_dim3 = "dim3({})".format(second_arg_clean)

    first_arg_raw_dim3 = first_arg_raw.replace(first_arg_clean, first_arg_dim3)
    second_arg_raw_dim3 = second_arg_raw.replace(second_arg_clean, second_arg_dim3)
    cuda_kernel = cuda_kernel.replace(first_arg_raw + second_arg_raw, first_arg_raw_dim3 + second_arg_raw_dim3)
    return cuda_kernel


def processKernelLaunches(string, stats):
    """ Replace the CUDA style Kernel launches with the HIP style kernel launches."""
    # Concat the namespace with the kernel names. (Find cleaner way of doing this later).
    string = re.sub(r'([ ]+)(detail?)::[ ]+\\\n[ ]+', lambda inp: "{0}{1}::".format(inp.group(1), inp.group(2)), string)

    def grab_method_and_template(in_kernel):
        # The positions for relevant kernel components.
        pos = {
            "kernel_launch": {"start": in_kernel["start"], "end": in_kernel["end"]},
            "kernel_name": {"start": -1, "end": -1},
            "template": {"start": -1, "end": -1}
        }

        # Count for balancing template
        count = {"<>": 0}

        # Status for whether we are parsing a certain item.
        START = 0
        AT_TEMPLATE = 1
        AFTER_TEMPLATE = 2
        AT_KERNEL_NAME = 3

        status = START

        # Parse the string character by character
        for i in range(pos["kernel_launch"]["start"] - 1, -1, -1):
            char = string[i]

            # Handle Templating Arguments
            if status == START or status == AT_TEMPLATE:
                if char == ">":
                    if status == START:
                        status = AT_TEMPLATE
                        pos["template"]["end"] = i
                    count["<>"] += 1

                if char == "<":
                    count["<>"] -= 1
                    if count["<>"] == 0 and (status == AT_TEMPLATE):
                        pos["template"]["start"] = i
                        status = AFTER_TEMPLATE

            # Handle Kernel Name
            if status != AT_TEMPLATE:
                if string[i].isalnum() or string[i] in  {'(', ')', '_', ':', '#'}:
                    if status != AT_KERNEL_NAME:
                        status = AT_KERNEL_NAME
                        pos["kernel_name"]["end"] = i

                    # Case: Kernel name starts the string.
                    if i == 0:
                        pos["kernel_name"]["start"] = 0

                        # Finished
                        return [(pos["kernel_name"]), (pos["template"]), (pos["kernel_launch"])]

                else:
                    # Potential ending point if we're already traversing a kernel's name.
                    if status == AT_KERNEL_NAME:
                        pos["kernel_name"]["start"] = i

                        # Finished
                        return [(pos["kernel_name"]), (pos["template"]), (pos["kernel_launch"])]

    def find_kernel_bounds(string):
        """Finds the starting and ending points for all kernel launches in the string."""
        kernel_end = 0
        kernel_positions = []

        # Continue until we cannot find any more kernels anymore.
        while string.find("<<<", kernel_end) != -1:
            # Get kernel starting position (starting from the previous ending point)
            kernel_start = string.find("<<<", kernel_end)

            # Get kernel ending position (adjust end point past the >>>)
            kernel_end = string.find(">>>", kernel_start) + 3
            if kernel_end <= 0:
                raise InputError("no kernel end found")

            # Add to list of traversed kernels
            kernel_positions.append({"start": kernel_start, "end": kernel_end,
                                     "group": string[kernel_start: kernel_end]})

        return kernel_positions

    # Grab positional ranges of all kernel launchces
    get_kernel_positions = [k for k in find_kernel_bounds(string)]
    output_string = string

    # Replace each CUDA kernel with a HIP kernel.
    for kernel in get_kernel_positions:
        # Get kernel components
        params = grab_method_and_template(kernel)

        # Find parenthesis after kernel launch
        parenthesis = string.find("(", kernel["end"])

        # Extract cuda kernel
        cuda_kernel = string[params[0]["start"]:parenthesis + 1]
        kernel_string = string[kernel['start']:kernel['end']]
        cuda_kernel_dim3 = add_dim3(kernel_string, cuda_kernel)
        # Keep number of kernel launch params consistent (grid dims, group dims, stream, dynamic shared size)
        num_klp = len(extract_arguments(0, kernel["group"].replace("<<<", "(").replace(">>>", ")")))

        hip_kernel = "hipLaunchKernelGGL(" + cuda_kernel_dim3[0:-1].replace(
            ">>>", ", 0" * (4 - num_klp) + ">>>").replace("<<<", ", ").replace(">>>", ", ")

        # Replace cuda kernel with hip kernel
        output_string = output_string.replace(cuda_kernel, hip_kernel)

        # Update the statistics
        stats["kernel_launches"].append(hip_kernel)

    return output_string


def find_closure_group(input_string, start, group):
    """Generalization for finding a balancing closure group

         if group = ["(", ")"], then finds the first balanced parantheses.
         if group = ["{", "}"], then finds the first balanced bracket.

    Given an input string, a starting position in the input string, and the group type,
    find_closure_group returns the positions of group[0] and group[1] as a tuple.

    Example:
        find_closure_group("(hi)", 0, ["(", ")"])

    Returns:
        0, 3
    """

    inside_parenthesis = False
    parens = 0
    pos = start
    p_start, p_end = -1, -1

    while pos < len(input_string):
        if input_string[pos] == group[0]:
            if inside_parenthesis is False:
                inside_parenthesis = True
                parens = 1
                p_start = pos
            else:
                parens += 1
        elif input_string[pos] == group[1] and inside_parenthesis:
            parens -= 1

            if parens == 0:
                p_end = pos
                return p_start, p_end

        pos += 1
    return None, None


def find_bracket_group(input_string, start):
    """Finds the first balanced parantheses."""
    return find_closure_group(input_string, start, group=["{", "}"])


def find_parentheses_group(input_string, start):
    """Finds the first balanced bracket."""
    return find_closure_group(input_string, start, group=["(", ")"])


def disable_asserts(input_string):
    """ Disables regular assert statements
    e.g. "assert(....)" -> "/*assert(....)*/"
    """
    output_string = input_string
    asserts = list(re.finditer(r"\bassert[ ]*\(", input_string))
    for assert_item in asserts:
        p_start, p_end = find_parentheses_group(input_string, assert_item.end() - 1)
        start = assert_item.start()
        output_string = output_string.replace(input_string[start:p_end + 1], "")
    return output_string


def replace_forceinline(input_string):
    """__forceinline__'d methods can cause 'symbol multiply defined' errors in HIP.
    Adding 'static' to all such methods leads to compilation errors, so
    replacing '__forceinline__' with 'inline' as a workaround
    https://github.com/ROCm-Developer-Tools/HIP/blob/master/docs/markdown/hip_faq.md#what-if-hip-generates-error-of-symbol-multiply-defined-only-on-amd-machine
    """
    output_string = input_string
    output_string = re.sub("__forceinline__", "inline", output_string)
    return output_string


def replace_math_functions(input_string):
    """ FIXME: Temporarily replace std:: invocations of math functions with non-std:: versions to prevent linker errors
        NOTE: This can lead to correctness issues when running tests, since the correct version of the math function (exp/expf) might not get called.
        Plan is to remove this function once HIP supports std:: math function calls inside device code
    """
    output_string = input_string
    for func in MATH_TRANSPILATIONS:
      output_string = output_string.replace(r'{}('.format(func), '{}('.format(MATH_TRANSPILATIONS[func]))

    return output_string


def hip_header_magic(input_string):
    """If the file makes kernel builtin calls and does not include the cuda_runtime.h header,
    then automatically add an #include to match the "magic" includes provided by NVCC.
    TODO:
        Update logic to ignore cases where the cuda_runtime.h is included by another file.
    """

    # Copy the input.
    output_string = input_string

    # Check if one of the following headers is already included.
    headers = ["hip/hip_runtime.h", "hip/hip_runtime_api.h"]
    if any(re.search(r'#include ("{0}"|<{0}>)'.format(ext), output_string) for ext in headers):
        return output_string

    # Rough logic to detect if we're inside device code
    hasDeviceLogic = "hipLaunchKernelGGL" in output_string
    hasDeviceLogic += "__global__" in output_string
    hasDeviceLogic += "__shared__" in output_string
    hasDeviceLogic += re.search(r"[:]?[:]?\b(__syncthreads)\b(\w*\()", output_string) is not None

    # If device logic found, provide the necessary header.
    if hasDeviceLogic:
        output_string = '#include "hip/hip_runtime.h"\n' + input_string

    return output_string


def replace_extern_shared(input_string):
    """Match extern __shared__ type foo[]; syntax and use HIP_DYNAMIC_SHARED() MACRO instead.
       https://github.com/ROCm-Developer-Tools/HIP/blob/master/docs/markdown/hip_kernel_language.md#__shared__
    Example:
        "extern __shared__ char smemChar[];" => "HIP_DYNAMIC_SHARED( char, smemChar)"
        "extern __shared__ unsigned char smem[];" => "HIP_DYNAMIC_SHARED( unsigned char, my_smem)"
    """
    output_string = input_string
    output_string = re.sub(
        r"extern\s+([\w\(\)]+)?\s*__shared__\s+([\w:<>\s]+)\s+(\w+)\s*\[\s*\]\s*;",
        lambda inp: "HIP_DYNAMIC_SHARED({0} {1}, {2})".format(
            inp.group(1) or "", inp.group(2), inp.group(3)), output_string)

    return output_string


def disable_function(input_string, function, replace_style):
    """ Finds and disables a function in a particular file.

    If type(function) == List
        function - The signature of the function to disable.
            e.g. ["bool", "overlappingIndices", "(const Tensor& t)"]
            disables function -> "bool overlappingIndices(const Tensor& t)"

    If type(function) == String
        function - Disables the function by name only.
            e.g. "overlappingIndices"

    replace_style - The style to use when stubbing functions.
    """
# void (*)(hcrngStateMtgp32 *, int, float *, double, double)
    info = {
        "function_start": -1,
        "function_end": -1,
        "bracket_count": 0
    }

    STARTED = 0
    INSIDE_FUNCTION = 1
    BRACKET_COMPLETE = 2

    STATE = STARTED

    if type(function) == list:
        # Extract components from function signature.
        func_info = {
            "return_type": function[0].strip(),
            "function_name": function[1].strip(),
            "function_args": function[2].strip()
        }

        # Create function string to search for
        function_string = "{0}{1}{2}".format(
            func_info["return_type"],
            func_info["function_name"],
            func_info["function_args"]
        )

        # Find the starting position for the function
        info["function_start"] = input_string.find(function_string)
    else:
        # Automatically detect signature.
        the_match = re.search(r"(((.*) (\*)?)({0})(\([^{{)]*\)))\s*{{".format(
            function.replace("(", "\(").replace(")", "\)")), input_string)
        if the_match is None:
            return input_string

        func_info = {
            "return_type": the_match.group(2).strip(),
            "function_name": the_match.group(5).strip(),
            "function_args": the_match.group(6).strip(),
        }

        # Find the starting position for the function
        info["function_start"] = the_match.start()
        function_string = the_match.group(1)

    # The function can't be found anymore.
    if info["function_start"] == -1:
        return input_string

    # Find function block start.
    pos = info["function_start"] + len(function_string) - 1
    while pos < len(input_string) and STATE != BRACKET_COMPLETE:
        if input_string[pos] == "{":
            if STATE != INSIDE_FUNCTION:
                STATE = INSIDE_FUNCTION
                info["bracket_count"] = 1
            else:
                info["bracket_count"] += 1
        elif input_string[pos] == "}":
            info["bracket_count"] -= 1

            if info["bracket_count"] == 0 and STATE == INSIDE_FUNCTION:
                STATE = BRACKET_COMPLETE
                info["function_end"] = pos

        pos += 1

    # Never found the function end. Corrupted file!
    if STATE != BRACKET_COMPLETE:
        return input_string

    # Preprocess the source by removing the function.
    function_body = input_string[info["function_start"]:info["function_end"] + 1]

    # Remove the entire function body
    if replace_style == disablefuncmode.REMOVE:
        output_string = input_string.replace(function_body, "")

    # Stub the function based off its return type.
    elif replace_style == disablefuncmode.STUB:
        # void return type
        if func_info["return_type"] == "void" or func_info["return_type"] == "static void":
            stub = "{0}{{\n}}".format(function_string)
        # pointer return type
        elif "*" in func_info["return_type"]:
            stub = "{0}{{\nreturn {1};\n}}".format(function_string, "NULL")  # nullptr
        else:
            stub = "{0}{{\n{1} stub_var;\nreturn stub_var;\n}}".format(function_string, func_info["return_type"])

        output_string = input_string.replace(function_body, stub)

    # Add HIP Preprocessors.
    elif replace_style == disablefuncmode.HCC_MACRO:
        output_string = input_string.replace(
            function_body,
            "#if !defined(__HIP_PLATFORM_HCC__)\n{0}\n#endif".format(function_body))

    # Add HIP Preprocessors.
    elif replace_style == disablefuncmode.DEVICE_MACRO:
        output_string = input_string.replace(
            function_body,
            "#if !defined(__HIP_DEVICE_COMPILE__)\n{0}\n#endif".format(function_body))

    # Throw an exception at runtime.
    elif replace_style == disablefuncmode.EXCEPTION:
        stub = "{0}{{\n{1};\n}}".format(
            function_string,
            'throw std::runtime_error("The function {0} is not implemented.")'.format(
                function_string.replace("\n", " ")))
        output_string = input_string.replace(function_body, stub)

    elif replace_style == disablefuncmode.ASSERT:
        stub = "{0}{{\n{1};\n}}".format(
            function_string,
            'assert(0)')
        output_string = input_string.replace(function_body, stub)

    elif replace_style == disablefuncmode.EMPTYBODY:
        stub = "{0}{{\n;\n}}".format(function_string)
        output_string = input_string.replace(function_body, stub)
    return output_string


def get_hip_file_path(filepath, hipify_caffe2, hip_suffix, extensions_to_hip_suffix):
    """ Returns the new name of the hipified file """
    if not hipify_caffe2:
        return filepath

    dirpath, filename = os.path.split(filepath)
    filename_without_ext, ext = os.path.splitext(filename)

    if 'gpu' in filename_without_ext:
        filename_without_ext = filename_without_ext.replace('gpu', 'hip')
    else:
        filename_without_ext += '_hip'

    # extensions are either specified, or .cu
    if ext in extensions_to_hip_suffix or ext == '.cu':
        ext = '.' + hip_suffix

    return os.path.join(dirpath, 'hip', filename_without_ext + ext)


def is_caffe2_gpu_file(filepath):
    filename = os.path.basename(filepath)
    _, ext = os.path.splitext(filename)
    return ('gpu' in filename or ext in ['.cu', '.cuh']) and ('cudnn' not in filename)


def preprocessor(filepath, stats, hipify_caffe2, hip_suffix, extensions_to_hip_suffix):
    """ Executes the CUDA -> HIP conversion on the specified file. """
    fin_path = filepath
    with open(fin_path, 'r') as fin:
        output_source = fin.read()

    fout_path = get_hip_file_path(filepath, hipify_caffe2, hip_suffix, extensions_to_hip_suffix)
    if not os.path.exists(os.path.dirname(fout_path)):
        os.makedirs(os.path.dirname(fout_path))

    with open(fout_path, 'w') as fout:
        # Perform type, method, constant replacements
        for mapping in CUDA_TO_HIP_MAPPINGS:
            for cuda_type, value in mapping.items():
                # Extract relevant information
                hip_type = value[0]
                meta_data = value[1:]

                if constants.API_CAFFE2 in meta_data and not hipify_caffe2:
                    continue

                if output_source.find(cuda_type) > -1:
                    # Check if supported
                    if constants.HIP_UNSUPPORTED in meta_data:
                        stats["unsupported_calls"].append((cuda_type, filepath))

                if cuda_type in output_source:
                    if hipify_caffe2:
                        pattern = r'({0})'.format(re.escape(cuda_type))
                    else:
                        pattern = r'(\b{0}\b)'.format(re.escape(cuda_type))
                    output_source = re.sub(pattern, hip_type, output_source)

        # Perform Kernel Launch Replacements
        output_source = processKernelLaunches(output_source, stats)

        # Disable asserts
        # if not filepath.endswith("THCGeneral.h.in"):
        #    output_source = disable_asserts(output_source)

        # Replace std:: with non-std:: versions
        if re.search(r"\.cu$", filepath) or re.search(r"\.cuh$", filepath):
          output_source = replace_math_functions(output_source)

        # Replace __forceinline__ with inline
        output_source = replace_forceinline(output_source)

        # Include header if device code is contained.
        output_source = hip_header_magic(output_source)

        # Replace the extern __shared__
        output_source = replace_extern_shared(output_source)

        fout.write(output_source)


def file_specific_replacement(filepath, search_string, replace_string, strict=False):
    with openf(filepath, "r+") as f:
        contents = f.read()
        if strict:
            contents = re.sub(r'\b({0})\b'.format(re.escape(search_string)), lambda x: replace_string, contents)
        else:
            contents = contents.replace(search_string, replace_string)
        f.seek(0)
        f.write(contents)
        f.truncate()


def file_add_header(filepath, header):
    with openf(filepath, "r+") as f:
        contents = f.read()
        if header[0] != "<" and header[-1] != ">":
            header = '"{0}"'.format(header)
        contents = ('#include {0} \n'.format(header)) + contents
        f.seek(0)
        f.write(contents)
        f.truncate()


def fix_static_global_kernels(in_txt):
    """Static global kernels in HIP results in a compilation error."""
    in_txt = in_txt.replace(" __global__ static", "__global__")
    return in_txt


def get_kernel_template_params(the_file, KernelDictionary, template_param_to_value):
    """Scan for __global__ kernel definitions then extract its argument types, and static cast as necessary"""
    # Read the kernel file.
    with openf(the_file, "r") as f:
        # Extract all kernels with their templates inside of the file
        string = f.read()

        get_kernel_definitions = [k for k in re.finditer(
            r"(template[ ]*<(.*)>\n.*\n?)?__global__ void[\n| ](\w+(\(.*\))?)\(", string)]

        # Create new launch syntax
        for kernel in get_kernel_definitions:
            template_arguments = kernel.group(2).split(",") if kernel.group(2) else ""
            template_arguments = [x.replace("template", "").replace("typename", "").strip() for x in template_arguments]
            kernel_name = kernel.group(3)

            # Kernel starting / ending positions
            arguments_start = kernel.end()
            argument_start_pos = arguments_start
            current_position = arguments_start + 1

            # Search for final parenthesis
            arguments = []
            closures = {"(": 1, "<": 0}
            while current_position < len(string):
                if string[current_position] == "(":
                    closures["("] += 1
                elif string[current_position] == ")":
                    closures["("] -= 1
                elif string[current_position] == "<":
                    closures["<"] += 1
                elif string[current_position] == ">":
                    closures["<"] -= 1

                # Finished all arguments
                if closures["("] == 0 and closures["<"] == 0:
                    # Add final argument
                    arguments.append({"start": argument_start_pos, "end": current_position})
                    break

                # Finished current argument
                if closures["("] == 1 and closures["<"] == 0 and string[current_position] == ",":
                    arguments.append({"start": argument_start_pos, "end": current_position})
                    argument_start_pos = current_position + 1

                current_position += 1

            # Grab range of arguments
            arguments_string = [string[arg["start"]: arg["end"]] for arg in arguments]

            argument_types = [None] * len(arguments_string)
            for arg_idx, arg in enumerate(arguments_string):
                for i in range(len(arg) - 1, -1, -1):
                    if arg[i] == "*" or arg[i] == " ":
                        argument_types[arg_idx] = re.sub(' +', ' ', arg[0:i + 1].replace("\n", "").strip())
                        break

            # Here we'll use the template_param_to_value dictionary to replace the PyTorch / Caffe2.
            if len(template_arguments) == 1 and template_arguments[0].strip() in template_param_to_value.keys():
                # Updates kernel
                kernel_with_template = "{0}<{1}>".format(
                    kernel_name, template_param_to_value[template_arguments[0].strip()])
            else:
                kernel_with_template = kernel_name
            formatted_args = {}
            for idx, arg_type in enumerate(argument_types):
                formatted_args[idx] = arg_type

            KernelDictionary[kernel_name] = {"kernel_with_template": kernel_with_template, "arg_types": formatted_args}

        # Extract generated kernels
        # curandStateMtgp32 *state, int size, T *result, ARG1
        for kernel in re.finditer(r"GENERATE_KERNEL([1-9])\((.*)\)", string):
            kernel_gen_type = int(kernel.group(1))
            kernel_name = kernel.group(2).split(",")[0]
            kernel_params = kernel.group(2).split(",")[1:]

            if kernel_gen_type == 1:
                kernel_args = {1: "int", 2: "{0} *".format(kernel_params[0]), 3: kernel_params[1]}

            if kernel_gen_type == 2:
                kernel_args = {1: "int", 2: "{0} *".format(kernel_params[0]), 3: kernel_params[1], 4: kernel_params[2]}

            # Argument at position 1 should be int
            KernelDictionary[kernel_name] = {"kernel_with_template": kernel_name, "arg_types": kernel_args}


def disable_unsupported_function_call(function, input_string, replacement):
    """Disables calls to an unsupported HIP function"""
    # Prepare output string
    output_string = input_string

    # Find all calls to the function
    calls = re.finditer(r"\b{0}\b".format(re.escape(function)), input_string)

    # Do replacements
    for call in calls:
        start = call.start()
        end = call.end()

        pos = end
        started_arguments = False
        bracket_count = 0
        while pos < len(input_string):
            if input_string[pos] == "(":
                if started_arguments is False:
                    started_arguments = True
                    bracket_count = 1
                else:
                    bracket_count += 1
            elif input_string[pos] == ")" and started_arguments:
                bracket_count -= 1

            if bracket_count == 0 and started_arguments:
                # Finished!
                break
            pos += 1

        function_call = input_string[start:pos + 1]
        output_string = output_string.replace(function_call, replacement)

    return output_string


def disable_module(input_file):
    """Disable a module entirely except for header includes."""
    with openf(input_file, "r+") as f:
        txt = f.read()
        last = list(re.finditer(r"#include .*\n", txt))[-1]
        end = last.end()

        disabled = "{0}#if !defined(__HIP_PLATFORM_HCC__)\n{1}\n#endif".format(txt[0:end], txt[end:])

        f.seek(0)
        f.write(disabled)
        f.truncate()


def extract_arguments(start, string):
    """ Return the list of arguments in the upcoming function parameter closure.
        Example:
        string (input): '(blocks, threads, 0, THCState_getCurrentStream(state))'
        arguments (output):
            '[{'start': 1, 'end': 7},
            {'start': 8, 'end': 16},
            {'start': 17, 'end': 19},
            {'start': 20, 'end': 53}]'
    """

    arguments = []
    closures = {
        "<": 0,
        "(": 0
    }
    current_position = start
    argument_start_pos = current_position + 1

    # Search for final parenthesis
    while current_position < len(string):
        if string[current_position] == "(":
            closures["("] += 1
        elif string[current_position] == ")":
            closures["("] -= 1
        elif string[current_position] == "<":
            closures["<"] += 1
        elif string[current_position] == ">" and string[current_position - 1] != "-" and closures["<"] > 0:
            closures["<"] -= 1

        # Finished all arguments
        if closures["("] == 0 and closures["<"] == 0:
            # Add final argument
            arguments.append({"start": argument_start_pos, "end": current_position})
            break

        # Finished current argument
        if closures["("] == 1 and closures["<"] == 0 and string[current_position] == ",":
            arguments.append({"start": argument_start_pos, "end": current_position})
            argument_start_pos = current_position + 1

        current_position += 1

    return arguments


# Add static_cast to ensure that the type of kernel arguments matches that in the corresponding kernel definition
def add_static_casts(filepath, KernelTemplateParams):
    """Add static casts to kernel launches in order to keep launch argument types and kernel definition types matching.

       Example:
           old_kernel_launch: ' createBatchGemmBuffer, grid, block, 0, THCState_getCurrentStream(state),
              (const scalar_t**)d_result, THCTensor_(data)(state, ra__),
              ra__->stride[0], num_batches'

           new_kernel_launch: ' createBatchGemmBuffer, grid, block, 0, THCState_getCurrentStream(state),
              (const scalar_t**)d_result, THCTensor_(data)(state, ra__),
              static_cast<int64_t>(ra__->stride[0]), static_cast<int64_t>(num_batches)'
    """

    # These are the types that generally have issues with hipKernelLaunch.
    static_cast_types = ["int", "const int", "int64_t", "THCIndex_t *",
                         "const int *", "ptrdiff_t", "long", "const int64_t*", "int64_t *", "double"]

    with openf(filepath, "r+") as fileobj:
        input_source = fileobj.read()
        new_output_source = input_source
        for kernel in re.finditer("hipLaunchKernelGGL\(", input_source):
            arguments = extract_arguments(kernel.end() - 1, input_source)

            # Check if we have templating + static_cast information
            argument_strings = [input_source[arg["start"]:arg["end"]] for arg in arguments]
            original_kernel_name_with_template = argument_strings[0].strip()
            kernel_name = original_kernel_name_with_template.split("<")[0].strip()
            ignore = ["upscale"]
            if kernel_name in KernelTemplateParams and kernel_name not in ignore:
                # Add template to the kernel
                # Add static_casts to relevant arguments
                kernel_name_with_template = KernelTemplateParams[kernel_name]["kernel_with_template"]
                argument_types = KernelTemplateParams[kernel_name]["arg_types"]

                # The first 5 arguments are simply (function, number blocks, dimension blocks, shared memory, stream)
                # old_kernel_launch_parameters - will contain the actual arguments to the function itself.
                old_kernel_launch_parameters = input_source[arguments[5]["start"]:arguments[-1]["end"]]
                new_kernel_launch_parameters = old_kernel_launch_parameters

                # full_old_kernel_launch - will contain the entire kernel launch closure.
                full_old_kernel_launch = input_source[arguments[0]["start"]:arguments[-1]["end"]]
                full_new_kernel_launch = full_old_kernel_launch

                kernel_params = argument_strings[5:]
                for arg_idx, arg in enumerate(kernel_params):
                    if arg_idx in argument_types:
                        the_type = argument_types[arg_idx]
                        the_arg = arg.replace("\n", "").replace("\\", "").strip()
                        # Not all types have issues with the hipLaunchKernelGGL.
                        if the_type in static_cast_types:
                            static_argument = "static_cast<{0}>({1})".format(the_type, the_arg)

                            def replace_arg(match):
                                return match.group(1) + static_argument + match.group(3)
                            # Update to static_cast, account for cases where argument is at start/end of string
                            new_kernel_launch_parameters = re.sub(r'(^|\W)({0})(\W|$)'.format(
                                re.escape(the_arg)), replace_arg, new_kernel_launch_parameters)

                # replace kernel arguments in full kernel launch arguments w/ static_cast ones
                full_new_kernel_launch = full_new_kernel_launch.replace(
                    old_kernel_launch_parameters, new_kernel_launch_parameters)

                # PyTorch Specific: Add template type
                # Here the template value will be resolved from <scalar_t> to <Dtype>.
                if "THCUNN" in filepath.split("/") and "generic" not in filepath.split("/"):
                    kernel_name_with_template = kernel_name_with_template.replace("<scalar_t>", "<Dtype>")

                full_new_kernel_launch = re.sub(r'\b{0}\b'.format(re.escape(original_kernel_name_with_template)),
                                                lambda x: kernel_name_with_template, full_new_kernel_launch)

                # Replace Launch
                new_output_source = new_output_source.replace(full_old_kernel_launch, full_new_kernel_launch)

        # Overwrite file contents
        fileobj.seek(0)
        fileobj.write(new_output_source)
        fileobj.truncate()
        fileobj.flush()

        # Flush to disk
        os.fsync(fileobj)


def str2bool(v):
    """ArgumentParser doesn't support type=bool. Thus, this helper method will convert
    from possible string types to True / False."""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    """Example invocation

    python hipify.py --project-directory /home/myproject/ --extensions cu cuh h cpp --output-directory /home/gains/
    """

    parser = argparse.ArgumentParser(
        description="The Python Hipify Script.")

    # required argument: user has to specify it before proceed
    # this is to avoid accidentally hipify files that are not intended
    parser.add_argument(
        '--project-directory',
        type=str,
        default=os.getcwd(),
        help="The root of the project.",
        required=True)

    parser.add_argument(
        '--show-detailed',
        type=str2bool,
        default=False,
        help="Show detailed summary of the hipification process.",
        required=False)

    parser.add_argument(
        '--extensions',
        nargs='+',
        default=[".cu", ".cuh", ".c", ".cpp", ".h", ".in", ".hpp"],
        help="The extensions for files to run the Hipify script over.",
        required=False)

    parser.add_argument(
        '--output-directory',
        type=str,
        default="",
        help="The directory to store the hipified project.",
        required=False)

    parser.add_argument(
        '--includes',
        nargs='+',
        default=[],
        help="The patterns of files that should be included.",
        required=False)

    parser.add_argument(
        '--json-settings',
        type=str,
        default="",
        help="The json file storing information for disabled functions and modules.",
        required=False)

    parser.add_argument(
        '--add-static-casts',
        type=str2bool,
        default=False,
        help="Whether to automatically add static_casts to kernel arguments.",
        required=False)

    parser.add_argument(
        '--hipify_caffe2',
        type=str2bool,
        default=False,
        help="Whether to hipify caffe2 source",
        required=False)

    parser.add_argument(
        '--ignores',
        nargs='+',
        default=[],
        help="list of patterns to ignore for hipifying")

    parser.add_argument(
        '--show-progress',
        type=str2bool,
        default=True,
        help="Whether to show the progress bar during the transpilation proecss.",
        required=False)

    parser.add_argument(
        '--hip-suffix',
        type=str,
        default='cc',
        help="The suffix for the hipified files",
        required=False)

    parser.add_argument(
        '--extensions-to-hip-suffix',
        type=str,
        default=('cc', 'cu'),
        help="Specify the file extensions whose suffix will be changed "
        + "to the new hip suffix",
        required=False)

    args = parser.parse_args()

    hipify(
        project_directory=args.project_directory,
        show_detailed=args.show_detailed,
        extensions=args.extensions,
        output_directory=args.output_directory,
        includes=args.includes,
        json_settings=args.json_settings,
        add_static_casts_option=args.add_static_casts,
        hipify_caffe2=args.hipify_caffe2,
        ignores=args.ignores,
        show_progress=args.show_progress,
        hip_suffix=args.hip_suffix,
        extensions_to_hip_suffix=args.extensions_to_hip_suffix)


def hipify(
    project_directory,
    show_detailed=False,
    extensions=(".cu", ".cuh", ".c", ".cpp", ".h", ".in", ".hpp"),
    output_directory="",
    includes=(),
    json_settings="",
    add_static_casts_option=False,
    hipify_caffe2=False,
    ignores=(),
    show_progress=True,
    hip_suffix='cc',
    extensions_to_hip_suffix=(),
):
    if project_directory == "":
        project_directory = os.getcwd()

    # Verify the project directory exists.
    if not os.path.exists(project_directory):
        print("The project folder specified does not exist.")
        sys.exit(1)

    # If no output directory, provide a default one.
    if not output_directory:
        project_directory.rstrip("/")
        output_directory = project_directory + "_amd"

    # Copy from project directory to output directory if not done already.
    if not os.path.exists(output_directory):
        shutil.copytree(project_directory, output_directory)

    # Open JSON file with disable information.
    if json_settings != "":
        with openf(json_settings, "r") as f:
            json_data = json.load(f)

        # Disable functions in certain files according to JSON description
        for disable_info in json_data["disabled_functions"]:
            filepath = os.path.join(output_directory, disable_info["path"])
            if "functions" in disable_info:
                functions = disable_info["functions"]
            else:
                functions = disable_info.get("functions", [])

            if "non_hip_functions" in disable_info:
                non_hip_functions = disable_info["non_hip_functions"]
            else:
                non_hip_functions = disable_info.get("non_hip_functions", [])

            if "non_device_functions" in disable_info:
                not_on_device_functions = disable_info["non_device_functions"]
            else:
                not_on_device_functions = disable_info.get("non_device_functions", [])

            with openf(filepath, "r+") as f:
                txt = f.read()
                for func in functions:
                    # TODO - Find fix assertions in HIP for device code.
                    txt = disable_function(txt, func, disablefuncmode.ASSERT)

                for func in non_hip_functions:
                    # Disable this function on HIP stack
                    txt = disable_function(txt, func, disablefuncmode.HCC_MACRO)

                for func in not_on_device_functions:
                    # Disable this function when compiling on Device
                    txt = disable_function(txt, func, disablefuncmode.DEVICE_MACRO)

                f.seek(0)
                f.write(txt)
                f.truncate()

        # Disable modules
        disable_modules = json_data["disabled_modules"]
        for module in disable_modules:
            disable_module(os.path.join(output_directory, module))

        # Disable unsupported HIP functions
        for disable in json_data["disable_unsupported_hip_calls"]:
            filepath = os.path.join(output_directory, disable["path"])
            if "functions" in disable:
                functions = disable["functions"]
            else:
                functions = disable.get("functions", [])

            if "constants" in disable:
                constants = disable["constants"]
            else:
                constants = disable.get("constants", [])

            if "s_constants" in disable:
                s_constants = disable["s_constants"]
            else:
                s_constants = disable.get("s_constants", [])

            if not os.path.exists(filepath):
                print("\n" + bcolors.WARNING + "JSON Warning: File {0} does not exist.".format(filepath) + bcolors.ENDC)
                continue

            with openf(filepath, "r+") as f:
                txt = f.read()

                # Disable HIP Functions
                for func in functions:
                    txt = disable_unsupported_function_call(func, txt, functions[func])

                # Disable Constants w\ Boundary.
                for const in constants:
                    txt = re.sub(r"\b{0}\b".format(re.escape(const)), constants[const], txt)

                # Disable Constants
                for s_const in s_constants:
                    txt = txt.replace(s_const, s_constants[s_const])

                # Save Changes
                f.seek(0)
                f.write(txt)
                f.truncate()

    all_files = list(matched_files_iter(output_directory, includes=includes,
                                        ignores=ignores, extensions=extensions,
                                        hipify_caffe2=hipify_caffe2))

    # Start Preprocessor
    preprocess(
        all_files,
        show_detailed=show_detailed,
        show_progress=show_progress,
        hipify_caffe2=hipify_caffe2,
        hip_suffix=hip_suffix,
        extensions_to_hip_suffix=extensions_to_hip_suffix)

    # Extract all of the kernel parameter and template type information.
    if add_static_casts_option:
        KernelTemplateParams = {}
        # ignore caffe2/utils/math_gpu.cu for static_casts as PowKernel
        # conflicts with kernel in caffe2/operators/pow_op.cu.
        # Need to figure out a long-term solution.
        for filepath in all_files:
            if "math_gpu.cu" not in os.path.basename(filepath):
                get_kernel_template_params(
                    filepath,
                    KernelTemplateParams,
                    CAFFE2_TEMPLATE_MAP if hipify_caffe2 else PYTORCH_TEMPLATE_MAP)

        # Execute the Clang Tool to Automatically add static casts
        for filepath in all_files:
            add_static_casts(
                get_hip_file_path(
                    filepath,
                    hipify_caffe2=hipify_caffe2,
                    hip_suffix=hip_suffix,
                    extensions_to_hip_suffix=extensions_to_hip_suffix),
                KernelTemplateParams)


if __name__ == '__main__':
    main()
