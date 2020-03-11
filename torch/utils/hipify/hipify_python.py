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

from . import constants
from .cuda_to_hip_mappings import CUDA_TO_HIP_MAPPINGS
from .cuda_to_hip_mappings import MATH_TRANSPILATIONS

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


def matched_files_iter(root_path, includes=('*',), ignores=(), extensions=(), out_of_place_only=False):
    def _fnmatch(filepath, patterns):
        return any(fnmatch.fnmatch(filepath, pattern) for pattern in patterns)

    def match_extensions(filename):
        """Helper method to see if filename ends with certain extension"""
        return any(filename.endswith(e) for e in extensions)

    exact_matches = set(includes)

    # This is a very rough heuristic; really, we want to avoid scanning
    # any file which is not checked into source control, but this script
    # needs to work even if you're in a Git or Hg checkout, so easier to
    # just blacklist the biggest time sinks that won't matter in the
    # end.
    for (abs_dirpath, dirs, filenames) in os.walk(root_path, topdown=True):
        rel_dirpath = os.path.relpath(abs_dirpath, root_path)
        if rel_dirpath == '.':
            # Blah blah blah O(n) blah blah
            if ".git" in dirs:
                dirs.remove(".git")
            if "build" in dirs:
                dirs.remove("build")
            if "third_party" in dirs:
                dirs.remove("third_party")
        for filename in filenames:
            filepath = os.path.join(rel_dirpath, filename)
            # We respect extensions, UNLESS you wrote the entire
            # filename verbatim, in which case we always accept it
            if (
                _fnmatch(filepath, includes)
                and (not _fnmatch(filepath, ignores))
                and (match_extensions(filepath) or filepath in exact_matches)
            ):
                if not is_pytorch_file(filepath) and not is_caffe2_gpu_file(filepath):
                    continue
                if out_of_place_only and not is_out_of_place(filepath):
                    continue
                yield filepath


def preprocess(
        output_directory,
        all_files,
        show_detailed=False,
        show_progress=True,
        hip_clang_launch=False,
        is_pytorch_extension=False):
    """
    Call preprocessor on selected files.

    Arguments)
        show_detailed - Show a detailed summary of the transpilation process.
    """

    # Preprocessing statistics.
    stats = {"unsupported_calls": [], "kernel_launches": []}

    for filepath in all_files:
        result = preprocessor(output_directory, filepath, stats, hip_clang_launch, is_pytorch_extension)
        # Show what happened
        if show_progress:
            print(
                filepath, "->",
                get_hip_file_path(filepath), result)

    print(bcolors.OKGREEN + "Successfully preprocessed all matching files." + bcolors.ENDC, file=sys.stderr)

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


RE_KERNEL_LAUNCH = re.compile(r'([ ]+)(detail?)::[ ]+\\\n[ ]+')


def processKernelLaunches(string, stats):
    """ Replace the CUDA style Kernel launches with the HIP style kernel launches."""
    # Concat the namespace with the kernel names. (Find cleaner way of doing this later).
    string = RE_KERNEL_LAUNCH.sub(lambda inp: "{0}{1}::".format(inp.group(1), inp.group(2)), string)

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
                if string[i].isalnum() or string[i] in {'(', ')', '_', ':', '#'}:
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

    # Grab positional ranges of all kernel launches
    get_kernel_positions = list(find_kernel_bounds(string))
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


RE_ASSERT = re.compile(r"\bassert[ ]*\(")


def replace_math_functions(input_string):
    """FIXME: Temporarily replace std:: invocations of math functions
        with non-std:: versions to prevent linker errors NOTE: This
        can lead to correctness issues when running tests, since the
        correct version of the math function (exp/expf) might not get
        called.  Plan is to remove this function once HIP supports
        std:: math function calls inside device code

    """
    output_string = input_string
    for func in MATH_TRANSPILATIONS:
        output_string = output_string.replace(r'{}('.format(func), '{}('.format(MATH_TRANSPILATIONS[func]))

    return output_string


RE_SYNCTHREADS = re.compile(r"[:]?[:]?\b(__syncthreads)\b(\w*\()")


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
    hasDeviceLogic += RE_SYNCTHREADS.search(output_string) is not None

    # If device logic found, provide the necessary header.
    if hasDeviceLogic:
        output_string = '#include "hip/hip_runtime.h"\n' + input_string

    return output_string


RE_EXTERN_SHARED = re.compile(r"extern\s+([\w\(\)]+)?\s*__shared__\s+([\w:<>\s]+)\s+(\w+)\s*\[\s*\]\s*;")


def replace_extern_shared(input_string):
    """Match extern __shared__ type foo[]; syntax and use HIP_DYNAMIC_SHARED() MACRO instead.
       https://github.com/ROCm-Developer-Tools/HIP/blob/master/docs/markdown/hip_kernel_language.md#__shared__
    Example:
        "extern __shared__ char smemChar[];" => "HIP_DYNAMIC_SHARED( char, smemChar)"
        "extern __shared__ unsigned char smem[];" => "HIP_DYNAMIC_SHARED( unsigned char, my_smem)"
    """
    output_string = input_string
    output_string = RE_EXTERN_SHARED.sub(
        lambda inp: "HIP_DYNAMIC_SHARED({0} {1}, {2})".format(
            inp.group(1) or "", inp.group(2), inp.group(3)), output_string)

    return output_string


def get_hip_file_path(filepath):
    """
    Returns the new name of the hipified file
    """
    # At the moment, some files are HIPified in place.  The predicate
    # is_out_of_place tells us if this is the case or not.
    if not is_out_of_place(filepath):
        return filepath

    dirpath, filename = os.path.split(filepath)
    root, ext = os.path.splitext(filename)

    # Here's the plan:
    #
    # In general, we need to disambiguate the HIPified filename so that
    # it gets a different name from the original Caffe2 filename, so
    # that we don't overwrite the original file.  (Additionally,
    # hcc historically had a bug where if you had two files with
    # the same basename, they would clobber each other.)
    #
    # There's a lot of different naming conventions across PyTorch
    # and Caffe2, but the general recipe is to convert occurrences
    # of cuda/gpu to hip, and add hip if there are no occurrences
    # of cuda/gpu anywhere.
    #
    # Concretely, we do the following:
    #
    #   - If there is a directory component named "cuda", replace
    #     it with "hip", AND
    #
    #   - If the file name contains "CUDA", replace it with "HIP", AND
    #
    # If NONE of the above occurred, then insert "hip" in the file path
    # as the direct parent folder of the file
    #
    # Furthermore, ALWAYS replace '.cu' with '.hip', because those files
    # contain CUDA kernels that needs to be hipified and processed with
    # hcc compiler
    #
    # This isn't set in stone; we might adjust this to support other
    # naming conventions.

    if ext == '.cu':
        ext = '.hip'

    orig_dirpath = dirpath

    dirpath = dirpath.replace('cuda', 'hip')
    dirpath = dirpath.replace('THC', 'THH')

    root = root.replace('cuda', 'hip')
    root = root.replace('CUDA', 'HIP')
    # Special case to handle caffe2/core/THCCachingAllocator
    if dirpath != "caffe2/core":
        root = root.replace('THC', 'THH')

    if dirpath == orig_dirpath:
        dirpath = os.path.join(dirpath, 'hip')

    return os.path.join(dirpath, root + ext)


def is_out_of_place(filepath):
    if filepath.startswith("torch/"):
        return False
    if filepath.startswith("tools/autograd/templates/"):
        return False
    return True


# Keep this synchronized with includes/ignores in build_amd.py
def is_pytorch_file(filepath):
    if filepath.startswith("aten/"):
        if filepath.startswith("aten/src/ATen/core/"):
            return False
        return True
    if filepath.startswith("torch/"):
        return True
    if filepath.startswith("tools/autograd/templates/"):
        return True
    return False


def is_caffe2_gpu_file(filepath):
    if filepath.startswith("c10/cuda"):
        return True
    filename = os.path.basename(filepath)
    _, ext = os.path.splitext(filename)
    return ('gpu' in filename or ext in ['.cu', '.cuh']) and ('cudnn' not in filename)


# Cribbed from https://stackoverflow.com/questions/42742810/speed-up-millions-of-regex-replacements-in-python-3/42789508#42789508
class Trie():
    """Regex::Trie in Python. Creates a Trie out of a list of words. The trie can be exported to a Regex pattern.
    The corresponding Regex should match much faster than a simple Regex union."""

    def __init__(self):
        self.data = {}

    def add(self, word):
        ref = self.data
        for char in word:
            ref[char] = char in ref and ref[char] or {}
            ref = ref[char]
        ref[''] = 1

    def dump(self):
        return self.data

    def quote(self, char):
        return re.escape(char)

    def _pattern(self, pData):
        data = pData
        if "" in data and len(data.keys()) == 1:
            return None

        alt = []
        cc = []
        q = 0
        for char in sorted(data.keys()):
            if isinstance(data[char], dict):
                try:
                    recurse = self._pattern(data[char])
                    alt.append(self.quote(char) + recurse)
                except Exception:
                    cc.append(self.quote(char))
            else:
                q = 1
        cconly = not len(alt) > 0

        if len(cc) > 0:
            if len(cc) == 1:
                alt.append(cc[0])
            else:
                alt.append('[' + ''.join(cc) + ']')

        if len(alt) == 1:
            result = alt[0]
        else:
            result = "(?:" + "|".join(alt) + ")"

        if q:
            if cconly:
                result += "?"
            else:
                result = "(?:%s)?" % result
        return result

    def pattern(self):
        return self._pattern(self.dump())


CAFFE2_TRIE = Trie()
CAFFE2_MAP = {}
PYTORCH_TRIE = Trie()
PYTORCH_MAP = {}
for mapping in CUDA_TO_HIP_MAPPINGS:
    for src, value in mapping.items():
        dst = value[0]
        meta_data = value[1:]
        if constants.API_CAFFE2 not in meta_data:
            PYTORCH_TRIE.add(src)
            PYTORCH_MAP[src] = dst
        if constants.API_PYTORCH not in meta_data:
            CAFFE2_TRIE.add(src)
            CAFFE2_MAP[src] = dst
RE_CAFFE2_PREPROCESSOR = re.compile(CAFFE2_TRIE.pattern())
RE_PYTORCH_PREPROCESSOR = re.compile(r'(?<=\W)({0})(?=\W)'.format(PYTORCH_TRIE.pattern()))

RE_QUOTE_HEADER = re.compile(r'#include "([^"]+)"')
RE_ANGLE_HEADER = re.compile(r'#include <([^>]+)>')
RE_THC_GENERIC_FILE = re.compile(r'#define THC_GENERIC_FILE "([^"]+)"')
RE_CU_SUFFIX = re.compile(r'\.cu\b')  # be careful not to pick up .cuh

def preprocessor(output_directory, filepath, stats, hip_clang_launch, is_pytorch_extension):
    """ Executes the CUDA -> HIP conversion on the specified file. """
    fin_path = os.path.join(output_directory, filepath)
    with open(fin_path, 'r') as fin:
        output_source = fin.read()

    fout_path = os.path.join(output_directory, get_hip_file_path(filepath))
    if not os.path.exists(os.path.dirname(fout_path)):
        os.makedirs(os.path.dirname(fout_path))

    # unsupported_calls statistics reporting is broken atm
    def pt_repl(m):
        return PYTORCH_MAP[m.group(0)]

    if is_pytorch_extension:
        output_source = RE_PYTORCH_PREPROCESSOR.sub(pt_repl, output_source)
    else:
        if is_pytorch_file(filepath):
            output_source = RE_PYTORCH_PREPROCESSOR.sub(pt_repl, output_source)
        else:
            def c2_repl(m):
                return CAFFE2_MAP[m.group(0)]
            output_source = RE_CAFFE2_PREPROCESSOR.sub(c2_repl, output_source)

    # Header rewrites
    def mk_repl(templ):
        def repl(m):
            f = m.group(1)
            if (
                f.startswith("ATen/cuda")
                or f.startswith("ATen/native/cuda")
                or f.startswith("ATen/native/quantized/cuda")
                or f.startswith("ATen/native/sparse/cuda")
                or f.startswith("THC/")
                or f.startswith("THCUNN/")
                or (f.startswith("THC") and not f.startswith("THCP"))
            ):
                return templ.format(get_hip_file_path(m.group(1)))
            return m.group(0)
        return repl
    output_source = RE_QUOTE_HEADER.sub(mk_repl('#include "{0}"'), output_source)
    output_source = RE_ANGLE_HEADER.sub(mk_repl('#include <{0}>'), output_source)
    output_source = RE_THC_GENERIC_FILE.sub(mk_repl('#define THC_GENERIC_FILE "{0}"'), output_source)

    # CMakeLists.txt rewrites
    if filepath.endswith('CMakeLists.txt'):
        output_source = output_source.replace('CUDA', 'HIP')
        output_source = output_source.replace('THC', 'THH')
        output_source = RE_CU_SUFFIX.sub('.hip', output_source)

    # Perform Kernel Launch Replacements
    if not hip_clang_launch:
        output_source = processKernelLaunches(output_source, stats)

    # Replace std:: with non-std:: versions
    if filepath.endswith(".cu") or filepath.endswith(".cuh"):
        output_source = replace_math_functions(output_source)

    # Include header if device code is contained.
    output_source = hip_header_magic(output_source)

    # Replace the extern __shared__
    output_source = replace_extern_shared(output_source)

    do_write = True
    if os.path.exists(fout_path):
        with open(fout_path, 'r') as fout_old:
            do_write = fout_old.read() != output_source
    if do_write:
        with open(fout_path, 'w') as fout:
            fout.write(output_source)
        return "ok"
    else:
        return "skipped"

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


RE_INCLUDE = re.compile(r"#include .*\n")


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


def str2bool(v):
    """ArgumentParser doesn't support type=bool. Thus, this helper method will convert
    from possible string types to True / False."""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def hipify(
    project_directory,
    show_detailed=False,
    extensions=(".cu", ".cuh", ".c", ".cc", ".cpp", ".h", ".in", ".hpp"),
    output_directory="",
    includes=(),
    out_of_place_only=False,
    ignores=(),
    show_progress=True,
    hip_clang_launch=False,
    is_pytorch_extension=False,
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

    all_files = list(matched_files_iter(output_directory, includes=includes,
                                        ignores=ignores, extensions=extensions,
                                        out_of_place_only=out_of_place_only))

    # Start Preprocessor
    preprocess(
        output_directory,
        all_files,
        show_detailed=show_detailed,
        show_progress=show_progress,
        hip_clang_launch=hip_clang_launch,
        is_pytorch_extension=is_pytorch_extension)
