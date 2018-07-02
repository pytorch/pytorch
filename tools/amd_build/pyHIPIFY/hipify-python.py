#!/usr/bin/python
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

import argparse
import constants
import re
import shutil
import sys
import os
import yaml

from functools import reduce
from cuda_to_hip_mappings import CUDA_TO_HIP_MAPPINGS


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


def update_progress_bar(total, progress):
    """
    Displays and updates a console progress bar.
    """
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


def filename_ends_with_extension(filename, extensions):
    """Helper method to see if filename ends with certain extension"""
    return reduce(lambda result, ext: filename.endswith("." + ext) or result, extensions, False)


def inside_included_directories(dirpath, rootpath, include_dirs):
    """Helper method to see if filename within included directories"""
    return reduce(lambda result, included_directory: re.match(r'(%s)\b' % os.path.join(rootpath, included_directory), dirpath) or result, include_dirs, None)


def walk_over_directory(rootpath, extensions, show_detailed=False, include_dirs=None):
    """
    Recursively walk over directory and call preprocessor on selected files.

    Arguments)
        extensions - A plist of file extensions ['cu', 'cuh', ..]

        include_dirs - Directories under the rootpath that should be included in the walk.

        show_detailed - Show a detailed summary of the transpilation process.
    """

    # Default argument for excluded directories.
    if include_dirs is None:
        include_dirs = []

    # Compute the total number of files to be traversed.
    total_files = 0
    for (dirpath, _dirnames, filenames) in os.walk(rootpath):
        if inside_included_directories(dirpath, rootpath, include_dirs):
            for filename in filenames:
                total_files += filename_ends_with_extension(filename, extensions)

    current_file = 0

    # Preprocessing statistics.
    stats = {"unsupported_calls": [], "kernel_launches": []}

    # Begin traversing the files.
    for (dirpath, _dirnames, filenames) in os.walk(rootpath, topdown=True):
        # Check if file ends with a valid extensions
        if not inside_included_directories(dirpath, rootpath, include_dirs):
            continue

        for filename in filenames:
            if filename_ends_with_extension(filename, extensions):
                # Construct the file's full path
                filepath = os.sep.join([dirpath, filename])

                # Execute the preprocessor on the specified file.
                preprocessor(filepath, stats)

                # Update the progress
                print(os.path.join(dirpath, filename))
                update_progress_bar(total_files, current_file)

                current_file += 1

    print(bcolors.OKGREEN + "Successfully preprocessed all matching files." + bcolors.ENDC)

    # Show detailed summary
    if show_detailed:
        compute_stats(stats, show_detailed)


def compute_stats(stats):
    unsupported_calls = set(cuda_call for (cuda_call, _filepath) in stats["unsupported_calls"])

    # Print the number of unsupported calls
    print("Total number of unsupported CUDA function calls: %d" % (len(unsupported_calls)))

    # Print the list of unsupported calls
    print(", ".join(unsupported_calls))

    # Print the number of kernel launches
    print("\nTotal number of replaced kernel launches: %d" % (len(stats["kernel_launches"])))


def processKernelLaunches(string, stats):
    """ Replace the CUDA style Kernel launches with the HIP style kernel launches."""
    # Concat the namespace with the kernel names. (Find cleaner way of doing this later).
    string = re.sub(r'([ ]+)(detail+)::[ ]+\\\n[ ]+', lambda inp: "%s%s::" % (inp.group(1), inp.group(2)), string)

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
                if string[i] == "(" or string[i] == ")" or string[i] == "_" or string[i].isalnum() or string[i] == ":":
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

        # Keep number of kernel launch params consistent (grid dims, group dims, stream, dynamic shared size)
        num_klp = len(extract_arguments(0, kernel["group"].replace("<<<", "(").replace(">>>", ")")))

        # Transform cuda kernel to hip kernel
        hip_kernel = "hipLaunchKernelGGL(" + cuda_kernel[0:-1].replace(">>>",
                                                                       ", 0" * (4 - num_klp) + ">>>").replace("<<<", ", ").replace(">>>", ", ")

        # Replace cuda kernel with hip kernel
        output_string = output_string.replace(cuda_kernel, hip_kernel)

        # Update the statistics
        stats["kernel_launches"].append(hip_kernel)

    return output_string


def find_parenthesis_end(input_string, start):
    inside_parenthesis = False
    parens = 0
    pos = start
    p_start, p_end = -1, -1

    while pos < len(input_string):
        if input_string[pos] == "(":
            if inside_parenthesis is False:
                inside_parenthesis = True
                parens = 1
                p_start = pos
            else:
                parens += 1
        elif input_string[pos] == ")" and inside_parenthesis:
            parens -= 1

            if parens == 0:
                p_end = pos
                return p_start, p_end

        pos += 1
    return None, None


def disable_asserts(input_string):
    """ Disables regular assert statements
    e.g. "assert(....)" -> "/*assert(....)*/"
    """
    output_string = input_string
    asserts = list(re.finditer(r"\bassert[ ]*\(", input_string))
    for assert_item in asserts:
        p_start, p_end = find_parenthesis_end(input_string, assert_item.end() - 1)
        start = assert_item.start()
        output_string = output_string.replace(input_string[start:p_end + 1], "")
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
        0 - Remove the function entirely (includes the signature).
        1 - Stub the function and return an empty object based off the type.
        2 - Add !defined(__HIP_PLATFORM_HCC__) preprocessors around the function.
        3 - Add !defined(__HIP_DEVICE_COMPILE__) preprocessors around the function.
        4 - Stub the function and throw an exception at runtime.
        5 - Stub the function and throw an assert(0).
        6 - Stub the function and keep an empty body.
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
        function_string = "%s%s%s" % (
            func_info["return_type"],
            func_info["function_name"],
            func_info["function_args"]
        )

        # Find the starting position for the function
        info["function_start"] = input_string.find(function_string)
    else:
        # Automatically detect signature.
        the_match = re.search(r"(((.*) (\*)?)(%s)(\([^{)]*\)))\s*{" %
                              (function.replace("(", "\(").replace(")", "\)")), input_string)
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
    if replace_style == 0:
        output_string = input_string.replace(function_body, "")

    # Stub the function based off its return type.
    elif replace_style == 1:
        # void return type
        if func_info["return_type"] == "void" or func_info["return_type"] == "static void":
            stub = "%s{\n}" % (function_string)
        # pointer return type
        elif "*" in func_info["return_type"]:
            stub = "%s{\nreturn %s;\n}" % (function_string, "NULL")  # nullptr
        else:
            stub = "%s{\n%s stub_var;\nreturn stub_var;\n}" % (function_string, func_info["return_type"])

        output_string = input_string.replace(function_body, stub)

    # Add HIP Preprocessors.
    elif replace_style == 2:
        output_string = input_string.replace(
            function_body,
            "#if !defined(__HIP_PLATFORM_HCC__)\n%s\n#endif" % function_body)

    # Add HIP Preprocessors.
    elif replace_style == 3:
        output_string = input_string.replace(
            function_body,
            "#if !defined(__HIP_DEVICE_COMPILE__)\n%s\n#endif" % function_body)

    # Throw an exception at runtime.
    elif replace_style == 4:
        stub = "%s{\n%s;\n}" % (
            function_string,
            'throw std::runtime_error("The function %s is not implemented.")' %
            function_string.replace("\n", " "))
        output_string = input_string.replace(function_body, stub)

    elif replace_style == 5:
        stub = "%s{\n%s;\n}" % (
            function_string,
            'assert(0)')
        output_string = input_string.replace(function_body, stub)

    elif replace_style == 6:
        stub = "%s{\n;\n}" % (function_string)
        output_string = input_string.replace(function_body, stub)
    return output_string


def preprocessor(filepath, stats):
    """ Executes the CUDA -> HIP conversion on the specified file. """
    with openf(filepath, "r+") as fileobj:
        output_source = fileobj.read()

        # Perform type, method, constant replacements
        for mapping in CUDA_TO_HIP_MAPPINGS:
            for cuda_type, value in mapping.items():
                # Extract relevant information
                hip_type = value[0]
                meta_data = value[1:]

                if output_source.find(cuda_type) > -1:
                    # Check if supported
                    if constants.HIP_UNSUPPORTED in meta_data:
                        stats["unsupported_calls"].append((cuda_type, filepath))

                if cuda_type in output_source:
                    output_source = re.sub(r'\b(%s)\b' % cuda_type, lambda x: hip_type, output_source)

        # Perform Kernel Launch Replacements
        output_source = processKernelLaunches(output_source, stats)

        # Disable asserts
        if not filepath.endswith("THCGeneral.h.in"):
            output_source = disable_asserts(output_source)

        # Overwrite file contents
        fileobj.seek(0)
        fileobj.write(output_source)
        fileobj.truncate()
        fileobj.flush()

        # Flush to disk
        os.fsync(fileobj)


def file_specific_replacement(filepath, search_string, replace_string, strict=False):
    with openf(filepath, "r+") as f:
        contents = f.read()
        if strict:
            contents = re.sub(r'\b(%s)\b' % search_string, lambda x: replace_string, contents)
        else:
            contents = contents.replace(search_string, replace_string)
        f.seek(0)
        f.write(contents)
        f.truncate()


def file_add_header(filepath, header):
    with openf(filepath, "r+") as f:
        contents = f.read()
        if header[0] != "<" and header[-1] != ">":
            header = '"%s"' % header
        contents = ('#include %s \n' % header) + contents
        f.seek(0)
        f.write(contents)
        f.truncate()


def fix_static_global_kernels(in_txt):
    """Static global kernels in HIP results in a compilation error."""
    in_txt = in_txt.replace(" __global__ static", "__global__")
    return in_txt


def get_kernel_template_params(the_file, KernelDictionary):
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
            if len(template_arguments) == 1 and template_arguments[0].strip() in ["Dtype", "T"]:
                # Updates kernel
                kernel_with_template = "%s<real>" % (kernel_name)
            else:
                kernel_with_template = kernel_name
            formatted_args = {}
            for idx, arg_type in enumerate(argument_types):
                formatted_args[idx] = arg_type

            KernelDictionary[kernel_name] = {"kernel_with_template": kernel_with_template, "arg_types": formatted_args}

        # Extract generated kernels
        get_generated_kernels = [k for k in re.finditer(r"GENERATE_KERNEL([1-9])\((.*)\)", string)]
        # curandStateMtgp32 *state, int size, T *result, ARG1
        for kernel in get_generated_kernels:
            kernel_gen_type = int(kernel.group(1))
            kernel_name = kernel.group(2).split(",")[0]
            kernel_params = kernel.group(2).split(",")[1:]

            if kernel_gen_type == 1:
                kernel_args = {1: "int", 2: "%s *" % kernel_params[0], 3: kernel_params[1]}

            if kernel_gen_type == 2:
                kernel_args = {1: "int", 2: "%s *" % kernel_params[0], 3: kernel_params[1], 4: kernel_params[2]}

            # Argument at position 1 should be int
            KernelDictionary[kernel_name] = {"kernel_with_template": kernel_name, "arg_types": kernel_args}


def disable_unsupported_function_call(function, input_string, replacement):
    """Disables calls to an unsupported HIP function"""
    # Prepare output string
    output_string = input_string

    # Find all calls to the function
    calls = re.finditer(r"\b%s\b" % function, input_string)

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

        disabled = "%s#if !defined(__HIP_PLATFORM_HCC__)\n%s\n#endif" % (txt[0:end], txt[end:])

        f.seek(0)
        f.write(disabled)
        f.truncate()


def extract_arguments(start, string):
    """ Return the list of arguments in the upcoming function parameter closure"""
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
        elif string[current_position] == ">" and string[current_position - 1] != "-":
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


def add_static_casts(directory, extensions, KernelTemplateParams):
    """Added necessary static casts to kernel launches due to issue in HIP"""
    # Add static_casts<> to all kernel launches.
    for (dirpath, _dirnames, filenames) in os.walk(directory):
        for filename in filenames:
            if filename_ends_with_extension(filename, extensions):
                filepath = os.sep.join([dirpath, filename])
                with openf(filepath, "r+") as fileobj:
                    input_source = fileobj.read()
                    new_output_source = input_source
                    get_kernel_definitions = [k for k in re.finditer("hipLaunchKernelGGL\(", input_source)]
                    for kernel in get_kernel_definitions:
                        arguments = extract_arguments(kernel.end() - 1, input_source)

                        # Check if we have templating + static_cast information
                        argument_strings = [input_source[arg["start"]:arg["end"]] for arg in arguments]
                        kernel_name = argument_strings[0].strip()
                        ignore = ["upscale"]
                        if kernel_name in KernelTemplateParams and kernel_name not in ignore:
                            # Add template to the kernel
                            # Add static_casts to relevant arguments
                            kernel_name_with_template = KernelTemplateParams[kernel_name]["kernel_with_template"]
                            argument_types = KernelTemplateParams[kernel_name]["arg_types"]

                            old_kernel_launch = input_source[arguments[0]["start"]:arguments[-1]["end"]]
                            new_kernel_launch = old_kernel_launch

                            kernel_params = argument_strings[5:]
                            for arg_idx, arg in enumerate(kernel_params):
                                if arg_idx in argument_types:
                                    arg = kernel_params[arg_idx].strip()
                                    the_type = argument_types[arg_idx]
                                    the_arg = arg.replace("\n", "").strip()
                                    if the_type in ["int", "const int", "int64_t", "THCIndex_t *", "const int *", "ptrdiff_t", "long", "const int64_t*", "int64_t *", "double"]:
                                        static_argument = "static_cast<%s>(%s)" % (the_type, the_arg)
                                        static_argument = arg.replace(the_arg, static_argument)

                                        # Update to static_cast
                                        new_kernel_launch = re.sub(r'\b(%s)\b' %
                                                                   arg, lambda x: static_argument, new_kernel_launch)

                            # Add template type
                            if "THCUNN" in filepath.split("/") and "generic" not in filepath.split("/"):
                                kernel_name_with_template = kernel_name_with_template.replace("<real>", "<Dtype>")
                            new_kernel_launch = re.sub(r'\b(%s)\b' % kernel_name,
                                                       lambda x: kernel_name_with_template, new_kernel_launch)

                            # Replace Launch
                            new_output_source = new_output_source.replace(old_kernel_launch, new_kernel_launch)

                    # Overwrite file contents
                    fileobj.seek(0)
                    fileobj.write(new_output_source)
                    fileobj.truncate()
                    fileobj.flush()

                    # Flush to disk
                    os.fsync(fileobj)


def main():
    """Example invocation

    python hipify.py --project-directory /home/myproject/ --extensions cu cuh h cpp --output-directory /home/gains/
    """

    parser = argparse.ArgumentParser(
        description="The Python Hipify Script.")

    parser.add_argument(
        '--project-directory',
        type=str,
        default=os.getcwd(),
        help="The root of the project.",
        required=True)

    parser.add_argument(
        '--show-detailed',
        type=bool,
        default=False,
        help="Show detailed summary of the hipification process.",
        required=False)

    parser.add_argument(
        '--extensions',
        nargs='+',
        default=["cu", "cuh", "c", "cpp", "h", "in", "hpp"],
        help="The extensions for files to run the Hipify script over.",
        required=False)

    parser.add_argument(
        '--output-directory',
        type=str,
        default="",
        help="The directory to store the hipified project.",
        required=False)

    parser.add_argument(
        '--include-dirs',
        nargs='+',
        default=[],
        help="The directories under the root that should be included.",
        required=False)

    parser.add_argument(
        '--yaml-settings',
        type=str,
        default="",
        help="The yaml file storing information for disabled functions and modules.",
        required=False)

    parser.add_argument(
        '--add-static-casts',
        type=bool,
        default=False,
        help="Whether to automatically add static_casts to kernel arguments.",
        required=False)

    args = parser.parse_args()

    # Verify the project directory exists.
    if not os.path.exists(args.project_directory):
        print("The project folder specified does not exist.")
        return

    # Make sure output directory exists.
    if not os.path.exists(args.output_directory):
        print("The output folder already exists.")
        return

    # If no output directory, provide a default one.
    if args.output_directory is "":
        args.project_directory = args.project_directory[0:-
                                                        1] if args.project_directory.endswith("/") else args.project_directory
        args.output_directory = args.project_directory + "_amd"

    # Copy from project directory to output directory if not done already.
    if not os.path.exists(args.output_directory):
        shutil.copytree(args.project_directory, args.output_directory)

    # Extract all of the kernel parameter and template type information.
    if args.add_static_casts:
        KernelTemplateParams = {}
        for (dirpath, _dirnames, filenames) in os.walk(args.output_directory):
            for filename in filenames:
                if filename_ends_with_extension(filename, args.extensions) and inside_included_directories(dirpath, args.output_directory, args.include_dirs):
                    the_file = os.sep.join([dirpath, filename])

                    # Store param information inside KernelTemplateParams
                    get_kernel_template_params(the_file, KernelTemplateParams)

    # Open YAML file with disable information.
    if args.yaml_settings != "":
        with openf(args.yaml_settings, "r") as lines:
            yaml_data = yaml.load(lines)

        # Disable functions in certain files according to YAML description
        for disable_info in yaml_data["disabled_functions"]:
            filepath = os.path.join(args.output_directory, disable_info["path"])
            if "functions" in disable_info:
                functions = disable_info["functions"]
            else:
                functions = []

            if "non_hip_functions" in disable_info:
                non_hip_functions = disable_info["non_hip_functions"]
            else:
                non_hip_functions = []

            if "non_device_functions" in disable_info:
                not_on_device_functions = disable_info["non_device_functions"]
            else:
                not_on_device_functions = []

            with openf(filepath, "r+") as f:
                txt = f.read()
                for func in functions:
                    # TODO - Find fix assertions in HIP for device code.
                    txt = disable_function(txt, func, 5)

                for func in non_hip_functions:
                    # Disable this function on HIP stack
                    txt = disable_function(txt, func, 2)

                for func in not_on_device_functions:
                    # Disable this function when compiling on Device
                    txt = disable_function(txt, func, 3)

                f.seek(0)
                f.write(txt)
                f.truncate()
                f.close()

        # Disable modules
        disable_modules = yaml_data["disabled_modules"]
        for module in disable_modules:
            disable_module(os.path.join(args.output_directory, module))

        # Disable unsupported HIP functions
        for disable in yaml_data["disable_unsupported_hip_calls"]:
            filepath = os.path.join(args.output_directory, disable["path"])
            if "functions" in disable:
                functions = disable["functions"]
            else:
                functions = []

            if "constants" in disable:
                constants = disable["constants"]
            else:
                constants = []

            if "s_constants" in disable:
                s_constants = disable["s_constants"]
            else:
                s_constants = []

            if not os.path.exists(filepath):
                print("\n" + bcolors.WARNING + "YAML Warning: File %s does not exist." % filepath + bcolors.ENDC)
                continue

            with openf(filepath, "r+") as f:
                txt = f.read()

                # Disable HIP Functions
                for func in functions:
                    txt = disable_unsupported_function_call(func, txt, functions[func])

                # Disable Constants w\ Boundary.
                for const in constants:
                    txt = re.sub(r"\b%s\b" % const, constants[const], txt)

                # Disable Constants
                for s_const in s_constants:
                    txt = txt.replace(s_const, s_constants[s_const])

                # Save Changes
                f.seek(0)
                f.write(txt)
                f.truncate()
                f.close()

    # Start Preprocessor
    walk_over_directory(
        args.output_directory,
        extensions=args.extensions,
        show_detailed=args.show_detailed,
        include_dirs=args.include_dirs)

    if args.add_static_casts:
        # Execute the Clang Tool to Automatically add static casts
        add_static_casts(args.output_directory, args.extensions, KernelTemplateParams)


if __name__ == '__main__':
    main()
