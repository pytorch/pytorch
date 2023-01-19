# pylint: disable=g-bad-file-header
# Copyright 2016 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Base library for configuring the C++ toolchain."""

def resolve_labels(repository_ctx, labels):
    """Resolves a collection of labels to their paths.

    Label resolution can cause the evaluation of Starlark functions to restart.
    For functions with side-effects (like the auto-configuration functions, which
    inspect the system and touch the file system), such restarts are costly.
    We cannot avoid the restarts, but we can minimize their penalty by resolving
    all labels upfront.

    Among other things, doing less work on restarts can cut analysis times by
    several seconds and may also prevent tickling kernel conditions that cause
    build failures.  See https://github.com/bazelbuild/bazel/issues/5196 for
    more details.

    Args:
      repository_ctx: The context with which to resolve the labels.
      labels: Labels to be resolved expressed as a list of strings.

    Returns:
      A dictionary with the labels as keys and their paths as values.
    """
    return dict([(label, repository_ctx.path(Label(label))) for label in labels])

def escape_string(arg):
    """Escape percent sign (%) in the string so it can appear in the Crosstool."""
    if arg != None:
        return str(arg).replace("%", "%%")
    else:
        return None

def split_escaped(string, delimiter):
    """Split string on the delimiter unless %-escaped.

    Examples:
      Basic usage:
        split_escaped("a:b:c", ":") -> [ "a", "b", "c" ]

      Delimeter that is not supposed to be splitten on has to be %-escaped:
        split_escaped("a%:b", ":") -> [ "a:b" ]

      Literal % can be represented by escaping it as %%:
        split_escaped("a%%b", ":") -> [ "a%b" ]

      Consecutive delimiters produce empty strings:
        split_escaped("a::b", ":") -> [ "a", "", "", "b" ]

    Args:
      string: The string to be split.
      delimiter: Non-empty string not containing %-sign to be used as a
          delimiter.

    Returns:
      A list of substrings.
    """
    if delimiter == "":
        fail("Delimiter cannot be empty")
    if delimiter.find("%") != -1:
        fail("Delimiter cannot contain %-sign")

    i = 0
    result = []
    accumulator = []
    length = len(string)
    delimiter_length = len(delimiter)

    if not string:
        return []

    # Iterate over the length of string since Starlark doesn't have while loops
    for _ in range(length):
        if i >= length:
            break
        if i + 2 <= length and string[i:i + 2] == "%%":
            accumulator.append("%")
            i += 2
        elif (i + 1 + delimiter_length <= length and
              string[i:i + 1 + delimiter_length] == "%" + delimiter):
            accumulator.append(delimiter)
            i += 1 + delimiter_length
        elif i + delimiter_length <= length and string[i:i + delimiter_length] == delimiter:
            result.append("".join(accumulator))
            accumulator = []
            i += delimiter_length
        else:
            accumulator.append(string[i])
            i += 1

    # Append the last group still in accumulator
    result.append("".join(accumulator))
    return result

def auto_configure_fail(msg):
    """Output failure message when auto configuration fails."""
    red = "\033[0;31m"
    no_color = "\033[0m"
    fail("\n%sAuto-Configuration Error:%s %s\n" % (red, no_color, msg))

def auto_configure_warning(msg):
    """Output warning message during auto configuration."""
    yellow = "\033[1;33m"
    no_color = "\033[0m"

    # buildifier: disable=print
    print("\n%sAuto-Configuration Warning:%s %s\n" % (yellow, no_color, msg))

def get_env_var(repository_ctx, name, default = None, enable_warning = True):
    """Find an environment variable in system path. Doesn't %-escape the value!

    Args:
      repository_ctx: The repository context.
      name: Name of the environment variable.
      default: Default value to be used when such environment variable is not present.
      enable_warning: Show warning if the variable is not present.
    Returns:
      value of the environment variable or default.
    """

    if name in repository_ctx.os.environ:
        return repository_ctx.os.environ[name]
    if default != None:
        if enable_warning:
            auto_configure_warning("'%s' environment variable is not set, using '%s' as default" % (name, default))
        return default
    return auto_configure_fail("'%s' environment variable is not set" % name)

def which(repository_ctx, cmd, default = None):
    """A wrapper around repository_ctx.which() to provide a fallback value. Doesn't %-escape the value!

    Args:
      repository_ctx: The repository context.
      cmd: name of the executable to resolve.
      default: Value to be returned when such executable couldn't be found.
    Returns:
      absolute path to the cmd or default when not found.
    """
    result = repository_ctx.which(cmd)
    return default if result == None else str(result)

def which_cmd(repository_ctx, cmd, default = None):
    """Find cmd in PATH using repository_ctx.which() and fail if cannot find it. Doesn't %-escape the cmd!

    Args:
      repository_ctx: The repository context.
      cmd: name of the executable to resolve.
      default: Value to be returned when such executable couldn't be found.
    Returns:
      absolute path to the cmd or default when not found.
    """
    result = repository_ctx.which(cmd)
    if result != None:
        return str(result)
    path = get_env_var(repository_ctx, "PATH")
    if default != None:
        auto_configure_warning("Cannot find %s in PATH, using '%s' as default.\nPATH=%s" % (cmd, default, path))
        return default
    auto_configure_fail("Cannot find %s in PATH, please make sure %s is installed and add its directory in PATH.\nPATH=%s" % (cmd, cmd, path))
    return str(result)

def execute(
        repository_ctx,
        command,
        environment = None,
        expect_failure = False):
    """Execute a command, return stdout if succeed and throw an error if it fails. Doesn't %-escape the result!

    Args:
      repository_ctx: The repository context.
      command: command to execute.
      environment: dictionary with environment variables to set for the command.
      expect_failure: True if the command is expected to fail.
    Returns:
      stdout of the executed command.
    """
    if environment:
        result = repository_ctx.execute(command, environment = environment)
    else:
        result = repository_ctx.execute(command)
    if expect_failure != (result.return_code != 0):
        if expect_failure:
            auto_configure_fail(
                "expected failure, command %s, stderr: (%s)" % (
                    command,
                    result.stderr,
                ),
            )
        else:
            auto_configure_fail(
                "non-zero exit code: %d, command %s, stderr: (%s)" % (
                    result.return_code,
                    command,
                    result.stderr,
                ),
            )
    stripped_stdout = result.stdout.strip()
    if not stripped_stdout:
        auto_configure_fail(
            "empty output from command %s, stderr: (%s)" % (command, result.stderr),
        )
    return stripped_stdout

def get_cpu_value(repository_ctx):
    """Compute the cpu_value based on the OS name. Doesn't %-escape the result!

    Args:
      repository_ctx: The repository context.
    Returns:
      One of (darwin, freebsd, x64_windows, ppc, s390x, arm, aarch64, k8, piii)
    """
    os_name = repository_ctx.os.name.lower()
    if os_name.startswith("mac os"):
        return "darwin"
    if os_name.find("freebsd") != -1:
        return "freebsd"
    if os_name.find("windows") != -1:
        return "x64_windows"

    # Use uname to figure out whether we are on x86_32 or x86_64
    result = repository_ctx.execute(["uname", "-m"])
    if result.stdout.strip() in ["power", "ppc64le", "ppc", "ppc64"]:
        return "ppc"
    if result.stdout.strip() in ["s390x"]:
        return "s390x"
    if result.stdout.strip() in ["arm", "armv7l"]:
        return "arm"
    if result.stdout.strip() in ["aarch64"]:
        return "aarch64"
    return "k8" if result.stdout.strip() in ["amd64", "x86_64", "x64"] else "piii"

def is_cc_configure_debug(repository_ctx):
    """Returns True if CC_CONFIGURE_DEBUG is set to 1."""
    env = repository_ctx.os.environ
    return "CC_CONFIGURE_DEBUG" in env and env["CC_CONFIGURE_DEBUG"] == "1"

def build_flags(flags):
    """Convert `flags` to a string of flag fields."""
    return "\n".join(["        flag: '" + flag + "'" for flag in flags])

def get_starlark_list(values):
    """Convert a list of string into a string that can be passed to a rule attribute."""
    if not values:
        return ""
    return "\"" + "\",\n    \"".join(values) + "\""

def auto_configure_warning_maybe(repository_ctx, msg):
    """Output warning message when CC_CONFIGURE_DEBUG is enabled."""
    if is_cc_configure_debug(repository_ctx):
        auto_configure_warning(msg)

def write_builtin_include_directory_paths(repository_ctx, cc, directories, file_suffix = ""):
    """Generate output file named 'builtin_include_directory_paths' in the root of the repository."""
    if get_env_var(repository_ctx, "BAZEL_IGNORE_SYSTEM_HEADERS_VERSIONS", "0", False) == "1":
        repository_ctx.file(
            "builtin_include_directory_paths" + file_suffix,
            """This file is generated by cc_configure and normally contains builtin include directories
that C++ compiler reported. But because BAZEL_IGNORE_SYSTEM_HEADERS_VERSIONS was set to 1,
header include directory paths are intentionally not put there.
""",
        )
    else:
        repository_ctx.file(
            "builtin_include_directory_paths" + file_suffix,
            """This file is generated by cc_configure and contains builtin include directories
that %s reported. This file is a dependency of every compilation action and
changes to it will be reflected in the action cache key. When some of these
paths change, Bazel will make sure to rerun the action, even though none of
declared action inputs or the action commandline changes.

%s
""" % (cc, "\n".join(directories)),
        )
