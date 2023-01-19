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
"""Configuring the C++ toolchain on Windows."""

load(
    ":lib_cc_configure.bzl",
    "auto_configure_fail",
    "auto_configure_warning",
    "auto_configure_warning_maybe",
    "escape_string",
    "execute",
    "resolve_labels",
    "write_builtin_include_directory_paths",
)

def _get_path_env_var(repository_ctx, name):
    """Returns a path from an environment variable.

    Removes quotes, replaces '/' with '\', and strips trailing '\'s."""
    if name in repository_ctx.os.environ:
        value = repository_ctx.os.environ[name]
        if value[0] == "\"":
            if len(value) == 1 or value[-1] != "\"":
                auto_configure_fail("'%s' environment variable has no trailing quote" % name)
            value = value[1:-1]
        if "/" in value:
            value = value.replace("/", "\\")
        if value[-1] == "\\":
            value = value.rstrip("\\")
        return value
    else:
        return None

def _get_temp_env(repository_ctx):
    """Returns the value of TMP, or TEMP, or if both undefined then C:\\Windows."""
    tmp = _get_path_env_var(repository_ctx, "TMP")
    if not tmp:
        tmp = _get_path_env_var(repository_ctx, "TEMP")
    if not tmp:
        tmp = "C:\\Windows\\Temp"
        auto_configure_warning(
            "neither 'TMP' nor 'TEMP' environment variables are set, using '%s' as default" % tmp,
        )
    return tmp

def _get_escaped_windows_msys_starlark_content(repository_ctx, use_mingw = False):
    """Return the content of msys cc toolchain rule."""
    msys_root = ""
    bazel_sh = _get_path_env_var(repository_ctx, "BAZEL_SH")
    if bazel_sh:
        bazel_sh = bazel_sh.replace("\\", "/").lower()
        tokens = bazel_sh.rsplit("/", 1)
        if tokens[0].endswith("/usr/bin"):
            msys_root = tokens[0][:len(tokens[0]) - len("usr/bin")]
        elif tokens[0].endswith("/bin"):
            msys_root = tokens[0][:len(tokens[0]) - len("bin")]

    prefix = "mingw64" if use_mingw else "usr"
    tool_path_prefix = escape_string(msys_root) + prefix
    tool_bin_path = tool_path_prefix + "/bin"
    tool_path = {}

    for tool in ["ar", "compat-ld", "cpp", "dwp", "gcc", "gcov", "ld", "nm", "objcopy", "objdump", "strip"]:
        if msys_root:
            tool_path[tool] = tool_bin_path + "/" + tool
        else:
            tool_path[tool] = "msys_gcc_installation_error.bat"
    tool_paths = ",\n        ".join(['"%s": "%s"' % (k, v) for k, v in tool_path.items()])
    include_directories = ('        "%s/",\n        ' % tool_path_prefix) if msys_root else ""
    return tool_paths, tool_bin_path, include_directories

def _get_system_root(repository_ctx):
    """Get System root path on Windows, default is C:\\Windows. Doesn't %-escape the result."""
    systemroot = _get_path_env_var(repository_ctx, "SYSTEMROOT")
    if not systemroot:
        systemroot = "C:\\Windows"
        auto_configure_warning_maybe(
            repository_ctx,
            "SYSTEMROOT is not set, using default SYSTEMROOT=C:\\Windows",
        )
    return escape_string(systemroot)

def _add_system_root(repository_ctx, env):
    """Running VCVARSALL.BAT and VCVARSQUERYREGISTRY.BAT need %SYSTEMROOT%\\\\system32 in PATH."""
    if "PATH" not in env:
        env["PATH"] = ""
    env["PATH"] = env["PATH"] + ";" + _get_system_root(repository_ctx) + "\\system32"
    return env

def _find_vc_path(repository_ctx):
    """Find Visual C++ build tools install path. Doesn't %-escape the result."""

    # 1. Check if BAZEL_VC or BAZEL_VS is already set by user.
    bazel_vc = _get_path_env_var(repository_ctx, "BAZEL_VC")
    if bazel_vc:
        if repository_ctx.path(bazel_vc).exists:
            return bazel_vc
        else:
            auto_configure_warning_maybe(
                repository_ctx,
                "%BAZEL_VC% is set to non-existent path, ignoring.",
            )

    bazel_vs = _get_path_env_var(repository_ctx, "BAZEL_VS")
    if bazel_vs:
        if repository_ctx.path(bazel_vs).exists:
            bazel_vc = bazel_vs + "\\VC"
            if repository_ctx.path(bazel_vc).exists:
                return bazel_vc
            else:
                auto_configure_warning_maybe(
                    repository_ctx,
                    "No 'VC' directory found under %BAZEL_VS%, ignoring.",
                )
        else:
            auto_configure_warning_maybe(
                repository_ctx,
                "%BAZEL_VS% is set to non-existent path, ignoring.",
            )

    auto_configure_warning_maybe(
        repository_ctx,
        "Neither %BAZEL_VC% nor %BAZEL_VS% are set, start looking for the latest Visual C++" +
        " installed.",
    )

    # 2. Check if VS%VS_VERSION%COMNTOOLS is set, if true then try to find and use
    # vcvarsqueryregistry.bat / VsDevCmd.bat to detect VC++.
    auto_configure_warning_maybe(repository_ctx, "Looking for VS%VERSION%COMNTOOLS environment variables, " +
                                                 "eg. VS140COMNTOOLS")
    for vscommontools_env, script in [
        ("VS160COMNTOOLS", "VsDevCmd.bat"),
        ("VS150COMNTOOLS", "VsDevCmd.bat"),
        ("VS140COMNTOOLS", "vcvarsqueryregistry.bat"),
        ("VS120COMNTOOLS", "vcvarsqueryregistry.bat"),
        ("VS110COMNTOOLS", "vcvarsqueryregistry.bat"),
        ("VS100COMNTOOLS", "vcvarsqueryregistry.bat"),
        ("VS90COMNTOOLS", "vcvarsqueryregistry.bat"),
    ]:
        if vscommontools_env not in repository_ctx.os.environ:
            continue
        script = _get_path_env_var(repository_ctx, vscommontools_env) + "\\" + script
        if not repository_ctx.path(script).exists:
            continue
        repository_ctx.file(
            "get_vc_dir.bat",
            "@echo off\n" +
            "call \"" + script + "\"\n" +
            "echo %VCINSTALLDIR%",
            True,
        )
        env = _add_system_root(repository_ctx, repository_ctx.os.environ)
        vc_dir = execute(repository_ctx, ["./get_vc_dir.bat"], environment = env)

        auto_configure_warning_maybe(repository_ctx, "Visual C++ build tools found at %s" % vc_dir)
        return vc_dir

    # 3. User might have purged all environment variables. If so, look for Visual C++ in registry.
    # Works for Visual Studio 2017 and older. (Does not work for Visual Studio 2019 Preview.)
    # TODO(laszlocsomor): check if "16.0" also has this registry key, after VS 2019 is released.
    auto_configure_warning_maybe(repository_ctx, "Looking for Visual C++ through registry")
    reg_binary = _get_system_root(repository_ctx) + "\\system32\\reg.exe"
    vc_dir = None
    for key, suffix in (("VC7", ""), ("VS7", "\\VC")):
        for version in ["15.0", "14.0", "12.0", "11.0", "10.0", "9.0", "8.0"]:
            if vc_dir:
                break
            result = repository_ctx.execute([reg_binary, "query", "HKEY_LOCAL_MACHINE\\SOFTWARE\\Wow6432Node\\Microsoft\\VisualStudio\\SxS\\" + key, "/v", version])
            auto_configure_warning_maybe(repository_ctx, "registry query result for VC %s:\n\nSTDOUT(start)\n%s\nSTDOUT(end)\nSTDERR(start):\n%s\nSTDERR(end)\n" %
                                                         (version, result.stdout, result.stderr))
            if not result.stderr:
                for line in result.stdout.split("\n"):
                    line = line.strip()
                    if line.startswith(version) and line.find("REG_SZ") != -1:
                        vc_dir = line[line.find("REG_SZ") + len("REG_SZ"):].strip() + suffix
    if vc_dir:
        auto_configure_warning_maybe(repository_ctx, "Visual C++ build tools found at %s" % vc_dir)
        return vc_dir

    # 4. Check default directories for VC installation
    auto_configure_warning_maybe(repository_ctx, "Looking for default Visual C++ installation directory")
    program_files_dir = _get_path_env_var(repository_ctx, "PROGRAMFILES(X86)")
    if not program_files_dir:
        program_files_dir = "C:\\Program Files (x86)"
        auto_configure_warning_maybe(
            repository_ctx,
            "'PROGRAMFILES(X86)' environment variable is not set, using '%s' as default" % program_files_dir,
        )
    for path in [
        "Microsoft Visual Studio\\2019\\Preview\\VC",
        "Microsoft Visual Studio\\2019\\BuildTools\\VC",
        "Microsoft Visual Studio\\2019\\Community\\VC",
        "Microsoft Visual Studio\\2019\\Professional\\VC",
        "Microsoft Visual Studio\\2019\\Enterprise\\VC",
        "Microsoft Visual Studio\\2017\\BuildTools\\VC",
        "Microsoft Visual Studio\\2017\\Community\\VC",
        "Microsoft Visual Studio\\2017\\Professional\\VC",
        "Microsoft Visual Studio\\2017\\Enterprise\\VC",
        "Microsoft Visual Studio 14.0\\VC",
    ]:
        path = program_files_dir + "\\" + path
        if repository_ctx.path(path).exists:
            vc_dir = path
            break

    if not vc_dir:
        auto_configure_warning_maybe(repository_ctx, "Visual C++ build tools not found.")
        return None
    auto_configure_warning_maybe(repository_ctx, "Visual C++ build tools found at %s" % vc_dir)
    return vc_dir

def _is_vs_2017_or_2019(vc_path):
    """Check if the installed VS version is Visual Studio 2017."""

    # In VS 2017 and 2019, the location of VC is like:
    # C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\
    # In VS 2015 or older version, it is like:
    # C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\
    return vc_path.find("2017") != -1 or vc_path.find("2019") != -1

def _find_vcvars_bat_script(repository_ctx, vc_path):
    """Find batch script to set up environment variables for VC. Doesn't %-escape the result."""
    if _is_vs_2017_or_2019(vc_path):
        vcvars_script = vc_path + "\\Auxiliary\\Build\\VCVARSALL.BAT"
    else:
        vcvars_script = vc_path + "\\VCVARSALL.BAT"

    if not repository_ctx.path(vcvars_script).exists:
        return None

    return vcvars_script

def _is_support_vcvars_ver(vc_full_version):
    """-vcvars_ver option is supported from version 14.11.25503 (VS 2017 version 15.3)."""
    version = [int(i) for i in vc_full_version.split(".")]
    min_version = [14, 11, 25503]
    return version >= min_version

def _is_support_winsdk_selection(repository_ctx, vc_path):
    """Windows SDK selection is supported with VC 2017 / 2019 or with full VS 2015 installation."""
    if _is_vs_2017_or_2019(vc_path):
        return True

    # By checking the source code of VCVARSALL.BAT in VC 2015, we know that
    # when devenv.exe or wdexpress.exe exists, VCVARSALL.BAT supports Windows SDK selection.
    vc_common_ide = repository_ctx.path(vc_path).dirname.get_child("Common7").get_child("IDE")
    for tool in ["devenv.exe", "wdexpress.exe"]:
        if vc_common_ide.get_child(tool).exists:
            return True
    return False

def setup_vc_env_vars(repository_ctx, vc_path, envvars = [], allow_empty = False, escape = True):
    """Get environment variables set by VCVARSALL.BAT script. Doesn't %-escape the result!

    Args:
        repository_ctx: the repository_ctx object
        vc_path: Visual C++ root directory
        envvars: list of envvars to retrieve; default is ["PATH", "INCLUDE", "LIB", "WINDOWSSDKDIR"]
        allow_empty: allow unset envvars; if False then report errors for those
        escape: if True, escape "\" as "\\" and "%" as "%%" in the envvar values

    Returns:
        dictionary of the envvars
    """
    if not envvars:
        envvars = ["PATH", "INCLUDE", "LIB", "WINDOWSSDKDIR"]

    vcvars_script = _find_vcvars_bat_script(repository_ctx, vc_path)
    if not vcvars_script:
        auto_configure_fail("Cannot find VCVARSALL.BAT script under %s" % vc_path)

    # Getting Windows SDK version set by user.
    # Only supports VC 2017 & 2019 and VC 2015 with full VS installation.
    winsdk_version = _get_winsdk_full_version(repository_ctx)
    if winsdk_version and not _is_support_winsdk_selection(repository_ctx, vc_path):
        auto_configure_warning(("BAZEL_WINSDK_FULL_VERSION=%s is ignored, " +
                                "because standalone Visual C++ Build Tools 2015 doesn't support specifying Windows " +
                                "SDK version, please install the full VS 2015 or use VC 2017/2019.") % winsdk_version)
        winsdk_version = ""

    # Get VC version set by user. Only supports VC 2017 & 2019.
    vcvars_ver = ""
    if _is_vs_2017_or_2019(vc_path):
        full_version = _get_vc_full_version(repository_ctx, vc_path)

        # Because VCVARSALL.BAT is from the latest VC installed, so we check if the latest
        # version supports -vcvars_ver or not.
        if _is_support_vcvars_ver(_get_latest_subversion(repository_ctx, vc_path)):
            vcvars_ver = "-vcvars_ver=" + full_version

    cmd = "\"%s\" amd64 %s %s" % (vcvars_script, winsdk_version, vcvars_ver)
    print_envvars = ",".join(["{k}=%{k}%".format(k = k) for k in envvars])
    repository_ctx.file(
        "get_env.bat",
        "@echo off\n" +
        ("call %s > NUL \n" % cmd) + ("echo %s \n" % print_envvars),
        True,
    )
    env = _add_system_root(repository_ctx, {k: "" for k in envvars})
    envs = execute(repository_ctx, ["./get_env.bat"], environment = env).split(",")
    env_map = {}
    for env in envs:
        key, value = env.split("=", 1)
        env_map[key] = escape_string(value.replace("\\", "\\\\")) if escape else value
    if not allow_empty:
        _check_env_vars(env_map, cmd, expected = envvars)
    return env_map

def _check_env_vars(env_map, cmd, expected):
    for env in expected:
        if not env_map.get(env):
            auto_configure_fail(
                "Setting up VC environment variables failed, %s is not set by the following command:\n    %s" % (env, cmd),
            )

def _get_latest_subversion(repository_ctx, vc_path):
    """Get the latest subversion of a VS 2017/2019 installation.

    For VS 2017 & 2019, there could be multiple versions of VC build tools.
    The directories are like:
      <vc_path>\\Tools\\MSVC\\14.10.24930\\bin\\HostX64\\x64
      <vc_path>\\Tools\\MSVC\\14.16.27023\\bin\\HostX64\\x64
    This function should return 14.16.27023 in this case."""
    versions = [path.basename for path in repository_ctx.path(vc_path + "\\Tools\\MSVC").readdir()]
    if len(versions) < 1:
        auto_configure_warning_maybe(repository_ctx, "Cannot find any VC installation under BAZEL_VC(%s)" % vc_path)
        return None

    # Parse the version string into integers, then sort the integers to prevent textual sorting.
    version_list = []
    for version in versions:
        parts = [int(i) for i in version.split(".")]
        version_list.append((parts, version))

    version_list = sorted(version_list)
    latest_version = version_list[-1][1]

    auto_configure_warning_maybe(repository_ctx, "Found the following VC verisons:\n%s\n\nChoosing the latest version = %s" % ("\n".join(versions), latest_version))
    return latest_version

def _get_vc_full_version(repository_ctx, vc_path):
    """Return the value of BAZEL_VC_FULL_VERSION if defined, otherwise the latest version."""
    if "BAZEL_VC_FULL_VERSION" in repository_ctx.os.environ:
        return repository_ctx.os.environ["BAZEL_VC_FULL_VERSION"]
    return _get_latest_subversion(repository_ctx, vc_path)

def _get_winsdk_full_version(repository_ctx):
    """Return the value of BAZEL_WINSDK_FULL_VERSION if defined, otherwise an empty string."""
    return repository_ctx.os.environ.get("BAZEL_WINSDK_FULL_VERSION", default = "")

def _find_msvc_tool(repository_ctx, vc_path, tool):
    """Find the exact path of a specific build tool in MSVC. Doesn't %-escape the result."""
    tool_path = None
    if _is_vs_2017_or_2019(vc_path):
        full_version = _get_vc_full_version(repository_ctx, vc_path)
        if full_version:
            tool_path = "%s\\Tools\\MSVC\\%s\\bin\\HostX64\\x64\\%s" % (vc_path, full_version, tool)
    else:
        # For VS 2015 and older version, the tools are under:
        # C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64
        tool_path = vc_path + "\\bin\\amd64\\" + tool

    if not tool_path or not repository_ctx.path(tool_path).exists:
        return None

    return tool_path.replace("\\", "/")

def _find_missing_vc_tools(repository_ctx, vc_path):
    """Check if any required tool is missing under given VC path."""
    missing_tools = []
    if not _find_vcvars_bat_script(repository_ctx, vc_path):
        missing_tools.append("VCVARSALL.BAT")

    for tool in ["cl.exe", "link.exe", "lib.exe", "ml64.exe"]:
        if not _find_msvc_tool(repository_ctx, vc_path, tool):
            missing_tools.append(tool)

    return missing_tools

def _is_support_debug_fastlink(repository_ctx, linker):
    """Run linker alone to see if it supports /DEBUG:FASTLINK."""
    if _use_clang_cl(repository_ctx):
        # LLVM's lld-link.exe doesn't support /DEBUG:FASTLINK.
        return False
    result = execute(repository_ctx, [linker], expect_failure = True)
    return result.find("/DEBUG[:{FASTLINK|FULL|NONE}]") != -1

def _find_llvm_path(repository_ctx):
    """Find LLVM install path."""

    # 1. Check if BAZEL_LLVM is already set by user.
    bazel_llvm = _get_path_env_var(repository_ctx, "BAZEL_LLVM")
    if bazel_llvm:
        return bazel_llvm

    auto_configure_warning_maybe(repository_ctx, "'BAZEL_LLVM' is not set, " +
                                                 "start looking for LLVM installation on machine.")

    # 2. Look for LLVM installation through registry.
    auto_configure_warning_maybe(repository_ctx, "Looking for LLVM installation through registry")
    reg_binary = _get_system_root(repository_ctx) + "\\system32\\reg.exe"
    llvm_dir = None
    result = repository_ctx.execute([reg_binary, "query", "HKEY_LOCAL_MACHINE\\SOFTWARE\\WOW6432Node\\LLVM\\LLVM"])
    auto_configure_warning_maybe(repository_ctx, "registry query result for LLVM:\n\nSTDOUT(start)\n%s\nSTDOUT(end)\nSTDERR(start):\n%s\nSTDERR(end)\n" %
                                                 (result.stdout, result.stderr))
    if not result.stderr:
        for line in result.stdout.split("\n"):
            line = line.strip()
            if line.startswith("(Default)") and line.find("REG_SZ") != -1:
                llvm_dir = line[line.find("REG_SZ") + len("REG_SZ"):].strip()
    if llvm_dir:
        auto_configure_warning_maybe(repository_ctx, "LLVM installation found at %s" % llvm_dir)
        return llvm_dir

    # 3. Check default directories for LLVM installation
    auto_configure_warning_maybe(repository_ctx, "Looking for default LLVM installation directory")
    program_files_dir = _get_path_env_var(repository_ctx, "PROGRAMFILES")
    if not program_files_dir:
        program_files_dir = "C:\\Program Files"
        auto_configure_warning_maybe(
            repository_ctx,
            "'PROGRAMFILES' environment variable is not set, using '%s' as default" % program_files_dir,
        )
    path = program_files_dir + "\\LLVM"
    if repository_ctx.path(path).exists:
        llvm_dir = path

    if not llvm_dir:
        auto_configure_warning_maybe(repository_ctx, "LLVM installation not found.")
        return None
    auto_configure_warning_maybe(repository_ctx, "LLVM installation found at %s" % llvm_dir)
    return llvm_dir

def _find_llvm_tool(repository_ctx, llvm_path, tool):
    """Find the exact path of a specific build tool in LLVM. Doesn't %-escape the result."""
    tool_path = llvm_path + "\\bin\\" + tool

    if not repository_ctx.path(tool_path).exists:
        return None

    return tool_path.replace("\\", "/")

def _use_clang_cl(repository_ctx):
    """Returns True if USE_CLANG_CL is set to 1."""
    return repository_ctx.os.environ.get("USE_CLANG_CL", default = "0") == "1"

def _find_missing_llvm_tools(repository_ctx, llvm_path):
    """Check if any required tool is missing under given LLVM path."""
    missing_tools = []
    for tool in ["clang-cl.exe", "lld-link.exe", "llvm-lib.exe"]:
        if not _find_llvm_tool(repository_ctx, llvm_path, tool):
            missing_tools.append(tool)

    return missing_tools

def _get_clang_version(repository_ctx, clang_cl):
    result = repository_ctx.execute([clang_cl, "-v"])
    if result.return_code != 0:
        auto_configure_fail("Failed to get clang version by running \"%s -v\"" % clang_cl)

    # Stderr should look like "clang version X.X.X ..."
    return result.stderr.splitlines()[0].split(" ")[2]

def _get_msys_mingw_vars(repository_ctx):
    """Get the variables we need to populate the msys/mingw toolchains."""
    tool_paths, tool_bin_path, inc_dir_msys = _get_escaped_windows_msys_starlark_content(repository_ctx)
    tool_paths_mingw, tool_bin_path_mingw, inc_dir_mingw = _get_escaped_windows_msys_starlark_content(repository_ctx, use_mingw = True)
    write_builtin_include_directory_paths(repository_ctx, "mingw", [inc_dir_mingw], file_suffix = "_mingw")
    msys_mingw_vars = {
        "%{cxx_builtin_include_directories}": inc_dir_msys,
        "%{mingw_cxx_builtin_include_directories}": inc_dir_mingw,
        "%{mingw_tool_bin_path}": tool_bin_path_mingw,
        "%{mingw_tool_paths}": tool_paths_mingw,
        "%{tool_bin_path}": tool_bin_path,
        "%{tool_paths}": tool_paths,
    }
    return msys_mingw_vars

def _get_msvc_vars(repository_ctx, paths):
    """Get the variables we need to populate the MSVC toolchains."""
    msvc_vars = dict()
    vc_path = _find_vc_path(repository_ctx)
    missing_tools = None
    if not vc_path:
        repository_ctx.template(
            "vc_installation_error.bat",
            paths["@rules_cc//cc/private/toolchain:vc_installation_error.bat.tpl"],
            {"%{vc_error_message}": ""},
        )
    else:
        missing_tools = _find_missing_vc_tools(repository_ctx, vc_path)
        if missing_tools:
            message = "\r\n".join([
                "echo. 1>&2",
                "echo Visual C++ build tools seems to be installed at %s 1>&2" % vc_path,
                "echo But Bazel can't find the following tools: 1>&2",
                "echo     %s 1>&2" % ", ".join(missing_tools),
                "echo. 1>&2",
            ])
            repository_ctx.template(
                "vc_installation_error.bat",
                paths["@rules_cc//cc/private/toolchain:vc_installation_error.bat.tpl"],
                {"%{vc_error_message}": message},
            )

    if not vc_path or missing_tools:
        write_builtin_include_directory_paths(repository_ctx, "msvc", [], file_suffix = "_msvc")
        msvc_vars = {
            "%{dbg_mode_debug_flag}": "/DEBUG",
            "%{fastbuild_mode_debug_flag}": "/DEBUG",
            "%{msvc_cl_path}": "vc_installation_error.bat",
            "%{msvc_cxx_builtin_include_directories}": "",
            "%{msvc_env_include}": "msvc_not_found",
            "%{msvc_env_lib}": "msvc_not_found",
            "%{msvc_env_path}": "msvc_not_found",
            "%{msvc_env_tmp}": "msvc_not_found",
            "%{msvc_lib_path}": "vc_installation_error.bat",
            "%{msvc_link_path}": "vc_installation_error.bat",
            "%{msvc_ml_path}": "vc_installation_error.bat",
        }
        return msvc_vars

    env = setup_vc_env_vars(repository_ctx, vc_path)
    escaped_paths = escape_string(env["PATH"])
    escaped_include_paths = escape_string(env["INCLUDE"])
    escaped_lib_paths = escape_string(env["LIB"])
    escaped_tmp_dir = escape_string(_get_temp_env(repository_ctx).replace("\\", "\\\\"))

    llvm_path = ""
    if _use_clang_cl(repository_ctx):
        llvm_path = _find_llvm_path(repository_ctx)
        if not llvm_path:
            auto_configure_fail("\nUSE_CLANG_CL is set to 1, but Bazel cannot find Clang installation on your system.\n" +
                                "Please install Clang via http://releases.llvm.org/download.html\n")
        cl_path = _find_llvm_tool(repository_ctx, llvm_path, "clang-cl.exe")
        link_path = _find_llvm_tool(repository_ctx, llvm_path, "lld-link.exe")
        if not link_path:
            link_path = _find_msvc_tool(repository_ctx, vc_path, "link.exe")
        lib_path = _find_llvm_tool(repository_ctx, llvm_path, "llvm-lib.exe")
        if not lib_path:
            lib_path = _find_msvc_tool(repository_ctx, vc_path, "lib.exe")
    else:
        cl_path = _find_msvc_tool(repository_ctx, vc_path, "cl.exe")
        link_path = _find_msvc_tool(repository_ctx, vc_path, "link.exe")
        lib_path = _find_msvc_tool(repository_ctx, vc_path, "lib.exe")

    msvc_ml_path = _find_msvc_tool(repository_ctx, vc_path, "ml64.exe")
    escaped_cxx_include_directories = []

    for path in escaped_include_paths.split(";"):
        if path:
            escaped_cxx_include_directories.append("\"%s\"" % path)
    if llvm_path:
        clang_version = _get_clang_version(repository_ctx, cl_path)
        clang_dir = llvm_path + "\\lib\\clang\\" + clang_version
        clang_include_path = (clang_dir + "\\include").replace("\\", "\\\\")
        escaped_cxx_include_directories.append("\"%s\"" % clang_include_path)
        clang_lib_path = (clang_dir + "\\lib\\windows").replace("\\", "\\\\")
        escaped_lib_paths = escaped_lib_paths + ";" + clang_lib_path

    support_debug_fastlink = _is_support_debug_fastlink(repository_ctx, link_path)

    write_builtin_include_directory_paths(repository_ctx, "msvc", escaped_cxx_include_directories, file_suffix = "_msvc")
    msvc_vars = {
        "%{dbg_mode_debug_flag}": "/DEBUG:FULL" if support_debug_fastlink else "/DEBUG",
        "%{fastbuild_mode_debug_flag}": "/DEBUG:FASTLINK" if support_debug_fastlink else "/DEBUG",
        "%{msvc_cl_path}": cl_path,
        "%{msvc_cxx_builtin_include_directories}": "        " + ",\n        ".join(escaped_cxx_include_directories),
        "%{msvc_env_include}": escaped_include_paths,
        "%{msvc_env_lib}": escaped_lib_paths,
        "%{msvc_env_path}": escaped_paths,
        "%{msvc_env_tmp}": escaped_tmp_dir,
        "%{msvc_lib_path}": lib_path,
        "%{msvc_link_path}": link_path,
        "%{msvc_ml_path}": msvc_ml_path,
    }
    return msvc_vars

def _get_clang_cl_vars(repository_ctx, paths, msvc_vars):
    """Get the variables we need to populate the clang-cl toolchains."""
    llvm_path = _find_llvm_path(repository_ctx)
    error_script = None
    if msvc_vars["%{msvc_cl_path}"] == "vc_installation_error.bat":
        error_script = "vc_installation_error.bat"
    elif not llvm_path:
        repository_ctx.template(
            "clang_installation_error.bat",
            paths["@rules_cc//cc/private/toolchain:clang_installation_error.bat.tpl"],
            {"%{clang_error_message}": ""},
        )
        error_script = "clang_installation_error.bat"
    else:
        missing_tools = _find_missing_llvm_tools(repository_ctx, llvm_path)
        if missing_tools:
            message = "\r\n".join([
                "echo. 1>&2",
                "echo LLVM/Clang seems to be installed at %s 1>&2" % llvm_path,
                "echo But Bazel can't find the following tools: 1>&2",
                "echo     %s 1>&2" % ", ".join(missing_tools),
                "echo. 1>&2",
            ])
            repository_ctx.template(
                "clang_installation_error.bat",
                paths["@rules_cc//cc/private/toolchain:clang_installation_error.bat.tpl"],
                {"%{clang_error_message}": message},
            )
            error_script = "clang_installation_error.bat"

    if error_script:
        write_builtin_include_directory_paths(repository_ctx, "clang-cl", [], file_suffix = "_clangcl")
        clang_cl_vars = {
            "%{clang_cl_cl_path}": error_script,
            "%{clang_cl_cxx_builtin_include_directories}": "",
            "%{clang_cl_dbg_mode_debug_flag}": "/DEBUG",
            "%{clang_cl_env_include}": "clang_cl_not_found",
            "%{clang_cl_env_lib}": "clang_cl_not_found",
            "%{clang_cl_env_path}": "clang_cl_not_found",
            "%{clang_cl_env_tmp}": "clang_cl_not_found",
            "%{clang_cl_fastbuild_mode_debug_flag}": "/DEBUG",
            "%{clang_cl_lib_path}": error_script,
            "%{clang_cl_link_path}": error_script,
            "%{clang_cl_ml_path}": error_script,
        }
        return clang_cl_vars

    clang_cl_path = _find_llvm_tool(repository_ctx, llvm_path, "clang-cl.exe")
    lld_link_path = _find_llvm_tool(repository_ctx, llvm_path, "lld-link.exe")
    llvm_lib_path = _find_llvm_tool(repository_ctx, llvm_path, "llvm-lib.exe")

    clang_version = _get_clang_version(repository_ctx, clang_cl_path)
    clang_dir = llvm_path + "\\lib\\clang\\" + clang_version
    clang_include_path = (clang_dir + "\\include").replace("\\", "\\\\")
    clang_lib_path = (clang_dir + "\\lib\\windows").replace("\\", "\\\\")

    clang_cl_include_directories = msvc_vars["%{msvc_cxx_builtin_include_directories}"] + (",\n        \"%s\"" % clang_include_path)
    write_builtin_include_directory_paths(repository_ctx, "clang-cl", [clang_cl_include_directories], file_suffix = "_clangcl")
    clang_cl_vars = {
        "%{clang_cl_cl_path}": clang_cl_path,
        "%{clang_cl_cxx_builtin_include_directories}": clang_cl_include_directories,
        # LLVM's lld-link.exe doesn't support /DEBUG:FASTLINK.
        "%{clang_cl_dbg_mode_debug_flag}": "/DEBUG",
        "%{clang_cl_env_include}": msvc_vars["%{msvc_env_include}"] + ";" + clang_include_path,
        "%{clang_cl_env_lib}": msvc_vars["%{msvc_env_lib}"] + ";" + clang_lib_path,
        "%{clang_cl_env_path}": msvc_vars["%{msvc_env_path}"],
        "%{clang_cl_env_tmp}": msvc_vars["%{msvc_env_tmp}"],
        "%{clang_cl_fastbuild_mode_debug_flag}": "/DEBUG",
        "%{clang_cl_lib_path}": llvm_lib_path,
        "%{clang_cl_link_path}": lld_link_path,
        "%{clang_cl_ml_path}": msvc_vars["%{msvc_ml_path}"],
    }
    return clang_cl_vars

def configure_windows_toolchain(repository_ctx):
    """Configure C++ toolchain on Windows.

    Args:
      repository_ctx: The repository context.
    """
    paths = resolve_labels(repository_ctx, [
        "@rules_cc//cc/private/toolchain:BUILD.windows.tpl",
        "@rules_cc//cc/private/toolchain:windows_cc_toolchain_config.bzl",
        "@rules_cc//cc/private/toolchain:armeabi_cc_toolchain_config.bzl",
        "@rules_cc//cc/private/toolchain:vc_installation_error.bat.tpl",
        "@rules_cc//cc/private/toolchain:msys_gcc_installation_error.bat",
        "@rules_cc//cc/private/toolchain:clang_installation_error.bat.tpl",
    ])

    repository_ctx.symlink(
        paths["@rules_cc//cc/private/toolchain:windows_cc_toolchain_config.bzl"],
        "windows_cc_toolchain_config.bzl",
    )
    repository_ctx.symlink(
        paths["@rules_cc//cc/private/toolchain:armeabi_cc_toolchain_config.bzl"],
        "armeabi_cc_toolchain_config.bzl",
    )
    repository_ctx.symlink(
        paths["@rules_cc//cc/private/toolchain:msys_gcc_installation_error.bat"],
        "msys_gcc_installation_error.bat",
    )

    template_vars = dict()
    msvc_vars = _get_msvc_vars(repository_ctx, paths)
    template_vars.update(msvc_vars)
    template_vars.update(_get_clang_cl_vars(repository_ctx, paths, msvc_vars))
    template_vars.update(_get_msys_mingw_vars(repository_ctx))

    repository_ctx.template(
        "BUILD",
        paths["@rules_cc//cc/private/toolchain:BUILD.windows.tpl"],
        template_vars,
    )
