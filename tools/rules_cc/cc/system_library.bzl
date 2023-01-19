"""system_library is a repository rule for importing system libraries"""

BAZEL_LIB_ADDITIONAL_PATHS_ENV_VAR = "BAZEL_LIB_ADDITIONAL_PATHS"
BAZEL_LIB_OVERRIDE_PATHS_ENV_VAR = "BAZEL_LIB_OVERRIDE_PATHS"
BAZEL_INCLUDE_ADDITIONAL_PATHS_ENV_VAR = "BAZEL_INCLUDE_ADDITIONAL_PATHS"
BAZEL_INCLUDE_OVERRIDE_PATHS_ENV_VAR = "BAZEL_INCLUDE_OVERRIDE_PATHS"
ENV_VAR_SEPARATOR = ","
ENV_VAR_ASSIGNMENT = "="

def _make_flags(flag_values, flag):
    flags = []
    if flag_values:
        for s in flag_values:
            flags.append(flag + s)
    return " ".join(flags)

def _split_env_var(repo_ctx, var_name):
    value = repo_ctx.os.environ.get(var_name)
    if value:
        assignments = value.split(ENV_VAR_SEPARATOR)
        dict = {}
        for assignment in assignments:
            pair = assignment.split(ENV_VAR_ASSIGNMENT)
            if len(pair) != 2:
                fail(
                    "Assignments should have form 'name=value', " +
                    "but encountered {} in env variable {}"
                        .format(assignment, var_name),
                )
            key, value = pair[0], pair[1]
            if not dict.get(key):
                dict[key] = []
            dict[key].append(value)
        return dict
    else:
        return {}

def _get_list_from_env_var(repo_ctx, var_name, key):
    return _split_env_var(repo_ctx, var_name).get(key, default = [])

def _execute_bash(repo_ctx, cmd):
    return repo_ctx.execute(["/bin/bash", "-c", cmd]).stdout.strip("\n")

def _find_linker(repo_ctx):
    ld = _execute_bash(repo_ctx, "which ld")
    lld = _execute_bash(repo_ctx, "which lld")
    if ld:
        return ld
    elif lld:
        return lld
    else:
        fail("No linker found")

def _find_compiler(repo_ctx):
    gcc = _execute_bash(repo_ctx, "which g++")
    clang = _execute_bash(repo_ctx, "which clang++")
    if gcc:
        return gcc
    elif clang:
        return clang
    else:
        fail("No compiler found")

def _find_lib_path(repo_ctx, lib_name, archive_names, lib_path_hints):
    override_paths = _get_list_from_env_var(
        repo_ctx,
        BAZEL_LIB_OVERRIDE_PATHS_ENV_VAR,
        lib_name,
    )
    additional_paths = _get_list_from_env_var(
        repo_ctx,
        BAZEL_LIB_ADDITIONAL_PATHS_ENV_VAR,
        lib_name,
    )

    # Directories will be searched in order
    path_flags = _make_flags(
        override_paths + lib_path_hints + additional_paths,
        "-L",
    )
    linker = _find_linker(repo_ctx)
    for archive_name in archive_names:
        cmd = """
              {} -verbose -l:{} {} 2>/dev/null | \\
              grep succeeded | \\
              head -1 | \\
              sed -e 's/^\\s*attempt to open //' -e 's/ succeeded\\s*$//'
              """.format(
            linker,
            archive_name,
            path_flags,
        )
        path = _execute_bash(repo_ctx, cmd)
        if path:
            return (archive_name, path)
    return (None, None)

def _find_header_path(repo_ctx, lib_name, header_name, includes):
    override_paths = _get_list_from_env_var(
        repo_ctx,
        BAZEL_INCLUDE_OVERRIDE_PATHS_ENV_VAR,
        lib_name,
    )
    additional_paths = _get_list_from_env_var(
        repo_ctx,
        BAZEL_INCLUDE_ADDITIONAL_PATHS_ENV_VAR,
        lib_name,
    )

    compiler = _find_compiler(repo_ctx)
    cmd = """
          print | \\
          {} -Wp,-v -x c++ - -fsyntax-only 2>&1 | \\
          sed -n -e '/^\\s\\+/p' | \\
          sed -e 's/^[ \t]*//'
          """.format(compiler)
    system_includes = _execute_bash(repo_ctx, cmd).split("\n")
    all_includes = (override_paths + includes +
                    system_includes + additional_paths)

    for directory in all_includes:
        cmd = """
              test -f "{dir}/{hdr}" && echo "{dir}/{hdr}"
              """.format(dir = directory, hdr = header_name)
        result = _execute_bash(repo_ctx, cmd)
        if result:
            return result
    return None

def _system_library_impl(repo_ctx):
    repo_name = repo_ctx.attr.name
    includes = repo_ctx.attr.includes
    hdrs = repo_ctx.attr.hdrs
    optional_hdrs = repo_ctx.attr.optional_hdrs
    deps = repo_ctx.attr.deps
    lib_path_hints = repo_ctx.attr.lib_path_hints
    static_lib_names = repo_ctx.attr.static_lib_names
    shared_lib_names = repo_ctx.attr.shared_lib_names

    static_lib_name, static_lib_path = _find_lib_path(
        repo_ctx,
        repo_name,
        static_lib_names,
        lib_path_hints,
    )
    shared_lib_name, shared_lib_path = _find_lib_path(
        repo_ctx,
        repo_name,
        shared_lib_names,
        lib_path_hints,
    )

    if not static_lib_path and not shared_lib_path:
        fail("Library {} could not be found".format(repo_name))

    hdr_names = []
    hdr_paths = []
    for hdr in hdrs:
        hdr_path = _find_header_path(repo_ctx, repo_name, hdr, includes)
        if hdr_path:
            repo_ctx.symlink(hdr_path, hdr)
            hdr_names.append(hdr)
            hdr_paths.append(hdr_path)
        else:
            fail("Could not find required header {}".format(hdr))

    for hdr in optional_hdrs:
        hdr_path = _find_header_path(repo_ctx, repo_name, hdr, includes)
        if hdr_path:
            repo_ctx.symlink(hdr_path, hdr)
            hdr_names.append(hdr)
            hdr_paths.append(hdr_path)

    hdrs_param = "hdrs = {},".format(str(hdr_names))

    # This is needed for the case when quote-includes and system-includes
    # alternate in the include chain, i.e.
    # #include <SDL2/SDL.h> -> #include "SDL_main.h"
    # -> #include <SDL2/_real_SDL_config.h> -> #include "SDL_platform.h"
    # The problem is that the quote-includes are assumed to be
    # in the same directory as the header they are included from -
    # they have no subdir prefix ("SDL2/") in their paths
    include_subdirs = {}
    for hdr in hdr_names:
        path_segments = hdr.split("/")
        path_segments.pop()
        current_path_segments = ["external", repo_name]
        for segment in path_segments:
            current_path_segments.append(segment)
            current_path = "/".join(current_path_segments)
            include_subdirs.update({current_path: None})

    includes_param = "includes = {},".format(str(include_subdirs.keys()))

    deps_names = []
    for dep in deps:
        dep_name = repr("@" + dep)
        deps_names.append(dep_name)
    deps_param = "deps = [{}],".format(",".join(deps_names))

    link_hdrs_command = "mkdir -p $(RULEDIR)/remote \n"
    remote_hdrs = []
    for path, hdr in zip(hdr_paths, hdr_names):
        remote_hdr = "remote/" + hdr
        remote_hdrs.append(remote_hdr)
        link_hdrs_command += "cp {path} $(RULEDIR)/{hdr}\n ".format(
            path = path,
            hdr = remote_hdr,
        )

    link_remote_static_lib_genrule = ""
    link_remote_shared_lib_genrule = ""
    remote_static_library_param = ""
    remote_shared_library_param = ""
    static_library_param = ""
    shared_library_param = ""

    if static_lib_path:
        repo_ctx.symlink(static_lib_path, static_lib_name)
        static_library_param = "static_library = \"{}\",".format(
            static_lib_name,
        )
        remote_static_library = "remote/" + static_lib_name
        link_library_command = """
mkdir -p $(RULEDIR)/remote && cp {path} $(RULEDIR)/{lib}""".format(
            path = static_lib_path,
            lib = remote_static_library,
        )
        remote_static_library_param = """
static_library = "remote_link_static_library","""
        link_remote_static_lib_genrule = """
genrule(
     name = "remote_link_static_library",
     outs = ["{remote_static_library}"],
     cmd = {link_library_command}
)
""".format(
            link_library_command = repr(link_library_command),
            remote_static_library = remote_static_library,
        )

    if shared_lib_path:
        repo_ctx.symlink(shared_lib_path, shared_lib_name)
        shared_library_param = "shared_library = \"{}\",".format(
            shared_lib_name,
        )
        remote_shared_library = "remote/" + shared_lib_name
        link_library_command = """
mkdir -p $(RULEDIR)/remote && cp {path} $(RULEDIR)/{lib}""".format(
            path = shared_lib_path,
            lib = remote_shared_library,
        )
        remote_shared_library_param = """
shared_library = "remote_link_shared_library","""
        link_remote_shared_lib_genrule = """
genrule(
        name = "remote_link_shared_library",
        outs = ["{remote_shared_library}"],
        cmd = {link_library_command}
)
""".format(
            link_library_command = repr(link_library_command),
            remote_shared_library = remote_shared_library,
        )

    repo_ctx.file(
        "BUILD",
        executable = False,
        content =
            """
load("@bazel_tools//tools/build_defs/cc:cc_import.bzl", "cc_import")
cc_import(
    name = "local_includes",
    {static_library}
    {shared_library}
    {hdrs}
    {deps}
    {includes}
)

genrule(
    name = "remote_link_headers",
    outs = {remote_hdrs},
    cmd = {link_hdrs_command}
)

{link_remote_static_lib_genrule}

{link_remote_shared_lib_genrule}

cc_import(
    name = "remote_includes",
    hdrs = [":remote_link_headers"],
    {remote_static_library}
    {remote_shared_library}
    {deps}
    {includes}
)

alias(
    name = "{name}",
    actual = select({{
        "@bazel_tools//src/conditions:remote": "remote_includes",
        "//conditions:default": "local_includes",
    }}),
    visibility = ["//visibility:public"],
)
""".format(
                static_library = static_library_param,
                shared_library = shared_library_param,
                hdrs = hdrs_param,
                deps = deps_param,
                hdr_names = str(hdr_names),
                link_hdrs_command = repr(link_hdrs_command),
                name = repo_name,
                includes = includes_param,
                remote_hdrs = remote_hdrs,
                link_remote_static_lib_genrule = link_remote_static_lib_genrule,
                link_remote_shared_lib_genrule = link_remote_shared_lib_genrule,
                remote_static_library = remote_static_library_param,
                remote_shared_library = remote_shared_library_param,
            ),
    )

system_library = repository_rule(
    implementation = _system_library_impl,
    local = True,
    remotable = True,
    environ = [
        BAZEL_INCLUDE_ADDITIONAL_PATHS_ENV_VAR,
        BAZEL_INCLUDE_OVERRIDE_PATHS_ENV_VAR,
        BAZEL_LIB_ADDITIONAL_PATHS_ENV_VAR,
        BAZEL_LIB_OVERRIDE_PATHS_ENV_VAR,
    ],
    attrs = {
        "deps": attr.string_list(doc = """
List of names of system libraries this target depends upon.
"""),
        "hdrs": attr.string_list(
            mandatory = True,
            allow_empty = False,
            doc = """
List of the library's public headers which must be imported.
""",
        ),
        "includes": attr.string_list(doc = """
List of directories that should be browsed when looking for headers.
"""),
        "lib_path_hints": attr.string_list(doc = """
List of directories that should be browsed when looking for library archives.
"""),
        "optional_hdrs": attr.string_list(doc = """
List of library's private headers.
"""),
        "shared_lib_names": attr.string_list(doc = """
List of possible shared library names in order of preference.
"""),
        "static_lib_names": attr.string_list(doc = """
List of possible static library names in order of preference.
"""),
    },
    doc =
        """system_library is a repository rule for importing system libraries

`system_library` is a repository rule for safely depending on system-provided
libraries on Linux. It can be used with remote caching and remote execution.
Under the hood it uses gcc/clang for finding the library files and headers
and symlinks them into the build directory. Symlinking allows Bazel to take
these files into account when it calculates a checksum of the project.
This prevents cache poisoning from happening.

Currently `system_library` requires two exeperimental flags:
--experimental_starlark_cc_import
--experimental_repo_remote_exec

A typical usage looks like this:
WORKSPACE
```
system_library(
    name = "jpeg",
    hdrs = ["jpeglib.h"],
    shared_lib_names = ["libjpeg.so, libjpeg.so.62"],
    static_lib_names = ["libjpeg.a"],
    includes = ["/usr/additional_includes"],
    lib_path_hints = ["/usr/additional_libs", "/usr/some/other_path"]
    optional_hdrs = [
        "jconfig.h",
        "jmorecfg.h",
    ],
)

system_library(
    name = "bar",
    hdrs = ["bar.h"],
    shared_lib_names = ["libbar.so"],
    deps = ["jpeg"]

)
```

BUILD
```
cc_binary(
    name = "foo",
    srcs = ["foo.cc"],
    deps = ["@bar"]
)
```

foo.cc
```
#include "jpeglib.h"
#include "bar.h"

[code using symbols from jpeglib and bar]
```

`system_library` requires users to specify at least one header
(as it makes no sense to import a library without headers).
Public headers of a library (i.e. those included in the user-written code,
like `jpeglib.h` in the example above) should be put in `hdrs` param, as they
are required for the library to work. However, some libraries may use more
"private" headers. They should be imported as well, but their names may differ
from system to system. They should be specified in the `optional_hdrs` param.
The build will not fail if some of them are not found, so it's safe to put a
superset there, containing all possible combinations of names for different
versions/distributions. It's up to the user to determine which headers are
required for the library to work.

One `system_library` target always imports exactly one library.
Users can specify many potential names for the library file,
as these names can differ from system to system. The order of names establishes
the order of preference. As some libraries can be linked both statically
and dynamically, the names of files of each kind can be specified separately.
`system_library` rule will try to find library archives of both kinds, but it's
up to the top-level target (for example, `cc_binary`) to decide which kind of
linking will be used.

`system_library` rule depends on gcc/clang (whichever is installed) for
finding the actual locations of library archives and headers.
Libraries installed in a standard way by a package manager
(`sudo apt install libjpeg-dev`) are usually placed in one of directories
searched by the compiler/linker by default - on Ubuntu library most archives
are stored in `/usr/lib/x86_64-linux-gnu/` and their headers in
`/usr/include/`. If the maintainer of a project expects the files
to be installed in a non-standard location, they can use the `includes`
parameter to add directories to the search path for headers
and `lib_path_hints` to add directories to the search path for library
archives.

User building the project can override or extend these search paths by
providing these environment variables to the build:
BAZEL_INCLUDE_ADDITIONAL_PATHS, BAZEL_INCLUDE_OVERRIDE_PATHS,
BAZEL_LIB_ADDITIONAL_PATHS, BAZEL_LIB_OVERRIDE_PATHS.
The syntax for setting the env variables is:
`<library>=<path>,<library>=<path2>`.
Users can provide multiple paths for one library by repeating this segment:
`<library>=<path>`.

So in order to build the example presented above but with custom paths for the
jpeg lib, one would use the following command:

```
bazel build //:foo \
  --experimental_starlark_cc_import \
  --experimental_repo_remote_exec \
  --action_env=BAZEL_LIB_OVERRIDE_PATHS=jpeg=/custom/libraries/path \
  --action_env=BAZEL_INCLUDE_OVERRIDE_PATHS=jpeg=/custom/include/path,jpeg=/inc
```

Some libraries can depend on other libraries. `system_library` rule provides
a `deps` parameter for specifying such relationships. `system_library` targets
can depend only on other system libraries.
""",
)
