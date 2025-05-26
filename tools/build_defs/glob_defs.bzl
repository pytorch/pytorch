# Only used for PyTorch open source BUCK build

"""Provides utility macros for working with globs."""

load("@bazel_skylib//lib:paths.bzl", "paths")

def subdir_glob(glob_specs, exclude = None, prefix = ""):
    """Returns a dict of sub-directory relative paths to full paths.

    The subdir_glob() function is useful for defining header maps for C/C++
    libraries which should be relative the given sub-directory.
    Given a list of tuples, the form of (relative-sub-directory, glob-pattern),
    it returns a dict of sub-directory relative paths to full paths.

    Please refer to native.glob() for explanations and examples of the pattern.

    Args:
      glob_specs: The array of tuples in form of
        (relative-sub-directory, glob-pattern inside relative-sub-directory).
        type: List[Tuple[str, str]]
      exclude: A list of patterns to identify files that should be removed
        from the set specified by the first argument. Defaults to [].
        type: Optional[List[str]]
      prefix: If is not None, prepends it to each key in the dictionary.
        Defaults to None.
        type: Optional[str]

    Returns:
      A dict of sub-directory relative paths to full paths.
    """
    if exclude == None:
        exclude = []

    results = []

    for dirpath, glob_pattern in glob_specs:
        results.append(
            _single_subdir_glob(dirpath, glob_pattern, exclude, prefix),
        )

    return _merge_maps(*results)

def _merge_maps(*file_maps):
    result = {}
    for file_map in file_maps:
        for key in file_map:
            if key in result and result[key] != file_map[key]:
                fail(
                    "Conflicting files in file search paths. " +
                    "\"%s\" maps to both \"%s\" and \"%s\"." %
                    (key, result[key], file_map[key]),
                )

            result[key] = file_map[key]

    return result

def _single_subdir_glob(dirpath, glob_pattern, exclude = None, prefix = None):
    if exclude == None:
        exclude = []
    results = {}
    files = native.glob([paths.join(dirpath, glob_pattern)], exclude = exclude)
    for f in files:
        if dirpath:
            key = f[len(dirpath) + 1:]
        else:
            key = f
        if prefix:
            key = paths.join(prefix, key)
        results[key] = f

    return results

# Using a flat list will trigger build errors on Android.
# cxx_library will generate an apple_library on iOS, a cxx_library on Android.
# Those rules have different behaviors. Using a map will make the behavior consistent.
#
def glob_private_headers(glob_patterns, exclude = []):
    result = {}
    headers = native.glob(glob_patterns, exclude = exclude)
    for header in headers:
        result[paths.basename(header)] = header
    return result

def glob(include, exclude = (), **kwargs):
    buildfile = native.read_config("buildfile", "name", "BUCK")
    subpkgs = [
        target[:-len(buildfile)] + "**/*"
        for target in native.glob(["*/**/" + buildfile])
    ]
    return native.glob(include, exclude = list(exclude) + subpkgs, **kwargs)
