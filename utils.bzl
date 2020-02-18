# https://github.com/LoopPerfect/buckaroo/wiki/Building-CMake-Projects

def extract(rule, path):
  name = 'extract-' + (
    rule.replace(':','').replace('//', '') + '-' +
    path
      .replace('/','-')
      .replace('.','-'))

  if not native.rule_exists(':'+name):
    filename = path.split('/')[-1]
    native.genrule(
      name = name,
      out = filename,
      cmd = 'cp $(location '+rule+')/'+ path +' $OUT')

  return ':'+name

def extractFolder(rule, path):
  name = 'extract-folder-' + (
    rule.replace(':','') + '-' +
    path
      .replace('/','-')
      .replace('.','-'))

  if not native.rule_exists(':'+name):
    native.genrule(
      name = name,
      out = 'out',
      cmd = 'mkdir $OUT && cd $OUT && cp -r $(location '+rule+')/'+ path +'/. .')

  return ':'+name

def pkgconfig(name, find, search = None, visibility = []):
  env = 'PKG_CONFIG_PATH=' + search + ' ' if search else ''

  native.genrule(
    name = name+'-flags',
    out = 'out.txt',
    cmd = env + 'pkg-config ' + find + ' --cflags > $OUT')

  native.genrule(
    name = name+'-linker',
    out = 'out.txt',
    cmd = env + 'pkg-config ' + find + ' --libs > $OUT')

  native.prebuilt_cxx_library(
    name = name,
    header_namespace = '',
      header_only = True,
      exported_preprocessor_flags = [
        '@$(location :'+name+'-flags)',
      ],
      exported_linker_flags = [
        '@$(location :'+name+'-linker)',
      ],
      visibility = visibility)


def cmake_args(args):
    # type: (Dict[str, str])
    args = ['-D{}={}'.format(name, "1" if value else "0") for name, value in args.items()]
    return ' '.join(args)


def cmake(name, srcs = [], options = [], targets = [], out = 'build', prefix = 'ROOT', jobs = 1, args = {}):
  jobs = str(jobs)

  make_cmd = 'make -j{} {}'.format(jobs, ' '.join(targets))
  print(make_cmd)
  native.genrule(
    name = name,
    srcs = srcs,
    out = out,
    cmd = ' && '.join([
      'mkdir $OUT $OUT/'+prefix,
      'cd $OUT',
      'cmake -DCMAKE_INSTALL_PREFIX:PATH=$OUT/{} {} $SRCDIR'.format(prefix, cmake_args(args)),
      # 'make -j{} {}'.format(jobs, ' '.join(targets))
      make_cmd
    ]
      # (['make -j'+ jobs + ' '.join(targets) ] if len(targets) else [])
    ))


"""Provides utility macros for working with globs."""

def _paths_join(path, *others):
    """Joins one or more path components."""
    result = path

    for p in others:
        if p.startswith("/"):  # absolute
            result = p
        elif not result or result.endswith("/"):
            result += p
        else:
            result += "/" + p

    return result

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
    files = native.glob([_paths_join(dirpath, glob_pattern)], exclude = exclude)
    for f in files:
        if dirpath:
            key = f[len(dirpath) + 1:]
        else:
            key = f
        if prefix:
            key = _paths_join(prefix, key)
        results[key] = f

    return results
