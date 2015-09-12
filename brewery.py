
import cPickle as pickle
from collections import defaultdict
import multiprocessing
import glob
import hashlib
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import traceback

from build_env import Env

# global variables
CAFFE2_RUN_TEST = False

class Colors(object):
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'

def BuildDebug(message, *args):
  # Note(Yangqing): if you want to know detailed message about the build,
  # uncomment the following line.
  print Colors.OKBLUE + 'DEBUG:', message % args, Colors.ENDC
  return

def BuildLog(message, *args):
  print Colors.OKGREEN + 'LOG:', message % args, Colors.ENDC

def BuildWarning(message, *args):
  print Colors.WARNING + 'WARNING:', message % args, Colors.ENDC

def BuildFatal(message, *args):
  print Colors.FAIL + 'FATAL:', message % args, Colors.ENDC
  print Colors.FAIL + 'Build exiting.' + Colors.ENDC
  Brewery.Finalize()
  sys.exit(1)

def BuildFatalIf(command, message, *args):
  if command:
    BuildFatal(message, *args)

_single_command_env = os.environ
if 'PYTHONPATH' not in _single_command_env:
  _single_command_env['PYTHONPATH'] = ''
_single_command_env['PYTHONPATH'] = (
    Env.GENDIR + ':' + _single_command_env['PYTHONPATH'])

def RunSingleCommand(command):
  BuildDebug(command)
  try:
    proc = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, env=_single_command_env)
    stdout, _ = proc.communicate()
    if proc.returncode:
      print stdout
    return proc.returncode
  except: # all exceptions caught here.
    e = sys.exc_info()[0]
    return str(e)

def Glob(patterns):
  """Globs all files with the given patterns, relative to the path of the BREW
  file."""
  files = []
  if type(patterns) is str:
    patterns = [patterns]
  for pattern in patterns:
    full_pattern = os.path.join(Brewery.CWD, pattern)
    files += glob.glob(full_pattern)
  prefix_len = len(Brewery.CWD) + 1
  return [f[prefix_len:] for f in files if os.path.isfile(f)]

def RectifyFileName(name):
  """Rectifies a build file name to its absolute name."""
  if name.startswith("//"):
    # Simply replace the "//" with the root folder.
    out_name = name[2:]
  else:
    # Add the current working directory.
    out_name = os.path.join(Brewery.CWD, name)
  # check if the name exists.
  BuildFatalIf(not os.path.exists(out_name), 'Cannot find file %s' % out_name)
  return out_name

def RectifyFileNames(names):
  return [RectifyFileName(n) for n in sorted(names)]

def RectifyTarget(name):
  """Rectifies a build target name."""
  if name.startswith("//"):
    return name
  elif name.startswith(":"):
    return Brewery.TARGET_PREFIX + name
  else:
    if Brewery.TARGET_PREFIX == '//':
      return Brewery.TARGET_PREFIX + name
    return Brewery.TARGET_PREFIX + ":" + name

def RectifyTargets(names):
  return [RectifyTarget(n) for n in sorted(names)]

def MakeGenDirs(rectified_srcs):
  for src in rectified_srcs:
    dst = os.path.join(Env.GENDIR, src)
    try:
      os.makedirs(os.path.dirname(dst))
    except OSError as e:
      pass

def CopyToGenDir(rectified_srcs):
  MakeGenDirs(rectified_srcs)
  for src in rectified_srcs:
    shutil.copyfile(src, GenFilename(src))

def GenFilename(name, new_ext=None, original_ext=None):
  if new_ext:
    if original_ext:
      new_name = name[:name.rfind(original_ext)] + new_ext
    else:
      new_name = name[:name.rfind('.') + 1] + new_ext
  else:
    new_name = name
  return os.path.join(Env.GENDIR, new_name)

def MergeOrderedObjs(dep_lists):
  added = set()
  output = []
  for dep_list in dep_lists:
    for item in dep_list[::-1]:
      if item not in added:
        added.add(item)
        output.insert(0, item)
  return output

class Brewery(object):
  # Targets store the dictionary from the target name to the build objects.
  _targets = dict()
  # Success stores whether a target is successfully built.
  _success = defaultdict(bool)
  # deps_map is a dictionary mapping each target to its dependents.
  _deps_map = dict()
  # signature_map is the map that stores the signatures for build targets.
  _signatures = defaultdict(str)
  _signature_filename = 'brewery.signature'
  # Pool is the compute pool that one can use to run a list of commands in
  # parallel.
  Pool = multiprocessing.Pool(Env.CPUS * 2)
  #Pool = multiprocessing.Pool(1)
  CWD = ''
  TARGET_PREFIX = '//'
  TMPDIR = ''

  def __init__(self):
    """Brewery is a singleton and should not be instantiated."""
    raise NotImplementedError(
        'Build system error: there shall only be one brewery.')

  @classmethod
  def InitBrewery(cls):
    """Initializes the brewery, e.g. loads the signatures currently built."""
    try:
      os.makedirs(Env.GENDIR)
    except OSError as e:
      pass
    cls.TMPDIR = tempfile.mkdtemp()
    if os.path.exists(os.path.join(Env.GENDIR, cls._signature_filename)):
      BuildDebug('Loading the signature file.')
      cls._signatures = pickle.load(
          open(os.path.join(Env.GENDIR, cls._signature_filename)))
    cls.FindAndParseBuildFiles()

  @classmethod
  def Finalize(cls):
    """Finalizes the brew process."""
    if os.path.exists(Env.GENDIR):
      BuildDebug('Saving the signature file.')
      pickle.dump(cls._signatures,
                  open(os.path.join(Env.GENDIR, cls._signature_filename), 'w'))
    else:
      BuildDebug('No gendir present. Exiting.')
    shutil.rmtree(cls.TMPDIR)

  @classmethod
  def Get(cls, name):
    return cls._targets[name]

  @classmethod
  def FindAndParseBuildFiles(cls):
    """Find and parse all the BREW files in the subfolders."""
    build_files = [os.path.join(d[2:], f)
                   for (d, _, files) in os.walk('.') if not d.startswith(Env.GENDIR)
                   for f in files if f.endswith('BREW')]
    for build_file in build_files:
      # Set the current working directory of the environment, and parse the build
      # file.
      BuildDebug("Parsing %s" % build_file)
      cls.SetCwd(os.path.dirname(build_file))
      execfile(build_file)
    cls.SetCwd('')
    return

  @classmethod
  def SetCwd(cls, cwd):
    if cwd and not os.path.isdir(cwd):
      # cwd should either be empty, or is a directory.
      raise RuntimeError('Setting an invalid cwd: %s' % cwd)
    cls.CWD = cwd
    cls.TARGET_PREFIX = '//' + cwd

  @classmethod
  def RunInParallel(cls, commands):
    fail = any(cls.Pool.map(RunSingleCommand, commands))
    sys.stdout.flush()
    if fail:
      BuildWarning('Command failed.')
    return not fail

  @classmethod
  def Register(cls, name, target):
    BuildFatalIf(name in cls._targets,
                 "%s already in build target.", name)
    BuildDebug("Registered build target %s, deps %s", name, str(target.deps))
    cls._targets[name] = target
    cls._deps_map[name] = target.deps + target.optional_deps

  @classmethod
  def _GetExecutionChain(cls, targets):
    """Gets the execution chain."""
    # First, verify all dependencies.
    for t in cls._targets:
      for d in cls._deps_map[t]:
        BuildFatalIf(d not in cls._targets,
            "Dependency %s for target %s does not exist.", d, t)
    if len(targets) == 0:
      targets = cls._targets
    else:
      # Get all targets that we need to build.
      seen_targets = set(targets)
      idx = 0
      while idx < len(targets):
        for d in cls._deps_map[targets[idx]]:
          if d not in seen_targets:
             seen_targets.add(d)
             targets.append(d)
        idx += 1
    # Now, create a topological order.
    inverse_deps_map = defaultdict(list)
    # Get the graph of all targets
    for t in targets:
      for d in cls._deps_map[t]:
        inverse_deps_map[d].append(t)
    deps_count = dict((t, len(cls._deps_map[t])) for t in targets)
    #BuildDebug("deps count: %s", str(deps_count))
    frontier = set(t for t in deps_count if deps_count[t] == 0)
    build_order = []
    while frontier:
      current = frontier.pop()
      #BuildDebug("processing %s", current)
      build_order.append(current)
      for t in inverse_deps_map[current]:
        deps_count[t] -= 1
        if deps_count[t] == 0:
          #BuildDebug('Add to frontier: %s', t)
          frontier.add(t)
    # If this does not cover all targets, the graph is not a DAG.
    BuildFatalIf(len(build_order) != len(targets),
                 "There are cycles in the dependency graph!")
    BuildDebug('Build order: %s', str(build_order))
    return build_order

  @classmethod
  def Signature(cls, target):
    # Returns the builtsignature of the current target.
    return cls._signatures[target]

  @classmethod
  def Success(cls, target):
    return cls._success[target]

  @classmethod
  def ClearSignature(cls, including_third_party=False):
    if including_third_party:
      cls._signatures = defaultdict(str)
    else:
      keys = cls._signatures.keys()
      for k in keys:
        if not k.startswith('//third_party'):
          del cls._signatures[k]

  @classmethod
  def Build(cls, targets):
    """Build all the targets, using their topological order."""
    BuildDebug("Start building.")
    build_order = cls._GetExecutionChain(targets)
    for t in build_order:
      BuildLog("Building %s", t)
      cls._success[t], changed, new_signature = (
          cls._targets[t].SetUpAndBuild(cls._signatures[t]))
      if cls._success[t]:
        cls._signatures[t] = new_signature
    # Finally, print a summary of the build results.
    succeeded = [key for key in cls._success if cls._success[key]]
    BuildDebug("Successfully built %d targets." % len(succeeded))
    #for key in cls._success:
    #  if cls._success[key]:
    #    BuildDebug(key)
    failed = [key for key in cls._success if not cls._success[key]]
    if len(failed) > 0:
      BuildWarning("Failed to build:")
      for key in failed:
        BuildWarning(key)

  @classmethod
  def Draw(cls):
    import pydot
    graph = pydot.Dot("brewery", rankdir="LR")
    nodes = {}
    node_style = {'shape': 'box', 'color': '#0F9D58', 'style': 'filled',
                  'fontcolor': '#FFFFFF'}
    for target_name in cls._targets:
      nodes[target_name] = pydot.Node('"' + target_name + '"', **node_style)
      graph.add_node(nodes[target_name])
    for target_name in cls._deps_map:
      for dep_name in cls._deps_map[target_name]:
        graph.add_edge(pydot.Edge(nodes[dep_name], nodes[target_name]))
    graph.write(graph.get_name() + '.dot', format='raw')
    with open(graph.get_name() + '.pdf', 'w') as fid:
      subprocess.call(['dot', '-Tpdf', graph.get_name() + '.dot'], stdout=fid)

class BuildTarget(object):
  """A build target that can be executed with the Build() function."""
  def __init__(self, name, srcs, other_files=[], deps=[], optional_deps=[]):
    self.name = RectifyTarget(name)
    self.srcs = RectifyFileNames(srcs)
    self.files = sorted(self.srcs + other_files)
    self.deps = sorted(RectifyTargets(deps))
    self.optional_deps = sorted(RectifyTargets(optional_deps))
    self.command_groups = []
    Brewery.Register(self.name, self)

  def GetSignature(self):
    """Generate the signature of the build object, and see if we need to
    rebuild it."""
    src_digest = ''.join([hashlib.sha256(open(f, 'rb').read()).hexdigest()
                           for f in self.files])
    dep_digest = ''.join([Brewery.Signature(d) for d in self.deps])
    command_digest = str(self.command_groups)
    return hashlib.sha256(src_digest + dep_digest + command_digest).hexdigest()

  def SetUpAndBuild(self, built_signature):
    # Add successful optional dependencies into deps.
    self.deps += [dep for dep in self.optional_deps
                  if Brewery.Success(dep)]
    self.SetUp()
    signature = self.GetSignature()
    if not all(Brewery.Success(d) for d in self.deps):
      BuildWarning("Not all dependencies have succeeded. Skipping build. "
                   "Failed dependencies: ")
      BuildWarning(str([d for d in self.deps if not Brewery.Success(d)]))
      return False, True, signature
    if signature != built_signature:
      success = self.Build()
      return success, True, signature
    return True, False, signature

  def SetUp(self):
    """Set up the build object's variables.

    This will always run even if the target has already been built. Anything
    that further dependencies will need should be implemented here.

    If your target just emits a set of shell commands, in SetUp() you can set
    self.command_groups and use the default Build function, which basically
    sends the command groups to a execution pool.
    """
    BuildFatal('Not implemented.')

  def Build(self):
    """Builds the target."""
    success = True
    for command_group in self.command_groups:
      success &= Brewery.RunInParallel(command_group)
      if not success:
        return False
    return True

class proto_library(BuildTarget):
  """Builds a protobuffer library.

  A protobuffer library builds a set of protobuffer source files to its cc and
  python source files, as well as the static library named "libname.a".
  """
  def __init__(self, name, srcs, deps=[], **kwargs):
    BuildTarget.__init__(self, name, srcs, deps=deps, **kwargs)

  def SetUp(self):
    MakeGenDirs(self.srcs)
    # proto_library depends on protoc, so it would need to add that to the
    # includes folder.
    pbcc_files = [GenFilename(filename, 'pb.cc') for filename in self.srcs]
    pbo_files = [GenFilename(filename, 'pb.o') for filename in self.srcs]
    proto_commands = [
      ' '.join([Env.PROTOC_BINARY, '-I.', '--cpp_out', Env.GENDIR,
                '--python_out', Env.GENDIR, filename])
      for filename in self.srcs]
    cpp_commands = [
        ' '.join([Env.CC, Env.CFLAGS, Env.INCLUDES, '-c', pbcc, '-o', pbo])
        for pbcc, pbo in zip(pbcc_files, pbo_files)]
    self.cc_obj_files = pbo_files
    self.cc_obj_files += MergeOrderedObjs(
        [Brewery.Get(dep).cc_obj_files for dep in self.deps])
    self.command_groups = [proto_commands, cpp_commands]


class cc_target(BuildTarget):
  def __init__(self, name, srcs, hdrs=[], deps=[], cflags=[], external_libs=[],
               build_binary=False, is_test=False, whole_archive=False,
               shared=False, **kwargs):
    self.hdrs = RectifyFileNames(hdrs)
    self.cflags = cflags
    self.external_libs = [
        '-l' + s if not s.startswith('-') else s for s in external_libs]
    self.build_binary = build_binary
    self.is_test = is_test
    self.whole_archive = whole_archive
    self.shared = shared
    BuildTarget.__init__(self, name, srcs, other_files=self.hdrs, deps=deps, **kwargs)

  def OutputName(self, is_library=False, is_shared=False):
    if len(self.srcs) == 0:
      # This is just a collection of dependencies, so we will not need
      # any output file. Returning an empty string.
      return ''
    name_split = self.name.split(':')
    if is_library:
      if is_shared:
        return os.path.join(
            Env.GENDIR, name_split[0][2:],
            'lib' + name_split[1] + Env.SHARED_LIB_EXT)
      else:
        return os.path.join(
            Env.GENDIR, name_split[0][2:], 'lib' + name_split[1] + '.a')
    else:
      return os.path.join(Env.GENDIR, name_split[0][2:], name_split[1])

  def SetUp(self):
    MakeGenDirs(self.srcs)
    CopyToGenDir(self.hdrs)
    archive_file = self.OutputName(is_library=True)
    self.cc_obj_files = MergeOrderedObjs(
        [Brewery.Get(dep).cc_obj_files for dep in self.deps] +
        [self.external_libs])

    if self.whole_archive:
      self.cc_obj_files.insert(0, Env.WHOLE_ARCHIVE_TEMPLATE % archive_file)
    else:
      self.cc_obj_files.insert(0, archive_file)
    if len(self.srcs) == 0:
      # There is nothing to build if there is no source files.
      self.command_groups = []
    else:
      obj_files = [GenFilename(src, 'o') for src in self.srcs]
      cpp_commands = [
          ' '.join([Env.CC, Env.CFLAGS, Env.INCLUDES, ' '.join(self.cflags),
                    '-c', src, '-o', obj])
          for src, obj in zip(self.srcs, obj_files)]
      # Create the archive
      link_commands = [
          ' '.join([Env.LINK_STATIC, archive_file] + obj_files)]
      if self.build_binary:
        link_binary_commands = [
            ' '.join([Env.LINK_BINARY, self.OutputName()] + self.cc_obj_files +
                     [Env.LINKFLAGS])]
        self.command_groups = [cpp_commands, link_commands, link_binary_commands]
      elif self.shared:
        link_shared_commands = [' '.join(
            [Env.LINK_SHARED, self.OutputName(is_library=True, is_shared=True)]
            + obj_files + self.cc_obj_files[1:] + [Env.LINKFLAGS])]
        self.command_groups = [cpp_commands, link_commands, link_shared_commands]
      else:
        self.command_groups = [cpp_commands, link_commands]
      if self.is_test and CAFFE2_RUN_TEST:
        # Add test command
        self.command_groups.append([
            ' '.join([self.OutputName(), '--caffe_test_root',
                      os.path.abspath(Env.GENDIR),
                      '--gtest_filter=-*.LARGE_*'])])


def cc_library(*args, **kwargs):
  return cc_target(*args, **kwargs)

def cc_binary(*args, **kwargs):
  return cc_target(*args, build_binary=True, **kwargs)

def cc_test(*args, **kwargs):
  if 'cflags' not in kwargs:
    kwargs['cflags'] = []
  kwargs['cflags'].append("-DGTEST_USE_OWN_TR1_TUPLE=1")
  return cc_target(
      *args, build_binary=True, is_test=True, whole_archive=True, **kwargs)

class mpi_test(cc_target):
  def __init__(self, *args, **kwargs):
    if 'cflags' not in kwargs:
      kwargs['cflags'] = []
    kwargs['cflags'].append("-DGTEST_USE_OWN_TR1_TUPLE=1")
    kwargs['build_binary'] = True
    kwargs['is_test'] = True
    kwargs['whole_archive'] = True
    cc_target.__init__(self, *args, **kwargs)
    self.mpi_size = kwargs['mpi_size'] if 'mpi_size' in kwargs else 4

  def SetUp(self):
    cc_target.SetUp(self)
    if CAFFE2_RUN_TEST:
      self.command_groups.append([
          ' '.join(['mpirun --allow-run-as-root -n',
                    str(self.mpi_size), self.OutputName(),
                    '--caffe_test_root', os.path.abspath(Env.GENDIR),
                    '--gtest_filter=-*.LARGE_*'])])


class cuda_library(BuildTarget):
  def __init__(self, name, srcs, hdrs=[], deps=[], cflags=[],
               whole_archive=False, **kwargs):
    self.hdrs = RectifyFileNames(hdrs)
    self.cflags = cflags
    self.whole_archive = whole_archive
    BuildTarget.__init__(self, name, srcs, other_files=self.hdrs, deps=deps, **kwargs)

  def OutputName(self, is_library=False):
    name_split = self.name.split(':')
    if is_library:
      return os.path.join(
          Env.GENDIR, name_split[0][2:], 'lib' + name_split[1] + '.a')
    else:
      return os.path.join(Env.GENDIR, name_split[0][2:], name_split[1])

  def SetUp(self):
    MakeGenDirs(self.srcs)
    CopyToGenDir(self.hdrs)
    obj_files = [GenFilename(src, 'cuo') for src in self.srcs]
    cpp_commands = [
        ' '.join([Env.NVCC, Env.NVCC_CFLAGS, Env.INCLUDES,
                  ' '.join(self.cflags), '-c', src, '-o', obj])
        for src, obj in zip(self.srcs, obj_files)]
    archive_file = self.OutputName(is_library=True)
    # Create the archive
    link_commands = [
        ' '.join([Env.LINK_STATIC, archive_file]
                 + obj_files)]
    if self.whole_archive:
      archive_file = Env.WHOLE_ARCHIVE_TEMPLATE % archive_file
    self.cc_obj_files = MergeOrderedObjs(
        [Brewery.Get(dep).cc_obj_files for dep in self.deps])
    # We will need to add nvidia link targets as well
    self.cc_obj_files.append(Env.NVCC_LINKS)
    self.cc_obj_files.insert(0, archive_file)
    self.command_groups = [cpp_commands, link_commands]


class filegroup(BuildTarget):
  def __init__(self, name, srcs, deps=[], **kwargs):
    self.cc_obj_files = []
    BuildTarget.__init__(self, name, srcs, deps=deps, **kwargs)

  def SetUp(self):
    CopyToGenDir(self.srcs)

def py_library(*args, **kwargs):
  return filegroup(*args, **kwargs)

def cc_headers(*args, **kwargs):
  return filegroup(*args, **kwargs)

class py_test(BuildTarget):
  def __init__(self, name, srcs, deps=[], **kwargs):
    self.cc_obj_files = []
    BuildTarget.__init__(self, name, srcs, deps=deps, **kwargs)

  def SetUp(self):
    CopyToGenDir(self.srcs)
    if len(self.srcs) > 1:
      raise RuntimeError('py_test should only take one python source file.')
    if CAFFE2_RUN_TEST:
      # Add test command
      self.command_groups = [
          ['python %s' % GenFilename(self.srcs[0])]]


class cc_thirdparty_target(BuildTarget):
  """thirdparty_target should only be used in third_party to build things with
  a pre-defined script. Note that this will also set the following values:
      cc_includes: the include folder needed for compiling dependent targets.
      cc_obj_files: the object files produced by the target.

  When building, this script will copy all stuff to a temporary directory, so
  that the original source tree is not affected.
  """
  def __init__(self, name, srcs, commands, cc_obj_files, deps=[], **kwargs):
    self.cwd = Brewery.CWD
    self.build_dir = os.path.join(Brewery.TMPDIR, Brewery.CWD)
    self.commands = [
        'SRCDIR=%s' % self.build_dir,
        'DSTDIR=%s' % os.path.join(os.path.abspath(Env.GENDIR), "third_party"),
        'CPUS=%d' % Env.CPUS,
        'cd %s' % self.build_dir,
    ] + commands
    self.cc_obj_files = [
        os.path.join(Env.GENDIR, "third_party", f)
        for f in cc_obj_files if not f.startswith('-l')] + [
        f for f in cc_obj_files if f.startswith('-l')]
    BuildTarget.__init__(self, name, srcs, deps=deps, **kwargs)

  def SetUp(self):
    self.cc_obj_files += MergeOrderedObjs(
        [Brewery.Get(dep).cc_obj_files for dep in self.deps])

  def Build(self):
    # First, copy all things to the temp directory
    if os.path.exists(self.build_dir):
      shutil.rmtree(self.build_dir)
    shutil.copytree(self.cwd, self.build_dir)
    BuildDebug("script: %s" % str(self.commands))

    proc = subprocess.Popen(' && '.join(self.commands), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, shell=True)
    stdout, _ = proc.communicate()
    if proc.returncode:
      BuildWarning("Script failed.")
      print stdout
      return False
    return True

class shell_script(BuildTarget):
  """Shell scripts are directly run to generate data files. It is run from the
  root of the gendir.
  """
  def __init__(self, name, srcs, commands, deps=[], **kwargs):
    self.cwd = Brewery.CWD
    self.commands = [
        'GENDIR=%s' % os.path.abspath(Env.GENDIR),
        'CWD=%s' % self.cwd,
        'cd %s' % os.path.abspath(Env.GENDIR),
    ] + commands
    BuildTarget.__init__(self, name, srcs, deps=deps, **kwargs)

  def SetUp(self):
    """A shell script should produce no cc_obj_files. This is here just so that
    a cc object can use shell_script as a data dependency.
    """
    CopyToGenDir(self.srcs)
    self.cc_obj_files = []

  def Build(self):
    BuildDebug("script: %s" % str(self.commands))
    proc = subprocess.Popen(' && '.join(self.commands), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, shell=True)
    stdout, _ = proc.communicate()
    if proc.returncode:
      BuildWarning("Script failed.")
      print stdout
      return False
    return True

################################################################################
# Below are functions during the main entry.
################################################################################

def main(argv):
  """The main entry of the build script."""
  BuildLog('Welcome to Caffe2. Running command: %s' % str(argv))
  Brewery.InitBrewery()
  if len(sys.argv) > 1:
    if sys.argv[1] == 'clean':
      for folder in ['caffe2', 'pycaffe2']:
        os.system('rm -rf ' + os.path.join(Env.GENDIR, folder))
      Brewery.ClearSignature()
    elif sys.argv[1] == 'reallyclean':
      os.system('rm -rf ' + Env.GENDIR)
      BuildLog('Finished cleaning.')
    elif sys.argv[1] == 'build':
      # Build all targets.
      targets = sys.argv[2:]
      Brewery.Build(targets)
    elif sys.argv[1] == 'test':
      global CAFFE2_RUN_TEST
      CAFFE2_RUN_TEST = True
      targets = sys.argv[2:]
      Brewery.Build(targets)
    elif sys.argv[1] == 'draw':
      # Draws the dependency graph.
      Brewery.Draw()
    else:
      BuildFatal('Unknown command: %s' % sys.argv[1])
  else:
    BuildLog('Finished parsing all build files without error.')
  Brewery.Finalize()

if __name__ == "__main__":
  main(sys.argv)
