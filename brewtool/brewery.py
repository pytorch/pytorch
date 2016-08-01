"""The build system scaffolding.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pickle
from collections import defaultdict
import glob
import hashlib
import multiprocessing
import os
from os.path import join
import re
import shlex
import shutil
import signal
import subprocess
import sys

from brewtool.logging import *
from brewtool import autoconfig


def MakedirSafe(directory):
    """Safely make a directory: if it does not exist, create it; if an Exception
    is thrown, check if the directory exists. If not, throw an error.
    """
    try:
        os.makedirs(directory)
    except Exception as e:
        if not os.path.exists(directory):
            BuildFatal('Cannot create directory: {0}. Exception: {1}',
                       directory, str(e))


def RunSingleCommand(command_and_env):
    """Runs a single command, and returns the return code and any info from
    the run.
    """
    command, env = command_and_env
    try:
        proc = subprocess.Popen(
            shlex.split(command), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, env=env)
        stdout, _ = proc.communicate()
        return proc.returncode, stdout.decode('utf-8')
    except OSError as e:
        out = 'Exception found in command {0}. Exception is: {1}.'.format(
            repr(command), str(e))
        return -1, out
    except Exception as e:
        out = 'Unhandled exception in command {0}. Exception is: {1}.'.format(
            repr(command), str(e))
        return -1, out

################################################################################
# The main brewery class.
################################################################################

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class Brewery(object):
    """A singleton class that will hold all the build targets."""
    # Targets store the dictionary from the target name to the build objects.
    _targets = dict()
    # Success stores whether a target is successfully built.
    _success = defaultdict(bool)
    # deps_map is a dictionary mapping each target to its dependents.
    _deps_map = dict()
    # signature_map is the map that stores the signatures for build targets.
    _signatures = defaultdict(str)
    # registered files - this will help us to find any file that is not
    # referred to by any of the targets.
    _registered_files = set()
    _SIGNATURE_FILENAME = 'brewery.signature'
    # Pool is the compute pool that one can use to run a list of commands in
    # parallel.
    Pool = multiprocessing.Pool(multiprocessing.cpu_count() * 2, init_worker)
    Env = None
    # The current working directory when working with build files. The target
    # prefix changes with cwd as well.
    CWD = ''
    
    # Whether in test_mode or not. In default no testing.
    is_test = False

    @classmethod
    def Get(cls, target):
        return cls._targets[target]

    @classmethod
    def InitBrewery(cls, config):
        """Initializes the brewery, e.g. loads the signatures currently built.
        """
        cls.Env = autoconfig.Env(config)
        # Makes the gen directory.
        MakedirSafe(cls.Env.GENDIR)
        # Load the signature file.
        signature_file = join(cls.Env.GENDIR, cls._SIGNATURE_FILENAME)
        if os.path.exists(signature_file):
            BuildDebug('Loading signatures.')
            with open(signature_file, 'rb') as fid:
                cls._signatures = pickle.load(fid)

    @classmethod
    def Finalize(cls):
        """Finalizes the brew process."""
        with open(join(cls.Env.GENDIR, cls._SIGNATURE_FILENAME), 'wb') as fid:
            BuildDebug('Saving signatures.')
            # we use protocol 2 in case one uses both python 2 and 3 to run
            # build in a mixed way. Unlikely, but just to be safe.
            pickle.dump(cls._signatures, fid, protocol=2)

    @classmethod
    def FindAndParseBuildFiles(cls):
        """Find and parse all the BREW files in the subfolders."""
        build_files = [
            join(d[2:], f) for (d, _, files) in os.walk('.')
            if not d.startswith('./gen')
            for f in files if f.endswith('BREW')]
        cls._registered_files.update(build_files)
        for build_file in build_files:
            # Set the current working directory, and parse the build file.
            BuildDebug("Parsing {0}", build_file)
            cls.CWD = os.path.dirname(build_file)
            try:
                exec(open(build_file).read())
            except Exception as e:
                BuildFatal('Error parsing build file {0}. Error is: {1}'
                           .format(build_file, str(e)))
        cls.CWD = ''

    @classmethod
    def RectifyFileName(cls, name):
        """Rectifies a build file name to its absolute path."""
        # Add the current working directory.
        out = os.path.join(cls.CWD, name)
        # check if the name exists.
        BuildFatalIf(not os.path.exists(out), 'File {0} does not exist.', out)
        return out

    @classmethod
    def RectifyTarget(cls, name):
        """Rectifies a build target name."""
        if name.startswith("//"):
            return name  # Nothing to do here. already rectified.
        elif name.startswith(":"):
            return "//" + cls.CWD + name
        else:
            return "//" + cls.CWD + ":" + name

    @classmethod
    def GenFilename(cls, name, new_ext=None):
        """Returns the corresponding filename under GENDIR.

        if new_ext is specified, the file extension is replaced with new_ext.
        """
        new_name = (name[:name.rfind('.') + 1] + new_ext) if new_ext else name
        return os.path.join(cls.Env.GENDIR, new_name)

    @classmethod
    def MakeGenDirs(cls, rectified_srcs):
        for src in rectified_srcs:
            dst = join(cls.Env.GENDIR, src)
            MakedirSafe(os.path.dirname(dst))

    @classmethod
    def CopyToGenDir(cls, rectified_srcs):
        cls.MakeGenDirs(rectified_srcs)
        for src in rectified_srcs:
            shutil.copyfile(src, cls.GenFilename(src))

    @classmethod
    def Register(cls, name, target):
        BuildFatalIf(name in cls._targets,
                     "{0} already in build target.", name)
        BuildDebug("Registered build target {0} deps {1}", name, str(target.deps))
        cls._targets[name] = target
        cls._registered_files.update(target.all_files)
        # Note that we do not add the dependencies to the deps_map yet during
        # the registration process, because Registration happens at the base
        # BuildTarget level, so it is possible that the derived classes may add
        # some more dependencies during its initialization. As a result, we will
        # only query the dependencies when all the parsing are finished and an
        # execution chain needs to be generated.

    @classmethod
    def _GetExecutionChain(cls, targets=None):
        """Gets the execution chain."""
        # First, verify all dependencies.
        for name, target in cls._targets.items():
            cls._deps_map[name] = target.deps + target.optional_deps
            for d in cls._deps_map[name]:
                BuildFatalIf(
                    d not in cls._targets,
                    "Dependency {0} for target {1} does not exist.", d, name)
        if targets is None or len(targets) == 0:
            targets = cls._targets
        else:
            # Get all targets that we need to build, including all dependencies.
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
        frontier = set(t for t in deps_count if deps_count[t] == 0)
        build_order = []
        while frontier:
            current = frontier.pop()
            build_order.append(current)
            for t in inverse_deps_map[current]:
                deps_count[t] -= 1
                if deps_count[t] == 0:
                    frontier.add(t)
        # If this does not cover all targets, the graph is not a DAG.
        BuildFatalIf(len(build_order) != len(targets),
                     "There are cycles in the dependency graph!")
        BuildDebug('Build order: {0}', str(build_order))
        return build_order

    @classmethod
    def Signature(cls, target):
        # Returns the builtsignature of the current target.
        return cls._signatures[target]

    @classmethod
    def Success(cls, target):
        return cls._success[target]

    @classmethod
    def Build(cls, targets):
        """Build all the targets, using their topological order."""
        BuildDebug("Actually start building.")
        build_order = cls._GetExecutionChain(targets)
        for t in build_order:
            BuildLog("Building {0}", t)
            cls._success[t], changed, new_signature = (
                cls._targets[t].SetUpAndBuild(cls._signatures[t]))
            if cls._success[t] and changed:
                BuildDebug("Updating signature for {0}: {1}.",
                           t, new_signature[:6])
                cls._signatures[t] = new_signature
        # Finally, print a summary of the build results.
        succeeded = [key for key in cls._success if cls._success[key]]
        BuildDebug("Successfully built {0} targets.", len(succeeded))
        failed = [key for key in cls._success if not cls._success[key]]
        failed.sort()
        if len(failed) > 0:
            BuildWarning("Failed to build:")
            for key in failed:
                BuildWarning(key)

    @classmethod
    def Run(cls, Config, argv):
        BuildLog("Brewing Caffe2. Running command:\n{0}", argv)
        cls.InitBrewery(Config)
        # Find and parse all the build files in caffe's library.
        cls.FindAndParseBuildFiles()
        command = argv[1] if len(argv) > 1 else 'build'
        if command == 'build':
            cls.Build(argv[2:])
            cls.Finalize()
        elif command == 'clean':
            os.system('rm -rf ' + cls.Env.GENDIR)
        elif command == 'test':
            cls.is_test = True
            cls.Build(argv[2:])
            cls.Finalize()
        else:
            BuildFatal('Unknown command: {0}', command)
        BuildLog("Brewing done.")
        BuildLog("Performing post-brewing checks.")
        # check if all files in the folder has been properly registered.
        all_files = set()
        for root, _, files in os.walk("caffe2"):
            all_files.update([os.path.join(root, f) for f in files])
        not_declared = list(all_files.difference(cls._registered_files))
        if len(not_declared):
            BuildWarning("You have the following files not being registered "
                         "by any BREW files:")
            not_declared.sort()
            for name in not_declared:
                print(name)


class BuildTarget(object):
    """A build target that can be executed with the Build() function."""
    def __init__(self, name, srcs, other_files=None, deps=None,
                 optional_deps=None):
        if other_files is None:
            other_files = []
        if deps is None:
            deps = []
        if optional_deps is None:
            optional_deps = []
        self.name = Brewery.RectifyTarget(name)
        self.srcs = [Brewery.RectifyFileName(n) for n in sorted(srcs)]
        self.all_files = self.srcs + [
            Brewery.RectifyFileName(n) for n in sorted(other_files)]
        self.deps = [Brewery.RectifyTarget(n) for n in sorted(deps)]
        self.optional_deps = [
            Brewery.RectifyTarget(n) for n in sorted(optional_deps)]
        self.command_groups = []
        Brewery.Register(self.name, self)

    def GetSignature(self):
        """Generate the signature of the build object, and see if we need to
        rebuild it.
        """
        src_digest = ''.join([hashlib.sha256(open(f, 'rb').read()).hexdigest()
                              for f in self.all_files])
        dep_digest = ''.join([Brewery.Signature(d) for d in self.deps])
        command_digest = str(self.command_groups)
        return hashlib.sha256(
            (src_digest + dep_digest + command_digest).encode('utf-8')).hexdigest()

    def SetUpAndBuild(self, built_signature):
        """SetUp and Build the target."""
        # Add successful optional dependencies into deps.
        self.deps += [dep for dep in self.optional_deps if Brewery.Success(dep)]
        # Sets up the current build.
        self.SetUp()
        signature = self.GetSignature()
        # If there are some dependencies failing to build, the current target
        # automatically fails.
        if any(not Brewery.Success(d) for d in self.deps):
            BuildWarning("Not all dependencies have succeeded. Skipping build. "
                         "Failed dependencies: ")
            BuildWarning(str([d for d in self.deps if not Brewery.Success(d)]))
            return False, True, signature
        # If the built signature is different from the current signature, we
        # will re-build the current target. Otherwise, nothing has changed so we
        # do not need to do anything.
        if signature != built_signature:
            BuildDebug("Signature changed: {0} -> {1}. Rebuild.",
                       built_signature[:6], signature[:6])
            # Need to actually build this target.
            return self.Build(), True, signature
        else:
            # Nothing has changed.
            return True, False, signature

    def SetUp(self):
        """Set up the build object's variables.

        This will always run even if the target has already been built. Anything
        that further dependencies will need should be implemented here.

        If your target just emits a set of shell commands, in SetUp() you can set
        self.command_groups and use the default Build function, which basically
        sends the command groups to a execution pool.
        """
        BuildFatal('SetUp not implemented for target {0}', self.name)

    def Build(self):
        for command_group in self.command_groups:
            BuildDebug("Stage:\n{0}", command_group)
            if len(command_group) == 0:
                continue
            try:
                run_stats = Brewery.Pool.map(
                    RunSingleCommand,
                    [(command, Brewery.Env.ENV) for command in command_group])
            except KeyboardInterrupt:
                BuildWarning("Received ctrl-c. Finishing.")
                Brewery.Pool.terminate()
                Brewery.Pool.join()
                sys.exit(1)
            if any([s[0] for s in run_stats]):
                # Something failed.
                BuildWarning("Build failed: {0}. Fail messages are as follows:",
                             self.name)
                for stat in run_stats:
                    if stat[0]:
                        BuildPrint('\n{0}\n', stat[1])
                return False
        return True

################################################################################
# Targets that we can use in the build script.
################################################################################


def Glob(patterns, excludes=None):
    """Globs all files with the given patterns, relative to the path of the BREW
    file."""
    excludes = excludes if excludes is not None else []
    files = set()
    if type(patterns) is str:
        patterns = [patterns]
    for pattern in patterns:
        full_pattern = os.path.join(Brewery.CWD, pattern)
        files.update(glob.glob(full_pattern))
    for pattern in excludes:
        pattern = pattern.replace("*", ".*")
        full_pattern = os.path.join(Brewery.CWD, pattern)
        files = [f for f in files if not re.match(full_pattern, f)]
    # If CWD is empty, we don't need to cut prefix; else, we will cut
    # CWD as well as the first separator symbol.
    prefix_len = len(Brewery.CWD) + 1 if len(Brewery.CWD) else 0
    return [f[prefix_len:] for f in files if os.path.isfile(f)]


def MergeOrderedObjs(dep_lists):
    added = set()
    output = []
    for dep_list in dep_lists:
        for item in dep_list[::-1]:
            if item not in added:
                added.add(item)
                output.insert(0, item)
    return output

################################################################################
# Protocol Buffer
################################################################################


# A hard-coded name that specifies the google protocol buffer compiler - this
# is required for proto libraries.
PROTOC_TARGET = "//third_party:protoc"
PROTOBUF_LITE_TARGET = "//third_party:protobuf_lite"
PROTOBUF_TARGET = "//third_party:protobuf"

# A hard-coded name that specifies the cuda dependency.
CUDA_TARGET = "//third_party:cuda"


class proto_library(BuildTarget):
    """Builds a protobuffer library.

    A protobuffer library builds a set of protobuffer source files to its cc and
    python source files, as well as the static library named "libname.a".
    """
    def __init__(self, name, srcs, deps=None, **kwargs):
        if deps is None:
            deps = []
        if (PROTOC_TARGET in deps or PROTOBUF_LITE_TARGET in deps
                or PROTOBUF_TARGET in deps):
            raise RuntimeError('')
            BuildFatal("For proto_library targets, you should not manually add "
                       "the third party protobuf targets. We automatically "
                       "figure it out depending on your compilation config.")
        # Determine what protobuf we want to depend on
        if Brewery.Env.Config.USE_LITE_PROTO:
            self.optimize_option = "LITE_RUNTIME"
            deps.append(PROTOBUF_LITE_TARGET)
        else:
            self.optimize_option = "SPEED"
            deps.append(PROTOBUF_TARGET)
        # PROTOC_TARGET is here just for controlling the build order.
        deps.append(PROTOC_TARGET)
        BuildTarget.__init__(self, name, srcs, deps=deps, **kwargs)

    def AddOptimizationOption(self, name):
        with open(name, 'r') as fid:
            lines = fid.read().decode('utf-8').split('\n')
        optimization_line = (
            'option optimize_for = {0};'.format(self.optimize_option))
        if lines[0].startswith('syntax'):
            lines.insert(1, optimization_line)
        else:
            lines.insert(0, optimization_line)
        with open(name, 'w') as fid:
            for line in lines:
                fid.write((line + '\n').encode('utf-8'))

    def SetUp(self):
        Brewery.CopyToGenDir(self.srcs)
        gen_srcs = [Brewery.GenFilename(s) for s in self.srcs]
        # Depending on whether we are building lite or full proto, we add the
        # optimization flags to the source file.
        for name in gen_srcs:
            self.AddOptimizationOption(name)
        pbcc_files = [Brewery.GenFilename(f, 'pb.cc') for f in gen_srcs]
        pbo_files = [Brewery.GenFilename(f, 'pb.o') for f in gen_srcs]
        self.cc_obj_files = pbo_files + MergeOrderedObjs(
            [Brewery.Get(dep).cc_obj_files for dep in self.deps])
        self.command_groups = [
            # protocol buffer commands
            [Brewery.Env.protoc(s) for s in gen_srcs],
            # cc commands
            [Brewery.Env.cc(s, d) for s, d in zip(pbcc_files, pbo_files)],
        ]

################################################################################
# C++
################################################################################


class cc_target(BuildTarget):
    def __init__(self, name, srcs, hdrs=None, deps=None, build_shared=False,
                 build_binary=False, is_test=False, whole_archive=False,
                 **kwargs):
        if hdrs is None:
            hdrs = []
        if deps is None:
            deps = []
        BuildTarget.__init__(self, name, srcs, other_files=hdrs,
                             deps=deps, **kwargs)
        self.hdrs = [Brewery.RectifyFileName(s) for s in hdrs]
        self.build_shared = build_shared
        self.build_binary = build_binary
        self.is_test = is_test
        self.whole_archive = whole_archive

    def _OutputName(self, is_library=False, is_shared=False):
        if len(self.srcs) == 0 and not is_shared:
            # This is just a collection of dependencies, so we will not produce
            # any output file. Returning an empty string.
            return ''
        name_split = self.name.split(':')
        if is_library:
            ext = Brewery.Env.SHARED_LIB_EXT if is_shared else '.a'
            return os.path.join(Brewery.Env.GENDIR, name_split[0][2:],
                                'lib' + name_split[1] + ext)
        else:
            # Return a binary name.
            return os.path.join(Brewery.Env.GENDIR, name_split[0][2:],
                                name_split[1])

    def SetUp(self):
        Brewery.MakeGenDirs(self.srcs)
        Brewery.CopyToGenDir(self.hdrs)
        # We will always create an archive file.
        archive_file = self._OutputName(True, False)

        if self.whole_archive:
            cc_obj_files = [Brewery.Env.whole_archive(archive_file)]
        else:
            cc_obj_files = [archive_file]
        cc_obj_files += MergeOrderedObjs(
            [Brewery.Get(dep).cc_obj_files for dep in self.deps])

        if self.build_binary:
            # A C++ binary should not be linked into another library - if this
            # happens, we will treat it as a control dependency and not really
            # link it.
            self.cc_obj_files = []
        else:
            self.cc_obj_files = cc_obj_files

        self.command_groups = []
        obj_files = [Brewery.GenFilename(s, 'o') for s in self.srcs]
        if len(self.srcs):
            # Build the source files, and remove the existing archive file.
            self.command_groups.append(
                [Brewery.Env.cc(s, o) for s, o in zip(self.srcs, obj_files)])
            # Remove the current archive
            self.command_groups.append(['rm -f ' + archive_file])
            # link the static library
            self.command_groups.append(
                [Brewery.Env.link_static(obj_files, archive_file)])

        if self.build_binary:
            # Build the binary.
            self.command_groups.append(
                [Brewery.Env.link_binary(
                    cc_obj_files, self._OutputName())])
        if self.build_shared:
            # Build the shared library
            self.command_groups.append(
                [Brewery.Env.link_shared(
                    cc_obj_files, self._OutputName(True, True))])

        if self.is_test and Brewery.is_test:
            # Runs the test.
            self.command_groups.append(
                [Brewery.Env.cc_test(self._OutputName())])


def cc_library(*args, **kwargs):
    return cc_target(*args, **kwargs)


def cc_binary(*args, **kwargs):
    return cc_target(*args, build_binary=True, **kwargs)


def cc_test(*args, **kwargs):
    return cc_target(*args, build_binary=True, is_test=True,
                     whole_archive=True, **kwargs)


class cc_headers(BuildTarget):
    def __init__(self, name, srcs, deps=None, **kwargs):
        BuildTarget.__init__(self, name, srcs, deps=deps, **kwargs)

    def SetUp(self):
        Brewery.CopyToGenDir(self.srcs)
        self.cc_obj_files = MergeOrderedObjs(
            [Brewery.Get(dep).cc_obj_files for dep in self.deps])


class python_cc_extension(BuildTarget):
    def __init__(self, name, srcs, hdrs=None, deps=None, **kwargs):
        if hdrs is None:
            hdrs = []
        self.hdrs = [Brewery.RectifyFileName(s) for s in hdrs]
        BuildTarget.__init__(self, name, srcs, other_files=hdrs,
                             deps=deps, **kwargs)

    def SetUp(self):
        Brewery.MakeGenDirs(self.srcs)
        Brewery.CopyToGenDir(self.hdrs)
        self.command_groups = []
        obj_files = [Brewery.GenFilename(s, 'o') for s in self.srcs]
        if len(self.srcs):
            # Build the source files, and remove the existing archive file.
            self.command_groups.append([Brewery.Env.pyext_cc(s, o)
                                        for s, o in zip(self.srcs, obj_files)])
        name_split = self.name.split(':')
        # TODO: avoid hard-code extension.
        self.command_groups.append(
            [Brewery.Env.pyext_link(
                obj_files + MergeOrderedObjs(
                    [Brewery.Get(dep).cc_obj_files for dep in self.deps]),
                os.path.join(Brewery.Env.GENDIR, name_split[0][2:],
                             'lib' + name_split[1] + ".so"))])


class mpi_test(cc_target):
    def __init__(self, *args, **kwargs):
        kwargs['build_binary'] = True
        kwargs['is_test'] = True
        kwargs['whole_archive'] = True
        cc_target.__init__(self, *args, **kwargs)
        self.mpi_size = kwargs['mpi_size'] if 'mpi_size' in kwargs else 4

    def SetUp(self):
        cc_target.SetUp(self)
        if Brewery.is_test:
            self.command_groups.append(
                [Brewery.Env.MPIRUN + ' -n ' + str(self.mpi_size) + ' ' +
                 Brewery.Env.cc_test(self._OutputName())])


class cuda_library(BuildTarget):
    def __init__(self, name, srcs, hdrs=None, deps=None,
                 whole_archive=False, **kwargs):
        if hdrs is None:
            hdrs = []
        BuildTarget.__init__(self, name, srcs, other_files=hdrs,
                             deps=deps, **kwargs)
        self.hdrs = [Brewery.RectifyFileName(s) for s in hdrs]
        self.whole_archive = whole_archive
        if CUDA_TARGET not in self.deps:
            self.deps.append(CUDA_TARGET)

    def SetUp(self):
        Brewery.MakeGenDirs(self.srcs)
        Brewery.CopyToGenDir(self.hdrs)
        name_split = self.name.split(':')
        archive_file = os.path.join(Brewery.Env.GENDIR, name_split[0][2:],
                                    'lib' + name_split[1] + '.a')
        if self.whole_archive:
            self.cc_obj_files = [Brewery.Env.whole_archive(archive_file)]
        else:
            self.cc_obj_files = [archive_file]
        self.cc_obj_files += MergeOrderedObjs(
            [Brewery.Get(dep).cc_obj_files for dep in self.deps])

        self.command_groups = []
        # Build the source files, and remove the existing archive file.
        obj_files = [Brewery.GenFilename(src, 'cuo') for src in self.srcs]
        self.command_groups.append(
            [Brewery.Env.nvcc(s, o) for s, o in zip(self.srcs, obj_files)])
        # Remove the current archive
        self.command_groups.append(['rm -f ' + archive_file])
        # link the static library
        self.command_groups.append(
            [Brewery.Env.link_static(obj_files, archive_file)])


class filegroup(BuildTarget):
    def __init__(self, name, srcs, deps=None, **kwargs):
        self.cc_obj_files = []
        BuildTarget.__init__(self, name, srcs, deps=deps, **kwargs)

    def SetUp(self):
        Brewery.CopyToGenDir(self.srcs)


def py_library(*args, **kwargs):
    return filegroup(*args, **kwargs)


class py_test(BuildTarget):
    def __init__(self, name, srcs, deps=None, **kwargs):
        BuildTarget.__init__(self, name, srcs, deps=deps, **kwargs)

    def SetUp(self):
        Brewery.CopyToGenDir(self.srcs)
        if len(self.srcs) > 1:
            raise RuntimeError('py_test should only take one python source file.')
        if Brewery.is_test:
            # Add test command
            self.command_groups = [
                ['python {0}'.format(Brewery.GenFilename(self.srcs[0]))]]


class shell_script(BuildTarget):
    """Shell scripts are directly run to generate data files. It is run from the
    root of the gendir.
    """
    def __init__(self, name, srcs, commands, deps=None, **kwargs):
        BuildTarget.__init__(self, name, srcs, deps=deps, **kwargs)
        self.cwd = Brewery.CWD
        self.commands = [
            'CAFFE2_CWD={0}'.format(self.cwd),
            'cd {0}'.format(os.path.abspath(Brewery.Env.GENDIR)),
        ] + commands
        self.cc_obj_files = []

    def SetUp(self):
        Brewery.CopyToGenDir(self.srcs)

    def Build(self):
        BuildDebug("script: {0}", '\n' + '\n'.join(self.commands))
        proc = subprocess.Popen(
            ' && '.join(self.commands), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, shell=True, env=Brewery.Env.ENV)
        stdout, _ = proc.communicate()
        if proc.returncode:
            BuildWarning("Script failed. Failure message:")
            BuildPrint("\n{0}\n", stdout.decode('utf-8'))
            return False
        return True


class cc_thirdparty_target(BuildTarget):
    """cc_thirdparty_target does nothing but specifying what link targets such
    third party library requires. It should not run any script.
    """
    def __init__(self, name, cc_obj_files, deps=None, **kwargs):
        BuildTarget.__init__(self, name, srcs=["BREW"], deps=deps, **kwargs)
        self.cc_obj_files = cc_obj_files

    def SetUp(self):
        self.cc_obj_files += MergeOrderedObjs(
            [Brewery.Get(dep).cc_obj_files for dep in self.deps])

    def Build(self):
        return True
