import os
import sys
import setuptools
import distutils
from contextlib import contextmanager
import subprocess

BUILD_DIR = 'build'


# on the fly create a ninja file in build/ and then
# run it when run() is called.
class NinjaBuilder(object):
    def __init__(self, name):
        import ninja
        if not os.path.exists(BUILD_DIR):
            os.mkdir(BUILD_DIR)
        self.ninja_program = os.path.join(ninja.BIN_DIR, 'ninja')
        self.name = name
        self.filename = os.path.join(BUILD_DIR, 'build.{}.ninja'.format(name))
        self.writer = ninja.Writer(open(self.filename, 'w'))
        self.writer.rule('do_cmd', '$cmd')
        self.writer.rule('compile', '$cmd')
        self.compdb_targets = []

    def run(self):
        import ninja
        self.writer.close()
        try:
            subprocess.check_call([self.ninja_program, '-f', self.filename])
        except subprocess.CalledProcessError as err:
            # avoid printing the setup.py stack trace because it obscures the
            # C++ errors.
            sys.stderr.write(str(err) + '\n')
            sys.exit(1)
        compile_db_path = os.path.join(BUILD_DIR, '{}_compile_commands.json'.format(self.name))
        with open(compile_db_path, 'w') as compile_db:
            subprocess.check_call([self.ninja_program, '-f', self.filename,
                                   '-t', 'compdb', 'compile'], stdout=compile_db)

        # weird build logic in build develop causes some things to be run
        # twice so make sure even after we run the command we still
        # reset this to a valid state
        # don't use the same name or you can't inspect the real ninja files
        self.__init__(self.name + "_")


class ninja_build_ext(setuptools.command.build_ext.build_ext):
    def _build_default(self, ext):
        return setuptools.command.build_ext.build_ext.build_extension(self, ext)

    def build_extension(self, ext):
        builder = NinjaBuilder(ext.name)

        @contextmanager
        def patch(obj, attr_name, val):
            orig_val = getattr(obj, attr_name)
            setattr(obj, attr_name, val)
            try:
                yield
            finally:
                setattr(obj, attr_name, orig_val)

        if os.sys.platform == 'win32':
            orig_compile = distutils.msvccompiler.MSVCCompiler.compile
            orig_link = distutils.msvccompiler.MSVCCompiler.link
        else:
            orig_compile = distutils.unixccompiler.UnixCCompiler._compile
            orig_link = distutils.unixccompiler.UnixCCompiler.link

        def win_compile(self, sources,
                        output_dir=None, macros=None, include_dirs=None, debug=0,
                        extra_preargs=None, extra_postargs=None, depends=None):
            # TODO: modify this function
            if not self.initialized:
                self.initialize()
            compile_info = self._setup_compile(output_dir, macros, include_dirs,
                                            sources, depends, extra_postargs)
            macros, objects, extra_postargs, pp_opts, build = compile_info

            compile_opts = extra_preargs or []
            compile_opts.append ('/c')
            if debug:
                compile_opts.extend(self.compile_options_debug)
            else:
                compile_opts.extend(self.compile_options)

            for obj in objects:
                try:
                    src, ext = build[obj]
                except KeyError:
                    continue
                if debug:
                    # pass the full pathname to MSVC in debug mode,
                    # this allows the debugger to find the source file
                    # without asking the user to browse for it
                    src = os.path.abspath(src)

                if ext in self._c_extensions:
                    input_opt = "/Tc" + src
                elif ext in self._cpp_extensions:
                    input_opt = "/Tp" + src
                elif ext in self._rc_extensions:
                    # compile .RC to .RES file
                    input_opt = src
                    output_opt = "/fo" + obj
                    try:
                        self.spawn([self.rc] + pp_opts +
                                [output_opt] + [input_opt])
                    except DistutilsExecError as msg:
                        raise CompileError(msg)
                    continue
                elif ext in self._mc_extensions:
                    # Compile .MC to .RC file to .RES file.
                    #   * '-h dir' specifies the directory for the
                    #     generated include file
                    #   * '-r dir' specifies the target directory of the
                    #     generated RC file and the binary message resource
                    #     it includes
                    #
                    # For now (since there are no options to change this),
                    # we use the source-directory for the include file and
                    # the build directory for the RC file and message
                    # resources. This works at least for win32all.
                    h_dir = os.path.dirname(src)
                    rc_dir = os.path.dirname(obj)
                    try:
                        # first compile .MC to .RC and .H file
                        self.spawn([self.mc] +
                                ['-h', h_dir, '-r', rc_dir] + [src])
                        base, _ = os.path.splitext (os.path.basename (src))
                        rc_file = os.path.join (rc_dir, base + '.rc')
                        # then compile .RC to .RES file
                        self.spawn([self.rc] +
                                ["/fo" + obj] + [rc_file])

                    except DistutilsExecError as msg:
                        raise CompileError(msg)
                    continue
                else:
                    # how to handle this file?
                    raise CompileError("Don't know how to compile %s to %s"
                                    % (src, obj))

                output_opt = "/Fo" + obj
                try:
                    self.spawn([self.cc] + compile_opts + pp_opts +
                            [input_opt, output_opt] +
                            extra_postargs)
                except DistutilsExecError as msg:
                    raise CompileError(msg)

            return objects

        def win_link(self, target_desc, objects,
                 output_filename, output_dir=None, libraries=None,
                 library_dirs=None, runtime_library_dirs=None,
                 export_symbols=None, debug=0, extra_preargs=None,
                 extra_postargs=None, build_temp=None, target_lang=None):

            # TODO: modify this function
            if not self.initialized:
                self.initialize()
            (objects, output_dir) = self._fix_object_args(objects, output_dir)
            fixed_args = self._fix_lib_args(libraries, library_dirs,
                                            runtime_library_dirs)
            (libraries, library_dirs, runtime_library_dirs) = fixed_args

            if runtime_library_dirs:
                self.warn ("I don't know what to do with 'runtime_library_dirs': "
                        + str (runtime_library_dirs))

            lib_opts = gen_lib_options(self,
                                    library_dirs, runtime_library_dirs,
                                    libraries)
            if output_dir is not None:
                output_filename = os.path.join(output_dir, output_filename)

            if self._need_link(objects, output_filename):
                if target_desc == CCompiler.EXECUTABLE:
                    if debug:
                        ldflags = self.ldflags_shared_debug[1:]
                    else:
                        ldflags = self.ldflags_shared[1:]
                else:
                    if debug:
                        ldflags = self.ldflags_shared_debug
                    else:
                        ldflags = self.ldflags_shared

                export_opts = []
                for sym in (export_symbols or []):
                    export_opts.append("/EXPORT:" + sym)

                ld_args = (ldflags + lib_opts + export_opts +
                        objects + ['/OUT:' + output_filename])

                # The MSVC linker generates .lib and .exp files, which cannot be
                # suppressed by any linker switches. The .lib files may even be
                # needed! Make sure they are generated in the temporary build
                # directory. Since they have different names for debug and release
                # builds, they can go into the same directory.
                if export_symbols is not None:
                    (dll_name, dll_ext) = os.path.splitext(
                        os.path.basename(output_filename))
                    implib_file = os.path.join(
                        os.path.dirname(objects[0]),
                        self.library_filename(dll_name))
                    ld_args.append ('/IMPLIB:' + implib_file)

                if extra_preargs:
                    ld_args[:0] = extra_preargs
                if extra_postargs:
                    ld_args.extend(extra_postargs)

                self.mkpath(os.path.dirname(output_filename))
                try:
                    self.spawn([self.linker] + ld_args)
                except DistutilsExecError as msg:
                    raise LinkError(msg)

            else:
                log.debug("skipping %s (up-to-date)", output_filename)

        def unix_compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
            depfile = os.path.splitext(obj)[0] + '.d'

            def spawn(cmd):
                builder.writer.build(
                    [obj], 'compile', [src],
                    variables={
                        'cmd': cmd,
                        'depfile': depfile,
                        'deps': 'gcc'
                    })

            extra_postargs = extra_postargs + ['-MMD', '-MF', depfile]
            with patch(self, 'spawn', spawn):
                orig_compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts)

        def unix_link(self, target_desc, objects,
                 output_filename, output_dir=None, libraries=None,
                 library_dirs=None, runtime_library_dirs=None,
                 export_symbols=None, debug=0, extra_preargs=None,
                 extra_postargs=None, build_temp=None, target_lang=None):

            builder.run()
            orig_link(self, target_desc, objects,
                      output_filename, output_dir, libraries,
                      library_dirs, runtime_library_dirs,
                      export_symbols, debug, extra_preargs,
                      extra_postargs, build_temp, target_lang)

        if os.sys.platform == 'win32':
            with patch(distutils.msvccompiler.MSVCCompiler, 'compile', win_compile):
                with patch(distutils.msvccompiler.MSVCCompiler, 'link', win_link):
                    with patch(self, 'force', True):
                        self._build_default(ext)
        else:
            with patch(distutils.unixccompiler.UnixCCompiler, '_compile', unix_compile):
                with patch(distutils.unixccompiler.UnixCCompiler, 'link', unix_link):
                    with patch(self, 'force', True):
                        self._build_default(ext)
