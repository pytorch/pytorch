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

        if self.compiler.compiler_type == 'msvc':
            import distutils.msvccompiler
            import distutils.msvc9compiler
            if sys.version[0] == 2:
                orig_compiler = distutils.msvc9compiler.MSVCCompiler
            else:
                orig_compiler = distutils._msvccompiler.MSVCCompiler
            orig_compile = orig_compiler.compile
            orig_link = orig_compiler.link
        else:
            orig_compiler = distutils.unixccompiler.UnixCCompiler
            orig_compile = orig_compiler._compile
            orig_link = orig_compiler.link

        def win_compile(self, sources,
                        output_dir=None, macros=None, include_dirs=None, debug=0,
                        extra_preargs=None, extra_postargs=None, depends=None):
            if not self.initialized:
                self.initialize()
            compile_info = self._setup_compile(output_dir, macros, include_dirs,
                                               sources, depends, extra_postargs)
            macros, objects, extra_postargs, pp_opts, build = compile_info

            compile_opts = extra_preargs or []
            compile_opts.append('/c')
            if debug:
                compile_opts.extend(self.compile_options_debug)
            else:
                compile_opts.extend(self.compile_options)

            add_cpp_opts = False

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

                def spawn(cmd):
                    builder.writer.build(
                        [obj], 'compile', [src],
                        variables={
                            'cmd': cmd,
                            'deps': 'msvc'
                        })

                if ext in self._c_extensions:
                    input_opt = "/Tc" + src
                elif ext in self._cpp_extensions:
                    input_opt = "/Tp" + src
                    add_cpp_opts = True
                elif ext in self._rc_extensions:
                    # compile .RC to .RES file
                    input_opt = src
                    output_opt = "/fo" + obj
                    try:
                        spawn([self.rc] + pp_opts +
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
                        spawn([self.mc] +
                              ['-h', h_dir, '-r', rc_dir] + [src])
                        base, _ = os.path.splitext(os.path.basename(src))
                        rc_file = os.path.join(rc_dir, base + '.rc')
                        # then compile .RC to .RES file
                        spawn([self.rc] +
                              ["/fo" + obj] + [rc_file])

                    except DistutilsExecError as msg:
                        raise CompileError(msg)
                    continue
                else:
                    # how to handle this file?
                    raise CompileError("Don't know how to compile %s to %s"
                                       % (src, obj))

                args = [self.cc] + compile_opts + pp_opts
                if add_cpp_opts:
                    args.append('/EHsc')
                args.append(input_opt)
                args.append("/Fo" + obj)
                args.extend(extra_postargs)

                try:
                    spawn(args)
                except DistutilsExecError as msg:
                    raise CompileError(msg)

            return objects

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

        def link(self, target_desc, objects,
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

        if self.compiler.compiler_type == 'msvc':
            _compile_func = win_compile
            _compile_func_name = 'compile'
        else:
            _compile_func = unix_compile
            _compile_func_name = '_compile'

        with patch(orig_compiler, _compile_func_name, _compile_func):
            with patch(orig_compiler, 'link', link):
                with patch(self, 'force', True):
                    self._build_default(ext)
