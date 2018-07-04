import re
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
            if sys.version_info < (3, ):
                orig_compiler = distutils.msvc9compiler.MSVCCompiler
            else:
                orig_compiler = distutils._msvccompiler.MSVCCompiler
            orig_compile = orig_compiler.compile
            orig_link = orig_compiler.link
            orig_spawn = orig_compiler.spawn
        else:
            import distutils.unixccompiler
            orig_compiler = distutils.unixccompiler.UnixCCompiler
            orig_compile = orig_compiler._compile
            orig_link = orig_compiler.link

        def win_compile(self, sources,
                        output_dir=None, macros=None, include_dirs=None, debug=0,
                        extra_preargs=None, extra_postargs=None, depends=None):

            def spawn(cmd):
                # Using regex to match src and obj

                src_regex = re.compile('/T(p|c)(.*)')
                src_list = [m.group(2) for m in (
                    src_regex.match(elem) for elem in cmd) if m]

                obj_regex = re.compile('/Fo(.*)')
                obj_list = [m.group(1) for m in (
                    obj_regex.match(elem) for elem in cmd) if m]

                if len(src_list) >= 1 and len(obj_list) >= 1:
                    src = src_list[0]
                    obj = obj_list[0]
                else:
                    # Cannot find src or obj, revert back to original style
                    return orig_spawn(cmd)

                quote_regex = re.compile('".*"')
                quote_list = [quote_regex.search(
                    arg) is not None for arg in cmd]
                no_quote = any(quote_list)

                if not no_quote:
                    from distutils.spawn import _nt_quote_args
                    cmd = _nt_quote_args(cmd)

                builder.writer.build(
                    [obj], 'compile', [src],
                    variables={
                        'cmd': cmd,
                        'deps': 'msvc'
                    })

            with patch(self, 'spawn', spawn):
                return orig_compile(self, sources,
                                    output_dir, macros, include_dirs, debug,
                                    extra_preargs, extra_postargs, depends)

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
