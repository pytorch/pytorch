import os
import sys
from distutils.command.build import build as old_build
from distutils.util import get_platform
from numpy.distutils.command.config_compiler import show_fortran_compilers

class build(old_build):

    sub_commands = [('config_cc',     lambda *args: True),
                    ('config_fc',     lambda *args: True),
                    ('build_src',     old_build.has_ext_modules),
                    ] + old_build.sub_commands

    user_options = old_build.user_options + [
        ('fcompiler=', None,
         "specify the Fortran compiler type"),
        ('warn-error', None,
         "turn all warnings into errors (-Werror)"),
        ]

    help_options = old_build.help_options + [
        ('help-fcompiler', None, "list available Fortran compilers",
         show_fortran_compilers),
        ]

    def initialize_options(self):
        old_build.initialize_options(self)
        self.fcompiler = None
        self.warn_error = False

    def finalize_options(self):
        build_scripts = self.build_scripts
        old_build.finalize_options(self)
        plat_specifier = ".{}-{}.{}".format(get_platform(), *sys.version_info[:2])
        if build_scripts is None:
            self.build_scripts = os.path.join(self.build_base,
                                              'scripts' + plat_specifier)

    def run(self):
        old_build.run(self)
