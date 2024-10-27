from __future__ import annotations

from collections.abc import Callable
from distutils.errors import DistutilsArgError
import inspect
import glob
import platform
import distutils.command.install as orig
from typing import Any, ClassVar, cast

import setuptools
from ..warnings import SetuptoolsDeprecationWarning, SetuptoolsWarning
from .bdist_egg import bdist_egg as bdist_egg_cls

# Prior to numpy 1.9, NumPy relies on the '_install' name, so provide it for
# now. See https://github.com/pypa/setuptools/issues/199/
_install = orig.install


class install(orig.install):
    """Use easy_install to install the package, w/dependencies"""

    user_options = orig.install.user_options + [
        ('old-and-unmanageable', None, "Try not to use this!"),
        (
            'single-version-externally-managed',
            None,
            "used by system package builders to create 'flat' eggs",
        ),
    ]
    boolean_options = orig.install.boolean_options + [
        'old-and-unmanageable',
        'single-version-externally-managed',
    ]
    # Type the same as distutils.command.install.install.sub_commands
    # Must keep the second tuple item potentially None due to invariance
    new_commands: ClassVar[list[tuple[str, Callable[[Any], bool] | None]]] = [
        ('install_egg_info', lambda self: True),
        ('install_scripts', lambda self: True),
    ]
    _nc = dict(new_commands)

    def initialize_options(self):
        SetuptoolsDeprecationWarning.emit(
            "setup.py install is deprecated.",
            """
            Please avoid running ``setup.py`` directly.
            Instead, use pypa/build, pypa/installer or other
            standards-based tools.
            """,
            see_url="https://blog.ganssle.io/articles/2021/10/setup-py-deprecated.html",
            # TODO: Document how to bootstrap setuptools without install
            #       (e.g. by unziping the wheel file)
            #       and then add a due_date to this warning.
        )

        super().initialize_options()
        self.old_and_unmanageable = None
        self.single_version_externally_managed = None

    def finalize_options(self):
        super().finalize_options()
        if self.root:
            self.single_version_externally_managed = True
        elif self.single_version_externally_managed:
            if not self.root and not self.record:
                raise DistutilsArgError(
                    "You must specify --record or --root when building system"
                    " packages"
                )

    def handle_extra_path(self):
        if self.root or self.single_version_externally_managed:
            # explicit backward-compatibility mode, allow extra_path to work
            return orig.install.handle_extra_path(self)

        # Ignore extra_path when installing an egg (or being run by another
        # command without --root or --single-version-externally-managed
        self.path_file = None
        self.extra_dirs = ''
        return None

    def run(self):
        # Explicit request for old-style install?  Just do it
        if self.old_and_unmanageable or self.single_version_externally_managed:
            return super().run()

        if not self._called_from_setup(inspect.currentframe()):
            # Run in backward-compatibility mode to support bdist_* commands.
            super().run()
        else:
            self.do_egg_install()

        return None

    @staticmethod
    def _called_from_setup(run_frame):
        """
        Attempt to detect whether run() was called from setup() or by another
        command.  If called by setup(), the parent caller will be the
        'run_command' method in 'distutils.dist', and *its* caller will be
        the 'run_commands' method.  If called any other way, the
        immediate caller *might* be 'run_command', but it won't have been
        called by 'run_commands'. Return True in that case or if a call stack
        is unavailable. Return False otherwise.
        """
        if run_frame is None:
            msg = "Call stack not available. bdist_* commands may fail."
            SetuptoolsWarning.emit(msg)
            if platform.python_implementation() == 'IronPython':
                msg = "For best results, pass -X:Frames to enable call stack."
                SetuptoolsWarning.emit(msg)
            return True

        frames = inspect.getouterframes(run_frame)
        for frame in frames[2:4]:
            (caller,) = frame[:1]
            info = inspect.getframeinfo(caller)
            caller_module = caller.f_globals.get('__name__', '')

            if caller_module == "setuptools.dist" and info.function == "run_command":
                # Starting from v61.0.0 setuptools overwrites dist.run_command
                continue

            return caller_module == 'distutils.dist' and info.function == 'run_commands'

        return False

    def do_egg_install(self):
        easy_install = self.distribution.get_command_class('easy_install')

        cmd = easy_install(
            self.distribution,
            args="x",
            root=self.root,
            record=self.record,
        )
        cmd.ensure_finalized()  # finalize before bdist_egg munges install cmd
        cmd.always_copy_from = '.'  # make sure local-dir eggs get installed

        # pick up setup-dir .egg files only: no .egg-info
        cmd.package_index.scan(glob.glob('*.egg'))

        self.run_command('bdist_egg')
        bdist_egg = cast(bdist_egg_cls, self.distribution.get_command_obj('bdist_egg'))
        args = [bdist_egg.egg_output]

        if setuptools.bootstrap_install_from:
            # Bootstrap self-installation of setuptools
            args.insert(0, setuptools.bootstrap_install_from)

        cmd.args = args
        cmd.run(show_deprecation=False)
        setuptools.bootstrap_install_from = None


# XXX Python 3.1 doesn't see _nc if this is inside the class
install.sub_commands = [
    cmd for cmd in orig.install.sub_commands if cmd[0] not in install._nc
] + install.new_commands
