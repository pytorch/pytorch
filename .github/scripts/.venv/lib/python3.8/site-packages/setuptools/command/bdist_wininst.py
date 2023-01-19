import distutils.command.bdist_wininst as orig
import warnings

from setuptools import SetuptoolsDeprecationWarning


class bdist_wininst(orig.bdist_wininst):
    def reinitialize_command(self, command, reinit_subcommands=0):
        """
        Supplement reinitialize_command to work around
        http://bugs.python.org/issue20819
        """
        cmd = self.distribution.reinitialize_command(
            command, reinit_subcommands)
        if command in ('install', 'install_lib'):
            cmd.install_lib = None
        return cmd

    def run(self):
        warnings.warn(
            "bdist_wininst is deprecated and will be removed in a future "
            "version. Use bdist_wheel (wheel packages) instead.",
            SetuptoolsDeprecationWarning
        )

        self._is_running = True
        try:
            orig.bdist_wininst.run(self)
        finally:
            self._is_running = False
