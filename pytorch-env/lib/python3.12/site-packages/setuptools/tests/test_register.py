from setuptools.command.register import register
from setuptools.dist import Distribution
from setuptools.errors import RemovedCommandError

from unittest import mock

import pytest


class TestRegister:
    def test_register_exception(self):
        """Ensure that the register command has been properly removed."""
        dist = Distribution()
        dist.dist_files = [(mock.Mock(), mock.Mock(), mock.Mock())]

        cmd = register(dist)

        with pytest.raises(RemovedCommandError):
            cmd.run()
