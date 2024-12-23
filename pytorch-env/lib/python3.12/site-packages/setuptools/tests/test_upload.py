from setuptools.command.upload import upload
from setuptools.dist import Distribution
from setuptools.errors import RemovedCommandError

from unittest import mock

import pytest


class TestUpload:
    def test_upload_exception(self):
        """Ensure that the register command has been properly removed."""
        dist = Distribution()
        dist.dist_files = [(mock.Mock(), mock.Mock(), mock.Mock())]

        cmd = upload(dist)

        with pytest.raises(RemovedCommandError):
            cmd.run()
