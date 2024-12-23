"""Tests for distutils.command.upload."""

import os
import unittest.mock as mock
from distutils.command import upload as upload_mod
from distutils.command.upload import upload
from distutils.core import Distribution
from distutils.errors import DistutilsError
from distutils.tests.test_config import PYPIRC, BasePyPIRCCommandTestCase
from urllib.request import HTTPError

import pytest

PYPIRC_LONG_PASSWORD = """\
[distutils]

index-servers =
    server1
    server2

[server1]
username:me
password:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa

[server2]
username:meagain
password: secret
realm:acme
repository:http://another.pypi/
"""


PYPIRC_NOPASSWORD = """\
[distutils]

index-servers =
    server1

[server1]
username:me
"""


class FakeOpen:
    def __init__(self, url, msg=None, code=None):
        self.url = url
        if not isinstance(url, str):
            self.req = url
        else:
            self.req = None
        self.msg = msg or 'OK'
        self.code = code or 200

    def getheader(self, name, default=None):
        return {
            'content-type': 'text/plain; charset=utf-8',
        }.get(name.lower(), default)

    def read(self):
        return b'xyzzy'

    def getcode(self):
        return self.code


@pytest.fixture(autouse=True)
def urlopen(request, monkeypatch):
    self = request.instance
    monkeypatch.setattr(upload_mod, 'urlopen', self._urlopen)
    self.next_msg = self.next_code = None


class TestUpload(BasePyPIRCCommandTestCase):
    def _urlopen(self, url):
        self.last_open = FakeOpen(url, msg=self.next_msg, code=self.next_code)
        return self.last_open

    def test_finalize_options(self):
        # new format
        self.write_file(self.rc, PYPIRC)
        dist = Distribution()
        cmd = upload(dist)
        cmd.finalize_options()
        for attr, waited in (
            ('username', 'me'),
            ('password', 'secret'),
            ('realm', 'pypi'),
            ('repository', 'https://upload.pypi.org/legacy/'),
        ):
            assert getattr(cmd, attr) == waited

    def test_saved_password(self):
        # file with no password
        self.write_file(self.rc, PYPIRC_NOPASSWORD)

        # make sure it passes
        dist = Distribution()
        cmd = upload(dist)
        cmd.finalize_options()
        assert cmd.password is None

        # make sure we get it as well, if another command
        # initialized it at the dist level
        dist.password = 'xxx'
        cmd = upload(dist)
        cmd.finalize_options()
        assert cmd.password == 'xxx'

    def test_upload(self, caplog):
        tmp = self.mkdtemp()
        path = os.path.join(tmp, 'xxx')
        self.write_file(path)
        command, pyversion, filename = 'xxx', '2.6', path
        dist_files = [(command, pyversion, filename)]
        self.write_file(self.rc, PYPIRC_LONG_PASSWORD)

        # lets run it
        pkg_dir, dist = self.create_dist(dist_files=dist_files)
        cmd = upload(dist)
        cmd.show_response = True
        cmd.ensure_finalized()
        cmd.run()

        # what did we send ?
        headers = dict(self.last_open.req.headers)
        assert int(headers['Content-length']) >= 2162
        content_type = headers['Content-type']
        assert content_type.startswith('multipart/form-data')
        assert self.last_open.req.get_method() == 'POST'
        expected_url = 'https://upload.pypi.org/legacy/'
        assert self.last_open.req.get_full_url() == expected_url
        data = self.last_open.req.data
        assert b'xxx' in data
        assert b'protocol_version' in data
        assert b'sha256_digest' in data
        assert (
            b'cd2eb0837c9b4c962c22d2ff8b5441b7b45805887f051d39bf133b583baf'
            b'6860' in data
        )
        if b'md5_digest' in data:
            assert b'f561aaf6ef0bf14d4208bb46a4ccb3ad' in data
        if b'blake2_256_digest' in data:
            assert (
                b'b6f289a27d4fe90da63c503bfe0a9b761a8f76bb86148565065f040be'
                b'6d1c3044cf7ded78ef800509bccb4b648e507d88dc6383d67642aadcc'
                b'ce443f1534330a' in data
            )

        # The PyPI response body was echoed
        results = caplog.messages
        assert results[-1] == 75 * '-' + '\nxyzzy\n' + 75 * '-'

    # bpo-32304: archives whose last byte was b'\r' were corrupted due to
    # normalization intended for Mac OS 9.
    def test_upload_correct_cr(self):
        # content that ends with \r should not be modified.
        tmp = self.mkdtemp()
        path = os.path.join(tmp, 'xxx')
        self.write_file(path, content='yy\r')
        command, pyversion, filename = 'xxx', '2.6', path
        dist_files = [(command, pyversion, filename)]
        self.write_file(self.rc, PYPIRC_LONG_PASSWORD)

        # other fields that ended with \r used to be modified, now are
        # preserved.
        pkg_dir, dist = self.create_dist(
            dist_files=dist_files, description='long description\r'
        )
        cmd = upload(dist)
        cmd.show_response = True
        cmd.ensure_finalized()
        cmd.run()

        headers = dict(self.last_open.req.headers)
        assert int(headers['Content-length']) >= 2172
        assert b'long description\r' in self.last_open.req.data

    def test_upload_fails(self, caplog):
        self.next_msg = "Not Found"
        self.next_code = 404
        with pytest.raises(DistutilsError):
            self.test_upload(caplog)

    @pytest.mark.parametrize(
        'exception,expected,raised_exception',
        [
            (OSError('oserror'), 'oserror', OSError),
            pytest.param(
                HTTPError('url', 400, 'httperror', {}, None),
                'Upload failed (400): httperror',
                DistutilsError,
                id="HTTP 400",
            ),
        ],
    )
    def test_wrong_exception_order(self, exception, expected, raised_exception, caplog):
        tmp = self.mkdtemp()
        path = os.path.join(tmp, 'xxx')
        self.write_file(path)
        dist_files = [('xxx', '2.6', path)]  # command, pyversion, filename
        self.write_file(self.rc, PYPIRC_LONG_PASSWORD)

        pkg_dir, dist = self.create_dist(dist_files=dist_files)

        with mock.patch(
            'distutils.command.upload.urlopen',
            new=mock.Mock(side_effect=exception),
        ):
            with pytest.raises(raised_exception):
                cmd = upload(dist)
                cmd.ensure_finalized()
                cmd.run()
            results = caplog.messages
            assert expected in results[-1]
            caplog.clear()
