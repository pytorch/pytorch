from __future__ import annotations

import builtins
import importlib
import os.path
import platform
import shutil
import stat
import struct
import sys
import sysconfig
from contextlib import suppress
from inspect import cleandoc
from unittest.mock import Mock
from zipfile import ZipFile

import jaraco.path
import pytest

import setuptools
from distutils.core import run_setup
from setuptools.command.bdist_wheel import (
    bdist_wheel,
    get_abi_tag,
    remove_readonly,
    remove_readonly_exc,
)
from setuptools.dist import Distribution
from packaging import tags

DEFAULT_FILES = {
    "dummy_dist-1.0.dist-info/top_level.txt",
    "dummy_dist-1.0.dist-info/METADATA",
    "dummy_dist-1.0.dist-info/WHEEL",
    "dummy_dist-1.0.dist-info/RECORD",
}
DEFAULT_LICENSE_FILES = {
    "LICENSE",
    "LICENSE.txt",
    "LICENCE",
    "LICENCE.txt",
    "COPYING",
    "COPYING.md",
    "NOTICE",
    "NOTICE.rst",
    "AUTHORS",
    "AUTHORS.txt",
}
OTHER_IGNORED_FILES = {
    "LICENSE~",
    "AUTHORS~",
}
SETUPPY_EXAMPLE = """\
from setuptools import setup

setup(
    name='dummy_dist',
    version='1.0',
)
"""


EXAMPLES = {
    "dummy-dist": {
        "setup.py": SETUPPY_EXAMPLE,
        "licenses": {"DUMMYFILE": ""},
        **dict.fromkeys(DEFAULT_LICENSE_FILES | OTHER_IGNORED_FILES, ""),
    },
    "simple-dist": {
        "setup.py": cleandoc(
            """
            from setuptools import setup

            setup(
                name="simple.dist",
                version="0.1",
                description="A testing distribution \N{SNOWMAN}",
                extras_require={"voting": ["beaglevote"]},
            )
            """
        ),
        "simpledist": "",
    },
    "complex-dist": {
        "setup.py": cleandoc(
            """
            from setuptools import setup

            setup(
                name="complex-dist",
                version="0.1",
                description="Another testing distribution \N{SNOWMAN}",
                long_description="Another testing distribution \N{SNOWMAN}",
                author="Illustrious Author",
                author_email="illustrious@example.org",
                url="http://example.org/exemplary",
                packages=["complexdist"],
                setup_requires=["setuptools"],
                install_requires=["quux", "splort"],
                extras_require={"simple": ["simple.dist"]},
                entry_points={
                    "console_scripts": [
                        "complex-dist=complexdist:main",
                        "complex-dist2=complexdist:main",
                    ],
                },
            )
            """
        ),
        "complexdist": {"__init__.py": "def main(): return"},
    },
    "headers-dist": {
        "setup.py": cleandoc(
            """
            from setuptools import setup

            setup(
                name="headers.dist",
                version="0.1",
                description="A distribution with headers",
                headers=["header.h"],
            )
            """
        ),
        "setup.cfg": "[bdist_wheel]\nuniversal=1",
        "headersdist.py": "",
        "header.h": "",
    },
    "commasinfilenames-dist": {
        "setup.py": cleandoc(
            """
            from setuptools import setup

            setup(
                name="testrepo",
                version="0.1",
                packages=["mypackage"],
                description="A test package with commas in file names",
                include_package_data=True,
                package_data={"mypackage.data": ["*"]},
            )
            """
        ),
        "mypackage": {
            "__init__.py": "",
            "data": {"__init__.py": "", "1,2,3.txt": ""},
        },
        "testrepo-0.1.0": {
            "mypackage": {"__init__.py": ""},
        },
    },
    "unicode-dist": {
        "setup.py": cleandoc(
            """
            from setuptools import setup

            setup(
                name="unicode.dist",
                version="0.1",
                description="A testing distribution \N{SNOWMAN}",
                packages=["unicodedist"],
                zip_safe=True,
            )
            """
        ),
        "unicodedist": {"__init__.py": "", "åäö_日本語.py": ""},
    },
    "utf8-metadata-dist": {
        "setup.cfg": cleandoc(
            """
            [metadata]
            name = utf8-metadata-dist
            version = 42
            author_email = "John X. Ãørçeč" <john@utf8.org>, Γαμα קּ 東 <gama@utf8.org>
            long_description = file: README.rst
            """
        ),
        "README.rst": "UTF-8 描述 説明",
    },
}


if sys.platform != "win32":
    # ABI3 extensions don't really work on Windows
    EXAMPLES["abi3extension-dist"] = {
        "setup.py": cleandoc(
            """
            from setuptools import Extension, setup

            setup(
                name="extension.dist",
                version="0.1",
                description="A testing distribution \N{SNOWMAN}",
                ext_modules=[
                    Extension(
                        name="extension", sources=["extension.c"], py_limited_api=True
                    )
                ],
            )
            """
        ),
        "setup.cfg": "[bdist_wheel]\npy_limited_api=cp32",
        "extension.c": "#define Py_LIMITED_API 0x03020000\n#include <Python.h>",
    }


def bdist_wheel_cmd(**kwargs):
    """Run command in the same process so that it is easier to collect coverage"""
    dist_obj = (
        run_setup("setup.py", stop_after="init")
        if os.path.exists("setup.py")
        else Distribution({"script_name": "%%build_meta%%"})
    )
    dist_obj.parse_config_files()
    cmd = bdist_wheel(dist_obj)
    for attr, value in kwargs.items():
        setattr(cmd, attr, value)
    cmd.finalize_options()
    return cmd


def mkexample(tmp_path_factory, name):
    basedir = tmp_path_factory.mktemp(name)
    jaraco.path.build(EXAMPLES[name], prefix=str(basedir))
    return basedir


@pytest.fixture(scope="session")
def wheel_paths(tmp_path_factory):
    build_base = tmp_path_factory.mktemp("build")
    dist_dir = tmp_path_factory.mktemp("dist")
    for name in EXAMPLES:
        example_dir = mkexample(tmp_path_factory, name)
        build_dir = build_base / name
        with jaraco.path.DirectoryStack().context(example_dir):
            bdist_wheel_cmd(bdist_dir=str(build_dir), dist_dir=str(dist_dir)).run()

    return sorted(str(fname) for fname in dist_dir.glob("*.whl"))


@pytest.fixture
def dummy_dist(tmp_path_factory):
    return mkexample(tmp_path_factory, "dummy-dist")


def test_no_scripts(wheel_paths):
    """Make sure entry point scripts are not generated."""
    path = next(path for path in wheel_paths if "complex_dist" in path)
    for entry in ZipFile(path).infolist():
        assert ".data/scripts/" not in entry.filename


def test_unicode_record(wheel_paths):
    path = next(path for path in wheel_paths if "unicode.dist" in path)
    with ZipFile(path) as zf:
        record = zf.read("unicode.dist-0.1.dist-info/RECORD")

    assert "åäö_日本語.py".encode() in record


UTF8_PKG_INFO = """\
Metadata-Version: 2.1
Name: helloworld
Version: 42
Author-email: "John X. Ãørçeč" <john@utf8.org>, Γαμα קּ 東 <gama@utf8.org>


UTF-8 描述 説明
"""


def test_preserve_unicode_metadata(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    egginfo = tmp_path / "dummy_dist.egg-info"
    distinfo = tmp_path / "dummy_dist.dist-info"

    egginfo.mkdir()
    (egginfo / "PKG-INFO").write_text(UTF8_PKG_INFO, encoding="utf-8")
    (egginfo / "dependency_links.txt").touch()

    class simpler_bdist_wheel(bdist_wheel):
        """Avoid messing with setuptools/distutils internals"""

        def __init__(self):
            pass

        @property
        def license_paths(self):
            return []

    cmd_obj = simpler_bdist_wheel()
    cmd_obj.egg2dist(egginfo, distinfo)

    metadata = (distinfo / "METADATA").read_text(encoding="utf-8")
    assert 'Author-email: "John X. Ãørçeč"' in metadata
    assert "Γαμα קּ 東 " in metadata
    assert "UTF-8 描述 説明" in metadata


def test_licenses_default(dummy_dist, monkeypatch, tmp_path):
    monkeypatch.chdir(dummy_dist)
    bdist_wheel_cmd(bdist_dir=str(tmp_path), universal=True).run()
    with ZipFile("dist/dummy_dist-1.0-py2.py3-none-any.whl") as wf:
        license_files = {
            "dummy_dist-1.0.dist-info/" + fname for fname in DEFAULT_LICENSE_FILES
        }
        assert set(wf.namelist()) == DEFAULT_FILES | license_files


def test_licenses_deprecated(dummy_dist, monkeypatch, tmp_path):
    dummy_dist.joinpath("setup.cfg").write_text(
        "[metadata]\nlicense_file=licenses/DUMMYFILE", encoding="utf-8"
    )
    monkeypatch.chdir(dummy_dist)

    bdist_wheel_cmd(bdist_dir=str(tmp_path), universal=True).run()

    with ZipFile("dist/dummy_dist-1.0-py2.py3-none-any.whl") as wf:
        license_files = {"dummy_dist-1.0.dist-info/DUMMYFILE"}
        assert set(wf.namelist()) == DEFAULT_FILES | license_files


@pytest.mark.parametrize(
    "config_file, config",
    [
        ("setup.cfg", "[metadata]\nlicense_files=licenses/*\n  LICENSE"),
        ("setup.cfg", "[metadata]\nlicense_files=licenses/*, LICENSE"),
        (
            "setup.py",
            SETUPPY_EXAMPLE.replace(
                ")", "  license_files=['licenses/DUMMYFILE', 'LICENSE'])"
            ),
        ),
    ],
)
def test_licenses_override(dummy_dist, monkeypatch, tmp_path, config_file, config):
    dummy_dist.joinpath(config_file).write_text(config, encoding="utf-8")
    monkeypatch.chdir(dummy_dist)
    bdist_wheel_cmd(bdist_dir=str(tmp_path), universal=True).run()
    with ZipFile("dist/dummy_dist-1.0-py2.py3-none-any.whl") as wf:
        license_files = {
            "dummy_dist-1.0.dist-info/" + fname for fname in {"DUMMYFILE", "LICENSE"}
        }
        assert set(wf.namelist()) == DEFAULT_FILES | license_files


def test_licenses_disabled(dummy_dist, monkeypatch, tmp_path):
    dummy_dist.joinpath("setup.cfg").write_text(
        "[metadata]\nlicense_files=\n", encoding="utf-8"
    )
    monkeypatch.chdir(dummy_dist)
    bdist_wheel_cmd(bdist_dir=str(tmp_path), universal=True).run()
    with ZipFile("dist/dummy_dist-1.0-py2.py3-none-any.whl") as wf:
        assert set(wf.namelist()) == DEFAULT_FILES


def test_build_number(dummy_dist, monkeypatch, tmp_path):
    monkeypatch.chdir(dummy_dist)
    bdist_wheel_cmd(bdist_dir=str(tmp_path), build_number="2", universal=True).run()
    with ZipFile("dist/dummy_dist-1.0-2-py2.py3-none-any.whl") as wf:
        filenames = set(wf.namelist())
        assert "dummy_dist-1.0.dist-info/RECORD" in filenames
        assert "dummy_dist-1.0.dist-info/METADATA" in filenames


EXTENSION_EXAMPLE = """\
#include <Python.h>

static PyMethodDef methods[] = {
  { NULL, NULL, 0, NULL }
};

static struct PyModuleDef module_def = {
  PyModuleDef_HEAD_INIT,
  "extension",
  "Dummy extension module",
  -1,
  methods
};

PyMODINIT_FUNC PyInit_extension(void) {
  return PyModule_Create(&module_def);
}
"""
EXTENSION_SETUPPY = """\
from __future__ import annotations

from setuptools import Extension, setup

setup(
    name="extension.dist",
    version="0.1",
    description="A testing distribution \N{SNOWMAN}",
    ext_modules=[Extension(name="extension", sources=["extension.c"])],
)
"""


@pytest.mark.filterwarnings(
    "once:Config variable '.*' is unset.*, Python ABI tag may be incorrect"
)
def test_limited_abi(monkeypatch, tmp_path, tmp_path_factory):
    """Test that building a binary wheel with the limited ABI works."""
    source_dir = tmp_path_factory.mktemp("extension_dist")
    (source_dir / "setup.py").write_text(EXTENSION_SETUPPY, encoding="utf-8")
    (source_dir / "extension.c").write_text(EXTENSION_EXAMPLE, encoding="utf-8")
    build_dir = tmp_path.joinpath("build")
    dist_dir = tmp_path.joinpath("dist")
    monkeypatch.chdir(source_dir)
    bdist_wheel_cmd(bdist_dir=str(build_dir), dist_dir=str(dist_dir)).run()


def test_build_from_readonly_tree(dummy_dist, monkeypatch, tmp_path):
    basedir = str(tmp_path.joinpath("dummy"))
    shutil.copytree(str(dummy_dist), basedir)
    monkeypatch.chdir(basedir)

    # Make the tree read-only
    for root, _dirs, files in os.walk(basedir):
        for fname in files:
            os.chmod(os.path.join(root, fname), stat.S_IREAD)

    bdist_wheel_cmd().run()


@pytest.mark.parametrize(
    "option, compress_type",
    list(bdist_wheel.supported_compressions.items()),
    ids=list(bdist_wheel.supported_compressions),
)
def test_compression(dummy_dist, monkeypatch, tmp_path, option, compress_type):
    monkeypatch.chdir(dummy_dist)
    bdist_wheel_cmd(bdist_dir=str(tmp_path), universal=True, compression=option).run()
    with ZipFile("dist/dummy_dist-1.0-py2.py3-none-any.whl") as wf:
        filenames = set(wf.namelist())
        assert "dummy_dist-1.0.dist-info/RECORD" in filenames
        assert "dummy_dist-1.0.dist-info/METADATA" in filenames
        for zinfo in wf.filelist:
            assert zinfo.compress_type == compress_type


def test_wheelfile_line_endings(wheel_paths):
    for path in wheel_paths:
        with ZipFile(path) as wf:
            wheelfile = next(fn for fn in wf.filelist if fn.filename.endswith("WHEEL"))
            wheelfile_contents = wf.read(wheelfile)
            assert b"\r" not in wheelfile_contents


def test_unix_epoch_timestamps(dummy_dist, monkeypatch, tmp_path):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")
    monkeypatch.chdir(dummy_dist)
    bdist_wheel_cmd(bdist_dir=str(tmp_path), build_number="2a", universal=True).run()
    with ZipFile("dist/dummy_dist-1.0-2a-py2.py3-none-any.whl") as wf:
        for zinfo in wf.filelist:
            assert zinfo.date_time >= (1980, 1, 1, 0, 0, 0)  # min epoch is used


def test_get_abi_tag_windows(monkeypatch):
    monkeypatch.setattr(tags, "interpreter_name", lambda: "cp")
    monkeypatch.setattr(sysconfig, "get_config_var", lambda x: "cp313-win_amd64")
    assert get_abi_tag() == "cp313"


def test_get_abi_tag_pypy_old(monkeypatch):
    monkeypatch.setattr(tags, "interpreter_name", lambda: "pp")
    monkeypatch.setattr(sysconfig, "get_config_var", lambda x: "pypy36-pp73")
    assert get_abi_tag() == "pypy36_pp73"


def test_get_abi_tag_pypy_new(monkeypatch):
    monkeypatch.setattr(sysconfig, "get_config_var", lambda x: "pypy37-pp73-darwin")
    monkeypatch.setattr(tags, "interpreter_name", lambda: "pp")
    assert get_abi_tag() == "pypy37_pp73"


def test_get_abi_tag_graalpy(monkeypatch):
    monkeypatch.setattr(
        sysconfig, "get_config_var", lambda x: "graalpy231-310-native-x86_64-linux"
    )
    monkeypatch.setattr(tags, "interpreter_name", lambda: "graalpy")
    assert get_abi_tag() == "graalpy231_310_native"


def test_get_abi_tag_fallback(monkeypatch):
    monkeypatch.setattr(sysconfig, "get_config_var", lambda x: "unknown-python-310")
    monkeypatch.setattr(tags, "interpreter_name", lambda: "unknown-python")
    assert get_abi_tag() == "unknown_python_310"


def test_platform_with_space(dummy_dist, monkeypatch):
    """Ensure building on platforms with a space in the name succeed."""
    monkeypatch.chdir(dummy_dist)
    bdist_wheel_cmd(plat_name="isilon onefs").run()


def test_rmtree_readonly(monkeypatch, tmp_path):
    """Verify onerr works as expected"""

    bdist_dir = tmp_path / "with_readonly"
    bdist_dir.mkdir()
    some_file = bdist_dir.joinpath("file.txt")
    some_file.touch()
    some_file.chmod(stat.S_IREAD)

    expected_count = 1 if sys.platform.startswith("win") else 0

    if sys.version_info < (3, 12):
        count_remove_readonly = Mock(side_effect=remove_readonly)
        shutil.rmtree(bdist_dir, onerror=count_remove_readonly)
        assert count_remove_readonly.call_count == expected_count
    else:
        count_remove_readonly_exc = Mock(side_effect=remove_readonly_exc)
        shutil.rmtree(bdist_dir, onexc=count_remove_readonly_exc)
        assert count_remove_readonly_exc.call_count == expected_count

    assert not bdist_dir.is_dir()


def test_data_dir_with_tag_build(monkeypatch, tmp_path):
    """
    Setuptools allow authors to set PEP 440's local version segments
    using ``egg_info.tag_build``. This should be reflected not only in the
    ``.whl`` file name, but also in the ``.dist-info`` and ``.data`` dirs.
    See pypa/setuptools#3997.
    """
    monkeypatch.chdir(tmp_path)
    files = {
        "setup.py": """
            from setuptools import setup
            setup(headers=["hello.h"])
            """,
        "setup.cfg": """
            [metadata]
            name = test
            version = 1.0

            [options.data_files]
            hello/world = file.txt

            [egg_info]
            tag_build = +what
            tag_date = 0
            """,
        "file.txt": "",
        "hello.h": "",
    }
    for file, content in files.items():
        with open(file, "w", encoding="utf-8") as fh:
            fh.write(cleandoc(content))

    bdist_wheel_cmd().run()

    # Ensure .whl, .dist-info and .data contain the local segment
    wheel_path = "dist/test-1.0+what-py3-none-any.whl"
    assert os.path.exists(wheel_path)
    entries = set(ZipFile(wheel_path).namelist())
    for expected in (
        "test-1.0+what.data/headers/hello.h",
        "test-1.0+what.data/data/hello/world/file.txt",
        "test-1.0+what.dist-info/METADATA",
        "test-1.0+what.dist-info/WHEEL",
    ):
        assert expected in entries

    for not_expected in (
        "test.data/headers/hello.h",
        "test-1.0.data/data/hello/world/file.txt",
        "test.dist-info/METADATA",
        "test-1.0.dist-info/WHEEL",
    ):
        assert not_expected not in entries


@pytest.mark.parametrize(
    "reported,expected",
    [("linux-x86_64", "linux_i686"), ("linux-aarch64", "linux_armv7l")],
)
@pytest.mark.skipif(
    platform.system() != "Linux", reason="Only makes sense to test on Linux"
)
def test_platform_linux32(reported, expected, monkeypatch):
    monkeypatch.setattr(struct, "calcsize", lambda x: 4)
    dist = setuptools.Distribution()
    cmd = bdist_wheel(dist)
    cmd.plat_name = reported
    cmd.root_is_pure = False
    _, _, actual = cmd.get_tag()
    assert actual == expected


def test_no_ctypes(monkeypatch) -> None:
    def _fake_import(name: str, *args, **kwargs):
        if name == "ctypes":
            raise ModuleNotFoundError(f"No module named {name}")

        return importlib.__import__(name, *args, **kwargs)

    with suppress(KeyError):
        monkeypatch.delitem(sys.modules, "wheel.macosx_libfile")

    # Install an importer shim that refuses to load ctypes
    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(ModuleNotFoundError, match="No module named ctypes"):
        import wheel.macosx_libfile  # noqa: F401

    # Unload and reimport the bdist_wheel command module to make sure it won't try to
    # import ctypes
    monkeypatch.delitem(sys.modules, "setuptools.command.bdist_wheel")

    import setuptools.command.bdist_wheel  # noqa: F401
