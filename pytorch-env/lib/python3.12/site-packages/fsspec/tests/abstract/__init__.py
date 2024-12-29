import os
from hashlib import md5

import pytest

from fsspec.implementations.local import LocalFileSystem
from fsspec.tests.abstract.copy import AbstractCopyTests  # noqa: F401
from fsspec.tests.abstract.get import AbstractGetTests  # noqa: F401
from fsspec.tests.abstract.put import AbstractPutTests  # noqa: F401


class BaseAbstractFixtures:
    """
    Abstract base class containing fixtures that are used by but never need to
    be overridden in derived filesystem-specific classes to run the abstract
    tests on such filesystems.
    """

    @pytest.fixture
    def fs_bulk_operations_scenario_0(self, fs, fs_join, fs_path):
        """
        Scenario on remote filesystem that is used for many cp/get/put tests.

        Cleans up at the end of each test it which it is used.
        """
        source = self._bulk_operations_scenario_0(fs, fs_join, fs_path)
        yield source
        fs.rm(source, recursive=True)

    @pytest.fixture
    def fs_glob_edge_cases_files(self, fs, fs_join, fs_path):
        """
        Scenario on remote filesystem that is used for glob edge cases cp/get/put tests.

        Cleans up at the end of each test it which it is used.
        """
        source = self._glob_edge_cases_files(fs, fs_join, fs_path)
        yield source
        fs.rm(source, recursive=True)

    @pytest.fixture
    def fs_dir_and_file_with_same_name_prefix(self, fs, fs_join, fs_path):
        """
        Scenario on remote filesystem that is used to check cp/get/put on directory
        and file with the same name prefixes.

        Cleans up at the end of each test it which it is used.
        """
        source = self._dir_and_file_with_same_name_prefix(fs, fs_join, fs_path)
        yield source
        fs.rm(source, recursive=True)

    @pytest.fixture
    def fs_10_files_with_hashed_names(self, fs, fs_join, fs_path):
        """
        Scenario on remote filesystem that is used to check cp/get/put files order
        when source and destination are lists.

        Cleans up at the end of each test it which it is used.
        """
        source = self._10_files_with_hashed_names(fs, fs_join, fs_path)
        yield source
        fs.rm(source, recursive=True)

    @pytest.fixture
    def fs_target(self, fs, fs_join, fs_path):
        """
        Return name of remote directory that does not yet exist to copy into.

        Cleans up at the end of each test it which it is used.
        """
        target = fs_join(fs_path, "target")
        yield target
        if fs.exists(target):
            fs.rm(target, recursive=True)

    @pytest.fixture
    def local_bulk_operations_scenario_0(self, local_fs, local_join, local_path):
        """
        Scenario on local filesystem that is used for many cp/get/put tests.

        Cleans up at the end of each test it which it is used.
        """
        source = self._bulk_operations_scenario_0(local_fs, local_join, local_path)
        yield source
        local_fs.rm(source, recursive=True)

    @pytest.fixture
    def local_glob_edge_cases_files(self, local_fs, local_join, local_path):
        """
        Scenario on local filesystem that is used for glob edge cases cp/get/put tests.

        Cleans up at the end of each test it which it is used.
        """
        source = self._glob_edge_cases_files(local_fs, local_join, local_path)
        yield source
        local_fs.rm(source, recursive=True)

    @pytest.fixture
    def local_dir_and_file_with_same_name_prefix(
        self, local_fs, local_join, local_path
    ):
        """
        Scenario on local filesystem that is used to check cp/get/put on directory
        and file with the same name prefixes.

        Cleans up at the end of each test it which it is used.
        """
        source = self._dir_and_file_with_same_name_prefix(
            local_fs, local_join, local_path
        )
        yield source
        local_fs.rm(source, recursive=True)

    @pytest.fixture
    def local_10_files_with_hashed_names(self, local_fs, local_join, local_path):
        """
        Scenario on local filesystem that is used to check cp/get/put files order
        when source and destination are lists.

        Cleans up at the end of each test it which it is used.
        """
        source = self._10_files_with_hashed_names(local_fs, local_join, local_path)
        yield source
        local_fs.rm(source, recursive=True)

    @pytest.fixture
    def local_target(self, local_fs, local_join, local_path):
        """
        Return name of local directory that does not yet exist to copy into.

        Cleans up at the end of each test it which it is used.
        """
        target = local_join(local_path, "target")
        yield target
        if local_fs.exists(target):
            local_fs.rm(target, recursive=True)

    def _glob_edge_cases_files(self, some_fs, some_join, some_path):
        """
        Scenario that is used for glob edge cases cp/get/put tests.
        Creates the following directory and file structure:

        📁 source
        ├── 📄 file1
        ├── 📄 file2
        ├── 📁 subdir0
        │   ├── 📄 subfile1
        │   ├── 📄 subfile2
        │   └── 📁 nesteddir
        │       └── 📄 nestedfile
        └── 📁 subdir1
            ├── 📄 subfile1
            ├── 📄 subfile2
            └── 📁 nesteddir
                └── 📄 nestedfile
        """
        source = some_join(some_path, "source")
        some_fs.touch(some_join(source, "file1"))
        some_fs.touch(some_join(source, "file2"))

        for subdir_idx in range(2):
            subdir = some_join(source, f"subdir{subdir_idx}")
            nesteddir = some_join(subdir, "nesteddir")
            some_fs.makedirs(nesteddir)
            some_fs.touch(some_join(subdir, "subfile1"))
            some_fs.touch(some_join(subdir, "subfile2"))
            some_fs.touch(some_join(nesteddir, "nestedfile"))

        return source

    def _bulk_operations_scenario_0(self, some_fs, some_join, some_path):
        """
        Scenario that is used for many cp/get/put tests. Creates the following
        directory and file structure:

        📁 source
        ├── 📄 file1
        ├── 📄 file2
        └── 📁 subdir
            ├── 📄 subfile1
            ├── 📄 subfile2
            └── 📁 nesteddir
                └── 📄 nestedfile
        """
        source = some_join(some_path, "source")
        subdir = some_join(source, "subdir")
        nesteddir = some_join(subdir, "nesteddir")
        some_fs.makedirs(nesteddir)
        some_fs.touch(some_join(source, "file1"))
        some_fs.touch(some_join(source, "file2"))
        some_fs.touch(some_join(subdir, "subfile1"))
        some_fs.touch(some_join(subdir, "subfile2"))
        some_fs.touch(some_join(nesteddir, "nestedfile"))
        return source

    def _dir_and_file_with_same_name_prefix(self, some_fs, some_join, some_path):
        """
        Scenario that is used to check cp/get/put on directory and file with
        the same name prefixes. Creates the following directory and file structure:

        📁 source
        ├── 📄 subdir.txt
        └── 📁 subdir
            └── 📄 subfile.txt
        """
        source = some_join(some_path, "source")
        subdir = some_join(source, "subdir")
        file = some_join(source, "subdir.txt")
        subfile = some_join(subdir, "subfile.txt")
        some_fs.makedirs(subdir)
        some_fs.touch(file)
        some_fs.touch(subfile)
        return source

    def _10_files_with_hashed_names(self, some_fs, some_join, some_path):
        """
        Scenario that is used to check cp/get/put files order when source and
        destination are lists. Creates the following directory and file structure:

        📁 source
        └── 📄 {hashed([0-9])}.txt
        """
        source = some_join(some_path, "source")
        for i in range(10):
            hashed_i = md5(str(i).encode("utf-8")).hexdigest()
            path = some_join(source, f"{hashed_i}.txt")
            some_fs.pipe(path=path, value=f"{i}".encode("utf-8"))
        return source


class AbstractFixtures(BaseAbstractFixtures):
    """
    Abstract base class containing fixtures that may be overridden in derived
    filesystem-specific classes to run the abstract tests on such filesystems.

    For any particular filesystem some of these fixtures must be overridden,
    such as ``fs`` and ``fs_path``, and others may be overridden if the
    default functions here are not appropriate, such as ``fs_join``.
    """

    @pytest.fixture
    def fs(self):
        raise NotImplementedError("This function must be overridden in derived classes")

    @pytest.fixture
    def fs_join(self):
        """
        Return a function that joins its arguments together into a path.

        Most fsspec implementations join paths in a platform-dependent way,
        but some will override this to always use a forward slash.
        """
        return os.path.join

    @pytest.fixture
    def fs_path(self):
        raise NotImplementedError("This function must be overridden in derived classes")

    @pytest.fixture(scope="class")
    def local_fs(self):
        # Maybe need an option for auto_mkdir=False?  This is only relevant
        # for certain implementations.
        return LocalFileSystem(auto_mkdir=True)

    @pytest.fixture
    def local_join(self):
        """
        Return a function that joins its arguments together into a path, on
        the local filesystem.
        """
        return os.path.join

    @pytest.fixture
    def local_path(self, tmpdir):
        return tmpdir

    @pytest.fixture
    def supports_empty_directories(self):
        """
        Return whether this implementation supports empty directories.
        """
        return True

    @pytest.fixture
    def fs_sanitize_path(self):
        return lambda x: x
