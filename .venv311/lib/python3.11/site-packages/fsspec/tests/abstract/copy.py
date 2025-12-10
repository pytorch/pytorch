from hashlib import md5
from itertools import product

import pytest

from fsspec.tests.abstract.common import GLOB_EDGE_CASES_TESTS


class AbstractCopyTests:
    def test_copy_file_to_existing_directory(
        self,
        fs,
        fs_join,
        fs_bulk_operations_scenario_0,
        fs_target,
        supports_empty_directories,
    ):
        # Copy scenario 1a
        source = fs_bulk_operations_scenario_0

        target = fs_target
        fs.mkdir(target)
        if not supports_empty_directories:
            # Force target directory to exist by adding a dummy file
            fs.touch(fs_join(target, "dummy"))
        assert fs.isdir(target)

        target_file2 = fs_join(target, "file2")
        target_subfile1 = fs_join(target, "subfile1")

        # Copy from source directory
        fs.cp(fs_join(source, "file2"), target)
        assert fs.isfile(target_file2)

        # Copy from sub directory
        fs.cp(fs_join(source, "subdir", "subfile1"), target)
        assert fs.isfile(target_subfile1)

        # Remove copied files
        fs.rm([target_file2, target_subfile1])
        assert not fs.exists(target_file2)
        assert not fs.exists(target_subfile1)

        # Repeat with trailing slash on target
        fs.cp(fs_join(source, "file2"), target + "/")
        assert fs.isdir(target)
        assert fs.isfile(target_file2)

        fs.cp(fs_join(source, "subdir", "subfile1"), target + "/")
        assert fs.isfile(target_subfile1)

    def test_copy_file_to_new_directory(
        self, fs, fs_join, fs_bulk_operations_scenario_0, fs_target
    ):
        # Copy scenario 1b
        source = fs_bulk_operations_scenario_0

        target = fs_target
        fs.mkdir(target)

        fs.cp(
            fs_join(source, "subdir", "subfile1"), fs_join(target, "newdir/")
        )  # Note trailing slash
        assert fs.isdir(target)
        assert fs.isdir(fs_join(target, "newdir"))
        assert fs.isfile(fs_join(target, "newdir", "subfile1"))

    def test_copy_file_to_file_in_existing_directory(
        self,
        fs,
        fs_join,
        fs_bulk_operations_scenario_0,
        fs_target,
        supports_empty_directories,
    ):
        # Copy scenario 1c
        source = fs_bulk_operations_scenario_0

        target = fs_target
        fs.mkdir(target)
        if not supports_empty_directories:
            # Force target directory to exist by adding a dummy file
            fs.touch(fs_join(target, "dummy"))
        assert fs.isdir(target)

        fs.cp(fs_join(source, "subdir", "subfile1"), fs_join(target, "newfile"))
        assert fs.isfile(fs_join(target, "newfile"))

    def test_copy_file_to_file_in_new_directory(
        self, fs, fs_join, fs_bulk_operations_scenario_0, fs_target
    ):
        # Copy scenario 1d
        source = fs_bulk_operations_scenario_0

        target = fs_target
        fs.mkdir(target)

        fs.cp(
            fs_join(source, "subdir", "subfile1"), fs_join(target, "newdir", "newfile")
        )
        assert fs.isdir(fs_join(target, "newdir"))
        assert fs.isfile(fs_join(target, "newdir", "newfile"))

    def test_copy_directory_to_existing_directory(
        self,
        fs,
        fs_join,
        fs_bulk_operations_scenario_0,
        fs_target,
        supports_empty_directories,
    ):
        # Copy scenario 1e
        source = fs_bulk_operations_scenario_0

        target = fs_target
        fs.mkdir(target)
        if not supports_empty_directories:
            # Force target directory to exist by adding a dummy file
            dummy = fs_join(target, "dummy")
            fs.touch(dummy)
        assert fs.isdir(target)

        for source_slash, target_slash in zip([False, True], [False, True]):
            s = fs_join(source, "subdir")
            if source_slash:
                s += "/"
            t = target + "/" if target_slash else target

            # Without recursive does nothing
            fs.cp(s, t)
            assert fs.ls(target, detail=False) == (
                [] if supports_empty_directories else [dummy]
            )

            # With recursive
            fs.cp(s, t, recursive=True)
            if source_slash:
                assert fs.isfile(fs_join(target, "subfile1"))
                assert fs.isfile(fs_join(target, "subfile2"))
                assert fs.isdir(fs_join(target, "nesteddir"))
                assert fs.isfile(fs_join(target, "nesteddir", "nestedfile"))
                assert not fs.exists(fs_join(target, "subdir"))

                fs.rm(
                    [
                        fs_join(target, "subfile1"),
                        fs_join(target, "subfile2"),
                        fs_join(target, "nesteddir"),
                    ],
                    recursive=True,
                )
            else:
                assert fs.isdir(fs_join(target, "subdir"))
                assert fs.isfile(fs_join(target, "subdir", "subfile1"))
                assert fs.isfile(fs_join(target, "subdir", "subfile2"))
                assert fs.isdir(fs_join(target, "subdir", "nesteddir"))
                assert fs.isfile(fs_join(target, "subdir", "nesteddir", "nestedfile"))

                fs.rm(fs_join(target, "subdir"), recursive=True)
            assert fs.ls(target, detail=False) == (
                [] if supports_empty_directories else [dummy]
            )

            # Limit recursive by maxdepth
            fs.cp(s, t, recursive=True, maxdepth=1)
            if source_slash:
                assert fs.isfile(fs_join(target, "subfile1"))
                assert fs.isfile(fs_join(target, "subfile2"))
                assert not fs.exists(fs_join(target, "nesteddir"))
                assert not fs.exists(fs_join(target, "subdir"))

                fs.rm(
                    [
                        fs_join(target, "subfile1"),
                        fs_join(target, "subfile2"),
                    ],
                    recursive=True,
                )
            else:
                assert fs.isdir(fs_join(target, "subdir"))
                assert fs.isfile(fs_join(target, "subdir", "subfile1"))
                assert fs.isfile(fs_join(target, "subdir", "subfile2"))
                assert not fs.exists(fs_join(target, "subdir", "nesteddir"))

                fs.rm(fs_join(target, "subdir"), recursive=True)
            assert fs.ls(target, detail=False) == (
                [] if supports_empty_directories else [dummy]
            )

    def test_copy_directory_to_new_directory(
        self,
        fs,
        fs_join,
        fs_bulk_operations_scenario_0,
        fs_target,
        supports_empty_directories,
    ):
        # Copy scenario 1f
        source = fs_bulk_operations_scenario_0

        target = fs_target
        fs.mkdir(target)

        for source_slash, target_slash in zip([False, True], [False, True]):
            s = fs_join(source, "subdir")
            if source_slash:
                s += "/"
            t = fs_join(target, "newdir")
            if target_slash:
                t += "/"

            # Without recursive does nothing
            fs.cp(s, t)
            if supports_empty_directories:
                assert fs.ls(target) == []
            else:
                with pytest.raises(FileNotFoundError):
                    fs.ls(target)

            # With recursive
            fs.cp(s, t, recursive=True)
            assert fs.isdir(fs_join(target, "newdir"))
            assert fs.isfile(fs_join(target, "newdir", "subfile1"))
            assert fs.isfile(fs_join(target, "newdir", "subfile2"))
            assert fs.isdir(fs_join(target, "newdir", "nesteddir"))
            assert fs.isfile(fs_join(target, "newdir", "nesteddir", "nestedfile"))
            assert not fs.exists(fs_join(target, "subdir"))

            fs.rm(fs_join(target, "newdir"), recursive=True)
            assert not fs.exists(fs_join(target, "newdir"))

            # Limit recursive by maxdepth
            fs.cp(s, t, recursive=True, maxdepth=1)
            assert fs.isdir(fs_join(target, "newdir"))
            assert fs.isfile(fs_join(target, "newdir", "subfile1"))
            assert fs.isfile(fs_join(target, "newdir", "subfile2"))
            assert not fs.exists(fs_join(target, "newdir", "nesteddir"))
            assert not fs.exists(fs_join(target, "subdir"))

            fs.rm(fs_join(target, "newdir"), recursive=True)
            assert not fs.exists(fs_join(target, "newdir"))

    def test_copy_glob_to_existing_directory(
        self,
        fs,
        fs_join,
        fs_bulk_operations_scenario_0,
        fs_target,
        supports_empty_directories,
    ):
        # Copy scenario 1g
        source = fs_bulk_operations_scenario_0

        target = fs_target
        fs.mkdir(target)
        if not supports_empty_directories:
            # Force target directory to exist by adding a dummy file
            dummy = fs_join(target, "dummy")
            fs.touch(dummy)
        assert fs.isdir(target)

        for target_slash in [False, True]:
            t = target + "/" if target_slash else target

            # Without recursive
            fs.cp(fs_join(source, "subdir", "*"), t)
            assert fs.isfile(fs_join(target, "subfile1"))
            assert fs.isfile(fs_join(target, "subfile2"))
            assert not fs.isdir(fs_join(target, "nesteddir"))
            assert not fs.exists(fs_join(target, "nesteddir", "nestedfile"))
            assert not fs.exists(fs_join(target, "subdir"))

            fs.rm(
                [
                    fs_join(target, "subfile1"),
                    fs_join(target, "subfile2"),
                ],
                recursive=True,
            )
            assert fs.ls(target, detail=False) == (
                [] if supports_empty_directories else [dummy]
            )

            # With recursive
            for glob, recursive in zip(["*", "**"], [True, False]):
                fs.cp(fs_join(source, "subdir", glob), t, recursive=recursive)
                assert fs.isfile(fs_join(target, "subfile1"))
                assert fs.isfile(fs_join(target, "subfile2"))
                assert fs.isdir(fs_join(target, "nesteddir"))
                assert fs.isfile(fs_join(target, "nesteddir", "nestedfile"))
                assert not fs.exists(fs_join(target, "subdir"))

                fs.rm(
                    [
                        fs_join(target, "subfile1"),
                        fs_join(target, "subfile2"),
                        fs_join(target, "nesteddir"),
                    ],
                    recursive=True,
                )
                assert fs.ls(target, detail=False) == (
                    [] if supports_empty_directories else [dummy]
                )

                # Limit recursive by maxdepth
                fs.cp(
                    fs_join(source, "subdir", glob), t, recursive=recursive, maxdepth=1
                )
                assert fs.isfile(fs_join(target, "subfile1"))
                assert fs.isfile(fs_join(target, "subfile2"))
                assert not fs.exists(fs_join(target, "nesteddir"))
                assert not fs.exists(fs_join(target, "subdir"))

                fs.rm(
                    [
                        fs_join(target, "subfile1"),
                        fs_join(target, "subfile2"),
                    ],
                    recursive=True,
                )
                assert fs.ls(target, detail=False) == (
                    [] if supports_empty_directories else [dummy]
                )

    def test_copy_glob_to_new_directory(
        self, fs, fs_join, fs_bulk_operations_scenario_0, fs_target
    ):
        # Copy scenario 1h
        source = fs_bulk_operations_scenario_0

        target = fs_target
        fs.mkdir(target)

        for target_slash in [False, True]:
            t = fs_join(target, "newdir")
            if target_slash:
                t += "/"

            # Without recursive
            fs.cp(fs_join(source, "subdir", "*"), t)
            assert fs.isdir(fs_join(target, "newdir"))
            assert fs.isfile(fs_join(target, "newdir", "subfile1"))
            assert fs.isfile(fs_join(target, "newdir", "subfile2"))
            assert not fs.exists(fs_join(target, "newdir", "nesteddir"))
            assert not fs.exists(fs_join(target, "newdir", "nesteddir", "nestedfile"))
            assert not fs.exists(fs_join(target, "subdir"))
            assert not fs.exists(fs_join(target, "newdir", "subdir"))

            fs.rm(fs_join(target, "newdir"), recursive=True)
            assert not fs.exists(fs_join(target, "newdir"))

            # With recursive
            for glob, recursive in zip(["*", "**"], [True, False]):
                fs.cp(fs_join(source, "subdir", glob), t, recursive=recursive)
                assert fs.isdir(fs_join(target, "newdir"))
                assert fs.isfile(fs_join(target, "newdir", "subfile1"))
                assert fs.isfile(fs_join(target, "newdir", "subfile2"))
                assert fs.isdir(fs_join(target, "newdir", "nesteddir"))
                assert fs.isfile(fs_join(target, "newdir", "nesteddir", "nestedfile"))
                assert not fs.exists(fs_join(target, "subdir"))
                assert not fs.exists(fs_join(target, "newdir", "subdir"))

                fs.rm(fs_join(target, "newdir"), recursive=True)
                assert not fs.exists(fs_join(target, "newdir"))

                # Limit recursive by maxdepth
                fs.cp(
                    fs_join(source, "subdir", glob), t, recursive=recursive, maxdepth=1
                )
                assert fs.isdir(fs_join(target, "newdir"))
                assert fs.isfile(fs_join(target, "newdir", "subfile1"))
                assert fs.isfile(fs_join(target, "newdir", "subfile2"))
                assert not fs.exists(fs_join(target, "newdir", "nesteddir"))
                assert not fs.exists(fs_join(target, "subdir"))
                assert not fs.exists(fs_join(target, "newdir", "subdir"))

                fs.rm(fs_join(target, "newdir"), recursive=True)
                assert not fs.exists(fs_join(target, "newdir"))

    @pytest.mark.parametrize(
        GLOB_EDGE_CASES_TESTS["argnames"],
        GLOB_EDGE_CASES_TESTS["argvalues"],
    )
    def test_copy_glob_edge_cases(
        self,
        path,
        recursive,
        maxdepth,
        expected,
        fs,
        fs_join,
        fs_glob_edge_cases_files,
        fs_target,
        fs_sanitize_path,
    ):
        # Copy scenario 1g
        source = fs_glob_edge_cases_files

        target = fs_target

        for new_dir, target_slash in product([True, False], [True, False]):
            fs.mkdir(target)

            t = fs_join(target, "newdir") if new_dir else target
            t = t + "/" if target_slash else t

            fs.copy(fs_join(source, path), t, recursive=recursive, maxdepth=maxdepth)

            output = fs.find(target)
            if new_dir:
                prefixed_expected = [
                    fs_sanitize_path(fs_join(target, "newdir", p)) for p in expected
                ]
            else:
                prefixed_expected = [
                    fs_sanitize_path(fs_join(target, p)) for p in expected
                ]
            assert sorted(output) == sorted(prefixed_expected)

            try:
                fs.rm(target, recursive=True)
            except FileNotFoundError:
                pass

    def test_copy_list_of_files_to_existing_directory(
        self,
        fs,
        fs_join,
        fs_bulk_operations_scenario_0,
        fs_target,
        supports_empty_directories,
    ):
        # Copy scenario 2a
        source = fs_bulk_operations_scenario_0

        target = fs_target
        fs.mkdir(target)
        if not supports_empty_directories:
            # Force target directory to exist by adding a dummy file
            dummy = fs_join(target, "dummy")
            fs.touch(dummy)
        assert fs.isdir(target)

        source_files = [
            fs_join(source, "file1"),
            fs_join(source, "file2"),
            fs_join(source, "subdir", "subfile1"),
        ]

        for target_slash in [False, True]:
            t = target + "/" if target_slash else target

            fs.cp(source_files, t)
            assert fs.isfile(fs_join(target, "file1"))
            assert fs.isfile(fs_join(target, "file2"))
            assert fs.isfile(fs_join(target, "subfile1"))

            fs.rm(
                [
                    fs_join(target, "file1"),
                    fs_join(target, "file2"),
                    fs_join(target, "subfile1"),
                ],
                recursive=True,
            )
            assert fs.ls(target, detail=False) == (
                [] if supports_empty_directories else [dummy]
            )

    def test_copy_list_of_files_to_new_directory(
        self, fs, fs_join, fs_bulk_operations_scenario_0, fs_target
    ):
        # Copy scenario 2b
        source = fs_bulk_operations_scenario_0

        target = fs_target
        fs.mkdir(target)

        source_files = [
            fs_join(source, "file1"),
            fs_join(source, "file2"),
            fs_join(source, "subdir", "subfile1"),
        ]

        fs.cp(source_files, fs_join(target, "newdir") + "/")  # Note trailing slash
        assert fs.isdir(fs_join(target, "newdir"))
        assert fs.isfile(fs_join(target, "newdir", "file1"))
        assert fs.isfile(fs_join(target, "newdir", "file2"))
        assert fs.isfile(fs_join(target, "newdir", "subfile1"))

    def test_copy_two_files_new_directory(
        self, fs, fs_join, fs_bulk_operations_scenario_0, fs_target
    ):
        # This is a duplicate of test_copy_list_of_files_to_new_directory and
        # can eventually be removed.
        source = fs_bulk_operations_scenario_0

        target = fs_target
        assert not fs.exists(target)
        fs.cp([fs_join(source, "file1"), fs_join(source, "file2")], target)

        assert fs.isdir(target)
        assert fs.isfile(fs_join(target, "file1"))
        assert fs.isfile(fs_join(target, "file2"))

    def test_copy_directory_without_files_with_same_name_prefix(
        self,
        fs,
        fs_join,
        fs_target,
        fs_dir_and_file_with_same_name_prefix,
        supports_empty_directories,
    ):
        # Create the test dirs
        source = fs_dir_and_file_with_same_name_prefix
        target = fs_target

        # Test without glob
        fs.cp(fs_join(source, "subdir"), target, recursive=True)

        assert fs.isfile(fs_join(target, "subfile.txt"))
        assert not fs.isfile(fs_join(target, "subdir.txt"))

        fs.rm([fs_join(target, "subfile.txt")])
        if supports_empty_directories:
            assert fs.ls(target) == []
        else:
            assert not fs.exists(target)

        # Test with glob
        fs.cp(fs_join(source, "subdir*"), target, recursive=True)

        assert fs.isdir(fs_join(target, "subdir"))
        assert fs.isfile(fs_join(target, "subdir", "subfile.txt"))
        assert fs.isfile(fs_join(target, "subdir.txt"))

    def test_copy_with_source_and_destination_as_list(
        self, fs, fs_target, fs_join, fs_10_files_with_hashed_names
    ):
        # Create the test dir
        source = fs_10_files_with_hashed_names
        target = fs_target

        # Create list of files for source and destination
        source_files = []
        destination_files = []
        for i in range(10):
            hashed_i = md5(str(i).encode("utf-8")).hexdigest()
            source_files.append(fs_join(source, f"{hashed_i}.txt"))
            destination_files.append(fs_join(target, f"{hashed_i}.txt"))

        # Copy and assert order was kept
        fs.copy(path1=source_files, path2=destination_files)

        for i in range(10):
            file_content = fs.cat(destination_files[i]).decode("utf-8")
            assert file_content == str(i)
