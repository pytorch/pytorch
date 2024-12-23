from hashlib import md5
from itertools import product

import pytest

from fsspec.implementations.local import make_path_posix
from fsspec.tests.abstract.common import GLOB_EDGE_CASES_TESTS


class AbstractGetTests:
    def test_get_file_to_existing_directory(
        self,
        fs,
        fs_join,
        fs_bulk_operations_scenario_0,
        local_fs,
        local_join,
        local_target,
    ):
        # Copy scenario 1a
        source = fs_bulk_operations_scenario_0

        target = local_target
        local_fs.mkdir(target)
        assert local_fs.isdir(target)

        target_file2 = local_join(target, "file2")
        target_subfile1 = local_join(target, "subfile1")

        # Copy from source directory
        fs.get(fs_join(source, "file2"), target)
        assert local_fs.isfile(target_file2)

        # Copy from sub directory
        fs.get(fs_join(source, "subdir", "subfile1"), target)
        assert local_fs.isfile(target_subfile1)

        # Remove copied files
        local_fs.rm([target_file2, target_subfile1])
        assert not local_fs.exists(target_file2)
        assert not local_fs.exists(target_subfile1)

        # Repeat with trailing slash on target
        fs.get(fs_join(source, "file2"), target + "/")
        assert local_fs.isdir(target)
        assert local_fs.isfile(target_file2)

        fs.get(fs_join(source, "subdir", "subfile1"), target + "/")
        assert local_fs.isfile(target_subfile1)

    def test_get_file_to_new_directory(
        self,
        fs,
        fs_join,
        fs_bulk_operations_scenario_0,
        local_fs,
        local_join,
        local_target,
    ):
        # Copy scenario 1b
        source = fs_bulk_operations_scenario_0

        target = local_target
        local_fs.mkdir(target)

        fs.get(
            fs_join(source, "subdir", "subfile1"), local_join(target, "newdir/")
        )  # Note trailing slash

        assert local_fs.isdir(target)
        assert local_fs.isdir(local_join(target, "newdir"))
        assert local_fs.isfile(local_join(target, "newdir", "subfile1"))

    def test_get_file_to_file_in_existing_directory(
        self,
        fs,
        fs_join,
        fs_bulk_operations_scenario_0,
        local_fs,
        local_join,
        local_target,
    ):
        # Copy scenario 1c
        source = fs_bulk_operations_scenario_0

        target = local_target
        local_fs.mkdir(target)

        fs.get(fs_join(source, "subdir", "subfile1"), local_join(target, "newfile"))
        assert local_fs.isfile(local_join(target, "newfile"))

    def test_get_file_to_file_in_new_directory(
        self,
        fs,
        fs_join,
        fs_bulk_operations_scenario_0,
        local_fs,
        local_join,
        local_target,
    ):
        # Copy scenario 1d
        source = fs_bulk_operations_scenario_0

        target = local_target
        local_fs.mkdir(target)

        fs.get(
            fs_join(source, "subdir", "subfile1"),
            local_join(target, "newdir", "newfile"),
        )
        assert local_fs.isdir(local_join(target, "newdir"))
        assert local_fs.isfile(local_join(target, "newdir", "newfile"))

    def test_get_directory_to_existing_directory(
        self,
        fs,
        fs_join,
        fs_bulk_operations_scenario_0,
        local_fs,
        local_join,
        local_target,
    ):
        # Copy scenario 1e
        source = fs_bulk_operations_scenario_0

        target = local_target
        local_fs.mkdir(target)
        assert local_fs.isdir(target)

        for source_slash, target_slash in zip([False, True], [False, True]):
            s = fs_join(source, "subdir")
            if source_slash:
                s += "/"
            t = target + "/" if target_slash else target

            # Without recursive does nothing
            fs.get(s, t)
            assert local_fs.ls(target) == []

            # With recursive
            fs.get(s, t, recursive=True)
            if source_slash:
                assert local_fs.isfile(local_join(target, "subfile1"))
                assert local_fs.isfile(local_join(target, "subfile2"))
                assert local_fs.isdir(local_join(target, "nesteddir"))
                assert local_fs.isfile(local_join(target, "nesteddir", "nestedfile"))
                assert not local_fs.exists(local_join(target, "subdir"))

                local_fs.rm(
                    [
                        local_join(target, "subfile1"),
                        local_join(target, "subfile2"),
                        local_join(target, "nesteddir"),
                    ],
                    recursive=True,
                )
            else:
                assert local_fs.isdir(local_join(target, "subdir"))
                assert local_fs.isfile(local_join(target, "subdir", "subfile1"))
                assert local_fs.isfile(local_join(target, "subdir", "subfile2"))
                assert local_fs.isdir(local_join(target, "subdir", "nesteddir"))
                assert local_fs.isfile(
                    local_join(target, "subdir", "nesteddir", "nestedfile")
                )

                local_fs.rm(local_join(target, "subdir"), recursive=True)
            assert local_fs.ls(target) == []

            # Limit recursive by maxdepth
            fs.get(s, t, recursive=True, maxdepth=1)
            if source_slash:
                assert local_fs.isfile(local_join(target, "subfile1"))
                assert local_fs.isfile(local_join(target, "subfile2"))
                assert not local_fs.exists(local_join(target, "nesteddir"))
                assert not local_fs.exists(local_join(target, "subdir"))

                local_fs.rm(
                    [
                        local_join(target, "subfile1"),
                        local_join(target, "subfile2"),
                    ],
                    recursive=True,
                )
            else:
                assert local_fs.isdir(local_join(target, "subdir"))
                assert local_fs.isfile(local_join(target, "subdir", "subfile1"))
                assert local_fs.isfile(local_join(target, "subdir", "subfile2"))
                assert not local_fs.exists(local_join(target, "subdir", "nesteddir"))

                local_fs.rm(local_join(target, "subdir"), recursive=True)
            assert local_fs.ls(target) == []

    def test_get_directory_to_new_directory(
        self,
        fs,
        fs_join,
        fs_bulk_operations_scenario_0,
        local_fs,
        local_join,
        local_target,
    ):
        # Copy scenario 1f
        source = fs_bulk_operations_scenario_0

        target = local_target
        local_fs.mkdir(target)

        for source_slash, target_slash in zip([False, True], [False, True]):
            s = fs_join(source, "subdir")
            if source_slash:
                s += "/"
            t = local_join(target, "newdir")
            if target_slash:
                t += "/"

            # Without recursive does nothing
            fs.get(s, t)
            assert local_fs.ls(target) == []

            # With recursive
            fs.get(s, t, recursive=True)
            assert local_fs.isdir(local_join(target, "newdir"))
            assert local_fs.isfile(local_join(target, "newdir", "subfile1"))
            assert local_fs.isfile(local_join(target, "newdir", "subfile2"))
            assert local_fs.isdir(local_join(target, "newdir", "nesteddir"))
            assert local_fs.isfile(
                local_join(target, "newdir", "nesteddir", "nestedfile")
            )
            assert not local_fs.exists(local_join(target, "subdir"))

            local_fs.rm(local_join(target, "newdir"), recursive=True)
            assert local_fs.ls(target) == []

            # Limit recursive by maxdepth
            fs.get(s, t, recursive=True, maxdepth=1)
            assert local_fs.isdir(local_join(target, "newdir"))
            assert local_fs.isfile(local_join(target, "newdir", "subfile1"))
            assert local_fs.isfile(local_join(target, "newdir", "subfile2"))
            assert not local_fs.exists(local_join(target, "newdir", "nesteddir"))
            assert not local_fs.exists(local_join(target, "subdir"))

            local_fs.rm(local_join(target, "newdir"), recursive=True)
            assert not local_fs.exists(local_join(target, "newdir"))

    def test_get_glob_to_existing_directory(
        self,
        fs,
        fs_join,
        fs_bulk_operations_scenario_0,
        local_fs,
        local_join,
        local_target,
    ):
        # Copy scenario 1g
        source = fs_bulk_operations_scenario_0

        target = local_target
        local_fs.mkdir(target)

        for target_slash in [False, True]:
            t = target + "/" if target_slash else target

            # Without recursive
            fs.get(fs_join(source, "subdir", "*"), t)
            assert local_fs.isfile(local_join(target, "subfile1"))
            assert local_fs.isfile(local_join(target, "subfile2"))
            assert not local_fs.isdir(local_join(target, "nesteddir"))
            assert not local_fs.exists(local_join(target, "nesteddir", "nestedfile"))
            assert not local_fs.exists(local_join(target, "subdir"))

            local_fs.rm(
                [
                    local_join(target, "subfile1"),
                    local_join(target, "subfile2"),
                ],
                recursive=True,
            )
            assert local_fs.ls(target) == []

            # With recursive
            for glob, recursive in zip(["*", "**"], [True, False]):
                fs.get(fs_join(source, "subdir", glob), t, recursive=recursive)
                assert local_fs.isfile(local_join(target, "subfile1"))
                assert local_fs.isfile(local_join(target, "subfile2"))
                assert local_fs.isdir(local_join(target, "nesteddir"))
                assert local_fs.isfile(local_join(target, "nesteddir", "nestedfile"))
                assert not local_fs.exists(local_join(target, "subdir"))

                local_fs.rm(
                    [
                        local_join(target, "subfile1"),
                        local_join(target, "subfile2"),
                        local_join(target, "nesteddir"),
                    ],
                    recursive=True,
                )
                assert local_fs.ls(target) == []

                # Limit recursive by maxdepth
                fs.get(
                    fs_join(source, "subdir", glob), t, recursive=recursive, maxdepth=1
                )
                assert local_fs.isfile(local_join(target, "subfile1"))
                assert local_fs.isfile(local_join(target, "subfile2"))
                assert not local_fs.exists(local_join(target, "nesteddir"))
                assert not local_fs.exists(local_join(target, "subdir"))

                local_fs.rm(
                    [
                        local_join(target, "subfile1"),
                        local_join(target, "subfile2"),
                    ],
                    recursive=True,
                )
                assert local_fs.ls(target) == []

    def test_get_glob_to_new_directory(
        self,
        fs,
        fs_join,
        fs_bulk_operations_scenario_0,
        local_fs,
        local_join,
        local_target,
    ):
        # Copy scenario 1h
        source = fs_bulk_operations_scenario_0

        target = local_target
        local_fs.mkdir(target)

        for target_slash in [False, True]:
            t = fs_join(target, "newdir")
            if target_slash:
                t += "/"

            # Without recursive
            fs.get(fs_join(source, "subdir", "*"), t)
            assert local_fs.isdir(local_join(target, "newdir"))
            assert local_fs.isfile(local_join(target, "newdir", "subfile1"))
            assert local_fs.isfile(local_join(target, "newdir", "subfile2"))
            assert not local_fs.exists(local_join(target, "newdir", "nesteddir"))
            assert not local_fs.exists(
                local_join(target, "newdir", "nesteddir", "nestedfile")
            )
            assert not local_fs.exists(local_join(target, "subdir"))
            assert not local_fs.exists(local_join(target, "newdir", "subdir"))

            local_fs.rm(local_join(target, "newdir"), recursive=True)
            assert local_fs.ls(target) == []

            # With recursive
            for glob, recursive in zip(["*", "**"], [True, False]):
                fs.get(fs_join(source, "subdir", glob), t, recursive=recursive)
                assert local_fs.isdir(local_join(target, "newdir"))
                assert local_fs.isfile(local_join(target, "newdir", "subfile1"))
                assert local_fs.isfile(local_join(target, "newdir", "subfile2"))
                assert local_fs.isdir(local_join(target, "newdir", "nesteddir"))
                assert local_fs.isfile(
                    local_join(target, "newdir", "nesteddir", "nestedfile")
                )
                assert not local_fs.exists(local_join(target, "subdir"))
                assert not local_fs.exists(local_join(target, "newdir", "subdir"))

                local_fs.rm(local_join(target, "newdir"), recursive=True)
                assert not local_fs.exists(local_join(target, "newdir"))

                # Limit recursive by maxdepth
                fs.get(
                    fs_join(source, "subdir", glob), t, recursive=recursive, maxdepth=1
                )
                assert local_fs.isdir(local_join(target, "newdir"))
                assert local_fs.isfile(local_join(target, "newdir", "subfile1"))
                assert local_fs.isfile(local_join(target, "newdir", "subfile2"))
                assert not local_fs.exists(local_join(target, "newdir", "nesteddir"))
                assert not local_fs.exists(local_join(target, "subdir"))
                assert not local_fs.exists(local_join(target, "newdir", "subdir"))

                local_fs.rm(local_fs.ls(target, detail=False), recursive=True)
                assert not local_fs.exists(local_join(target, "newdir"))

    @pytest.mark.parametrize(
        GLOB_EDGE_CASES_TESTS["argnames"],
        GLOB_EDGE_CASES_TESTS["argvalues"],
    )
    def test_get_glob_edge_cases(
        self,
        path,
        recursive,
        maxdepth,
        expected,
        fs,
        fs_join,
        fs_glob_edge_cases_files,
        local_fs,
        local_join,
        local_target,
    ):
        # Copy scenario 1g
        source = fs_glob_edge_cases_files

        target = local_target

        for new_dir, target_slash in product([True, False], [True, False]):
            local_fs.mkdir(target)

            t = local_join(target, "newdir") if new_dir else target
            t = t + "/" if target_slash else t

            fs.get(fs_join(source, path), t, recursive=recursive, maxdepth=maxdepth)

            output = local_fs.find(target)
            if new_dir:
                prefixed_expected = [
                    make_path_posix(local_join(target, "newdir", p)) for p in expected
                ]
            else:
                prefixed_expected = [
                    make_path_posix(local_join(target, p)) for p in expected
                ]
            assert sorted(output) == sorted(prefixed_expected)

            try:
                local_fs.rm(target, recursive=True)
            except FileNotFoundError:
                pass

    def test_get_list_of_files_to_existing_directory(
        self,
        fs,
        fs_join,
        fs_bulk_operations_scenario_0,
        local_fs,
        local_join,
        local_target,
    ):
        # Copy scenario 2a
        source = fs_bulk_operations_scenario_0

        target = local_target
        local_fs.mkdir(target)

        source_files = [
            fs_join(source, "file1"),
            fs_join(source, "file2"),
            fs_join(source, "subdir", "subfile1"),
        ]

        for target_slash in [False, True]:
            t = target + "/" if target_slash else target

            fs.get(source_files, t)
            assert local_fs.isfile(local_join(target, "file1"))
            assert local_fs.isfile(local_join(target, "file2"))
            assert local_fs.isfile(local_join(target, "subfile1"))

            local_fs.rm(
                [
                    local_join(target, "file1"),
                    local_join(target, "file2"),
                    local_join(target, "subfile1"),
                ],
                recursive=True,
            )
            assert local_fs.ls(target) == []

    def test_get_list_of_files_to_new_directory(
        self,
        fs,
        fs_join,
        fs_bulk_operations_scenario_0,
        local_fs,
        local_join,
        local_target,
    ):
        # Copy scenario 2b
        source = fs_bulk_operations_scenario_0

        target = local_target
        local_fs.mkdir(target)

        source_files = [
            fs_join(source, "file1"),
            fs_join(source, "file2"),
            fs_join(source, "subdir", "subfile1"),
        ]

        fs.get(source_files, local_join(target, "newdir") + "/")  # Note trailing slash
        assert local_fs.isdir(local_join(target, "newdir"))
        assert local_fs.isfile(local_join(target, "newdir", "file1"))
        assert local_fs.isfile(local_join(target, "newdir", "file2"))
        assert local_fs.isfile(local_join(target, "newdir", "subfile1"))

    def test_get_directory_recursive(
        self, fs, fs_join, fs_path, local_fs, local_join, local_target
    ):
        # https://github.com/fsspec/filesystem_spec/issues/1062
        # Recursive cp/get/put of source directory into non-existent target directory.
        src = fs_join(fs_path, "src")
        src_file = fs_join(src, "file")
        fs.mkdir(src)
        fs.touch(src_file)

        target = local_target

        # get without slash
        assert not local_fs.exists(target)
        for loop in range(2):
            fs.get(src, target, recursive=True)
            assert local_fs.isdir(target)

            if loop == 0:
                assert local_fs.isfile(local_join(target, "file"))
                assert not local_fs.exists(local_join(target, "src"))
            else:
                assert local_fs.isfile(local_join(target, "file"))
                assert local_fs.isdir(local_join(target, "src"))
                assert local_fs.isfile(local_join(target, "src", "file"))

        local_fs.rm(target, recursive=True)

        # get with slash
        assert not local_fs.exists(target)
        for loop in range(2):
            fs.get(src + "/", target, recursive=True)
            assert local_fs.isdir(target)
            assert local_fs.isfile(local_join(target, "file"))
            assert not local_fs.exists(local_join(target, "src"))

    def test_get_directory_without_files_with_same_name_prefix(
        self,
        fs,
        fs_join,
        local_fs,
        local_join,
        local_target,
        fs_dir_and_file_with_same_name_prefix,
    ):
        # Create the test dirs
        source = fs_dir_and_file_with_same_name_prefix
        target = local_target

        # Test without glob
        fs.get(fs_join(source, "subdir"), target, recursive=True)

        assert local_fs.isfile(local_join(target, "subfile.txt"))
        assert not local_fs.isfile(local_join(target, "subdir.txt"))

        local_fs.rm([local_join(target, "subfile.txt")])
        assert local_fs.ls(target) == []

        # Test with glob
        fs.get(fs_join(source, "subdir*"), target, recursive=True)

        assert local_fs.isdir(local_join(target, "subdir"))
        assert local_fs.isfile(local_join(target, "subdir", "subfile.txt"))
        assert local_fs.isfile(local_join(target, "subdir.txt"))

    def test_get_with_source_and_destination_as_list(
        self,
        fs,
        fs_join,
        local_fs,
        local_join,
        local_target,
        fs_10_files_with_hashed_names,
    ):
        # Create the test dir
        source = fs_10_files_with_hashed_names
        target = local_target

        # Create list of files for source and destination
        source_files = []
        destination_files = []
        for i in range(10):
            hashed_i = md5(str(i).encode("utf-8")).hexdigest()
            source_files.append(fs_join(source, f"{hashed_i}.txt"))
            destination_files.append(
                make_path_posix(local_join(target, f"{hashed_i}.txt"))
            )

        # Copy and assert order was kept
        fs.get(rpath=source_files, lpath=destination_files)

        for i in range(10):
            file_content = local_fs.cat(destination_files[i]).decode("utf-8")
            assert file_content == str(i)
