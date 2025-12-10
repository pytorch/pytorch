import pathlib
from typing import List

import pytest

from sympy.testing.runtests_pytest import (
    make_absolute_path,
    sympy_dir,
    update_args_with_paths,
)


class TestMakeAbsolutePath:

    @staticmethod
    @pytest.mark.parametrize(
        'partial_path', ['sympy', 'sympy/core', 'sympy/nonexistant_directory'],
    )
    def test_valid_partial_path(partial_path: str):
        """Paths that start with `sympy` are valid."""
        _ = make_absolute_path(partial_path)

    @staticmethod
    @pytest.mark.parametrize(
        'partial_path', ['not_sympy', 'also/not/sympy'],
    )
    def test_invalid_partial_path_raises_value_error(partial_path: str):
        """A `ValueError` is raises on paths that don't start with `sympy`."""
        with pytest.raises(ValueError):
            _ = make_absolute_path(partial_path)


class TestUpdateArgsWithPaths:

    @staticmethod
    def test_no_paths():
        """If no paths are passed, only `sympy` and `doc/src` are appended.

        `sympy` and `doc/src` are the `testpaths` stated in `pytest.ini`. They
        need to be manually added as if any path-related arguments are passed
        to `pytest.main` then the settings in `pytest.ini` may be ignored.

        """
        paths = []
        args = update_args_with_paths(paths=paths, keywords=None, args=[])
        expected = [
            str(pathlib.Path(sympy_dir(), 'sympy')),
            str(pathlib.Path(sympy_dir(), 'doc/src')),
        ]
        assert args == expected

    @staticmethod
    @pytest.mark.parametrize(
        'path',
        ['sympy/core/tests/test_basic.py', '_basic']
    )
    def test_one_file(path: str):
        """Single files/paths, full or partial, are matched correctly."""
        args = update_args_with_paths(paths=[path], keywords=None, args=[])
        expected = [
            str(pathlib.Path(sympy_dir(), 'sympy/core/tests/test_basic.py')),
        ]
        assert args == expected

    @staticmethod
    def test_partial_path_from_root():
        """Partial paths from the root directly are matched correctly."""
        args = update_args_with_paths(paths=['sympy/functions'], keywords=None, args=[])
        expected = [str(pathlib.Path(sympy_dir(), 'sympy/functions'))]
        assert args == expected

    @staticmethod
    def test_multiple_paths_from_root():
        """Multiple paths, partial or full, are matched correctly."""
        paths = ['sympy/core/tests/test_basic.py', 'sympy/functions']
        args = update_args_with_paths(paths=paths, keywords=None, args=[])
        expected = [
            str(pathlib.Path(sympy_dir(), 'sympy/core/tests/test_basic.py')),
            str(pathlib.Path(sympy_dir(), 'sympy/functions')),
        ]
        assert args == expected

    @staticmethod
    @pytest.mark.parametrize(
        'paths, expected_paths',
        [
            (
                ['/core', '/util'],
                [
                    'doc/src/modules/utilities',
                    'doc/src/reference/public/utilities',
                    'sympy/core',
                    'sympy/logic/utilities',
                    'sympy/utilities',
                ]
            ),
        ]
    )
    def test_multiple_paths_from_non_root(paths: List[str], expected_paths: List[str]):
        """Multiple partial paths are matched correctly."""
        args = update_args_with_paths(paths=paths, keywords=None, args=[])
        assert len(args) == len(expected_paths)
        for arg, expected in zip(sorted(args), expected_paths):
            assert expected in arg

    @staticmethod
    @pytest.mark.parametrize(
        'paths',
        [

            [],
            ['sympy/physics'],
            ['sympy/physics/mechanics'],
            ['sympy/physics/mechanics/tests'],
            ['sympy/physics/mechanics/tests/test_kane3.py'],
        ]
    )
    def test_string_as_keyword(paths: List[str]):
        """String keywords are matched correctly."""
        keywords = ('bicycle', )
        args = update_args_with_paths(paths=paths, keywords=keywords, args=[])
        expected_args = ['sympy/physics/mechanics/tests/test_kane3.py::test_bicycle']
        assert len(args) == len(expected_args)
        for arg, expected in zip(sorted(args), expected_args):
            assert expected in arg

    @staticmethod
    @pytest.mark.parametrize(
        'paths',
        [

            [],
            ['sympy/core'],
            ['sympy/core/tests'],
            ['sympy/core/tests/test_sympify.py'],
        ]
    )
    def test_integer_as_keyword(paths: List[str]):
        """Integer keywords are matched correctly."""
        keywords = ('3538', )
        args = update_args_with_paths(paths=paths, keywords=keywords, args=[])
        expected_args = ['sympy/core/tests/test_sympify.py::test_issue_3538']
        assert len(args) == len(expected_args)
        for arg, expected in zip(sorted(args), expected_args):
            assert expected in arg

    @staticmethod
    def test_multiple_keywords():
        """Multiple keywords are matched correctly."""
        keywords = ('bicycle', '3538')
        args = update_args_with_paths(paths=[], keywords=keywords, args=[])
        expected_args = [
            'sympy/core/tests/test_sympify.py::test_issue_3538',
            'sympy/physics/mechanics/tests/test_kane3.py::test_bicycle',
        ]
        assert len(args) == len(expected_args)
        for arg, expected in zip(sorted(args), expected_args):
            assert expected in arg

    @staticmethod
    def test_keyword_match_in_multiple_files():
        """Keywords are matched across multiple files."""
        keywords = ('1130', )
        args = update_args_with_paths(paths=[], keywords=keywords, args=[])
        expected_args = [
            'sympy/integrals/tests/test_heurisch.py::test_heurisch_symbolic_coeffs_1130',
            'sympy/utilities/tests/test_lambdify.py::test_python_div_zero_issue_11306',
        ]
        assert len(args) == len(expected_args)
        for arg, expected in zip(sorted(args), expected_args):
            assert expected in arg
