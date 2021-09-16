import contextlib
import os
import typing
from typing import Iterator, List, Optional
import unittest
import unittest.mock

import tools.setup_helpers.env as unused_but_resolving_circular_import
import tools.setup_helpers.cmake


class TestCMake(unittest.TestCase):

    def test_build_max_jobs(self) -> None:
        with env_var('MAX_JOBS', '8'):
            build_args = self._cmake_build_and_get_args()

        self.assertListEqual(build_args[-3:], ['--', '-j', '8'])

    def test_build_with_ninja(self) -> None:
        with unittest.mock.patch.object(tools.setup_helpers.cmake, 'USE_NINJA', True):
            build_args = self._cmake_build_and_get_args()

        self.assertNotIn('--', build_args)
        self.assertNotIn('-j', build_args)

    def test_build_no_ninja(self) -> None:
        with unittest.mock.patch.object(tools.setup_helpers.cmake, 'USE_NINJA', False):
            with unittest.mock.patch('multiprocessing.cpu_count') as mock_cpu_count:
                mock_cpu_count.return_value = 13
                build_args = self._cmake_build_and_get_args()

        self.assertListEqual(build_args[-3:], ['--', '-j', '13'])

    def _cmake_build_and_get_args(self) -> List[str]:
        """Runs CMake.build() but then returns the arguments."""
        cmake = tools.setup_helpers.cmake.CMake()

        with unittest.mock.patch.object(cmake, 'run') as cmake_run:
            cmake.build({})

        cmake_run.assert_called_once()
        call, = cmake_run.mock_calls
        build_args, _ = call.args
        assert isinstance(build_args, list), build_args
        assert all(isinstance(x, str) for x in build_args), build_args
        return typing.cast(List[str], build_args)


@contextlib.contextmanager
def env_var(key: str, value: Optional[str]) -> Iterator[None]:
    """Sets/clears an environment variable within a Python context."""
    # Get the previous value and then override it.
    previous_value = os.environ.get(key)
    set_env_var(key, value)
    try:
        yield
    finally:
        # Restore to previous value.
        set_env_var(key, previous_value)


def set_env_var(key: str, value: Optional[str]) -> None:
    """Sets/clears an environment variable."""
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
