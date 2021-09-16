import os
from typing import List, Optional
import unittest
import unittest.mock

import tools.setup_helpers.cmake


class TestCMake(unittest.TestCase):

    def test_build_max_jobs(self) -> None:
        with EnvVar('MAX_JOBS', '8'):
            build_args = self._cmake_build_and_get_args()

        self.assertListEqual(build_args[-3:], ['--', '-j', '8'])

    def test_build_with_ninja(self) -> None:
        build_args = self._cmake_build_and_get_args()

        self.assertNotIn('--', build_args)
        self.assertNotIn('-j', build_args)

    @unittest.mock.patch('tools.setup_helpers.cmake.USE_NINJA', new_callable=lambda: False)
    @unittest.mock.patch('multiprocessing.cpu_count')
    def test_build_no_ninja(self, mock_cpu_count, mock_use_ninja) -> None:
        mock_cpu_count.return_value = 13
        build_args = self._cmake_build_and_get_args()

        self.assertListEqual(build_args[-3:], ['--', '-j', '13'])

    def _cmake_build_and_get_args(self) -> List[str]:
        """Runs CMake.build() but then returns the arguments."""
        cmake = tools.setup_helpers.cmake.CMake()
        cmake.run = unittest.mock.MagicMock(name='run')

        cmake.build({})

        cmake.run.assert_called_once()
        call, = cmake.run.mock_calls
        build_args, _ = call.args
        return build_args


class EnvVar:
    """Sets/clears an environment variable within a Python context."""
    def __init__(self, key: str, value: Optional[str]):
        self.key = key
        self.value = value

    def __enter__(self):
        """Sets the environment variable, remembering the previous value, if any."""
        self.previous_value = os.environ.get(self.key)
        set_env_var(self.key, self.value)

    def __exit__(self, type, value, traceback):
        """Restores the environment to the previous state."""
        set_env_var(self.key, self.previous_value)


def set_env_var(key: str, value: Optional[str]) -> None:
    """Sets/clears an environment variable."""
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
