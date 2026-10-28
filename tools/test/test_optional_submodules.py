from __future__ import annotations

from subprocess import CalledProcessError
from unittest import main, mock, TestCase

from tools import optional_submodules


EIGEN_URL = "https://gitlab.com/libeigen/eigen"


class TestCheckoutByTag(TestCase):
    def _checkout(self, side_effect, **kwargs):
        with (
            mock.patch.object(
                optional_submodules, "check_call", side_effect=side_effect
            ) as check_call,
            mock.patch.object(optional_submodules.time, "sleep") as sleep,
            mock.patch.object(optional_submodules.shutil, "rmtree"),
        ):
            try:
                optional_submodules._checkout_by_tag(EIGEN_URL, "5.0.1", **kwargs)
            finally:
                self.check_call, self.sleep = check_call, sleep

    def test_retries_until_clone_succeeds(self):
        failure = CalledProcessError(128, "git")
        self._checkout([failure, failure, None])
        self.assertEqual(self.check_call.call_count, 3)
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [10, 20])

    def test_raises_after_last_attempt(self):
        failure = CalledProcessError(128, "git")
        with self.assertRaises(CalledProcessError):
            self._checkout([failure] * 3, attempts=3)
        self.assertEqual(self.check_call.call_count, 3)
        self.assertEqual(self.sleep.call_count, 2)

    def test_no_retry_on_success(self):
        self._checkout([None])
        self.assertEqual(self.check_call.call_count, 1)
        self.sleep.assert_not_called()


if __name__ == "__main__":
    main()
