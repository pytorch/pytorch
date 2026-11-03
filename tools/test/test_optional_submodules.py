from __future__ import annotations

import tempfile
from pathlib import Path
from subprocess import CalledProcessError, CompletedProcess
from unittest import main, mock, TestCase

from tools import optional_submodules


EIGEN_URL = "https://gitlab.com/libeigen/eigen"
GITLAB_LOAD = (
    "remote: GitLab is currently unable to handle this request due to load\n"
    "fatal: unable to access 'https://gitlab.com/libeigen/eigen/': "
    "The requested URL returned error: 503\n"
)
NO_SUCH_TAG = "fatal: Remote branch 5.0.1 not found in upstream origin\n"
NO_DNS = "fatal: unable to access 'https://gitlab.com/': Could not resolve host\n"


def result(returncode: int = 0, stderr: str = "") -> CompletedProcess:
    return CompletedProcess(["git"], returncode, stdout="", stderr=stderr)


FAILURE = result(128, GITLAB_LOAD)


class TestClone(TestCase):
    def _checkout(self, side_effect, **kwargs):
        with (
            mock.patch.object(
                optional_submodules, "run", side_effect=side_effect
            ) as run,
            mock.patch.object(optional_submodules.time, "sleep") as sleep,
            mock.patch("sys.stderr"),
        ):
            try:
                optional_submodules._checkout_by_tag(EIGEN_URL, "5.0.1", **kwargs)
            finally:
                self.run_, self.sleep = run, sleep

    def test_retries_until_clone_succeeds(self):
        self._checkout([FAILURE, FAILURE, result()])
        self.assertEqual(self.run_.call_count, 3)
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [10, 20])

    def test_survives_seven_failures_with_the_default_attempts(self):
        self._checkout([FAILURE] * 7 + [result()])
        self.assertEqual(self.run_.call_count, 8)
        self.assertEqual(self.sleep.call_count, 7)

    def test_delay_is_capped(self):
        with self.assertRaises(CalledProcessError):
            self._checkout([FAILURE] * 6, attempts=6)
        self.assertEqual(
            [c.args[0] for c in self.sleep.call_args_list], [10, 20, 40, 60, 60]
        )

    def test_raises_after_last_attempt(self):
        with self.assertRaises(CalledProcessError) as raised:
            self._checkout([FAILURE] * 3, attempts=3)
        self.assertEqual(raised.exception.returncode, 128)
        self.assertEqual(raised.exception.stderr, GITLAB_LOAD)
        self.assertEqual(self.run_.call_count, 3)
        self.assertEqual(self.sleep.call_count, 2)

    def test_no_retry_on_success(self):
        self._checkout([result()])
        self.assertEqual(self.run_.call_count, 1)
        self.sleep.assert_not_called()

    def test_permanent_errors_fail_without_retrying(self):
        for stderr in (NO_SUCH_TAG, NO_DNS, ""):
            with self.subTest(stderr=stderr):
                with self.assertRaises(CalledProcessError):
                    self._checkout([result(128, stderr)])
                self.assertEqual(self.run_.call_count, 1)
                self.sleep.assert_not_called()

    def test_clone_command(self):
        self._checkout([result()])
        self.run_.assert_called_once()
        self.assertEqual(
            self.run_.call_args.args[0],
            ["git", "clone", "--depth", "1", "--branch", "5.0.1", EIGEN_URL],
        )
        self.assertEqual(
            self.run_.call_args.kwargs["cwd"], optional_submodules.third_party_path
        )


class TestCheckoutNcclEp(TestCase):
    def test_clone_is_retried_then_checked_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            third_party = Path(tmp)
            (third_party / "nccl_ep").mkdir()
            (third_party / "nccl_ep" / "pin.txt").write_text("abc123\n")
            with (
                mock.patch.object(optional_submodules, "third_party_path", third_party),
                mock.patch.object(
                    optional_submodules, "run", side_effect=[FAILURE, result()]
                ) as run,
                mock.patch.object(optional_submodules, "check_call") as check_call,
                mock.patch.object(optional_submodules.time, "sleep"),
                mock.patch("sys.stderr"),
            ):
                optional_submodules.checkout_nccl_ep()
        self.assertEqual(run.call_count, 2)
        self.assertEqual(
            run.call_args.args[0][:3], ["git", "clone", "--no-checkout"]
        )
        check_call.assert_called_once_with(
            ["git", "checkout", "--detach", "abc123"],
            cwd=third_party / "nccl-extensions",
        )


if __name__ == "__main__":
    main()
