from __future__ import annotations

import io
import tempfile
from pathlib import Path
from subprocess import CalledProcessError, CompletedProcess, PIPE
from unittest import main, mock, TestCase

from tools import optional_submodules


EIGEN_URL = "https://gitlab.com/libeigen/eigen"
CLONE = ["git", "clone", "--depth", "1", "--branch", "5.0.1", EIGEN_URL]
GITLAB_LOAD = (
    "remote: GitLab is currently unable to handle this request due to load\n"
    "fatal: unable to access 'https://gitlab.com/libeigen/eigen/': "
    "The requested URL returned error: 503\n"
)

# One realistic git message per alternative of the retry pattern, plus the
# case-insensitive match.
TRANSIENT = {
    "gitlab load page": "remote: GitLab is currently unable to handle this request due to load",
    "get 503": "fatal: unable to access 'https://x/': The requested URL returned error: 503",
    "get 429": "fatal: unable to access 'https://x/': The requested URL returned error: 429",
    "post 502": "error: RPC failed; HTTP 502 curl 22 The requested URL returned error: 502",
    "post 429": "error: RPC failed; HTTP 429 curl 22 The requested URL returned error: 429",
    "transfer broke": "error: RPC failed; curl 56 OpenSSL SSL_read: Connection reset by peer, errno 104",
    "early eof": "fatal: early EOF",
    "unexpected disconnect": "fatal: unexpected disconnect while reading sideband packet",
    "gnutls": "fatal: unable to access 'https://x/': GnuTLS recv error (-54): Error in the pull function.",
    "reset while reading": "fatal: unable to access 'https://x/': OpenSSL SSL_read: Connection reset by peer, errno 104",
    "syscall while reading": "fatal: unable to access 'https://x/': OpenSSL SSL_read: SSL_ERROR_SYSCALL, errno 104",
    "case": "FATAL: EARLY eof",
}
PERMANENT = {
    "missing tag": "fatal: Remote branch 5.0.1 not found in upstream origin",
    "get 404": "fatal: unable to access 'https://x/': The requested URL returned error: 404",
    "get 403": "fatal: unable to access 'https://x/': The requested URL returned error: 403",
    "post 403": "error: RPC failed; HTTP 403 curl 22 The requested URL returned error: 403",
    "no dns": "fatal: unable to access 'https://gitlab.com/': Could not resolve host: gitlab.com",
    "connect timeout": "fatal: unable to access 'https://gitlab.com/': Failed to connect to gitlab.com port 443 after 130000 ms: Connection timed out",
    "handshake reset": "fatal: unable to access 'https://x/': OpenSSL SSL_connect: Connection reset by peer in connection to gitlab.com:443",
    "handshake syscall": "fatal: unable to access 'https://x/': OpenSSL SSL_connect: SSL_ERROR_SYSCALL in connection to gitlab.com:443",
    "unwritable": "fatal: could not create work tree dir 'eigen': Permission denied",
    "empty": "",
}


def result(returncode: int = 0, stderr: str = "") -> CompletedProcess:
    return CompletedProcess(["git"], returncode, stdout="", stderr=stderr)


FAILURE = result(128, GITLAB_LOAD)


class TestIsTransientCloneError(TestCase):
    def test_transient_messages_are_retried(self):
        for name, stderr in TRANSIENT.items():
            with self.subTest(name):
                self.assertTrue(optional_submodules._is_transient_clone_error(stderr))

    def test_permanent_messages_are_not(self):
        for name, stderr in PERMANENT.items():
            with self.subTest(name):
                self.assertFalse(optional_submodules._is_transient_clone_error(stderr))

    def test_a_connect_time_line_does_not_hide_a_transient_one(self):
        stderr = PERMANENT["handshake reset"] + "\n" + TRANSIENT["early eof"]
        self.assertTrue(optional_submodules._is_transient_clone_error(stderr))


class TestClone(TestCase):
    def _clone(self, side_effect, *, cmd=CLONE, **kwargs):
        with (
            mock.patch.object(
                optional_submodules, "run", side_effect=side_effect
            ) as run,
            mock.patch.object(optional_submodules.time, "sleep") as sleep,
            mock.patch("sys.stderr", new_callable=io.StringIO) as stderr,
        ):
            try:
                optional_submodules._clone(cmd, **kwargs)
            finally:
                self.run_, self.sleep, self.stderr = run, sleep, stderr.getvalue()

    def test_retries_until_clone_succeeds(self):
        self._clone([FAILURE, FAILURE, result()])
        self.assertEqual(self.run_.call_count, 3)
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [10, 20])

    def test_survives_seven_failures_with_the_default_attempts(self):
        self._clone([FAILURE] * 7 + [result()])
        self.assertEqual(self.run_.call_count, 8)
        self.assertEqual(self.sleep.call_count, 7)

    def test_delay_is_capped(self):
        with self.assertRaises(CalledProcessError) as raised:
            self._clone([FAILURE] * 6, attempts=6)
        self.assertEqual(raised.exception.returncode, 128)
        self.assertEqual(raised.exception.stderr, GITLAB_LOAD)
        self.assertEqual(
            [c.args[0] for c in self.sleep.call_args_list], [10, 20, 40, 60, 60]
        )

    def test_raises_after_last_attempt(self):
        with self.assertRaises(CalledProcessError) as raised:
            self._clone([FAILURE] * 3, attempts=3)
        self.assertEqual(raised.exception.returncode, 128)
        self.assertEqual(raised.exception.cmd, CLONE)
        self.assertEqual(raised.exception.stderr, GITLAB_LOAD)
        self.assertEqual(self.run_.call_count, 3)
        self.assertEqual(self.sleep.call_count, 2)

    def test_no_retry_on_success(self):
        self._clone([result()])
        self.assertEqual(self.run_.call_count, 1)
        self.sleep.assert_not_called()

    def test_permanent_errors_fail_without_retrying(self):
        for name, stderr in PERMANENT.items():
            with self.subTest(name):
                with self.assertRaises(CalledProcessError) as raised:
                    self._clone([result(128, stderr)])
                self.assertEqual(raised.exception.returncode, 128)
                self.assertEqual(raised.exception.stderr, stderr)
                self.assertEqual(self.run_.call_count, 1)
                self.sleep.assert_not_called()

    def test_attempts_must_be_positive(self):
        for attempts in (0, -1):
            with self.subTest(attempts):
                with self.assertRaises(ValueError):
                    self._clone([result()], attempts=attempts)
                self.run_.assert_not_called()

    def test_git_runs_with_captured_text_output_in_the_c_locale(self):
        with mock.patch.dict(
            optional_submodules.os.environ, {"PATH": "/bin", "https_proxy": "p:1"}
        ):
            self._clone([result()], cwd=Path("/tmp/x"))
        self.run_.assert_called_once()
        self.assertEqual(self.run_.call_args.args, (CLONE,))
        kwargs = self.run_.call_args.kwargs
        self.assertEqual(kwargs["cwd"], Path("/tmp/x"))
        self.assertEqual(kwargs["stderr"], PIPE)
        self.assertIs(kwargs["text"], True)
        self.assertEqual(kwargs["errors"], "replace")
        # The rest of the environment (PATH, proxy settings) is kept.
        self.assertEqual(kwargs["env"]["LC_ALL"], "C")
        self.assertEqual(kwargs["env"]["PATH"], "/bin")
        self.assertEqual(kwargs["env"]["https_proxy"], "p:1")

    def test_gits_stderr_is_still_printed(self):
        self._clone([FAILURE, FAILURE, result(0, "Cloning into 'eigen'...\n")])
        self.assertEqual(self.stderr, GITLAB_LOAD * 2 + "Cloning into 'eigen'...\n")


class TestCheckoutByTag(TestCase):
    def test_clones_the_tag_into_third_party(self):
        with mock.patch.object(optional_submodules, "_clone") as clone:
            optional_submodules._checkout_by_tag(EIGEN_URL, "5.0.1")
        clone.assert_called_once_with(CLONE, cwd=optional_submodules.third_party_path)


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
                mock.patch("sys.stderr", new_callable=io.StringIO),
            ):
                optional_submodules.checkout_nccl_ep()
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args.args[0][:3], ["git", "clone", "--no-checkout"])
        check_call.assert_called_once_with(
            ["git", "checkout", "--detach", "abc123"],
            cwd=third_party / "nccl-extensions",
        )


if __name__ == "__main__":
    main()
