# Owner(s): ["module: hub"]

import contextlib
import errno
import io
import linecache
import os
import tempfile
import threading
import time
import unittest
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from filelock import Timeout

import torch
import torch.hub as hub
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_SANDCASTLE,
    IS_WINDOWS,
    parametrize,
    retry,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)
from torch.utils._filelock import FileLock


def sum_of_state_dict(state_dict):
    s = 0
    for v in state_dict.values():
        s += v.sum()
    return s


SUM_OF_HUB_EXAMPLE = 431080
TORCHHUB_EXAMPLE_RELEASE_URL = (
    "https://github.com/ailzhang/torchhub_example/releases/download/0.1/mnist_init_ones"
)


@unittest.skipIf(IS_SANDCASTLE, "Sandcastle cannot ping external")
class TestHub(TestCase):
    def setUp(self):
        super().setUp()
        self.previous_hub_dir = torch.hub.get_dir()
        self.tmpdir = tempfile.TemporaryDirectory("hub_dir")
        torch.hub.set_dir(self.tmpdir.name)
        self.trusted_list_path = os.path.join(torch.hub.get_dir(), "trusted_list")

    def tearDown(self):
        super().tearDown()
        torch.hub.set_dir(self.previous_hub_dir)  # probably not needed, but can't hurt
        self.tmpdir.cleanup()

    def _assert_trusted_list_is_empty(self):
        with open(self.trusted_list_path) as f:
            lines = f.readlines()
            if lines:
                raise AssertionError(f"Expected empty trusted list, got {lines}")

    def _assert_in_trusted_list(self, line):
        with open(self.trusted_list_path) as f:
            if line not in (l.strip() for l in f):
                raise AssertionError(f"Expected {line} in trusted list")

    @retry(Exception, tries=3)
    def test_load_from_github(self):
        hub_model = hub.load(
            "ailzhang/torchhub_example",
            "mnist",
            source="github",
            pretrained=True,
            verbose=False,
        )
        self.assertEqual(sum_of_state_dict(hub_model.state_dict()), SUM_OF_HUB_EXAMPLE)

    @retry(Exception, tries=3)
    def test_load_from_local_dir(self):
        local_dir = hub._get_cache_or_reload(
            "ailzhang/torchhub_example",
            force_reload=False,
            trust_repo=True,
        )
        hub_model = hub.load(
            local_dir, "mnist", source="local", pretrained=True, verbose=False
        )
        self.assertEqual(sum_of_state_dict(hub_model.state_dict()), SUM_OF_HUB_EXAMPLE)

    @retry(Exception, tries=3)
    def test_load_from_branch(self):
        hub_model = hub.load(
            "ailzhang/torchhub_example:ci/test_slash",
            "mnist",
            pretrained=True,
            verbose=False,
        )
        self.assertEqual(sum_of_state_dict(hub_model.state_dict()), SUM_OF_HUB_EXAMPLE)

    @retry(Exception, tries=3)
    def test_get_set_dir(self):
        previous_hub_dir = torch.hub.get_dir()
        with tempfile.TemporaryDirectory("hub_dir") as tmpdir:
            torch.hub.set_dir(tmpdir)
            self.assertEqual(torch.hub.get_dir(), tmpdir)
            self.assertNotEqual(previous_hub_dir, tmpdir)

            hub_model = hub.load(
                "ailzhang/torchhub_example", "mnist", pretrained=True, verbose=False
            )
            self.assertEqual(
                sum_of_state_dict(hub_model.state_dict()), SUM_OF_HUB_EXAMPLE
            )
            if not os.path.exists(
                os.path.join(tmpdir, "ailzhang_torchhub_example_master")
            ):
                raise AssertionError(
                    "Expected torchhub_example_master directory to exist"
                )

        # Test that set_dir properly calls expanduser()
        # non-regression test for https://github.com/pytorch/pytorch/issues/69761
        new_dir = os.path.join("~", "hub")
        torch.hub.set_dir(new_dir)
        self.assertEqual(torch.hub.get_dir(), os.path.expanduser(new_dir))

    @retry(Exception, tries=3)
    def test_list_entrypoints(self):
        entry_lists = hub.list("ailzhang/torchhub_example", trust_repo=True)
        self.assertObjectIn("mnist", entry_lists)

    @retry(Exception, tries=3)
    def test_download_url_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = os.path.join(tmpdir, "temp")
            hub.download_url_to_file(TORCHHUB_EXAMPLE_RELEASE_URL, f, progress=False)
            loaded_state = torch.load(f)
            self.assertEqual(sum_of_state_dict(loaded_state), SUM_OF_HUB_EXAMPLE)
            # Check that the downloaded file has default file permissions
            f_ref = os.path.join(tmpdir, "reference")
            open(f_ref, "w").close()
            expected_permissions = oct(os.stat(f_ref).st_mode & 0o777)
            actual_permissions = oct(os.stat(f).st_mode & 0o777)
            if actual_permissions != expected_permissions:
                raise AssertionError(
                    f"Expected permissions {expected_permissions}, got {actual_permissions}"
                )

    @retry(Exception, tries=3)
    def test_load_state_dict_from_url(self):
        loaded_state = hub.load_state_dict_from_url(TORCHHUB_EXAMPLE_RELEASE_URL)
        self.assertEqual(sum_of_state_dict(loaded_state), SUM_OF_HUB_EXAMPLE)

        # with name
        file_name = "the_file_name"
        loaded_state = hub.load_state_dict_from_url(
            TORCHHUB_EXAMPLE_RELEASE_URL, file_name=file_name
        )
        expected_file_path = os.path.join(torch.hub.get_dir(), "checkpoints", file_name)
        self.assertTrue(os.path.exists(expected_file_path))
        self.assertEqual(sum_of_state_dict(loaded_state), SUM_OF_HUB_EXAMPLE)

        # with safe weight_only
        loaded_state = hub.load_state_dict_from_url(
            TORCHHUB_EXAMPLE_RELEASE_URL, weights_only=True
        )
        self.assertEqual(sum_of_state_dict(loaded_state), SUM_OF_HUB_EXAMPLE)

    @retry(Exception, tries=3)
    def test_load_legacy_zip_checkpoint(self):
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            hub_model = hub.load(
                "ailzhang/torchhub_example", "mnist_zip", pretrained=True, verbose=False
            )
            self.assertEqual(
                sum_of_state_dict(hub_model.state_dict()), SUM_OF_HUB_EXAMPLE
            )
            if not any(
                "will be deprecated in favor of default zipfile" in str(w) for w in ws
            ):
                raise AssertionError("Expected deprecation warning for legacy zip")

    # Test the default zipfile serialization format produced by >=1.6 release.
    @retry(Exception, tries=3)
    def test_load_zip_1_6_checkpoint(self):
        hub_model = hub.load(
            "ailzhang/torchhub_example",
            "mnist_zip_1_6",
            pretrained=True,
            verbose=False,
            trust_repo=True,
        )
        self.assertEqual(sum_of_state_dict(hub_model.state_dict()), SUM_OF_HUB_EXAMPLE)

    @retry(Exception, tries=3)
    def test_hub_parse_repo_info(self):
        # If the branch is specified we just parse the input and return
        self.assertEqual(torch.hub._parse_repo_info("a/b:c"), ("a", "b", "c"))
        # For torchvision, the default branch is main
        self.assertEqual(
            torch.hub._parse_repo_info("pytorch/vision"), ("pytorch", "vision", "main")
        )
        # For the torchhub_example repo, the default branch is still master
        self.assertEqual(
            torch.hub._parse_repo_info("ailzhang/torchhub_example"),
            ("ailzhang", "torchhub_example", "master"),
        )

    @retry(Exception, tries=3)
    def test_load_commit_from_forked_repo(self):
        with self.assertRaisesRegex(ValueError, "If it's a commit from a forked repo"):
            torch.hub.load("pytorch/vision:4e2c216", "resnet18")

    @retry(Exception, tries=3)
    @patch("builtins.input", return_value="")
    def test_trust_repo_false_emptystring(self, patched_input):
        with self.assertRaisesRegex(Exception, "Untrusted repository."):
            torch.hub.load(
                "ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo=False
            )
        self._assert_trusted_list_is_empty()
        patched_input.assert_called_once()

        patched_input.reset_mock()
        with self.assertRaisesRegex(Exception, "Untrusted repository."):
            torch.hub.load(
                "ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo=False
            )
        self._assert_trusted_list_is_empty()
        patched_input.assert_called_once()

    @retry(Exception, tries=3)
    @patch("builtins.input", return_value="no")
    def test_trust_repo_false_no(self, patched_input):
        with self.assertRaisesRegex(Exception, "Untrusted repository."):
            torch.hub.load(
                "ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo=False
            )
        self._assert_trusted_list_is_empty()
        patched_input.assert_called_once()

        patched_input.reset_mock()
        with self.assertRaisesRegex(Exception, "Untrusted repository."):
            torch.hub.load(
                "ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo=False
            )
        self._assert_trusted_list_is_empty()
        patched_input.assert_called_once()

    @skipIfTorchDynamo(msg="https://github.com/pytorch/pytorch/issues/159627")
    @retry(Exception, tries=3)
    @patch("builtins.input", return_value="y")
    def test_trusted_repo_false_yes(self, patched_input):
        torch.hub.load("ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo=False)
        self._assert_in_trusted_list("ailzhang_torchhub_example")
        patched_input.assert_called_once()

        # Loading a second time with "check", we don't ask for user input
        patched_input.reset_mock()
        torch.hub.load("ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo="check")
        patched_input.assert_not_called()

        # Loading again with False, we still ask for user input
        patched_input.reset_mock()
        torch.hub.load("ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo=False)
        patched_input.assert_called_once()

    @retry(Exception, tries=3)
    @patch("builtins.input", return_value="no")
    def test_trust_repo_check_no(self, patched_input):
        with self.assertRaisesRegex(Exception, "Untrusted repository."):
            torch.hub.load(
                "ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo="check"
            )
        self._assert_trusted_list_is_empty()
        patched_input.assert_called_once()

        patched_input.reset_mock()
        with self.assertRaisesRegex(Exception, "Untrusted repository."):
            torch.hub.load(
                "ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo="check"
            )
        patched_input.assert_called_once()

    @skipIfTorchDynamo(msg="https://github.com/pytorch/pytorch/issues/155617")
    @retry(Exception, tries=3)
    @patch("builtins.input", return_value="y")
    def test_trust_repo_check_yes(self, patched_input):
        torch.hub.load("ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo="check")
        self._assert_in_trusted_list("ailzhang_torchhub_example")
        patched_input.assert_called_once()

        # Loading a second time with "check", we don't ask for user input
        patched_input.reset_mock()
        torch.hub.load("ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo="check")
        patched_input.assert_not_called()

    @retry(Exception, tries=3)
    def test_trust_repo_true(self):
        torch.hub.load("ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo=True)
        self._assert_in_trusted_list("ailzhang_torchhub_example")

    @retry(Exception, tries=3)
    def test_trust_repo_builtin_trusted_owners(self):
        torch.hub.load("pytorch/vision", "resnet18", trust_repo="check")
        self._assert_trusted_list_is_empty()

    @retry(Exception, tries=3)
    def test_trust_repo_legacy(self):
        # We first download a repo and then delete the allowlist file
        # Then we check that the repo is indeed trusted without a prompt,
        # because it was already downloaded in the past.
        torch.hub.load("ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo=True)
        os.remove(self.trusted_list_path)

        torch.hub.load("ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo="check")

        self._assert_trusted_list_is_empty()

    def _create_malicious_zip(self, filename, malicious_path):
        """Helper to create a zip file with a malicious path entry."""
        with zipfile.ZipFile(filename, "w") as zf:
            # Create an entry with malicious path
            zf.writestr(malicious_path, b"malicious content")

    def _create_legacy_malicious_zip(self, filename, malicious_path):
        """Helper to create a legacy-style zip file (single entry) with malicious path."""
        with zipfile.ZipFile(filename, "w") as zf:
            # Create a single entry with malicious path for legacy format
            zf.writestr(malicious_path, b"malicious content")

    def test_safe_extract_zip_blocks_directory_traversal(self):
        """Test that _safe_extract_zip blocks directory traversal attacks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            malicious_zip = os.path.join(tmpdir, "malicious.zip")
            extract_dir = os.path.join(tmpdir, "extract")
            os.makedirs(extract_dir)

            # Test various directory traversal patterns
            malicious_paths = [
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32\\config\\sam",
                "../../tmp/malicious_script.py",
                "../legitimate_looking_file.txt",
                "subdir/../../../etc/shadow",
            ]

            for malicious_path in malicious_paths:
                with self.subTest(path=malicious_path):
                    self._create_malicious_zip(malicious_zip, malicious_path)

                    with zipfile.ZipFile(malicious_zip) as zf:
                        with self.assertRaises(ValueError) as cm:
                            hub._safe_extract_zip(zf, extract_dir)

                        # Verify the error message indicates directory traversal
                        self.assertIn("directory traversal", str(cm.exception))

                    os.remove(malicious_zip)

    def test_safe_extract_zip_blocks_absolute_paths(self):
        """Test that _safe_extract_zip blocks absolute path attacks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            malicious_zip = os.path.join(tmpdir, "malicious.zip")
            extract_dir = os.path.join(tmpdir, "extract")
            os.makedirs(extract_dir)

            # Test absolute paths (both Unix and Windows style)
            malicious_paths = [
                "/etc/passwd",
                "/tmp/malicious_file",
                "C:\\Windows\\System32\\config\\sam",
                "\\etc\\passwd",
                "/home/user/.ssh/authorized_keys",
            ]

            for malicious_path in malicious_paths:
                with self.subTest(path=malicious_path):
                    self._create_malicious_zip(malicious_zip, malicious_path)

                    with zipfile.ZipFile(malicious_zip) as zf:
                        with self.assertRaises(ValueError) as cm:
                            hub._safe_extract_zip(zf, extract_dir)

                        # Verify the error message indicates absolute path
                        self.assertIn("absolute path", str(cm.exception))

                    os.remove(malicious_zip)

    def test_legacy_zip_load_zipslip_protection(self):
        """Test that _legacy_zip_load is protected against zipslip attacks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            malicious_zip = os.path.join(tmpdir, "malicious_legacy.zip")
            model_dir = os.path.join(tmpdir, "models")
            os.makedirs(model_dir)

            # Create a legacy-style malicious zip (single entry with traversal)
            malicious_path = "../../../etc/passwd"
            self._create_legacy_malicious_zip(malicious_zip, malicious_path)

            # Test that _legacy_zip_load rejects the malicious zip
            with self.assertRaises(ValueError) as cm:
                hub._legacy_zip_load(malicious_zip, model_dir, None, False)

            self.assertIn("directory traversal", str(cm.exception))

    def test_get_cache_or_reload_zipslip_protection(self):
        """Test that repository zip extraction is protected against zipslip attacks."""
        # This is more of an integration test that would require creating a malicious
        # GitHub repository archive, which is complex. Instead, we'll test the
        # _safe_extract_zip function directly in the context it would be used.

        with tempfile.TemporaryDirectory() as tmpdir:
            # Simulate what _get_cache_or_reload does
            cached_file = os.path.join(tmpdir, "test_repo.zip")
            hub_dir = os.path.join(tmpdir, "hub")
            os.makedirs(hub_dir)

            # Create a zip that looks like a GitHub repository but has malicious content
            malicious_entries = [
                "legitimate_repo_name-master/hubconf.py",  # Legitimate entry
                "../../../etc/passwd",  # Malicious entry
            ]

            with zipfile.ZipFile(cached_file, "w") as zf:
                zf.writestr(malicious_entries[0], "# hubconf content")
                zf.writestr(malicious_entries[1], "malicious content")

            # Test that safe extraction rejects this
            with zipfile.ZipFile(cached_file) as cached_zipfile:
                with self.assertRaises(ValueError) as cm:
                    hub._safe_extract_zip(cached_zipfile, hub_dir)

                self.assertIn("directory traversal", str(cm.exception))

    def _run_concurrently(self, fns):
        # Slow down the download so that, without locking, every worker would
        # see an empty cache and start its own download. With the short lock
        # timeout, a lock that is never released makes the workers fall back
        # within seconds, which _assert_lock_used catches, instead of hanging.
        real_download = hub.download_url_to_file
        count_lock = threading.Lock()
        num_downloads = 0

        def slow_download(*args, **kwargs):
            nonlocal num_downloads
            with count_lock:
                num_downloads += 1
            time.sleep(0.5)
            return real_download(*args, **kwargs)

        barrier = threading.Barrier(len(fns))

        def worker(fn):
            barrier.wait()
            return fn()

        with (
            self._assert_lock_used(),
            patch.object(hub, "_CACHE_LOCK_TIMEOUT_S", 5),
            patch.object(hub, "download_url_to_file", side_effect=slow_download),
            ThreadPoolExecutor(len(fns)) as pool,
        ):
            results = list(pool.map(worker, fns))
        return results, num_downloads

    @contextlib.contextmanager
    def _assert_lock_used(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            yield
        fallbacks = [w for w in caught if ".lock" in str(w.message)]
        self.assertEqual(fallbacks, [])

    def _save_weights(self):
        src = os.path.join(self.tmpdir.name, "weights.pt")
        torch.save({"w": torch.arange(10)}, src)
        return Path(src).as_uri(), os.path.join(self.tmpdir.name, "checkpoints")

    def _make_hub_archive(self, repo, entries=None, top_dir=None):
        archive_dir = os.path.join(self.tmpdir.name, "archives")
        os.makedirs(archive_dir, exist_ok=True)
        archive = os.path.join(archive_dir, repo + ".zip")
        # Mimic a GitHub archive: a single top-level directory.
        top_dir = top_dir or f"{repo}-main"
        if entries is None:
            entries = {f"{top_dir}/hubconf.py": f"name = {repo!r}\n"}
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr(f"{top_dir}/", "")
            for name, content in entries.items():
                zf.writestr(name, content)
        return Path(archive).as_uri()

    def _load_repo(self, repo, force_reload=False, skip_validation=True, verbose=False):
        return hub._get_cache_or_reload(
            f"owner/{repo}:main",
            force_reload=force_reload,
            trust_repo=True,
            verbose=verbose,
            skip_validation=skip_validation,
        )

    def _assert_no_leftovers(self):
        names = os.listdir(hub.get_dir())
        leftovers = [
            n
            for n in names
            if not n.endswith(".lock") and (n.endswith(".zip") or "~" in n)
        ]
        self.assertEqual(leftovers, [])

    def test_load_state_dict_from_url_concurrent(self):
        url, model_dir = self._save_weights()

        def load():
            return hub.load_state_dict_from_url(
                url, model_dir=model_dir, progress=False
            )

        results, num_downloads = self._run_concurrently([load] * 4)
        self.assertEqual(num_downloads, 1)
        for state_dict in results:
            self.assertEqual(state_dict["w"], torch.arange(10))

    def test_get_cache_or_reload_concurrent(self):
        repos = ["repo", "other"]
        urls = {repo: self._make_hub_archive(repo) for repo in repos}
        loads = [lambda r=repo: self._load_repo(r) for repo in repos for _ in range(3)]
        with patch.object(
            hub, "_git_archive_link", side_effect=lambda owner, name, ref: urls[name]
        ):
            results, num_downloads = self._run_concurrently(loads)

        # One download per repo, although both repos are on the same branch.
        self.assertEqual(num_downloads, len(repos))
        hub_dir = hub.get_dir()
        expected = [
            os.path.join(hub_dir, f"owner_{r}_main") for r in repos for _ in range(3)
        ]
        self.assertEqual(results, expected)
        for repo in repos:
            repo_dir = os.path.join(hub_dir, f"owner_{repo}_main")
            self.assertEqual(os.listdir(repo_dir), ["hubconf.py"])
            with open(os.path.join(repo_dir, "hubconf.py")) as f:
                self.assertEqual(f.read(), f"name = {repo!r}\n")
        self._assert_no_leftovers()

    def test_get_cache_or_reload_force_reload(self):
        archives = [
            self._make_hub_archive("repo"),
            self._make_hub_archive(
                "v2", {"repo-main/hubconf.py": "name = 'v2'\n"}, "repo-main"
            ),
        ]
        real_download = hub.download_url_to_file
        with (
            patch.object(
                hub, "_git_archive_link", side_effect=lambda *args: archives[0]
            ),
            patch.object(
                hub, "download_url_to_file", side_effect=real_download
            ) as mock_download,
        ):
            repo_dir = self._load_repo("repo")
            self._load_repo("repo")
            self.assertEqual(mock_download.call_count, 1)
            archives.pop(0)
            self.assertEqual(self._load_repo("repo", force_reload=True), repo_dir)
            self.assertEqual(mock_download.call_count, 2)
        self.assertEqual(os.listdir(repo_dir), ["hubconf.py"])
        with open(os.path.join(repo_dir, "hubconf.py")) as f:
            self.assertEqual(f.read(), "name = 'v2'\n")
        self._assert_no_leftovers()

    def test_get_cache_or_reload_failed_extraction_cleans_up(self):
        url = self._make_hub_archive(
            "repo", {"repo-main/hubconf.py": "", "../../../evil.py": ""}
        )
        with (
            patch.object(hub, "_git_archive_link", return_value=url),
            self.assertRaisesRegex(ValueError, "directory traversal"),
        ):
            self._load_repo("repo")
        self.assertFalse(os.path.exists(os.path.join(hub.get_dir(), "owner_repo_main")))
        self._assert_no_leftovers()

    def test_load_state_dict_from_url_failed_download_releases_lock(self):
        url, model_dir = self._save_weights()
        real_download = hub.download_url_to_file
        waiting = threading.Event()
        calls = []

        class StderrRecorder(io.StringIO):
            def write(self, s):
                if "Waiting for another process" in s:
                    waiting.set()
                return super().write(s)

        def flaky_download(*args, **kwargs):
            calls.append(None)
            if len(calls) > 1:
                return real_download(*args, **kwargs)
            # Fail only once the other worker is blocked on the lock.
            if not waiting.wait(timeout=60):
                raise AssertionError("the other worker never waited for the lock")
            raise RuntimeError("download failed")

        def load(_):
            try:
                # progress=True so that the waiting worker reports it.
                return hub.load_state_dict_from_url(
                    url, model_dir=model_dir, progress=True
                )
            except RuntimeError as e:
                return e

        with (
            self._assert_lock_used(),
            patch.object(hub, "_CACHE_LOCK_TIMEOUT_S", 5),
            patch("sys.stderr", new=StderrRecorder()),
            patch.object(hub, "download_url_to_file", side_effect=flaky_download),
            ThreadPoolExecutor(2) as pool,
        ):
            outcomes = list(pool.map(load, range(2)))

        # The worker whose download failed releases the lock, and the one that
        # was waiting for it downloads the file itself.
        self.assertEqual(len(calls), 2)
        errors = [o for o in outcomes if isinstance(o, RuntimeError)]
        state_dicts = [o for o in outcomes if not isinstance(o, RuntimeError)]
        self.assertEqual(len(errors), 1)
        self.assertEqual(len(state_dicts), 1)
        self.assertEqual(state_dicts[0]["w"], torch.arange(10))

    def test_load_state_dict_from_url_without_lock(self):
        url, model_dir = self._save_weights()
        with (
            patch.object(FileLock, "__enter__", side_effect=NotImplementedError),
            self.assertWarnsRegex(UserWarning, "Could not lock"),
        ):
            state_dict = hub.load_state_dict_from_url(
                url, model_dir=model_dir, progress=False
            )
        self.assertEqual(state_dict["w"], torch.arange(10))

    def test_get_cache_or_reload_concurrent_validates_once(self):
        url = self._make_hub_archive("repo")
        with (
            patch.object(hub, "_git_archive_link", return_value=url),
            patch.object(hub, "_validate_not_a_forked_repo") as mock_validate,
        ):
            _, num_downloads = self._run_concurrently(
                [lambda: self._load_repo("repo", skip_validation=False)] * 4
            )
        # Validation runs under the lock, so workers that find the repo already
        # extracted skip its GitHub API calls.
        self.assertEqual(num_downloads, 1)
        self.assertEqual(mock_validate.call_count, 1)

    def test_get_cache_or_reload_two_refs_same_commit(self):
        # Two refs at the same commit unpack to the same top-level directory.
        urls = {
            ref: self._make_hub_archive(
                f"repo_{ref}",
                {"repo-abc123/hubconf.py": f"ref = {ref!r}\n"},
                "repo-abc123",
            )
            for ref in ("main", "v1")
        }

        def load(ref):
            return hub._get_cache_or_reload(
                f"owner/repo:{ref}",
                force_reload=False,
                trust_repo=True,
                verbose=False,
                skip_validation=True,
            )

        # Where the old code extracted both archives; it must stay untouched.
        sentinel = os.path.join(hub.get_dir(), "repo-abc123")
        os.makedirs(sentinel)
        with patch.object(
            hub, "_git_archive_link", side_effect=lambda owner, name, ref: urls[ref]
        ):
            results, num_downloads = self._run_concurrently(
                [lambda: load("main"), lambda: load("v1")]
            )
        self.assertEqual(num_downloads, 2)
        for ref, repo_dir in zip(("main", "v1"), results):
            with open(os.path.join(repo_dir, "hubconf.py")) as f:
                self.assertEqual(f.read(), f"ref = {ref!r}\n")
        self.assertEqual(os.listdir(sentinel), [])

    def test_get_cache_or_reload_unlocked_keeps_published_repo(self):
        url = self._make_hub_archive("repo")
        repo_dir = os.path.join(hub.get_dir(), "owner_repo_main")
        real_download = hub.download_url_to_file

        def download_while_another_process_publishes(*args, **kwargs):
            os.makedirs(repo_dir)
            with open(os.path.join(repo_dir, "hubconf.py"), "w") as f:
                f.write("name = 'other'\n")
            return real_download(*args, **kwargs)

        error = OSError(errno.ENOLCK, "No locks available")
        with (
            patch.object(hub, "_git_archive_link", return_value=url),
            patch.object(FileLock, "__enter__", side_effect=error),
            patch.object(
                hub,
                "download_url_to_file",
                side_effect=download_while_another_process_publishes,
            ),
            self.assertWarnsRegex(UserWarning, "No locks available"),
        ):
            self.assertEqual(self._load_repo("repo"), repo_dir)
        # The other process's copy is neither replaced nor nested into.
        self.assertEqual(os.listdir(repo_dir), ["hubconf.py"])
        with open(os.path.join(repo_dir, "hubconf.py")) as f:
            self.assertEqual(f.read(), "name = 'other'\n")
        self._assert_no_leftovers()

    def test_get_cache_or_reload_removes_stale_work_dir(self):
        hub_dir = hub.get_dir()
        stale = os.path.join(hub_dir, "owner_repo_main~killed01")
        os.makedirs(os.path.join(stale, "repo-main"))
        old = time.time() - 2 * hub._CACHE_LOCK_TIMEOUT_S
        for path in (stale, os.path.join(stale, "repo-main")):
            os.utime(path, (old, old))
        # A download in progress, possibly without the lock: its directory is
        # old, but the file being downloaded into it was just written.
        live = os.path.join(hub_dir, "owner_repo_main~running")
        os.makedirs(live)
        open(os.path.join(live, "archive.zip.partial"), "w").close()
        os.utime(live, (old, old))
        # The cached repo of ref "main.1".
        other_ref = os.path.join(hub_dir, "owner_repo_main.1")
        os.makedirs(other_ref)
        url = self._make_hub_archive("repo")
        with patch.object(hub, "_git_archive_link", return_value=url):
            self._load_repo("repo")
        self.assertFalse(os.path.exists(stale))
        self.assertTrue(os.path.isdir(live))
        self.assertTrue(os.path.isdir(other_ref))

    @parametrize("progress", [True, False])
    def test_load_state_dict_from_url_lock_timeout(self, progress):
        url, model_dir = self._save_weights()
        os.makedirs(model_dir)
        stderr = io.StringIO()
        # The pool is outermost, so the lock is released before it joins the worker.
        with (
            ThreadPoolExecutor(1) as pool,
            FileLock(os.path.join(model_dir, "weights.pt.lock")),
            patch.object(hub, "_CACHE_LOCK_TIMEOUT_S", 0.2),
            patch("sys.stderr", new=stderr),
            self.assertWarnsRegex(UserWarning, "for more than 0.2 s"),
        ):
            # Another thread, so that the held lock looks like another process's.
            state_dict = pool.submit(
                hub.load_state_dict_from_url,
                url,
                model_dir=model_dir,
                progress=progress,
            ).result(timeout=60)
        self.assertEqual(state_dict["w"], torch.arange(10))
        self.assertEqual("Waiting for another process" in stderr.getvalue(), progress)

    def test_get_cache_or_reload_lock_timeout(self):
        url = self._make_hub_archive("repo")
        repo_dir = os.path.join(hub.get_dir(), "owner_repo_main")
        os.makedirs(hub.get_dir(), exist_ok=True)
        stderr = io.StringIO()
        with (
            ThreadPoolExecutor(1) as pool,
            FileLock(repo_dir + ".lock"),
            patch.object(hub, "_CACHE_LOCK_TIMEOUT_S", 0.2),
            patch.object(hub, "_git_archive_link", return_value=url),
            patch("sys.stderr", new=stderr),
            self.assertWarnsRegex(UserWarning, "for more than 0.2 s"),
        ):
            load = pool.submit(self._load_repo, "repo", verbose=True)
            self.assertEqual(load.result(timeout=60), repo_dir)
        self.assertIn("Waiting for another process", stderr.getvalue())
        self.assertEqual(os.listdir(repo_dir), ["hubconf.py"])
        self._assert_no_leftovers()

    def test_get_cache_or_reload_failed_publish_raises(self):
        url = self._make_hub_archive("repo")
        repo_dir = os.path.join(hub.get_dir(), "owner_repo_main")
        real_rename = os.rename

        def rename(src, dst, **kwargs):
            if dst == repo_dir:
                raise PermissionError(errno.EACCES, "Access is denied", dst)
            return real_rename(src, dst, **kwargs)

        with (
            patch.object(hub, "_git_archive_link", return_value=url),
            patch("os.rename", side_effect=rename),
            self.assertRaises(PermissionError),
        ):
            self._load_repo("repo")
        self.assertFalse(os.path.exists(repo_dir))
        self._assert_no_leftovers()

    def test_load_state_dict_from_url_unlockable_falls_back_at_once(self):
        url, model_dir = self._save_weights()
        error = OSError(errno.ENOLCK, "No locks available")
        # filelock 3.x reports ENOLCK as a held lock; the probe must not wait.
        busy = Timeout(os.path.join(model_dir, "weights.pt.lock"))
        with (
            patch.object(hub, "_lock_unavailable", return_value=error),
            patch.object(FileLock, "__enter__", side_effect=[busy, AssertionError]),
            self.assertWarnsRegex(UserWarning, "No locks available"),
        ):
            state_dict = hub.load_state_dict_from_url(
                url, model_dir=model_dir, progress=False
            )
        self.assertEqual(state_dict["w"], torch.arange(10))

    @unittest.skipIf(IS_WINDOWS, "flock probe is Unix only")
    def test_lock_unavailable_tells_unlockable_from_busy(self):
        lock_path = os.path.join(self.tmpdir.name, "entry.lock")
        enolck = OSError(errno.ENOLCK, "No locks available")
        with patch("fcntl.flock", side_effect=enolck):
            self.assertIs(hub._lock_unavailable(lock_path), enolck)
        with patch("fcntl.flock", side_effect=BlockingIOError(errno.EWOULDBLOCK, "")):
            self.assertIsNone(hub._lock_unavailable(lock_path))
        self.assertIsNone(hub._lock_unavailable(lock_path))

    @unittest.skipIf(IS_WINDOWS, "flock probe is Unix only")
    def test_lock_unavailable_reports_unwritable_lock_file(self):
        if os.geteuid() == 0:
            self.skipTest("root can open any lock file")
        lock_path = os.path.join(self.tmpdir.name, "entry.lock")
        # A lock file created by another user without group write access.
        open(lock_path, "w").close()
        os.chmod(lock_path, 0o444)
        error = hub._lock_unavailable(lock_path)
        self.assertIsInstance(error, PermissionError)

    @unittest.skipIf(IS_WINDOWS, "flock probe is Unix only")
    def test_lock_unavailable_does_not_follow_symlinks(self):
        target = os.path.join(self.tmpdir.name, "victim")
        lock_path = os.path.join(self.tmpdir.name, "entry.lock")
        os.symlink(target, lock_path)
        self.assertEqual(hub._lock_unavailable(lock_path).errno, errno.ELOOP)
        self.assertFalse(os.path.exists(target))

    @unittest.skipIf(not IS_WINDOWS, "Windows probe")
    def test_lock_unavailable_reports_unwritable_dir(self):
        lock_path = os.path.join(self.tmpdir.name, "entry.lock")
        denied = PermissionError(errno.EACCES, "Access is denied")
        with patch("tempfile.mkstemp", side_effect=denied):
            self.assertIs(hub._lock_unavailable(lock_path), denied)

    def test_lock_fallback_warning_points_at_caller(self):
        url, model_dir = self._save_weights()
        hubconf = "def model():\n    'Doc.'\n    return 1\n"
        repo_url = self._make_hub_archive("repo", {"repo-main/hubconf.py": hubconf})
        repo = "owner/repo:main"
        kwargs = {"force_reload": True, "skip_validation": True, "trust_repo": True}
        calls = {
            "hub.list(": lambda: hub.list(repo, verbose=False, **kwargs),
            "hub.help(": lambda: hub.help(repo, "model", **kwargs),
            "hub.load(": lambda: hub.load(repo, "model", verbose=False, **kwargs),
            "hub.load_state_dict_from_url(": lambda: hub.load_state_dict_from_url(
                url, model_dir=model_dir, progress=False
            ),
        }
        error = OSError(errno.ENOLCK, "No locks available")
        with (
            patch.object(hub, "_git_archive_link", return_value=repo_url),
            patch.object(FileLock, "__enter__", side_effect=error),
        ):
            for name, call in calls.items():
                with self.assertWarnsRegex(UserWarning, "Could not lock") as cm:
                    call()
                # The line of the API call itself, not of a frame above it.
                self.assertEqual(cm.filename, __file__)
                self.assertIn(name, linecache.getline(cm.filename, cm.lineno))

    def test_warm_cache_does_not_lock(self):
        url, model_dir = self._save_weights()
        repo_url = self._make_hub_archive("repo")
        with patch.object(hub, "_git_archive_link", return_value=repo_url):
            hub.load_state_dict_from_url(url, model_dir=model_dir, progress=False)
            repo_dir = self._load_repo("repo")
            with patch.object(hub, "_cache_lock", side_effect=AssertionError):
                hub.load_state_dict_from_url(url, model_dir=model_dir, progress=False)
                self.assertEqual(self._load_repo("repo"), repo_dir)


instantiate_parametrized_tests(TestHub)

if __name__ == "__main__":
    run_tests()
