"""Unit tests for pytorch_rocm_ci_debug_env helpers (no network, no docker)."""

from __future__ import annotations

import argparse
import base64
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytorch_rocm_ci_debug_env as mod  # noqa: E402
from cli_common import CliError  # noqa: E402
from pytorch_rocm_ci_debug_env import (  # noqa: E402
    CONTAINER_HOME_DIR,
    CONTAINER_REPO_DIR,
    DEBUG_ENV_NAME_INFIX,
    artifact_zip_url,
    collect_build_job_label,
    collect_docker_image_stem,
    container_setup_script,
    debug_env_container_name,
    filesystem_safe_commit_segment,
    filesystem_safe_label,
    find_rocm_build_jobs,
    format_byte_size,
    ghcr_image_ref,
    github_contents_url,
    is_static_github_actions_string,
    launching_user_id,
    rocm_build_job_label,
)


class TestFilesystemSafeCommitSegment(unittest.TestCase):
    def test_alnum_preserved(self) -> None:
        self.assertEqual(filesystem_safe_commit_segment("deadbeef"), "deadbeef")

    def test_slashes_replaced(self) -> None:
        self.assertEqual(filesystem_safe_commit_segment("abc/def"), "abc_def")

    def test_empty_becomes_unknown(self) -> None:
        self.assertEqual(filesystem_safe_commit_segment("   "), "unknown")


class TestLaunchingUserId(unittest.TestCase):
    def test_uses_getpass_login_name(self) -> None:
        with mock.patch.object(mod.getpass, "getuser", return_value="fadiallo"):
            self.assertEqual(launching_user_id(), "fadiallo")

    def test_strips_whitespace(self) -> None:
        with mock.patch.object(mod.getpass, "getuser", return_value="  faa  "):
            self.assertEqual(launching_user_id(), "faa")

    def test_falls_back_to_uid_when_getuser_raises(self) -> None:
        with mock.patch.object(mod.getpass, "getuser", side_effect=KeyError("no name")), \
                mock.patch.object(mod.os, "getuid", return_value=1000, create=True):
            self.assertEqual(launching_user_id(), "1000")

    def test_falls_back_to_uid_when_getuser_blank(self) -> None:
        with mock.patch.object(mod.getpass, "getuser", return_value="   "), \
                mock.patch.object(mod.os, "getuid", return_value=1234, create=True):
            self.assertEqual(launching_user_id(), "1234")


class TestDebugEnvContainerName(unittest.TestCase):
    def test_matches_convention(self) -> None:
        name = debug_env_container_name("fadiallo", "abcdef1234567890", "trunk")
        self.assertEqual(name, f"fadiallo_{DEBUG_ENV_NAME_INFIX}_abcdef123456_trunk")

    def test_uses_short_12_char_sha(self) -> None:
        name = debug_env_container_name("u", "0123456789abcdef" * 2, "wf")
        self.assertIn("_0123456789ab_", name)

    def test_sanitizes_unsafe_chars(self) -> None:
        name = debug_env_container_name("first last", "abc/def", "roc m")
        self.assertEqual(name, f"first_last_{DEBUG_ENV_NAME_INFIX}_abc_def_roc_m")

    def test_prefixes_when_leading_char_not_alphanumeric(self) -> None:
        name = debug_env_container_name("_weird", "deadbeef", "trunk")
        self.assertTrue(name[0].isalnum())
        self.assertTrue(name.startswith("x_"))

    def test_empty_segments_use_fallbacks(self) -> None:
        name = debug_env_container_name("   ", "", "")
        self.assertEqual(name, f"user_{DEBUG_ENV_NAME_INFIX}_commit_workflow")


class TestDockerContainerNameInUse(unittest.TestCase):
    def test_true_when_docker_lists_id(self) -> None:
        completed = mock.MagicMock(returncode=0, stdout="cid123\n", stderr="")
        with mock.patch.object(mod.subprocess, "run", return_value=completed) as run:
            self.assertTrue(mod.docker_container_name_in_use("foo"))
        self.assertEqual(list(run.call_args.args[0]), ["docker", "ps", "-aq", "--filter", "name=^foo$"])

    def test_false_when_no_output(self) -> None:
        completed = mock.MagicMock(returncode=0, stdout="\n", stderr="")
        with mock.patch.object(mod.subprocess, "run", return_value=completed):
            self.assertFalse(mod.docker_container_name_in_use("foo"))

    def test_false_when_docker_errors(self) -> None:
        completed = mock.MagicMock(returncode=1, stdout="", stderr="boom")
        with mock.patch.object(mod.subprocess, "run", return_value=completed):
            self.assertFalse(mod.docker_container_name_in_use("foo"))


class TestGithubContentsUrl(unittest.TestCase):
    def test_builds_contents_url_with_ref(self) -> None:
        url = github_contents_url(".github/workflows/trunk.yml", "abc123")
        self.assertEqual(
            url,
            "https://api.github.com/repos/pytorch/pytorch/contents/"
            ".github/workflows/trunk.yml?ref=abc123",
        )


class TestGithubTreeEntrySha(unittest.TestCase):
    def test_returns_sha_of_named_entry(self) -> None:
        listing = [
            {"name": "aotriton.txt", "sha": "deadbeef"},
            {"name": "docker", "sha": "7ed27d6a861850867bbd15d52665f27fb821857f"},
        ]
        with mock.patch.object(mod, "github_get_json", return_value=listing):
            sha = mod.github_tree_entry_sha(".ci", "docker", "commit", None)
        self.assertEqual(sha, "7ed27d6a861850867bbd15d52665f27fb821857f")

    def test_missing_entry_raises(self) -> None:
        with mock.patch.object(mod, "github_get_json", return_value=[{"name": "x", "sha": "y"}]):
            with self.assertRaises(CliError):
                mod.github_tree_entry_sha(".ci", "docker", "commit", None)


class TestResolveWorkflowViaGithub(unittest.TestCase):
    WORKFLOW_YAML = """
name: trunk
jobs:
  linux-jammy-rocm-py3_10-mi355-build:
    name: linux-jammy-rocm-py3.10-mi355
    with:
      docker-image-name: ci-image:pytorch-linux-jammy-rocm-n-py3
"""

    def _fake_github_get_json(self, url: str, token: object) -> object:
        if "contents/.github/workflows/trunk.yml" in url:
            encoded = base64.b64encode(self.WORKFLOW_YAML.encode("utf-8")).decode("ascii")
            return {"content": encoded, "encoding": "base64"}
        if "contents/.ci?" in url:
            return [{"name": "docker", "sha": "treesha123"}]
        if "/commits/" in url:
            return {"sha": "fullheadsha456"}
        raise AssertionError(f"unexpected url: {url}")

    def test_resolves_all_fields_without_checkout(self) -> None:
        with mock.patch.object(mod, "github_get_json", side_effect=self._fake_github_get_json):
            resolved = mod.resolve_workflow_via_github("abc123", "trunk", None)
        self.assertEqual(resolved.image_stem, "ci-image:pytorch-linux-jammy-rocm-n-py3")
        self.assertEqual(resolved.build_label, "linux-jammy-rocm-py3.10-mi355")
        self.assertEqual(resolved.docker_directory_sha, "treesha123")
        self.assertEqual(resolved.head_sha, "fullheadsha456")
        self.assertEqual(resolved.workflow_path.name, "trunk.yml")


class TestContainerSetupScript(unittest.TestCase):
    def test_host_mode_has_no_clone_or_download(self) -> None:
        script = container_setup_script(CONTAINER_REPO_DIR)
        self.assertNotIn("git clone", script)
        self.assertNotIn("curl", script)
        self.assertIn(f"cd {CONTAINER_REPO_DIR}", script)
        self.assertIn("unzip -o -q artifacts.zip", script)
        self.assertIn("pip install *.whl", script)

    def test_in_container_mode_clones_and_downloads(self) -> None:
        script = container_setup_script(
            CONTAINER_REPO_DIR,
            clone_url="https://github.com/pytorch/pytorch.git",
            commit="abc123",
            artifact_url="https://example.com/artifacts.zip",
        )
        self.assertIn(f"rm -rf {CONTAINER_REPO_DIR}", script)
        self.assertIn(
            f"git clone --filter=blob:none --no-checkout "
            f"https://github.com/pytorch/pytorch.git {CONTAINER_REPO_DIR}",
            script,
        )
        self.assertIn("git checkout --detach abc123", script)
        self.assertIn('curl -fSL -o artifacts.zip "https://example.com/artifacts.zip"', script)
        self.assertIn("pip install *.whl", script)

    def test_test_dir_is_single_repo_relative_path(self) -> None:
        # Unified layout: repo root is the work dir, so the test dir is always <work_dir>/test
        # with no multi-branch fallback.
        script = container_setup_script(CONTAINER_REPO_DIR)
        self.assertIn(f"cd {CONTAINER_REPO_DIR}/test", script)
        self.assertNotIn("../../test", script)
        self.assertNotIn("elif", script)

    def test_repo_dir_is_under_home(self) -> None:
        self.assertEqual(CONTAINER_REPO_DIR, f"{CONTAINER_HOME_DIR}/pytorch")
        self.assertEqual(CONTAINER_HOME_DIR, "/var/lib/jenkins")

    def test_creates_workspace_symlink_in_all_modes(self) -> None:
        # The /workspace/pytorch -> work_dir symlink is in the shared block, so every mode gets it.
        host_script = container_setup_script(CONTAINER_REPO_DIR)
        in_container_script = container_setup_script(
            CONTAINER_REPO_DIR,
            clone_url="https://github.com/pytorch/pytorch.git",
            commit="abc123",
            artifact_url="https://example.com/artifacts.zip",
        )
        for script in (host_script, in_container_script):
            self.assertIn("sudo mkdir -p /workspace", script)
            self.assertIn(f"sudo ln -sfn {CONTAINER_REPO_DIR} /workspace/pytorch", script)


class TestFilesystemSafeLabel(unittest.TestCase):
    def test_plain_label_preserved(self) -> None:
        self.assertEqual(filesystem_safe_label("linux-jammy-rocm-py3.10-mi355"),
                         "linux-jammy-rocm-py3.10-mi355")

    def test_unsafe_chars_replaced(self) -> None:
        self.assertEqual(filesystem_safe_label('a/b\\c:d*e?'), "a_b_c_d_e_")

    def test_empty_becomes_job(self) -> None:
        self.assertEqual(filesystem_safe_label("   "), "job")


class TestIsStaticGithubActionsString(unittest.TestCase):
    def test_plain_string_is_static(self) -> None:
        self.assertTrue(is_static_github_actions_string("ci-image:foo"))

    def test_expression_is_not_static(self) -> None:
        self.assertFalse(is_static_github_actions_string("${{ matrix.docker-image }}"))

    def test_empty_is_not_static(self) -> None:
        self.assertFalse(is_static_github_actions_string("   "))


class TestFindRocmBuildJobs(unittest.TestCase):
    def test_matches_rocm_build_excludes_cuda(self) -> None:
        workflow = {
            "jobs": {
                "linux-jammy-rocm-py3_10-build": {"name": "linux-jammy-rocm-py3.10"},
                "linux-jammy-cuda-build": {"name": "linux-jammy-cuda"},
                "linux-jammy-rocm-py3_10-test": {"name": "rocm-test"},
            }
        }
        matched = find_rocm_build_jobs(workflow)
        self.assertEqual([jid for jid, _ in matched], ["linux-jammy-rocm-py3_10-build"])

    def test_no_jobs_mapping_returns_empty(self) -> None:
        self.assertEqual(find_rocm_build_jobs({}), [])
        self.assertEqual(find_rocm_build_jobs({"jobs": "nope"}), [])


class TestRocmBuildJobLabel(unittest.TestCase):
    def test_static_name_wins(self) -> None:
        self.assertEqual(
            rocm_build_job_label("the-id", {"name": "linux-jammy-rocm-py3.10-mi355"}),
            "linux-jammy-rocm-py3.10-mi355",
        )

    def test_falls_back_to_job_id(self) -> None:
        self.assertEqual(rocm_build_job_label("the-id", {}), "the-id")

    def test_expression_name_falls_back_to_job_id(self) -> None:
        self.assertEqual(rocm_build_job_label("the-id", {"name": "${{ x }}"}), "the-id")


class TestCollectBuildJobLabel(unittest.TestCase):
    PATH = Path("trunk.yml")

    def test_no_rocm_build_job_raises(self) -> None:
        workflow = {"jobs": {"linux-cuda-build": {"name": "cuda"}}}
        with self.assertRaisesRegex(CliError, "no ROCm build job"):
            collect_build_job_label(workflow, workflow_path=self.PATH)

    def test_more_than_one_rocm_build_job_raises(self) -> None:
        workflow = {
            "jobs": {
                "a-rocm-build": {"name": "rocm-a"},
                "b-rocm-build": {"name": "rocm-b"},
            }
        }
        with self.assertRaisesRegex(CliError, "expected exactly one ROCm build job"):
            collect_build_job_label(workflow, workflow_path=self.PATH)

    def test_single_rocm_build_job_label(self) -> None:
        workflow = {"jobs": {"x-rocm-build": {"name": "linux-rocm-mi355"}}}
        self.assertEqual(
            collect_build_job_label(workflow, workflow_path=self.PATH), "linux-rocm-mi355"
        )


class TestCollectDockerImageStem(unittest.TestCase):
    PATH = Path("trunk.yml")

    def _wf(self, with_block: object) -> dict:
        return {"jobs": {"x-rocm-build": {"name": "rocm", "with": with_block}}}

    def test_missing_with_block_raises(self) -> None:
        workflow = {"jobs": {"x-rocm-build": {"name": "rocm"}}}
        with self.assertRaisesRegex(CliError, "no 'with:' mapping"):
            collect_docker_image_stem(workflow, workflow_path=self.PATH)

    def test_missing_docker_image_name_raises(self) -> None:
        with self.assertRaisesRegex(CliError, "missing docker-image-name"):
            collect_docker_image_stem(self._wf({}), workflow_path=self.PATH)

    def test_non_static_docker_image_name_raises(self) -> None:
        with self.assertRaisesRegex(CliError, "not a static string"):
            collect_docker_image_stem(
                self._wf({"docker-image-name": "${{ matrix.img }}"}), workflow_path=self.PATH
            )

    def test_static_docker_image_name_returned(self) -> None:
        stem = collect_docker_image_stem(
            self._wf({"docker-image-name": " ci-image:pytorch-linux-jammy-rocm-n-py3 "}),
            workflow_path=self.PATH,
        )
        self.assertEqual(stem, "ci-image:pytorch-linux-jammy-rocm-n-py3")


class TestAcquireRepository(unittest.TestCase):
    def test_missing_repo_dir_raises(self) -> None:
        with self.assertRaisesRegex(CliError, "not a directory"):
            mod.acquire_repository("abc123", Path("/no/such/dir/at/all"))

    def test_repo_dir_pointing_at_file_raises(self) -> None:
        with tempfile.NamedTemporaryFile() as handle:
            with self.assertRaisesRegex(CliError, "not a directory"):
                mod.acquire_repository("abc123", Path(handle.name))


class TestGhcrImageRefAndArtifactUrl(unittest.TestCase):
    def test_ghcr_image_ref(self) -> None:
        self.assertEqual(
            ghcr_image_ref("ci-image:pytorch-linux-jammy-rocm-n-py3", "7ed27d6a"),
            "ghcr.io/pytorch/ci-image:pytorch-linux-jammy-rocm-n-py3-7ed27d6a",
        )

    def test_artifact_zip_url(self) -> None:
        self.assertEqual(
            artifact_zip_url(123, "linux-jammy-rocm-py3.10-mi355"),
            "https://gha-artifacts.s3.amazonaws.com/pytorch/pytorch/123/"
            "linux-jammy-rocm-py3.10-mi355/artifacts.zip",
        )


class TestFormatByteSize(unittest.TestCase):
    def test_units(self) -> None:
        self.assertEqual(format_byte_size(512), "512 B")
        self.assertEqual(format_byte_size(2048), "2.00 KiB")
        self.assertEqual(format_byte_size(5 * (1 << 20)), "5.00 MiB")
        self.assertEqual(format_byte_size(3 * (1 << 30)), "3.00 GiB")

    def test_negative_clamped(self) -> None:
        self.assertEqual(format_byte_size(-10), "-10 B")


class TestGithubApiHelpers(unittest.TestCase):
    def test_github_file_text_decodes_base64(self) -> None:
        encoded = base64.b64encode(b"hello").decode("ascii")
        with mock.patch.object(
            mod, "github_get_json", return_value={"content": encoded, "encoding": "base64"}
        ):
            self.assertEqual(mod.github_file_text("p", "c", None), "hello")

    def test_github_file_text_rejects_non_base64(self) -> None:
        with mock.patch.object(
            mod, "github_get_json", return_value={"content": "x", "encoding": "utf-8"}
        ):
            with self.assertRaisesRegex(CliError, "not base64"):
                mod.github_file_text("p", "c", None)

    def test_github_commit_full_sha(self) -> None:
        with mock.patch.object(mod, "github_get_json", return_value={"sha": "fullsha"}):
            self.assertEqual(mod.github_commit_full_sha("c", None), "fullsha")

    def test_github_commit_full_sha_missing_raises(self) -> None:
        with mock.patch.object(mod, "github_get_json", return_value={}):
            with self.assertRaisesRegex(CliError, "missing sha"):
                mod.github_commit_full_sha("c", None)

    def test_http_error_becomes_clierror(self) -> None:
        err = urllib.error.HTTPError("http://x", 404, "Not Found", {}, None)
        with mock.patch.object(mod, "github_get_json", side_effect=err):
            with self.assertRaisesRegex(CliError, "HTTP 404"):
                mod.github_commit_full_sha("c", None)


class TestLatestWorkflowRunId(unittest.TestCase):
    def test_picks_newest_run(self) -> None:
        payload = {
            "workflow_runs": [
                {"id": 1, "created_at": "2024-01-01T00:00:00Z"},
                {"id": 2, "created_at": "2024-02-01T00:00:00Z"},
            ]
        }
        with mock.patch.object(mod, "github_get_json", return_value=payload):
            self.assertEqual(mod.latest_workflow_run_id("trunk.yml", "sha", None), 2)

    def test_empty_runs_raises(self) -> None:
        with mock.patch.object(mod, "github_get_json", return_value={"workflow_runs": []}):
            with self.assertRaisesRegex(CliError, "no workflow_runs"):
                mod.latest_workflow_run_id("trunk.yml", "sha", None)

    def test_http_error_raises_clierror(self) -> None:
        err = urllib.error.HTTPError("http://x", 500, "boom", {}, None)
        with mock.patch.object(mod, "github_get_json", side_effect=err):
            with self.assertRaises(CliError):
                mod.latest_workflow_run_id("trunk.yml", "sha", None)


class TestDockerRunMountAndWorkdir(unittest.TestCase):
    def _capture_command(self, repo: object, container_name: str = "cname") -> list[str]:
        completed = mock.MagicMock(returncode=0, stdout="cid123\n", stderr="")
        with mock.patch.object(mod, "host_render_gid", return_value=None), mock.patch.object(
            mod.subprocess, "run", return_value=completed
        ) as run:
            mod.docker_run_pytorch_container(
                "img:tag", repo, "headsha", "master", container_name
            )
        return list(run.call_args.args[0])

    def test_name_flag_always_added(self) -> None:
        command = self._capture_command(None, "faa_pytorch_rocm_ci_debug_env_abc_trunk")
        nidx = command.index("--name")
        self.assertEqual(command[nidx + 1], "faa_pytorch_rocm_ci_debug_env_abc_trunk")

    def test_host_mode_mounts_repo_at_repo_dir(self) -> None:
        command = self._capture_command(Path("/tmp/clone"))
        self.assertIn("-v", command)
        self.assertIn(f"/tmp/clone:{CONTAINER_REPO_DIR}", command)
        widx = command.index("-w")
        self.assertEqual(command[widx + 1], CONTAINER_REPO_DIR)

    def test_in_container_mode_has_no_bind_mount(self) -> None:
        command = self._capture_command(None)
        self.assertNotIn("-v", command)
        widx = command.index("-w")
        self.assertEqual(command[widx + 1], CONTAINER_HOME_DIR)


class TestChownScope(unittest.TestCase):
    def test_install_wheel_chowns_repo_dir(self) -> None:
        with mock.patch.object(mod, "docker_exec_chown_workspace") as chown, mock.patch.object(
            mod, "_docker_exec_bash"
        ):
            mod.docker_exec_install_wheel("cid")
        chown.assert_called_once_with("cid", CONTAINER_REPO_DIR)

    def test_clone_and_install_chowns_home_dir(self) -> None:
        with mock.patch.object(mod, "docker_exec_chown_workspace") as chown, mock.patch.object(
            mod, "_docker_exec_bash"
        ):
            mod.docker_exec_clone_and_install_wheel("cid", "abc123", "https://x/artifacts.zip")
        chown.assert_called_once_with("cid", CONTAINER_HOME_DIR)

    def test_chown_targets_given_dir(self) -> None:
        completed = mock.MagicMock(returncode=0, stdout="", stderr="")
        with mock.patch.object(mod.subprocess, "run", return_value=completed) as run:
            mod.docker_exec_chown_workspace("cid", CONTAINER_REPO_DIR)
        self.assertEqual(list(run.call_args.args[0])[-1], CONTAINER_REPO_DIR)


class TestModeSelection(unittest.TestCase):
    @staticmethod
    def _args(*, host_mapped: bool = False, repo_dir: object = None) -> argparse.Namespace:
        return argparse.Namespace(host_mapped=host_mapped, repo_dir=repo_dir)

    def test_default_is_in_container(self) -> None:
        self.assertTrue(mod.use_in_container_mode(self._args()))

    def test_host_mapped_is_not_in_container(self) -> None:
        self.assertFalse(mod.use_in_container_mode(self._args(host_mapped=True)))

    def test_repo_dir_is_not_in_container(self) -> None:
        self.assertFalse(mod.use_in_container_mode(self._args(repo_dir=Path("/tmp/clone"))))

    def test_default_resolves_via_github(self) -> None:
        args = argparse.Namespace(
            host_mapped=False, repo_dir=None, commit="abc123", workflow="trunk"
        )
        sentinel = object()
        with mock.patch.object(
            mod, "resolve_workflow_via_github", return_value=sentinel
        ) as via_github, mock.patch.object(mod, "acquire_repository") as acquire:
            repo_root, temp_clone, resolved = mod.resolve_for_mode(args, None)
        via_github.assert_called_once_with("abc123", "trunk", None)
        acquire.assert_not_called()
        self.assertIsNone(repo_root)
        self.assertIsNone(temp_clone)
        self.assertIs(resolved, sentinel)


class TestMutualExclusion(unittest.TestCase):
    def test_repo_dir_and_host_mapped_fail_cleanly(self) -> None:
        argv = [
            "prog",
            "7f426f9ce0341aac19f5343b39fc492174de02aa",
            "trunk",
            "--repo-dir",
            "/tmp",
            "--host-mapped",
        ]
        with mock.patch.object(sys, "argv", argv):
            with mock.patch("sys.stderr"):
                rc = mod.main()
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
