"""Unit tests for cli.lib.pytorch.workflow_parser."""
# ruff: noqa: S101

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pathlib import Path

import pytest
from cli.lib.pytorch.workflow_parser import (
    _build_image,
    _build_runner,
    _strip_gha,
    parse_workflow,
    resolve_workflow_path,
)


# ---------------------------------------------------------------------------
# _strip_gha
# ---------------------------------------------------------------------------


class TestStripGha:
    def test_removes_expression(self):
        assert _strip_gha("${{ foo.bar }}linux.2xlarge") == "linux.2xlarge"

    def test_no_expression(self):
        assert _strip_gha("linux.2xlarge") == "linux.2xlarge"

    def test_multiple_expressions(self):
        assert _strip_gha("${{ a }}${{ b }}x") == "x"

    def test_empty(self):
        assert _strip_gha("") == ""


# ---------------------------------------------------------------------------
# _build_runner / _build_image
# ---------------------------------------------------------------------------


class TestBuildRunner:
    def test_explicit_runner(self):
        job = {"with": {"runner": "linux.4xlarge"}}
        assert _build_runner(job) == "linux.4xlarge"

    def test_gha_prefix_stripped(self):
        job = {"with": {"runner": "${{ prefix }}linux.4xlarge"}}
        assert _build_runner(job) == "linux.4xlarge"

    def test_default_when_missing(self):
        assert _build_runner({}) == "linux.2xlarge"


class TestBuildImage:
    def test_returns_image(self):
        job = {
            "with": {"docker-image-name": "ci-image:pytorch-linux-jammy-py3.10-gcc11"}
        }
        assert _build_image(job) == "ci-image:pytorch-linux-jammy-py3.10-gcc11"

    def test_empty_when_missing(self):
        assert _build_image({}) == ""

    def test_gha_expression_stripped(self):
        job = {
            "with": {"docker-image-name": "${{ steps.calculate.outputs.docker-image }}"}
        }
        assert _build_image(job) == ""


# ---------------------------------------------------------------------------
# parse_workflow
# ---------------------------------------------------------------------------

_WORKFLOW_YAML = textwrap.dedent("""\
    jobs:
      linux-jammy-py3_10-gcc11-build:
        with:
          build-environment: linux-jammy-py3.10-gcc11
          docker-image-name: ci-image:pytorch-linux-jammy-py3.10-gcc11
          test-matrix: |
            { include: [
              { config: "default", shard: 1, num_shards: 2, runner: "${{ prefix }}linux.2xlarge" },
              { config: "default", shard: 2, num_shards: 2, runner: "${{ prefix }}linux.2xlarge" },
              { config: "numpy_2_x", shard: 1, num_shards: 1, runner: "${{ prefix }}linux.2xlarge" },
              { config: "rocm_only", shard: 1, num_shards: 1, runner: "${{ prefix }}linux.rocm.gpu" },
            ]}
      linux-cuda12-build:
        with:
          build-environment: linux-cuda12.4
          docker-image-name: ci-image:pytorch-linux-cuda12.4
          runner: linux.4xlarge
          test-matrix: |
            { include: [
              { config: "default", shard: 1, num_shards: 1, runner: "${{ prefix }}linux.2xlarge" },
            ]}
      not-a-build-job:
        with:
          build-environment: should-be-ignored
          test-matrix: |
            { include: [{ config: "x", runner: "linux.2xlarge" }] }
""")


@pytest.fixture
def workflow_file(tmp_path: Path) -> Path:
    p = tmp_path / "pull.yml"
    p.write_text(_WORKFLOW_YAML)
    return p


class TestParseWorkflow:
    def test_basic_entries(self, workflow_file):
        results = parse_workflow(str(workflow_file), runner_filter="linux.2xlarge")
        configs = [(r["build_env"], r["config"]) for r in results]
        assert ("linux-jammy-py3.10-gcc11", "default") in configs
        assert ("linux-jammy-py3.10-gcc11", "numpy_2_x") in configs
        assert ("linux-cuda12.4", "default") in configs

    def test_runner_filter_excludes_rocm(self, workflow_file):
        results = parse_workflow(str(workflow_file), runner_filter="linux.2xlarge")
        configs = [r["config"] for r in results]
        assert "rocm_only" not in configs

    def test_no_filter_includes_all(self, workflow_file):
        results = parse_workflow(str(workflow_file), runner_filter="")
        configs = [r["config"] for r in results]
        assert "rocm_only" in configs

    def test_deduplication(self, workflow_file):
        results = parse_workflow(str(workflow_file), runner_filter="linux.2xlarge")
        # "default" for linux-jammy should appear only once despite two shards
        default_entries = [
            r
            for r in results
            if r["build_env"] == "linux-jammy-py3.10-gcc11" and r["config"] == "default"
        ]
        assert len(default_entries) == 1

    def test_build_image_populated(self, workflow_file):
        results = parse_workflow(str(workflow_file), runner_filter="linux.2xlarge")
        entry = next(r for r in results if r["build_env"] == "linux-jammy-py3.10-gcc11")
        assert entry["build_image"] == "ci-image:pytorch-linux-jammy-py3.10-gcc11"

    def test_build_runner_explicit(self, workflow_file):
        results = parse_workflow(str(workflow_file), runner_filter="linux.2xlarge")
        entry = next(r for r in results if r["build_env"] == "linux-cuda12.4")
        assert entry["build_runner"] == "linux.4xlarge"

    def test_build_runner_default(self, workflow_file):
        results = parse_workflow(str(workflow_file), runner_filter="linux.2xlarge")
        entry = next(r for r in results if r["build_env"] == "linux-jammy-py3.10-gcc11")
        assert entry["build_runner"] == "linux.2xlarge"

    def test_non_build_jobs_ignored(self, workflow_file):
        results = parse_workflow(str(workflow_file), runner_filter="linux.2xlarge")
        envs = [r["build_env"] for r in results]
        assert "should-be-ignored" not in envs

    def test_gha_expressions_stripped_from_test_runner(self, workflow_file):
        results = parse_workflow(str(workflow_file), runner_filter="linux.2xlarge")
        for r in results:
            assert "${{" not in r["test_runner"]

    def test_empty_workflow(self, tmp_path):
        p = tmp_path / "empty.yml"
        p.write_text("jobs: {}\n")
        assert parse_workflow(str(p)) == []


# ---------------------------------------------------------------------------
# resolve_workflow_path
# ---------------------------------------------------------------------------


class TestResolveWorkflowPath:
    def test_full_path(self, tmp_path):
        p = tmp_path / "myflow.yml"
        p.write_text("")
        result = resolve_workflow_path(str(p))
        assert result == p

    def test_stem_resolves_with_yml(self, tmp_path, monkeypatch):
        workflows_dir = tmp_path / ".github" / "workflows"
        workflows_dir.mkdir(parents=True)
        (workflows_dir / "pull.yml").write_text("")
        monkeypatch.chdir(tmp_path)
        from cli.lib.pytorch import workflow_parser

        monkeypatch.setattr(workflow_parser, "WORKFLOWS_DIR", workflows_dir)
        result = resolve_workflow_path("pull")
        assert result == workflows_dir / "pull.yml"

    def test_not_found_raises(self, tmp_path, monkeypatch):
        workflows_dir = tmp_path / ".github" / "workflows"
        workflows_dir.mkdir(parents=True)
        from cli.lib.pytorch import workflow_parser

        monkeypatch.setattr(workflow_parser, "WORKFLOWS_DIR", workflows_dir)
        with pytest.raises(RuntimeError, match="not found"):
            resolve_workflow_path("nonexistent")
