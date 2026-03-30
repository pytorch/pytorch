"""Git and PR utilities for remote execution."""

from __future__ import annotations

import json
import logging
import re
import subprocess

logger = logging.getLogger(__name__)


class CommitResolver:
    """Resolve commit SHA and repo URL for RE submission.

    Safety checks: blocks if workdir is dirty or local HEAD
    doesn't match remote PR HEAD.
    """

    def __init__(self, repo: str) -> None:
        self.repo = repo

    def resolve(self, pr: int | None = None, commit: str | None = None) -> dict:
        """Return {"sha": ..., "repo": ...} for RE submission."""
        if commit:
            return {"sha": commit, "repo": self.repo}

        self.check_clean_workdir()

        is_ghstack = False
        if pr is None:
            ghstack_pr = self._detect_pr_from_ghstack()
            if ghstack_pr is not None:
                pr = ghstack_pr
                is_ghstack = True
            else:
                pr = self._detect_pr_from_branch()
            logger.info("Auto-detected PR #%d from current branch", pr)

        info = self.pr_info(pr)

        # ghstack: local orig commit differs from PR head commit by design
        if not is_ghstack:
            local_sha = self.local_head()
            if local_sha != info["sha"]:
                raise RuntimeError(
                    f"Local HEAD ({local_sha[:12]}) does not match "
                    f"PR #{pr} HEAD ({info['sha'][:12]}). "
                    "Push your changes before running RE."
                )

        logger.info("PR #%d -> %s (%s)", pr, info["sha"][:12], info["repo"])
        return {"sha": info["sha"], "repo": info["repo"]}

    def pr_info(self, pr: int) -> dict:
        """Get commit SHA and repo URL from a PR number."""
        out = subprocess.run(
            [
                "gh", "pr", "view", str(pr),
                "--repo", self.repo.replace("https://github.com/", "").replace(".git", ""),
                "--json", "headRefOid,headRefName,headRepository,headRepositoryOwner",
            ],
            capture_output=True, text=True, check=True,
        )
        data = json.loads(out.stdout)
        owner = data["headRepositoryOwner"]["login"]
        repo_name = data["headRepository"]["name"]
        return {
            "sha": data["headRefOid"],
            "branch": data["headRefName"],
            "repo": f"https://github.com/{owner}/{repo_name}.git",
        }

    def _detect_pr_from_branch(self) -> int:
        """Detect PR number from current branch name."""
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        out = subprocess.run(
            [
                "gh", "pr", "list",
                "--repo", self.repo.replace("https://github.com/", "").replace(".git", ""),
                "--head", branch,
                "--json", "number",
            ],
            capture_output=True, text=True,
        )
        if out.returncode != 0:
            raise RuntimeError(
                "No PR found for current branch. "
                "Push your branch and open a PR first, or pass --pr explicitly."
            )
        prs = json.loads(out.stdout)
        if not prs:
            raise RuntimeError(
                f"No open PR found for branch '{branch}'. "
                "Push your branch and open a PR first, or pass --pr explicitly."
            )
        return prs[0]["number"]

    def _detect_pr_from_ghstack(self) -> int | None:
        """Detect PR number from ghstack Pull-Request trailer in HEAD commit."""
        out = subprocess.run(
            ["git", "log", "-1", "--format=%B"],
            capture_output=True, text=True, check=True,
        )
        match = re.search(r"^Pull Request resolved:\s*https://github\.com/.+/pull/(\d+)", out.stdout, re.MULTILINE)
        if match:
            pr = int(match.group(1))
            logger.info("Detected ghstack PR #%d from commit trailer", pr)
            return pr
        return None

    def local_head(self) -> str:
        """Get local HEAD commit SHA."""
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()

    def check_clean_workdir(self) -> None:
        """Ensure no uncommitted changes exist."""
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        )
        if out.stdout.strip():
            raise RuntimeError(
                "You have uncommitted local changes. "
                "Commit or stash them before running RE."
            )
