from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from trusted_config import (
    EXTRA_OWNERSHIP_METADATA_PATH,
    load_extra_ownership_metadata,
    load_team_members,
    TEAM_MEMBERS_PATH,
)


REPOSITORY = "pytorch/ciforge"
COMMIT_SHA = "a" * 40


class LocalOwnershipLoadingTest(unittest.TestCase):
    def write_documents(self, root: Path) -> dict[str, bytes]:
        documents = {
            EXTRA_OWNERSHIP_METADATA_PATH: json.dumps(
                {"autograd": {"description": "Owns autograd behavior."}}
            ).encode(),
            TEAM_MEMBERS_PATH: json.dumps({"autograd": ["@reviewer"]}).encode(),
        }
        for relative_path, content in documents.items():
            path = root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        return documents

    def assert_source(
        self, *, source: dict[str, str], relative_path: str, content: bytes
    ) -> None:
        header = f"blob {len(content)}\0".encode()
        self.assertEqual(
            source,
            {
                "repository": REPOSITORY,
                "path": relative_path,
                "ref": COMMIT_SHA,
                "blob_sha": hashlib.sha1(header + content).hexdigest(),
            },
        )

    def test_loads_each_configuration_and_computes_git_blob_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            documents = self.write_documents(root)
            metadata = load_extra_ownership_metadata(
                repository_root=root, repo=REPOSITORY, ref=COMMIT_SHA
            )
            members = load_team_members(
                repository_root=root, repo=REPOSITORY, ref=COMMIT_SHA
            )

        self.assertEqual(
            metadata["owners"],
            {
                "autograd": {
                    "description": "Owns autograd behavior.",
                    "bypass_intake_criteria": None,
                }
            },
        )
        self.assertEqual(
            members["members"]["autograd"],
            ["@reviewer"],
        )
        self.assert_source(
            source=metadata["source"],
            relative_path=EXTRA_OWNERSHIP_METADATA_PATH,
            content=documents[EXTRA_OWNERSHIP_METADATA_PATH],
        )
        self.assert_source(
            source=members["source"],
            relative_path=TEAM_MEMBERS_PATH,
            content=documents[TEAM_MEMBERS_PATH],
        )

    def test_analysis_loader_reads_only_extra_team_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            documents = self.write_documents(root)
            (root / TEAM_MEMBERS_PATH).unlink()
            metadata = load_extra_ownership_metadata(
                repository_root=root, repo=REPOSITORY, ref=COMMIT_SHA
            )

        self.assertEqual(
            metadata["owners"]["autograd"]["description"], "Owns autograd behavior."
        )
        self.assert_source(
            source=metadata["source"],
            relative_path=EXTRA_OWNERSHIP_METADATA_PATH,
            content=documents[EXTRA_OWNERSHIP_METADATA_PATH],
        )

    def test_rejects_oversized_document(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_documents(root)
            with (
                mock.patch("trusted_config.MAX_CONFIG_BYTES", 8),
                self.assertRaisesRegex(RuntimeError, "size is invalid"),
            ):
                load_extra_ownership_metadata(
                    repository_root=root, repo=REPOSITORY, ref=COMMIT_SHA
                )

    def test_rejects_symlinked_document(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside.json"
            outside.write_text("{}")
            path = root / EXTRA_OWNERSHIP_METADATA_PATH
            path.parent.mkdir(parents=True)
            path.symlink_to(outside)
            with self.assertRaisesRegex(RuntimeError, "config file is unavailable"):
                load_extra_ownership_metadata(
                    repository_root=root, repo=REPOSITORY, ref=COMMIT_SHA
                )

    def test_preserves_strict_document_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_documents(root)
            path = root / EXTRA_OWNERSHIP_METADATA_PATH
            path.write_text('{"schema_version": 1, "owners": {}}')
            with self.assertRaisesRegex(RuntimeError, "entry is invalid"):
                load_extra_ownership_metadata(
                    repository_root=root, repo=REPOSITORY, ref=COMMIT_SHA
                )
            member_path = root / TEAM_MEMBERS_PATH
            member_path.write_text('{"schema_version": 1, "teams": {}}')
            with self.assertRaisesRegex(RuntimeError, "roster is invalid"):
                load_team_members(repository_root=root, repo=REPOSITORY, ref=COMMIT_SHA)

    def test_bypass_intake_is_optional_per_owner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_documents(root)
            document = {
                "autograd": {
                    "description": "Owns autograd behavior.",
                    "bypass_intake_criteria": "PRs that fix incorrect gradient formulas.",
                },
                "nn": {"description": "Owns torch.nn modules."},
            }
            (root / EXTRA_OWNERSHIP_METADATA_PATH).write_text(json.dumps(document))
            metadata = load_extra_ownership_metadata(
                repository_root=root, repo=REPOSITORY, ref=COMMIT_SHA
            )

        self.assertEqual(
            metadata["owners"],
            {
                "autograd": document["autograd"],
                "nn": {
                    "description": "Owns torch.nn modules.",
                    "bypass_intake_criteria": None,
                },
            },
        )

    def test_rejects_malformed_owner_entries(self) -> None:
        cases = {
            "entry is invalid": [
                {"autograd": "Owns autograd behavior."},
                {"autograd": {"bypass_intake_criteria": "PRs that fix gradients."}},
                {"autograd": {"description": "Owns autograd.", "labels": ["x"]}},
                {"Not-An-ID": {"description": "Owns autograd."}},
            ],
            "description is invalid": [{"autograd": {"description": "  "}}],
            "bypass_intake_criteria is invalid": [
                {
                    "autograd": {
                        "description": "Owns autograd.",
                        "bypass_intake_criteria": "",
                    }
                },
                {
                    "autograd": {
                        "description": "Owns autograd.",
                        "bypass_intake_criteria": 1,
                    }
                },
                {
                    "autograd": {
                        "description": "Owns autograd.",
                        "bypass_intake_criteria": None,
                    }
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_documents(root)
            path = root / EXTRA_OWNERSHIP_METADATA_PATH
            for message, documents in cases.items():
                for document in documents:
                    path.write_text(json.dumps(document))
                    with (
                        self.subTest(document=document),
                        self.assertRaisesRegex(RuntimeError, message),
                    ):
                        load_extra_ownership_metadata(
                            repository_root=root, repo=REPOSITORY, ref=COMMIT_SHA
                        )

    def test_rejects_empty_reviewer_roster(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_documents(root)
            (root / TEAM_MEMBERS_PATH).write_text('{"autograd": []}')
            with self.assertRaisesRegex(RuntimeError, "roster is invalid"):
                load_team_members(repository_root=root, repo=REPOSITORY, ref=COMMIT_SHA)

    def test_checked_in_metadata_and_roster_define_the_same_teams(self) -> None:
        root = Path(__file__).resolve().parents[3]

        metadata = load_extra_ownership_metadata(
            repository_root=root, repo=REPOSITORY, ref=COMMIT_SHA
        )
        members = load_team_members(
            repository_root=root, repo=REPOSITORY, ref=COMMIT_SHA
        )

        self.assertEqual(set(metadata["owners"]), set(members["members"]))


if __name__ == "__main__":
    unittest.main()
