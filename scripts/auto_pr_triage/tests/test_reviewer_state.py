from __future__ import annotations

import unittest
from unittest import mock

from github_api import PullRequestRef
from reviewer_state import (
    fetch_requested_reviewer_handles,
    fetch_submitted_review_state,
)


REPOSITORY = "pytorch/ciforge"


class ReviewerStateTest(unittest.TestCase):
    def test_requested_reviewers_include_manual_requests(self) -> None:
        github = mock.Mock()
        github.json.return_value = {
            "users": [
                {"login": "soulitzer"},
                {"login": "manual-reviewer"},
            ],
            "teams": [{"slug": "compiler"}],
        }

        reviewers = fetch_requested_reviewer_handles(
            PullRequestRef(github=github, repo=REPOSITORY, number=999)
        )

        self.assertEqual(
            reviewers,
            {"@soulitzer", "@manual-reviewer", "@pytorch/compiler"},
        )

    def test_submitted_reviewers_include_qualifying_reviews_from_any_revision(
        self,
    ) -> None:
        github = mock.Mock()
        github.graphql.return_value = {
            "repository": {
                "pullRequest": {
                    "reviews": {
                        "nodes": [
                            {
                                "author": {"login": "reviewer"},
                                "commit": {"oid": "h" * 40},
                                "state": "APPROVED",
                            },
                            {
                                "author": {"login": "old-reviewer"},
                                "commit": {"oid": "o" * 40},
                                "state": "COMMENTED",
                            },
                            {
                                "author": {"login": "commenter"},
                                "commit": {"oid": "h" * 40},
                                "state": "COMMENTED",
                            },
                            {
                                "author": {"login": "change-requester"},
                                "commit": {"oid": "h" * 40},
                                "state": "CHANGES_REQUESTED",
                            },
                            {
                                "author": {"login": "pending"},
                                "commit": {"oid": "h" * 40},
                                "state": "PENDING",
                            },
                            {
                                "author": {"login": "dismissed"},
                                "commit": {"oid": "h" * 40},
                                "state": "DISMISSED",
                            },
                            {
                                "author": {"login": "missing-commit"},
                                "commit": None,
                                "state": "APPROVED",
                            },
                            {
                                "author": None,
                                "commit": {"oid": "h" * 40},
                                "state": "COMMENTED",
                            },
                        ],
                        "pageInfo": {"endCursor": None, "hasNextPage": False},
                    }
                }
            }
        }

        reviewers = fetch_submitted_review_state(
            PullRequestRef(github=github, repo=REPOSITORY, number=999)
        )

        self.assertEqual(
            reviewers,
            frozenset(
                {
                    "@change-requester",
                    "@commenter",
                    "@missing-commit",
                    "@old-reviewer",
                    "@reviewer",
                }
            ),
        )
        self.assertNotIn("onBehalfOf", github.graphql.call_args.kwargs["query"])

    def test_submitted_reviewers_reject_malformed_response(self) -> None:
        github = mock.Mock()
        github.graphql.return_value = {
            "repository": {"pullRequest": {"reviews": {"nodes": [None]}}}
        }

        with self.assertRaisesRegex(RuntimeError, "response is incomplete"):
            fetch_submitted_review_state(
                PullRequestRef(github=github, repo=REPOSITORY, number=999)
            )


if __name__ == "__main__":
    unittest.main()
