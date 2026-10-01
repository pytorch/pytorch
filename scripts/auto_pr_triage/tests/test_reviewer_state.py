from __future__ import annotations

import unittest
from unittest import mock

from github_api import PullRequestRef
from reviewer_state import (
    choose_round_robin_member,
    fetch_assigned_member,
    fetch_latest_labeled_pull_requests,
    fetch_requested_codeowner_handles,
    fetch_requested_reviewer_handles,
    fetch_round_robin_cursors,
    fetch_submitted_review_state,
    next_round_robin_member,
    stable_fallback_member,
)


REPOSITORY = "pytorch/ciforge"


class FakeGitHub:
    def __init__(self, responses: list[object]) -> None:
        self.responses = iter(responses)
        self.calls: list[str] = []

    def json(self, endpoint: str) -> object:
        self.calls.append(endpoint)
        return next(self.responses)


def owner(name: str, *members: str) -> dict[str, object]:
    return {"label": f"owner: {name}", "members": list(members)}


def selection(*, reviewer: str, reason: str) -> dict[str, str]:
    return {"reviewer": reviewer, "selection_reason": reason}


def round_robin(
    *, github: FakeGitHub, owners: dict[str, dict[str, object]], ineligible: set[str]
) -> dict[str, dict[str, str] | None]:
    rosters = {
        name: (spec["label"], tuple(spec["members"])) for name, spec in owners.items()
    }
    cursors = fetch_round_robin_cursors(
        PullRequestRef(github=github, repo=REPOSITORY, number=9), owners=rosters
    )
    choices = {
        name: choose_round_robin_member(
            repo=REPOSITORY,
            current_number=9,
            owner=name,
            members=rosters[name][1],
            cursor=cursors[name],
            ineligible_reviewers=ineligible,
        )
        for name in owners
    }
    return {
        name: choice and selection(reviewer=choice[0], reason=choice[1])
        for name, choice in choices.items()
    }


def owner_event(
    *,
    event_id: int,
    event: str = "labeled",
    number: int = 7,
    name: str = "autograd",
) -> dict[str, object]:
    return {
        "id": event_id,
        "event": event,
        "label": {"name": f"owner: {name}"},
        "issue": {"number": number, "pull_request": {}},
    }


def assignment_timeline(
    *, label_id: int, reviewer: str = "first"
) -> list[dict[str, object]]:
    return [
        {
            "id": label_id - 1,
            "event": "review_requested",
            "requested_reviewer": {"login": reviewer},
        },
        {"id": label_id, "event": "labeled"},
    ]


class RoundRobinTest(unittest.TestCase):
    def test_zero_one_and_two_member_rotation(self) -> None:
        cases = (
            ((), None, set(), None),
            (("@first",), None, set(), "@first"),
            (("@first",), "@first", set(), "@first"),
            (("@first", "@second"), None, set(), "@first"),
            (("@first", "@second"), "@first", set(), "@second"),
            (("@first", "@second"), "@second", set(), "@first"),
            (
                ("@first", "@second", "@third"),
                "@first",
                {"@second"},
                "@third",
            ),
        )
        for members, latest, ineligible, expected in cases:
            with self.subTest(members=members, latest=latest):
                self.assertEqual(
                    next_round_robin_member(
                        members=members,
                        latest_reviewer=latest,
                        ineligible_reviewers=ineligible,
                    ),
                    expected,
                )

    def test_zero_owners_needs_no_history(self) -> None:
        github = FakeGitHub([])

        self.assertEqual(
            round_robin(github=github, owners={}, ineligible=set()),
            {},
        )
        self.assertEqual(github.calls, [])

    def test_collects_only_active_codeowner_requests(self) -> None:
        github = mock.Mock()
        github.graphql.side_effect = [
            {
                "repository": {
                    "pullRequest": {
                        "reviewRequests": {
                            "nodes": [
                                {
                                    "asCodeOwner": True,
                                    "requestedReviewer": {
                                        "__typename": "User",
                                        "login": "alice",
                                    },
                                },
                                {
                                    "asCodeOwner": False,
                                    "requestedReviewer": {
                                        "__typename": "User",
                                        "login": "manual",
                                    },
                                },
                            ],
                            "pageInfo": {
                                "endCursor": "next",
                                "hasNextPage": True,
                            },
                        }
                    }
                }
            },
            {
                "repository": {
                    "pullRequest": {
                        "reviewRequests": {
                            "nodes": [
                                {
                                    "asCodeOwner": True,
                                    "requestedReviewer": {
                                        "__typename": "Team",
                                        "slug": "compiler",
                                    },
                                }
                            ],
                            "pageInfo": {
                                "endCursor": None,
                                "hasNextPage": False,
                            },
                        }
                    }
                }
            },
        ]

        self.assertEqual(
            fetch_requested_codeowner_handles(
                PullRequestRef(github=github, repo="pytorch/ciforge", number=9)
            ),
            frozenset({"@alice", "@pytorch/compiler"}),
        )
        self.assertEqual(github.graphql.call_count, 2)

    def test_rejects_codeowner_request_without_provenance(self) -> None:
        github = mock.Mock()
        github.graphql.return_value = {
            "repository": {
                "pullRequest": {
                    "reviewRequests": {
                        "nodes": [
                            {
                                "requestedReviewer": {
                                    "__typename": "User",
                                    "login": "alice",
                                }
                            }
                        ],
                        "pageInfo": {
                            "endCursor": None,
                            "hasNextPage": False,
                        },
                    }
                }
            }
        }

        with self.assertRaisesRegex(RuntimeError, "provenance"):
            fetch_requested_codeowner_handles(
                PullRequestRef(github=github, repo="pytorch/ciforge", number=9)
            )

    def test_latest_team_label_records_pr_and_event(self) -> None:
        github = FakeGitHub(
            [
                [
                    {
                        "id": 91,
                        "event": "labeled",
                        "label": {"name": "owner: autograd"},
                        "issue": {"number": 7, "pull_request": {}},
                    }
                ]
            ]
        )

        self.assertEqual(
            fetch_latest_labeled_pull_requests(
                PullRequestRef(github=github, repo="pytorch/ciforge", number=9),
                team_labels={"autograd": "owner: autograd"},
            ),
            ({"autograd": (7, 91)}, {}),
        )

    def test_bounded_repository_history_recovers_from_durable_label(self) -> None:
        github = FakeGitHub(
            [[{"event": "commented"}] * 100 for _ in range(10)]
            + [
                [{"number": 7, "pull_request": {}}],
                [
                    {
                        "id": 91,
                        "event": "labeled",
                        "label": {"name": "owner: autograd"},
                    }
                ],
            ]
        )

        self.assertEqual(
            fetch_latest_labeled_pull_requests(
                PullRequestRef(github=github, repo="pytorch/ciforge", number=9),
                team_labels={"autograd": "owner: autograd"},
            ),
            (
                {"autograd": (7, 91)},
                {
                    7: [
                        {
                            "id": 91,
                            "event": "labeled",
                            "label": {"name": "owner: autograd"},
                        }
                    ]
                },
            ),
        )

    def test_unused_label_bootstraps_after_bounded_repository_history(self) -> None:
        github = FakeGitHub([[{"event": "commented"}] * 100 for _ in range(10)] + [[]])

        self.assertEqual(
            fetch_latest_labeled_pull_requests(
                PullRequestRef(github=github, repo="pytorch/ciforge", number=9),
                team_labels={"autograd": "owner: autograd"},
            ),
            ({}, {}),
        )
        self.assertEqual(
            github.calls[-1],
            "repos/pytorch/ciforge/issues?state=all&labels=owner%3A%20autograd&per_page=100&page=1",
        )

    def test_removed_label_history(self) -> None:
        cases = (
            (
                "removed only",
                [
                    [
                        owner_event(event_id=92, event="unlabeled"),
                        owner_event(event_id=91),
                    ]
                ],
                "@first",
                "round_robin_initial",
            ),
            (
                "fall back to older active label",
                [
                    [
                        owner_event(event_id=102, event="unlabeled", number=8),
                        owner_event(event_id=101, number=8),
                        owner_event(event_id=91),
                    ],
                    assignment_timeline(label_id=91),
                ],
                "@second",
                "round_robin_next",
            ),
            (
                "new label after removal is active",
                [
                    [
                        owner_event(event_id=103, number=8),
                        owner_event(event_id=102, event="unlabeled", number=8),
                        owner_event(event_id=101, number=8),
                    ],
                    [
                        *assignment_timeline(label_id=101),
                        {"id": 102, "event": "unlabeled"},
                        {"id": 103, "event": "labeled"},
                    ],
                ],
                "@second",
                "round_robin_next",
            ),
        )
        owners = {"autograd": owner("autograd", "@first", "@second")}
        for name, responses, expected, expected_reason in cases:
            github = FakeGitHub(responses)
            with self.subTest(name=name):
                self.assertEqual(
                    round_robin(github=github, owners=owners, ineligible=set()),
                    {"autograd": selection(reviewer=expected, reason=expected_reason)},
                )

    def test_removed_state_is_preserved_across_event_pages(self) -> None:
        first_page = [
            owner_event(event_id=102, event="unlabeled", number=8),
            *([{"event": "commented"}] * 99),
        ]
        github = FakeGitHub(
            [
                first_page,
                [
                    owner_event(event_id=101, number=8),
                    owner_event(event_id=91),
                ],
                assignment_timeline(label_id=91),
            ]
        )

        self.assertEqual(
            round_robin(
                github=github,
                owners={"autograd": owner("autograd", "@first", "@second")},
                ineligible=set(),
            ),
            {"autograd": selection(reviewer="@second", reason="round_robin_next")},
        )

    def test_two_owners_mix_history_and_bootstrap(self) -> None:
        github = FakeGitHub(
            [
                [owner_event(event_id=91)],
                assignment_timeline(label_id=91),
            ]
        )

        self.assertEqual(
            round_robin(
                github=github,
                owners={
                    "autograd": owner("autograd", "@first", "@second"),
                    "compiler": owner("compiler", "@third", "@fourth"),
                },
                ineligible=set(),
            ),
            {
                "autograd": selection(reviewer="@second", reason="round_robin_next"),
                "compiler": selection(reviewer="@third", reason="round_robin_initial"),
            },
        )

    def test_assignment_comes_from_request_before_team_label(self) -> None:
        timeline = [
            {
                "id": 80,
                "event": "review_requested",
                "requested_reviewer": {"login": "first"},
            },
            {
                "id": 90,
                "event": "review_requested",
                "requested_reviewer": {"login": "second"},
            },
            {"id": 91, "event": "labeled"},
            {
                "id": 92,
                "event": "review_requested",
                "requested_reviewer": {"login": "third"},
            },
        ]

        self.assertEqual(
            fetch_assigned_member(
                timeline=timeline,
                label_event_id=91,
                members=("@first", "@second", "@third"),
            ),
            "@second",
        )

    def test_team_label_without_assignment_returns_none(self) -> None:
        self.assertIsNone(
            fetch_assigned_member(
                timeline=[{"id": 91, "event": "labeled"}],
                label_event_id=91,
                members=("@first", "@second"),
            )
        )

    def test_invalid_assignment_uses_stable_fallback(self) -> None:
        owners = {"autograd": owner("autograd", "@first", "@second", "@third")}
        github = FakeGitHub(
            [
                [owner_event(event_id=91)],
                [{"id": 91, "event": "labeled"}],
            ]
        )
        expected = stable_fallback_member(
            repo="pytorch/ciforge",
            current_number=9,
            owner="autograd",
            members=("@first", "@second", "@third"),
            ineligible_reviewers={"@second"},
        )

        self.assertEqual(
            round_robin(github=github, owners=owners, ineligible={"@second"}),
            {"autograd": selection(reviewer=expected, reason="stable_fallback")},
        )

    def test_missing_label_event_does_not_use_fallback(self) -> None:
        github = FakeGitHub(
            [
                [owner_event(event_id=91)],
                [
                    {
                        "id": 90,
                        "event": "review_requested",
                        "requested_reviewer": {"login": "first"},
                    }
                ],
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "label event is absent"):
            round_robin(
                github=github,
                owners={"autograd": owner("autograd", "@first", "@second")},
                ineligible=set(),
            )

    def test_stable_fallback_is_reproducible_and_skips_ineligible_members(self) -> None:
        args = {
            "repo": "pytorch/ciforge",
            "current_number": 9,
            "owner": "autograd",
            "members": ("@first", "@second", "@third"),
            "ineligible_reviewers": {"@second"},
        }

        first = stable_fallback_member(**args)

        self.assertEqual(first, stable_fallback_member(**args))
        self.assertIn(first, {"@first", "@third"})

    def test_stable_fallback_requires_an_eligible_member(self) -> None:
        self.assertIsNone(
            stable_fallback_member(
                repo="pytorch/ciforge",
                current_number=9,
                owner="autograd",
                members=("@first", "@second"),
                ineligible_reviewers={"@first", "@second"},
            )
        )

    def test_round_robin_advances_from_recorded_assignment(self) -> None:
        github = FakeGitHub(
            [
                [
                    {
                        "id": 91,
                        "event": "labeled",
                        "label": {"name": "owner: autograd"},
                        "issue": {"number": 7, "pull_request": {}},
                    }
                ],
                [
                    {
                        "id": 90,
                        "event": "review_requested",
                        "requested_reviewer": {"login": "first"},
                    },
                    {"id": 91, "event": "labeled"},
                ],
            ]
        )

        self.assertEqual(
            round_robin(
                github=github,
                owners={
                    "autograd": {
                        "label": "owner: autograd",
                        "members": ["@first", "@second"],
                    }
                },
                ineligible=set(),
            ),
            {"autograd": selection(reviewer="@second", reason="round_robin_next")},
        )

    def test_teams_on_one_prior_pr_share_one_timeline_fetch(self) -> None:
        github = FakeGitHub(
            [
                [
                    {
                        "id": 91,
                        "event": "labeled",
                        "label": {"name": "owner: autograd"},
                        "issue": {"number": 7, "pull_request": {}},
                    },
                    {
                        "id": 92,
                        "event": "labeled",
                        "label": {"name": "owner: compiler"},
                        "issue": {"number": 7, "pull_request": {}},
                    },
                ],
                [
                    {
                        "id": 90,
                        "event": "review_requested",
                        "requested_reviewer": {"login": "first"},
                    },
                    {"id": 91, "event": "labeled"},
                    {"id": 92, "event": "labeled"},
                ],
            ]
        )

        self.assertEqual(
            round_robin(
                github=github,
                owners={
                    "autograd": {
                        "label": "owner: autograd",
                        "members": ["@first", "@second"],
                    },
                    "compiler": {
                        "label": "owner: compiler",
                        "members": ["@first", "@second"],
                    },
                },
                ineligible=set(),
            ),
            {
                "autograd": selection(reviewer="@second", reason="round_robin_next"),
                "compiler": selection(reviewer="@second", reason="round_robin_next"),
            },
        )
        timeline = "repos/pytorch/ciforge/issues/7/timeline?per_page=100&page=1"
        self.assertEqual(github.calls.count(timeline), 1)


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
