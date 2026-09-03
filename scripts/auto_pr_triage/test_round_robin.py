from __future__ import annotations

import unittest
from unittest import mock

from github_reviews import (
    fetch_assigned_member,
    fetch_latest_labeled_pull_requests,
    fetch_requested_codeowner_handles,
    fetch_round_robin_reviewers,
    next_round_robin_member,
    stable_fallback_member,
)


class FakeGitHub:
    def __init__(self, responses: list[object]) -> None:
        self.responses = iter(responses)
        self.calls: list[str] = []

    def json(self, endpoint: str) -> object:
        self.calls.append(endpoint)
        return next(self.responses)


def owner(name: str, *members: str) -> dict[str, object]:
    return {"label": f"owner: {name}", "members": list(members)}


def selection(reviewer: str, reason: str) -> dict[str, str]:
    return {"reviewer": reviewer, "selection_reason": reason}


def owner_event(
    event_id: int,
    event: str = "labeled",
    *,
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
    label_id: int, reviewer: str = "first"
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
                    next_round_robin_member(members, latest, ineligible),
                    expected,
                )

    def test_zero_owners_needs_no_history(self) -> None:
        github = FakeGitHub([])

        self.assertEqual(
            fetch_round_robin_reviewers(
                github,
                "pytorch/ciforge",
                9,
                {},
                set(),
            ),
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
            fetch_requested_codeowner_handles(github, "pytorch/ciforge", 9),
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
            fetch_requested_codeowner_handles(github, "pytorch/ciforge", 9)

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
                github,
                "pytorch/ciforge",
                {"autograd": "owner: autograd"},
                9,
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
                github,
                "pytorch/ciforge",
                {"autograd": "owner: autograd"},
                9,
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
                github,
                "pytorch/ciforge",
                {"autograd": "owner: autograd"},
                9,
            ),
            ({}, {}),
        )
        self.assertEqual(
            github.calls[-1],
            "repos/pytorch/ciforge/issues?state=all&labels="
            "owner%3A%20autograd&per_page=100&page=1",
        )

    def test_removed_label_history(self) -> None:
        cases = (
            (
                "removed only",
                [[owner_event(92, "unlabeled"), owner_event(91)]],
                "@first",
                "round_robin_initial",
            ),
            (
                "fall back to older active label",
                [
                    [
                        owner_event(102, "unlabeled", number=8),
                        owner_event(101, number=8),
                        owner_event(91),
                    ],
                    assignment_timeline(91),
                ],
                "@second",
                "round_robin_next",
            ),
            (
                "new label after removal is active",
                [
                    [
                        owner_event(103, number=8),
                        owner_event(102, "unlabeled", number=8),
                        owner_event(101, number=8),
                    ],
                    [
                        *assignment_timeline(101),
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
            github = FakeGitHub([{"name": "owner: autograd"}, *responses])
            with self.subTest(name=name):
                self.assertEqual(
                    fetch_round_robin_reviewers(
                        github, "pytorch/ciforge", 9, owners, set()
                    ),
                    {"autograd": selection(expected, expected_reason)},
                )

    def test_removed_state_is_preserved_across_event_pages(self) -> None:
        first_page = [
            owner_event(102, "unlabeled", number=8),
            *([{"event": "commented"}] * 99),
        ]
        github = FakeGitHub(
            [
                {"name": "owner: autograd"},
                first_page,
                [
                    owner_event(101, number=8),
                    owner_event(91),
                ],
                assignment_timeline(91),
            ]
        )

        self.assertEqual(
            fetch_round_robin_reviewers(
                github,
                "pytorch/ciforge",
                9,
                {"autograd": owner("autograd", "@first", "@second")},
                set(),
            ),
            {"autograd": selection("@second", "round_robin_next")},
        )

    def test_two_owners_mix_history_and_bootstrap(self) -> None:
        github = FakeGitHub(
            [
                {"name": "owner: autograd"},
                {"name": "owner: compiler"},
                [owner_event(91)],
                assignment_timeline(91),
            ]
        )

        self.assertEqual(
            fetch_round_robin_reviewers(
                github,
                "pytorch/ciforge",
                9,
                {
                    "autograd": owner("autograd", "@first", "@second"),
                    "compiler": owner("compiler", "@third", "@fourth"),
                },
                set(),
            ),
            {
                "autograd": selection("@second", "round_robin_next"),
                "compiler": selection("@third", "round_robin_initial"),
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
                timeline,
                91,
                ("@first", "@second", "@third"),
            ),
            "@second",
        )

    def test_team_label_without_assignment_returns_none(self) -> None:
        self.assertIsNone(
            fetch_assigned_member(
                [{"id": 91, "event": "labeled"}],
                91,
                ("@first", "@second"),
            )
        )

    def test_invalid_assignment_uses_stable_fallback(self) -> None:
        owners = {"autograd": owner("autograd", "@first", "@second", "@third")}
        github = FakeGitHub(
            [
                {"name": "owner: autograd"},
                [owner_event(91)],
                [{"id": 91, "event": "labeled"}],
            ]
        )
        expected = stable_fallback_member(
            "pytorch/ciforge",
            9,
            "autograd",
            ("@first", "@second", "@third"),
            {"@second"},
        )

        with mock.patch("builtins.print") as warning:
            self.assertEqual(
                fetch_round_robin_reviewers(
                    github,
                    "pytorch/ciforge",
                    9,
                    owners,
                    {"@second"},
                ),
                {"autograd": selection(expected, "stable_fallback")},
            )

        self.assertIn("using stable fallback", warning.call_args.args[0])

    def test_missing_label_event_does_not_use_fallback(self) -> None:
        github = FakeGitHub(
            [
                {"name": "owner: autograd"},
                [owner_event(91)],
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
            fetch_round_robin_reviewers(
                github,
                "pytorch/ciforge",
                9,
                {"autograd": owner("autograd", "@first", "@second")},
                set(),
            )

    def test_stable_fallback_is_reproducible_and_skips_ineligible_members(self) -> None:
        args = (
            "pytorch/ciforge",
            9,
            "autograd",
            ("@first", "@second", "@third"),
            {"@second"},
        )

        first = stable_fallback_member(*args)

        self.assertEqual(first, stable_fallback_member(*args))
        self.assertIn(first, {"@first", "@third"})

    def test_stable_fallback_requires_an_eligible_member(self) -> None:
        self.assertIsNone(
            stable_fallback_member(
                "pytorch/ciforge",
                9,
                "autograd",
                ("@first", "@second"),
                {"@first", "@second"},
            )
        )

    def test_round_robin_advances_from_recorded_assignment(self) -> None:
        github = FakeGitHub(
            [
                {"name": "owner: autograd"},
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
            fetch_round_robin_reviewers(
                github,
                "pytorch/ciforge",
                9,
                {
                    "autograd": {
                        "label": "owner: autograd",
                        "members": ["@first", "@second"],
                    }
                },
                set(),
            ),
            {"autograd": selection("@second", "round_robin_next")},
        )

    def test_teams_on_one_prior_pr_share_one_timeline_fetch(self) -> None:
        github = FakeGitHub(
            [
                {"name": "owner: autograd"},
                {"name": "owner: compiler"},
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
            fetch_round_robin_reviewers(
                github,
                "pytorch/ciforge",
                9,
                {
                    "autograd": {
                        "label": "owner: autograd",
                        "members": ["@first", "@second"],
                    },
                    "compiler": {
                        "label": "owner: compiler",
                        "members": ["@first", "@second"],
                    },
                },
                set(),
            ),
            {
                "autograd": selection("@second", "round_robin_next"),
                "compiler": selection("@second", "round_robin_next"),
            },
        )
        timeline = "repos/pytorch/ciforge/issues/7/timeline?per_page=100&page=1"
        self.assertEqual(github.calls.count(timeline), 1)

    def test_missing_team_label_disables_round_robin(self) -> None:
        github = FakeGitHub([{"message": "Not Found"}])

        with self.assertRaisesRegex(RuntimeError, "label is unavailable"):
            fetch_round_robin_reviewers(
                github,
                "pytorch/ciforge",
                9,
                {
                    "autograd": {
                        "label": "owner: autograd",
                        "members": ["@first", "@second"],
                    }
                },
                set(),
            )


if __name__ == "__main__":
    unittest.main()
