#!/usr/bin/env python3
# Tests implemented in this file are relying on GitHub GraphQL APIs
# In order to avoid test flakiness, results of the queries
# are cached in gql_mocks.json
# PyTorch Lint workflow does not have GITHUB_TOKEN defined to avoid
# flakiness, so if you are making changes to merge_rules or
# GraphQL queries in trymerge.py, please make sure to delete `gql_mocks.json`
# And re-run the test locally with ones PAT

import json
from pr_updates_submodules import gh_get_pr_updated_submodules
from test_trymerge import mocked_gh_graphql
from unittest import TestCase, main, mock
from typing import Any

class TestPRUpdatesSubmodules(TestCase):
    @mock.patch('pr_updates_submodules.gh_graphql', side_effect=mocked_gh_graphql)
    def test_pr_with_no_updates(self, *args: Any) -> None:
        rc = gh_get_pr_updated_submodules("pytorch", "pytorch", 95045)
        self.assertEqual(rc, [])
    @mock.patch('pr_updates_submodules.gh_graphql', side_effect=mocked_gh_graphql)
    def test_pr_updates_ideep(self, *args: Any) -> None:
        rc = gh_get_pr_updated_submodules("pytorch", "pytorch", 94939)
        self.assertEqual(rc, ["third_party/ideep"])


if __name__ == "__main__":
    main()
