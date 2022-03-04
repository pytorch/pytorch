#!/usr/bin/env python3
import json
import os
from hashlib import sha256
from trymerge import gh_graphql, GitHubPR
from typing import Any
from unittest import TestCase, main, mock

def mocked_gh_graphql(query: str, **kwargs: Any) -> Any:
    gql_db_fname = os.path.join(os.path.dirname(__file__), "gql_mocks.json")

    def get_mocked_queries() -> Any:
        if not os.path.exists(gql_db_fname):
            return {}
        with open(gql_db_fname, encoding="utf-8") as f:
            return json.load(f)

    def save_mocked_queries(obj: Any) -> None:
        with open(gql_db_fname, encoding="utf-8", mode="w") as f:
            json.dump(obj, f)

    key = f"query_sha={sha256(query.encode('utf-8')).hexdigest()} " + " ".join([f"{k}={kwargs[k]}" for k in sorted(kwargs.keys())])
    mocked_queries = get_mocked_queries()

    if key in mocked_queries:
        return mocked_queries[key]

    rc = gh_graphql(query, **kwargs)
    mocked_queries[key] = rc

    save_mocked_queries(mocked_queries)

    return rc


class TestGitHubPR(TestCase):
    @mock.patch('trymerge.gh_graphql', side_effect=mocked_gh_graphql)
    def test_get_author_null(self, mocked_gql: Any) -> None:
        """ Tests that PR author can be computed
            If reply contains NULL
        """
        pr = GitHubPR("pytorch", "pytorch", 71759)
        author = pr.get_author()
        self.assertTrue(author is not None)
        self.assertTrue("@" in author)

    @mock.patch('trymerge.gh_graphql', side_effect=mocked_gh_graphql)
    def test_large_diff(self, mocked_gql: Any) -> None:
        "Tests that PR with 100+ files can be fetched"
        pr = GitHubPR("pytorch", "pytorch", 73099)
        self.assertTrue(pr.get_changed_files_count() > 100)
        flist = pr.get_changed_files()
        self.assertEqual(len(flist), pr.get_changed_files_count())


if __name__ == "__main__":
    main()
