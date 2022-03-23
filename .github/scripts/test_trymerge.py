#!/usr/bin/env python3
import json
import os
from hashlib import sha256
from trymerge import find_matching_merge_rule, gh_graphql, GitHubPR
from gitutils import get_git_remote_name, get_git_repo_dir, GitRepo
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
            json.dump(obj, f, indent=2)
            f.write("\n")

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
    def test_match_rules(self, mocked_gql: Any) -> None:
        "Tests that PR passes merge rules"
        pr = GitHubPR("pytorch", "pytorch", 71759)
        repo = GitRepo(get_git_repo_dir(), get_git_remote_name())
        self.assertTrue(find_matching_merge_rule(pr, repo) is not None)

    @mock.patch('trymerge.gh_graphql', side_effect=mocked_gh_graphql)
    def test_get_last_comment(self, mocked_gql: Any) -> None:
        "Tests that last comment can be fetched"
        pr = GitHubPR("pytorch", "pytorch", 71759)
        comment = pr.get_last_comment()
        self.assertEqual(comment.author_login, "github-actions")
        self.assertIsNone(comment.editor_login)
        self.assertTrue("You've committed this PR" in comment.body_text)

    @mock.patch('trymerge.gh_graphql', side_effect=mocked_gh_graphql)
    def test_get_author_null(self, mocked_gql: Any) -> None:
        """ Tests that PR author can be computed
            If reply contains NULL
        """
        pr = GitHubPR("pytorch", "pytorch", 71759)
        author = pr.get_author()
        self.assertTrue(author is not None)
        self.assertTrue("@" in author)
        self.assertTrue(pr.get_diff_revision() is None)

    @mock.patch('trymerge.gh_graphql', side_effect=mocked_gh_graphql)
    def test_large_diff(self, mocked_gql: Any) -> None:
        "Tests that PR with 100+ files can be fetched"
        pr = GitHubPR("pytorch", "pytorch", 73099)
        self.assertTrue(pr.get_changed_files_count() > 100)
        flist = pr.get_changed_files()
        self.assertEqual(len(flist), pr.get_changed_files_count())

    @mock.patch('trymerge.gh_graphql', side_effect=mocked_gh_graphql)
    def test_internal_changes(self, mocked_gql: Any) -> None:
        "Tests that PR with internal changes is detected"
        pr = GitHubPR("pytorch", "pytorch", 73969)
        self.assertTrue(pr.has_internal_changes())

    @mock.patch('trymerge.gh_graphql', side_effect=mocked_gh_graphql)
    def test_checksuites_pagination(self, mocked_gql: Any) -> None:
        "Tests that PR with lots of checksuits can be fetched"
        pr = GitHubPR("pytorch", "pytorch", 73811)
        self.assertGreater(len(pr.get_checkrun_conclusions()), 0)


if __name__ == "__main__":
    main()
