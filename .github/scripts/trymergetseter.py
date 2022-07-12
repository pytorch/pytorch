from gitutils import GitRepo, get_git_repo_dir, get_git_remote_name
from trymerge import GitHubPR, find_matching_merge_rule, validate_land_time_checks
import os
import json
# GITHUB_TOKEN=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
os.environ['GITHUB_TOKEN'] = "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"
pr_num = 81119
commit = "6882717f73deffb692219ccd1fd6db258d8ed684"
repo = GitRepo(get_git_repo_dir(), get_git_remote_name())
org, project = repo.gh_owner_and_name()
pr = GitHubPR(org, project, pr_num, commit)
checks = pr.get_checkrun_conclusions()
land_checks = pr.get_land_checkrun_conclusions()
# print(json.dumps(land_checks))
# validate_land_time_checks(pr)

# print(land_checks)
find_matching_merge_rule(pr, repo, False, False)

# merge(pr_num, repo,
#       dry_run=True,
#       force=False,
#       on_green=False,
#       mandatory_only=False,
# )
