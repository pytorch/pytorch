# These are for jobs that are supposed to run only on the base branch and not
# on PRs
NON_PR_FILTERS = {
    "branches": {
        "only": [
            "master",
            r"/ci-all\/.*/",
            r"/release\/.*/",
        ]
    }
}
