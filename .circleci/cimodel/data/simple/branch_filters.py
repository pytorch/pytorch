NON_PR_BRANCH_LIST = [
    "master",
    r"/ci-all\/.*/",
    r"/release\/.*/",
]


def gen_branch_filter_dict():
    return {
        "branches": {
            "only": NON_PR_BRANCH_LIST,
        },
    }
