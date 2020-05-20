NON_PR_BRANCH_LIST = [
    "master",
    r"/ci-all\/.*/",
    r"/release\/.*/",
]


def gen_branches_only_filter_dict(branches_list=NON_PR_BRANCH_LIST):
    return {
        "branches": {
            "only": branches_list,
        },
    }
