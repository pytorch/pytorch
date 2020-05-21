"""
Run this job on everything since it is
the dependency for everything.
"""


class SetupJob:
    def __init__(self, name):
        self.name = name

    def gen_tree(self):

        # Note that these include-everything glob patterns are required; it is not equivalent to
        # omit the filters entirely because tags are not included by default.
        # And if you specify tags then you also need to specify branches since
        # specifying tags disincludes branches by default.
        props_dict = {
            "filters": {
                "tags": {
                    "only": "/.*/",
                },
                "branches": {
                    "only": "/.*/",
                },
            },
        }

        return [{self.name: props_dict}]


WORKFLOW_DATA = [
    SetupJob("setup"),
]


def get_workflow_jobs():
    return [item.gen_tree() for item in WORKFLOW_DATA]
