from typing import OrderedDict


def get_workflow_job():
    return [
        OrderedDict(
            {
                "upload_test_stats": OrderedDict(
                    {
                        "name": "upload test status",
                        "requires": [
                            "macos-12-py3-x86-64-test-1-2-default",
                            "macos-12-py3-x86-64-test-2-2-default",
                            "macos-12-py3-x86-64-test-1-1-functorch",
                        ],
                    }
                )
            }
        ),
    ]
