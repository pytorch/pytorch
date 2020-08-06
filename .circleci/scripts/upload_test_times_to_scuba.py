import json
import os
from glob import glob
from xml.dom import minidom

import requests


class TestCase:
    def __init__(self, dom):
        self.class_name = str(dom.attributes["classname"].value)
        self.name = str(dom.attributes["name"].value)
        self.time = float(dom.attributes["time"].value)
        self.errored = len(dom.getElementsByTagName("error")) > 0
        self.failed = len(dom.getElementsByTagName("failure")) > 0
        self.skipped = len(dom.getElementsByTagName("skipped")) > 0

    def __str__(self):
        return f"{self.class_name}.{self.name}"

    def get_test_status(self):
        if self.errored:
            return "error"
        elif self.failed:
            return "failed"
        elif self.skipped:
            return "skipped"
        else:
            return "passed"


def send_to_scuba(test_case):
    print(f"Uploading {test_case}...")
    access_token = os.environ.get("SCRIBE_GRAPHQL_ACCESS_TOKEN")
    if not access_token:
        raise ValueError("Can't find access token from environment variable")
    url = "https://graph.facebook.com/scribe_logs"
    r = requests.post(
        url,
        data={
            "access_token": access_token,
            "logs": json.dumps(
                [
                    {
                        "category": "perfpipe_pytorch_test_times",
                        "message": {
                            "config_name": os.environ.get("CIRCLE_JOB"),
                            "test_class": test_case.class_name,
                            "test_name": test_case.name,
                            "time_taken": test_case.time,
                            "status": test_case.get_test_status,
                        },
                        "line_escape": False,
                    }
                ]
            ),
        },
    )
    print(r.text)
    r.raise_for_status()


def parse_report(path):
    dom = minidom.parse(path)
    for test_case in dom.getElementsByTagName("testcase"):
        yield TestCase(test_case)


if __name__ == "__main__":
    reports = glob(os.path.join("test", "**", "*.xml"), recursive=True)
    tests_by_class = dict()
    for report in reports:
        for test_case in parse_report(report):
            send_to_scuba(test_case)
