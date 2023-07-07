import gzip
import io
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import boto3

from tools.stats.upload_stats_lib import upload_to_s3

PYTEST_CACHE_KEY_PREFIX = "additional_caching_info"
BUCKET = "gha-artifacts"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


if __name__ == "__main__":

    commit_sha = subprocess.check_output("git rev-parse HEAD".split()).decode("utf-8")
    pr_num = os.environ["PR_NUM"]

    invoking_files = set()
    for xml_report in Path(REPO_ROOT).glob("test/test-reports/**/*.xml"):
        root = ET.parse(xml_report)

        for test_suite in root.iter("testsuite"):
            if int(test_suite.attrib.get("failures", 0)) > 0 or int(test_suite.attrib.get("errors", 0)) > 0:
                invoking_files.add(xml_report.parent.name)
    print(invoking_files)
    if len(invoking_files) > 0:
        upload_to_s3(BUCKET, f"commit/{commit_sha}" invoking_files)
        if pr_num:
            upload_to_s3(BUCKET, f"commit/{pr_num}" invoking_files)
