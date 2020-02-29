import glob
import json
import logging
import os
import os.path
import re
import sys
import time

import requests


def get_size(file_dir):
    try:
        # we should only expect one file, if no, something is wrong
        file_name = glob.glob(os.path.join(file_dir, "*"))[0]
        return os.stat(file_name).st_size
    except:
        logging.exception(f"error getting file from: {file_dir}")
        return 0


def build_message(size):
    pkg_type, py_ver, cu_ver, *_ = os.environ.get("BUILD_ENVIRONMENT", "").split() + [
        None,
        None,
        None,
    ]
    os_name = os.uname()[0].lower()
    if os_name == "darwin":
        os_name = "macos"
    return {
        "normal": {
            "os": os_name,
            "pkg_type": pkg_type,
            "py_ver": py_ver,
            "cu_ver": cu_ver,
            "pr": os.environ.get("CIRCLE_PR_NUMBER"),
            "build_num": os.environ.get("CIRCLE_BUILD_NUM"),
            "sha1": os.environ.get("CIRCLE_SHA1"),
            "branch": os.environ.get("CIRCLE_BRANCH"),
        },
        "int": {
            "time": int(time.time()),
            "size": size,
            "commit_time": int(os.environ.get("COMMIT_TIME", "0")),
        },
    }


def send_message(message):
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
                        "category": "perfpipe_pytorch_binary_size",
                        "message": json.dumps(message),
                        "line_escape": False,
                    }
                ]
            ),
        },
    )
    print(r.text)
    r.raise_for_status()


if __name__ == "__main__":
    file_dir = os.environ.get(
        "PYTORCH_FINAL_PACKAGE_DIR", "/home/circleci/project/final_pkgs"
    )
    if len(sys.argv) == 2:
        file_dir = sys.argv[1]
    print("checking dir: " + file_dir)
    size = get_size(file_dir)
    if size != 0:
        try:
            send_message(build_message(size))
        except:
            logging.exception("can't send message")
