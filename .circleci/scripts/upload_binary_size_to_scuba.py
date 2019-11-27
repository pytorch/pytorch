import json
import os
import requests
import re
import time
import glob
import sys


def get_size(file_dir="/home/circleci/project/final_pkgs"):
    try:
        # we should only expect one file, if no, something is wrong
        file_name = glob.glob(file_dir)[0]
        return os.stat(file_name).st_size
    except Exception as e:
        print(e)
        return 0


def build_message(size):
    pkg_type, py_ver, cu_ver = os.env.get("BUILD_ENVIRONMENT", "N/A N/A N/A").split()
    os_name = os.uname()[0].lower()
    pr = os.env.get("CIRCLE_PR_NUMBER", "n/a")
    build_num = os.env.get("CIRCLE_BUILD_NUM", "n/a")
    sha1 = os.env.get("CIRCLE_SHA1", "n/a")
    if os_name == "darwin":
        os_name = "macos"
    return {
        "normal": {
            "build_info": json.dumps(
                {
                    "os": os,
                    "pkg_type": pkg_type,
                    "py_ver": py_ver,
                    "cu_ver": cu_ver,
                    "pr": pr,
                    "build": build_num,
                    "sha1": sha1,
                    "size": size,
                }
            )
        },
        "int": {"time": int(time.time())},
    }


def send_message(message):
    access_token = os.environ.get("access_token")
    if not access_token:
        raise ValueError("Can't find access token from environment")
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
    file_dir = "/home/circleci/project/final_pkgs"
    if len(sys.argv) == 2:
        file_dir = sys.argv[1]
    size = get_size(file_dir)
    if size != 0:
        try:
            send_message(build_message(size))
        except Exception as e:
            print(e)
