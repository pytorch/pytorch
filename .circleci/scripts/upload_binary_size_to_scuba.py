import glob
import json
import logging
import os
import os.path
import pathlib
import re
import sys
import time
import zipfile

import requests


def get_size(file_dir):
    try:
        # we should only expect one file, if no, something is wrong
        file_name = glob.glob(os.path.join(file_dir, "*"))[0]
        return os.stat(file_name).st_size
    except Exception:
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
            "workflow_id": os.environ.get("CIRCLE_WORKFLOW_ID"),
        },
        "int": {
            "time": int(time.time()),
            "size": size,
            "commit_time": int(os.environ.get("COMMIT_TIME", "0")),
            "run_duration": int(time.time() - os.path.getmtime(os.path.realpath(__file__))),
        },
    }


def send_message(messages):
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
                    for message in messages
                ]
            ),
        },
    )
    print(r.text)
    r.raise_for_status()


def report_android_sizes(file_dir):
    def gen_sizes():
        # we should only expect one file, if no, something is wrong
        aar_files = list(pathlib.Path(file_dir).rglob("pytorch_android-*.aar"))
        if len(aar_files) != 1:
            logging.exception(f"error getting aar files from: {file_dir} / {aar_files}")
            return

        aar_file = aar_files[0]
        zf = zipfile.ZipFile(aar_file)
        for info in zf.infolist():
            # Scan ".so" libs in `jni` folder. Examples:
            # jni/arm64-v8a/libfbjni.so
            # jni/arm64-v8a/libpytorch_jni.so
            m = re.match(r"^jni/([^/]+)/(.*\.so)$", info.filename)
            if not m:
                continue
            arch, lib = m.groups()
            # report per architecture library size
            yield [arch, lib, info.compress_size, info.file_size]

        # report whole package size
        yield ["aar", aar_file.name, os.stat(aar_file).st_size, 0]

    def gen_messages():
        android_build_type = os.environ.get("ANDROID_BUILD_TYPE")
        for arch, lib, comp_size, uncomp_size in gen_sizes():
            print(android_build_type, arch, lib, comp_size, uncomp_size)
            yield {
                "normal": {
                    "os": "android",
                    # TODO: create dedicated columns
                    "pkg_type": "{}/{}/{}".format(android_build_type, arch, lib),
                    "cu_ver": "",  # dummy value for derived field `build_name`
                    "py_ver": "",  # dummy value for derived field `build_name`
                    "pr": os.environ.get("CIRCLE_PR_NUMBER"),
                    "build_num": os.environ.get("CIRCLE_BUILD_NUM"),
                    "sha1": os.environ.get("CIRCLE_SHA1"),
                    "branch": os.environ.get("CIRCLE_BRANCH"),
                    "workflow_id": os.environ.get("CIRCLE_WORKFLOW_ID"),
                },
                "int": {
                    "time": int(time.time()),
                    "commit_time": int(os.environ.get("COMMIT_TIME", "0")),
                    "run_duration": int(time.time() - os.path.getmtime(os.path.realpath(__file__))),
                    "size": comp_size,
                    "raw_size": uncomp_size,
                },
            }

    send_message(list(gen_messages()))


if __name__ == "__main__":
    file_dir = os.environ.get(
        "PYTORCH_FINAL_PACKAGE_DIR", "/home/circleci/project/final_pkgs"
    )
    if len(sys.argv) == 2:
        file_dir = sys.argv[1]
    print("checking dir: " + file_dir)

    if "-android" in os.environ.get("BUILD_ENVIRONMENT", ""):
        report_android_sizes(file_dir)
    else:
        size = get_size(file_dir)
        if size != 0:
            try:
                send_message([build_message(size)])
            except Exception:
                logging.exception("can't send message")
