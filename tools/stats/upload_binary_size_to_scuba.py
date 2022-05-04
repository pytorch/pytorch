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

from typing import Any, Dict, Generator, List
from tools.stats.scribe import (
    send_to_scribe,
    rds_write,
    register_rds_schema,
    schema_from_sample,
)


def get_size(file_dir: str) -> int:
    try:
        # we should only expect one file, if no, something is wrong
        file_name = glob.glob(os.path.join(file_dir, "*"))[0]
        return os.stat(file_name).st_size
    except Exception:
        logging.exception(f"error getting file from: {file_dir}")
        return 0


def base_data() -> Dict[str, Any]:
    return {
        "run_duration_seconds": int(
            time.time() - os.path.getmtime(os.path.realpath(__file__))
        ),
    }


def build_message(size: int) -> Dict[str, Any]:
    build_env_split: List[Any] = os.environ.get("BUILD_ENVIRONMENT", "").split()
    pkg_type, py_ver, cu_ver, *_ = build_env_split + [None, None, None]
    os_name = os.uname()[0].lower()
    if os_name == "darwin":
        os_name = "macos"

    return {
        "normal": {
            "os": os_name,
            "pkg_type": pkg_type,
            "py_ver": py_ver,
            "cu_ver": cu_ver,
            "pr": os.environ.get("PR_NUMBER", os.environ.get("CIRCLE_PR_NUMBER")),
            # This is the only place where we use directly CIRCLE_BUILD_NUM, everywhere else CIRCLE_* vars
            # are used as fallback, there seems to be no direct analogy between circle build number and GHA IDs
            "build_num": os.environ.get("CIRCLE_BUILD_NUM"),
            "sha1": os.environ.get("SHA1", os.environ.get("CIRCLE_SHA1")),
            "branch": os.environ.get("BRANCH", os.environ.get("CIRCLE_BRANCH")),
            "workflow_id": os.environ.get(
                "WORKFLOW_ID", os.environ.get("CIRCLE_WORKFLOW_ID")
            ),
        },
        "int": {
            "time": int(time.time()),
            "size": size,
            "commit_time": int(os.environ.get("COMMIT_TIME", "0")),
            "run_duration": int(
                time.time() - os.path.getmtime(os.path.realpath(__file__))
            ),
        },
    }


def send_message(messages: List[Dict[str, Any]]) -> None:
    logs = json.dumps(
        [
            {
                "category": "perfpipe_pytorch_binary_size",
                "message": json.dumps(message),
                "line_escape": False,
            }
            for message in messages
        ]
    )
    res = send_to_scribe(logs)
    print(res)


def report_android_sizes(file_dir: str) -> None:
    def gen_sizes() -> Generator[List[Any], None, None]:
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

    def gen_messages() -> Generator[Dict[str, Any], None, None]:
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
                    "pr": os.environ.get(
                        "PR_NUMBER", os.environ.get("CIRCLE_PR_NUMBER")
                    ),
                    # This is the only place where we use directly CIRCLE_BUILD_NUM, everywhere else CIRCLE_* vars
                    # are used as fallback, there seems to be no direct analogy between circle build number and GHA IDs
                    "build_num": os.environ.get("CIRCLE_BUILD_NUM"),
                    "sha1": os.environ.get("SHA1", os.environ.get("CIRCLE_SHA1")),
                    "branch": os.environ.get("BRANCH", os.environ.get("CIRCLE_BRANCH")),
                    "workflow_id": os.environ.get(
                        "WORKFLOW_ID", os.environ.get("CIRCLE_WORKFLOW_ID")
                    ),
                },
                "int": {
                    "time": int(time.time()),
                    "commit_time": int(os.environ.get("COMMIT_TIME", "0")),
                    "run_duration": int(
                        time.time() - os.path.getmtime(os.path.realpath(__file__))
                    ),
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

    if os.getenv("IS_GHA", "0") == "1":
        sample_lib = {
            "library": "abcd",
            "size": 1234,
        }
        sample_data = {
            **base_data(),
            **sample_lib,
        }
        register_rds_schema("binary_size", schema_from_sample(sample_data))

    if "-android" in os.environ.get("BUILD_ENVIRONMENT", ""):
        report_android_sizes(file_dir)
    else:
        if os.getenv("IS_GHA", "0") == "1":
            build_path = pathlib.Path("build") / "lib"
            libraries = [
                (path.name, os.stat(path).st_size) for path in build_path.glob("*")
            ]
            data = []
            for name, size in libraries:
                if name.strip() == "":
                    continue
                library_data = {
                    "library": name,
                    "size": size,
                }
                data.append({**base_data(), **library_data})
            rds_write("binary_size", data)
            print(json.dumps(data, indent=2))
        else:
            print("checking dir: " + file_dir)
            size = get_size(file_dir)
            # Sending the message anyway if no size info is collected.
            try:
                send_message([build_message(size)])
            except Exception:
                logging.exception("can't send message")
