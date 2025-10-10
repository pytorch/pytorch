from __future__ import annotations

import os
from enum import Enum
from pathlib import Path


# <project folder>
HOME_DIR = os.environ["HOME"]
TOOLS_FOLDER = str(Path(__file__).resolve().parents[2])


# <profile folder>
PROFILE_DIR = os.path.join(TOOLS_FOLDER, "profile")
JSON_FOLDER_BASE_DIR = os.path.join(PROFILE_DIR, "json")
MERGED_FOLDER_BASE_DIR = os.path.join(PROFILE_DIR, "merged")
SUMMARY_FOLDER_DIR = os.path.join(PROFILE_DIR, "summary")

# <log path>
LOG_DIR = os.path.join(PROFILE_DIR, "log")


# test type, DO NOT change the name, it should be consistent with [buck query --output-attribute] result
class TestType(Enum):
    CPP = "cxx_test"
    PY = "python_test"


class Test:
    name: str
    target_pattern: str
    test_set: str  # like __aten__
    test_type: TestType

    def __init__(
        self, name: str, target_pattern: str, test_set: str, test_type: TestType
    ) -> None:
        self.name = name
        self.target_pattern = target_pattern
        self.test_set = test_set
        self.test_type = test_type


TestList = list[Test]
TestStatusType = dict[str, set[str]]


# option
class Option:
    need_build: bool = False
    need_run: bool = False
    need_merge: bool = False
    need_export: bool = False
    need_summary: bool = False
    need_pytest: bool = False


# test platform
class TestPlatform(Enum):
    FBCODE = "fbcode"
    OSS = "oss"


# compiler type
class CompilerType(Enum):
    CLANG = "clang"
    GCC = "gcc"
