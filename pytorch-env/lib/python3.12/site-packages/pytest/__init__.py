# PYTHON_ARGCOMPLETE_OK
"""pytest: unit and functional testing with Python."""

from __future__ import annotations

from _pytest import __version__
from _pytest import version_tuple
from _pytest._code import ExceptionInfo
from _pytest.assertion import register_assert_rewrite
from _pytest.cacheprovider import Cache
from _pytest.capture import CaptureFixture
from _pytest.config import cmdline
from _pytest.config import Config
from _pytest.config import console_main
from _pytest.config import ExitCode
from _pytest.config import hookimpl
from _pytest.config import hookspec
from _pytest.config import main
from _pytest.config import PytestPluginManager
from _pytest.config import UsageError
from _pytest.config.argparsing import OptionGroup
from _pytest.config.argparsing import Parser
from _pytest.debugging import pytestPDB as __pytestPDB
from _pytest.doctest import DoctestItem
from _pytest.fixtures import fixture
from _pytest.fixtures import FixtureDef
from _pytest.fixtures import FixtureLookupError
from _pytest.fixtures import FixtureRequest
from _pytest.fixtures import yield_fixture
from _pytest.freeze_support import freeze_includes
from _pytest.legacypath import TempdirFactory
from _pytest.legacypath import Testdir
from _pytest.logging import LogCaptureFixture
from _pytest.main import Dir
from _pytest.main import Session
from _pytest.mark import Mark
from _pytest.mark import MARK_GEN as mark
from _pytest.mark import MarkDecorator
from _pytest.mark import MarkGenerator
from _pytest.mark import param
from _pytest.monkeypatch import MonkeyPatch
from _pytest.nodes import Collector
from _pytest.nodes import Directory
from _pytest.nodes import File
from _pytest.nodes import Item
from _pytest.outcomes import exit
from _pytest.outcomes import fail
from _pytest.outcomes import importorskip
from _pytest.outcomes import skip
from _pytest.outcomes import xfail
from _pytest.pytester import HookRecorder
from _pytest.pytester import LineMatcher
from _pytest.pytester import Pytester
from _pytest.pytester import RecordedHookCall
from _pytest.pytester import RunResult
from _pytest.python import Class
from _pytest.python import Function
from _pytest.python import Metafunc
from _pytest.python import Module
from _pytest.python import Package
from _pytest.python_api import approx
from _pytest.python_api import raises
from _pytest.recwarn import deprecated_call
from _pytest.recwarn import WarningsRecorder
from _pytest.recwarn import warns
from _pytest.reports import CollectReport
from _pytest.reports import TestReport
from _pytest.runner import CallInfo
from _pytest.stash import Stash
from _pytest.stash import StashKey
from _pytest.terminal import TestShortLogReport
from _pytest.tmpdir import TempPathFactory
from _pytest.warning_types import PytestAssertRewriteWarning
from _pytest.warning_types import PytestCacheWarning
from _pytest.warning_types import PytestCollectionWarning
from _pytest.warning_types import PytestConfigWarning
from _pytest.warning_types import PytestDeprecationWarning
from _pytest.warning_types import PytestExperimentalApiWarning
from _pytest.warning_types import PytestRemovedIn9Warning
from _pytest.warning_types import PytestReturnNotNoneWarning
from _pytest.warning_types import PytestUnhandledCoroutineWarning
from _pytest.warning_types import PytestUnhandledThreadExceptionWarning
from _pytest.warning_types import PytestUnknownMarkWarning
from _pytest.warning_types import PytestUnraisableExceptionWarning
from _pytest.warning_types import PytestWarning


set_trace = __pytestPDB.set_trace


__all__ = [
    "__version__",
    "approx",
    "Cache",
    "CallInfo",
    "CaptureFixture",
    "Class",
    "cmdline",
    "Collector",
    "CollectReport",
    "Config",
    "console_main",
    "deprecated_call",
    "Dir",
    "Directory",
    "DoctestItem",
    "exit",
    "ExceptionInfo",
    "ExitCode",
    "fail",
    "File",
    "fixture",
    "FixtureDef",
    "FixtureLookupError",
    "FixtureRequest",
    "freeze_includes",
    "Function",
    "hookimpl",
    "HookRecorder",
    "hookspec",
    "importorskip",
    "Item",
    "LineMatcher",
    "LogCaptureFixture",
    "main",
    "mark",
    "Mark",
    "MarkDecorator",
    "MarkGenerator",
    "Metafunc",
    "Module",
    "MonkeyPatch",
    "OptionGroup",
    "Package",
    "param",
    "Parser",
    "PytestAssertRewriteWarning",
    "PytestCacheWarning",
    "PytestCollectionWarning",
    "PytestConfigWarning",
    "PytestDeprecationWarning",
    "PytestExperimentalApiWarning",
    "PytestRemovedIn9Warning",
    "PytestReturnNotNoneWarning",
    "Pytester",
    "PytestPluginManager",
    "PytestUnhandledCoroutineWarning",
    "PytestUnhandledThreadExceptionWarning",
    "PytestUnknownMarkWarning",
    "PytestUnraisableExceptionWarning",
    "PytestWarning",
    "raises",
    "RecordedHookCall",
    "register_assert_rewrite",
    "RunResult",
    "Session",
    "set_trace",
    "skip",
    "Stash",
    "StashKey",
    "version_tuple",
    "TempdirFactory",
    "TempPathFactory",
    "Testdir",
    "TestReport",
    "TestShortLogReport",
    "UsageError",
    "WarningsRecorder",
    "warns",
    "xfail",
    "yield_fixture",
]
