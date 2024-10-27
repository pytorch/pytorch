"""Indirection for time functions.

We intentionally grab some "time" functions internally to avoid tests mocking "time" to affect
pytest runtime information (issue #185).

Fixture "mock_timing" also interacts with this module for pytest's own tests.
"""

from __future__ import annotations

from time import perf_counter
from time import sleep
from time import time


__all__ = ["perf_counter", "sleep", "time"]
