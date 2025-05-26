# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional


"""
This file is not meant to be used directly. It serves as a stub to allow
other files to be safely imported without requiring the installation of
the 'etcd' library. The classes and methods here raise exceptions to
indicate that the real 'etcd' module is needed.
"""


class EtcdStubError(ImportError):
    """Custom exception to indicate that the real etcd module is required."""

    def __init__(self) -> None:
        super().__init__("The 'etcd' module is required but not installed.")


class EtcdAlreadyExist(Exception):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise EtcdStubError


class EtcdCompareFailed(Exception):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise EtcdStubError


class EtcdKeyNotFound(Exception):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise EtcdStubError


class EtcdWatchTimedOut(Exception):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise EtcdStubError


class EtcdEventIndexCleared(Exception):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise EtcdStubError


class EtcdException(Exception):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise EtcdStubError


class EtcdResult:
    def __init__(self) -> None:
        raise EtcdStubError


class Client:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise EtcdStubError

    def read(self, key: str) -> None:
        raise EtcdStubError

    def write(
        self, key: str, value: Any, ttl: Optional[int] = None, **kwargs: Any
    ) -> None:
        raise EtcdStubError

    def test_and_set(
        self, key: str, value: Any, prev_value: Any, ttl: Optional[int] = None
    ) -> None:
        raise EtcdStubError
