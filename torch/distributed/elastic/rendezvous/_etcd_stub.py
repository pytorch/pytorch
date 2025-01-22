# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


class EtcdStubError(ImportError):
    """Custom exception to indicate that the real etcd module is required."""

    def __init__(self):
        super().__init__("The 'etcd' module is required but not installed.")


class EtcdAlreadyExist(Exception):
    def __init__(self, *args, **kwargs):
        raise EtcdStubError()


class EtcdCompareFailed(Exception):
    def __init__(self, *args, **kwargs):
        raise EtcdStubError()


class EtcdKeyNotFound(Exception):
    def __init__(self, *args, **kwargs):
        raise EtcdStubError()


class EtcdWatchTimedOut(Exception):
    def __init__(self, *args, **kwargs):
        raise EtcdStubError()


class EtcdEventIndexCleared(Exception):
    def __init__(self, *args, **kwargs):
        raise EtcdStubError()


class EtcdException(Exception):
    def __init__(self, *args, **kwargs):
        raise EtcdStubError()


class EtcdResult:
    def __init__(self):
        raise EtcdStubError()


class Client:
    def __init__(self, *args, **kwargs):
        raise EtcdStubError()

    def read(self, key):
        raise EtcdStubError()

    def write(self, key, value, ttl=None, **kwargs):
        raise EtcdStubError()

    def test_and_set(self, key, value, prev_value, ttl=None):
        raise EtcdStubError()
