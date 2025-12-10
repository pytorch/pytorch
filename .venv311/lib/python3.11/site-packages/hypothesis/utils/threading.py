# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import threading
from collections.abc import Callable
from typing import Any


class ThreadLocal:
    """
    Manages thread-local state. ThreadLocal forwards getattr and setattr to a
    threading.local() instance. The passed kwargs defines the available attributes
    on the threadlocal and their default values.

    The only supported names to geattr and setattr are the keys of the passed kwargs.
    """

    def __init__(self, **kwargs: Callable) -> None:
        for name, value in kwargs.items():
            if not callable(value):
                raise TypeError(f"Attribute {name} must be a callable. Got {value}")

        self.__initialized = False
        self.__kwargs = kwargs
        self.__threadlocal = threading.local()
        self.__initialized = True

    def __getattr__(self, name: str) -> Any:
        if name not in self.__kwargs:
            raise AttributeError(f"No attribute {name}")

        if not hasattr(self.__threadlocal, name):
            default = self.__kwargs[name]()
            setattr(self.__threadlocal, name, default)

        return getattr(self.__threadlocal, name)

    def __setattr__(self, name: str, value: Any) -> None:
        # disable attribute-forwarding while initializing
        if "_ThreadLocal__initialized" not in self.__dict__ or not self.__initialized:
            super().__setattr__(name, value)
        else:
            if name not in self.__kwargs:
                raise AttributeError(f"No attribute {name}")
            setattr(self.__threadlocal, name, value)
