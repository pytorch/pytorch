#!/usr/bin/env python
#
# Copyright (c) 2016-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
#
#

# Based on the software developed by:
# Copyright (c) 2008,2016 david decotigny (Pool of threads)
# Copyright (c) 2006-2008, R Oudkerk (multiprocessing.Pool)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of author nor the names of any contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
#

from __future__ import print_function
import time
import threading

from .api import *
from .pool import *


def test(arg=None):
    if arg == "-v":
        def say(*x):
            print(*x)
    else:
        def say(*x):
            pass
    say("Start Pool testing")

    get_tid = lambda: threading.current_thread().ident

    def return42():
        return 42

    def f(x):
        return x * x

    def work(mseconds):
        res = str(mseconds)
        if mseconds < 0:
            mseconds = -mseconds
        say("[%d] Start to work for %fms..." % (get_tid(), mseconds*10))
        time.sleep(mseconds/100.)
        say("[%d] Work done (%fms)." % (get_tid(), mseconds*10))
        return res

    ### Test copy/pasted from multiprocessing
    pool = Pool(4)  # start worker threads

    # edge cases
    assert pool.map(return42, []) == []
    assert pool.apply_async(return42, []).get() == 42
    assert pool.apply(return42, []) == 42
    assert list(pool.imap(return42, iter([]))) == []
    assert list(pool.imap_unordered(return42, iter([]))) == []
    assert pool.map_async(return42, []).get() == []
    assert list(pool.imap_async(return42, iter([])).get()) == []
    assert list(pool.imap_unordered_async(return42, iter([])).get()) == []

    # basic tests
    result = pool.apply_async(f, (10,))  # evaluate "f(10)" asynchronously
    assert result.get(timeout=1) == 100  # ... unless slow computer
    assert list(pool.map(f, range(10))) == list(map(f, range(10)))
    it = pool.imap(f, range(10))
    assert next(it) == 0
    assert next(it) == 1
    assert next(it) == 4

    # Test apply_sync exceptions
    result = pool.apply_async(time.sleep, (3,))
    try:
        say(result.get(timeout=1))  # raises `TimeoutError`
    except TimeoutError:
        say("Good. Got expected timeout exception.")
    else:
        assert False, "Expected exception !"
    assert result.get() is None  # sleep() returns None

    def cb(s):
        say("Result ready: %s" % s)

    # Test imap()
    assert list(pool.imap(work, range(10, 3, -1), chunksize=4)) == list(map(
        str, range(10, 3, -1)))

    # Test imap_unordered()
    assert sorted(pool.imap_unordered(work, range(10, 3, -1))) == sorted(map(
        str, range(10, 3, -1)))

    # Test map_async()
    result = pool.map_async(work, range(10), callback=cb)
    try:
        result.get(timeout=0.01)  # raises `TimeoutError`
    except TimeoutError:
        say("Good. Got expected timeout exception.")
    else:
        assert False, "Expected exception !"
    say(result.get())

    # Test imap_async()
    result = pool.imap_async(work, range(3, 10), callback=cb)
    try:
        result.get(timeout=0.01)  # raises `TimeoutError`
    except TimeoutError:
        say("Good. Got expected timeout exception.")
    else:
        assert False, "Expected exception !"
    for i in result.get():
        say("Item:", i)
    say("### Loop again:")
    for i in result.get():
        say("Item2:", i)

    # Test imap_unordered_async()
    result = pool.imap_unordered_async(work, range(10, 3, -1), callback=cb)
    try:
        say(result.get(timeout=0.01))  # raises `TimeoutError`
    except TimeoutError:
        say("Good. Got expected timeout exception.")
    else:
        assert False, "Expected exception !"
    for i in result.get():
        say("Item1:", i)
    for i in result.get():
        say("Item2:", i)
    r = result.get()
    for i in r:
        say("Item3:", i)
    for i in r:
        say("Item4:", i)
    for i in r:
        say("Item5:", i)

    #
    # The case for the exceptions
    #

    # Exceptions in imap_unordered_async()
    result = pool.imap_unordered_async(work, range(2, -10, -1), callback=cb)
    time.sleep(3)
    try:
        for i in result.get():
            say("Got item:", i)
    except (IOError, ValueError):
        say("Good. Got expected exception")

    # Exceptions in imap_async()
    result = pool.imap_async(work, range(2, -10, -1), callback=cb)
    time.sleep(3)
    try:
        for i in result.get():
            say("Got item:", i)
    except (IOError, ValueError):
        say("Good. Got expected exception")

    # Stop the test: need to stop the pool !!!
    pool.terminate()
    pool.join()


