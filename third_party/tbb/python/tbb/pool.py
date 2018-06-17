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

# @brief Python Pool implementation based on TBB with monkey-patching
#
# See http://docs.python.org/dev/library/multiprocessing.html
# Differences: added imap_async and imap_unordered_async, and terminate()
# has to be called explicitly (it's not registered by atexit).
#
# The general idea is that we submit works to a workqueue, either as
# single Jobs (one function to call), or JobSequences (batch of
# Jobs). Each Job is associated with an ApplyResult object which has 2
# states: waiting for the Job to complete, or Ready. Instead of
# waiting for the jobs to finish, we wait for their ApplyResult object
# to become ready: an event mechanism is used for that.
# When we apply a function to several arguments in "parallel", we need
# a way to wait for all/part of the Jobs to be processed: that's what
# "collectors" are for; they group and wait for a set of ApplyResult
# objects. Once a collector is ready to be used, we can use a
# CollectorIterator to iterate over the result values it's collecting.
#
# The methods of a Pool object use all these concepts and expose
# them to their caller in a very simple way.

import sys
import threading
import traceback
from .api import *

__all__ = ["Pool", "TimeoutError"]
__doc__ = """
Standard Python Pool implementation based on Python API
for Intel(R) Threading Building Blocks library (Intel(R) TBB)
"""


class TimeoutError(Exception):
    """Raised when a result is not available within the given timeout"""
    pass


class Pool(object):
    """
    The Pool class provides standard multiprocessing.Pool interface
    which is mapped onto Intel(R) TBB tasks executing in its thread pool
    """

    def __init__(self, nworkers=0, name="Pool"):
        """
        \param nworkers (integer) number of worker threads to start
        \param name (string) prefix for the worker threads' name
        """
        self._closed = False
        self._tasks = task_group()
        self._pool = [None,]*default_num_threads()  # Dask asks for len(_pool)

    def apply(self, func, args=(), kwds=dict()):
        """Equivalent of the apply() builtin function. It blocks till
        the result is ready."""
        return self.apply_async(func, args, kwds).get()

    def map(self, func, iterable, chunksize=None):
        """A parallel equivalent of the map() builtin function. It
        blocks till the result is ready.

        This method chops the iterable into a number of chunks which
        it submits to the process pool as separate tasks. The
        (approximate) size of these chunks can be specified by setting
        chunksize to a positive integer."""
        return self.map_async(func, iterable, chunksize).get()

    def imap(self, func, iterable, chunksize=1):
        """
        An equivalent of itertools.imap().

        The chunksize argument is the same as the one used by the
        map() method. For very long iterables using a large value for
        chunksize can make the job complete much faster than
        using the default value of 1.

        Also if chunksize is 1 then the next() method of the iterator
        returned by the imap() method has an optional timeout
        parameter: next(timeout) will raise processing.TimeoutError if
        the result cannot be returned within timeout seconds.
        """
        collector = OrderedResultCollector(as_iterator=True)
        self._create_sequences(func, iterable, chunksize, collector)
        return iter(collector)

    def imap_unordered(self, func, iterable, chunksize=1):
        """The same as imap() except that the ordering of the results
        from the returned iterator should be considered
        arbitrary. (Only when there is only one worker process is the
        order guaranteed to be "correct".)"""
        collector = UnorderedResultCollector()
        self._create_sequences(func, iterable, chunksize, collector)
        return iter(collector)

    def apply_async(self, func, args=(), kwds=dict(), callback=None):
        """A variant of the apply() method which returns an
        ApplyResult object.

        If callback is specified then it should be a callable which
        accepts a single argument. When the result becomes ready,
        callback is applied to it (unless the call failed). callback
        should complete immediately since otherwise the thread which
        handles the results will get blocked."""
        assert not self._closed  # No lock here. We assume it's atomic...
        apply_result = ApplyResult(callback=callback)
        job = Job(func, args, kwds, apply_result)
        self._tasks.run(job)
        return apply_result

    def map_async(self, func, iterable, chunksize=None, callback=None):
        """A variant of the map() method which returns a ApplyResult
        object.

        If callback is specified then it should be a callable which
        accepts a single argument. When the result becomes ready
        callback is applied to it (unless the call failed). callback
        should complete immediately since otherwise the thread which
        handles the results will get blocked."""
        apply_result = ApplyResult(callback=callback)
        collector    = OrderedResultCollector(apply_result, as_iterator=False)
        if not self._create_sequences(func, iterable, chunksize, collector):
          apply_result._set_value([])
        return apply_result

    def imap_async(self, func, iterable, chunksize=None, callback=None):
        """A variant of the imap() method which returns an ApplyResult
        object that provides an iterator (next method(timeout)
        available).

        If callback is specified then it should be a callable which
        accepts a single argument. When the resulting iterator becomes
        ready, callback is applied to it (unless the call
        failed). callback should complete immediately since otherwise
        the thread which handles the results will get blocked."""
        apply_result = ApplyResult(callback=callback)
        collector    = OrderedResultCollector(apply_result, as_iterator=True)
        if not self._create_sequences(func, iterable, chunksize, collector):
          apply_result._set_value(iter([]))
        return apply_result

    def imap_unordered_async(self, func, iterable, chunksize=None,
                             callback=None):
        """A variant of the imap_unordered() method which returns an
        ApplyResult object that provides an iterator (next
        method(timeout) available).

        If callback is specified then it should be a callable which
        accepts a single argument. When the resulting iterator becomes
        ready, callback is applied to it (unless the call
        failed). callback should complete immediately since otherwise
        the thread which handles the results will get blocked."""
        apply_result = ApplyResult(callback=callback)
        collector    = UnorderedResultCollector(apply_result)
        if not self._create_sequences(func, iterable, chunksize, collector):
          apply_result._set_value(iter([]))
        return apply_result

    def close(self):
        """Prevents any more tasks from being submitted to the
        pool. Once all the tasks have been completed the worker
        processes will exit."""
        # No lock here. We assume it's sufficiently atomic...
        self._closed = True

    def terminate(self):
        """Stops the worker processes immediately without completing
        outstanding work. When the pool object is garbage collected
        terminate() will be called immediately."""
        self.close()
        self._tasks.cancel()

    def join(self):
        """Wait for the worker processes to exit. One must call
        close() or terminate() before using join()."""
        self._tasks.wait()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.join()

    def __del__(self):
        self.terminate()
        self.join()

    def _create_sequences(self, func, iterable, chunksize, collector):
        """
        Create callable objects to process and pushes them on the
        work queue. Each work unit is meant to process a slice of
        iterable of size chunksize. If collector is specified, then
        the ApplyResult objects associated with the jobs will notify
        collector when their result becomes ready.

        \return the list callable objects (basically: JobSequences)
        pushed onto the work queue
        """
        assert not self._closed  # No lock here. We assume it's atomic...
        it_ = iter(iterable)
        exit_loop = False
        sequences = []
        while not exit_loop:
            seq = []
            for _ in range(chunksize or 1):
                try:
                    arg = next(it_)
                except StopIteration:
                    exit_loop = True
                    break
                apply_result = ApplyResult(collector)
                job = Job(func, (arg,), {}, apply_result)
                seq.append(job)
            if seq:
                sequences.append(JobSequence(seq))
        for t in sequences:
            self._tasks.run(t)
        return sequences


class Job:
    """A work unit that corresponds to the execution of a single function"""

    def __init__(self, func, args, kwds, apply_result):
        """
        \param func/args/kwds used to call the function
        \param apply_result ApplyResult object that holds the result
        of the function call
        """
        self._func = func
        self._args = args
        self._kwds = kwds
        self._result = apply_result

    def __call__(self):
        """
        Call the function with the args/kwds and tell the ApplyResult
        that its result is ready. Correctly handles the exceptions
        happening during the execution of the function
        """
        try:
            result = self._func(*self._args, **self._kwds)
        except:
            self._result._set_exception()
        else:
            self._result._set_value(result)


class JobSequence:
    """A work unit that corresponds to the processing of a continuous
    sequence of Job objects"""

    def __init__(self, jobs):
        self._jobs = jobs

    def __call__(self):
        """
        Call all the Job objects that have been specified
        """
        for job in self._jobs:
            job()


class ApplyResult(object):
    """An object associated with a Job object that holds its result:
    it's available during the whole life the Job and after, even when
    the Job didn't process yet. It's possible to use this object to
    wait for the result/exception of the job to be available.

    The result objects returns by the Pool::*_async() methods are of
    this type"""

    def __init__(self, collector=None, callback=None):
        """
        \param collector when not None, the notify_ready() method of
        the collector will be called when the result from the Job is
        ready
        \param callback when not None, function to call when the
        result becomes available (this is the paramater passed to the
        Pool::*_async() methods.
        """
        self._success = False
        self._event = threading.Event()
        self._data = None
        self._collector = None
        self._callback = callback

        if collector is not None:
            collector.register_result(self)
            self._collector = collector

    def get(self, timeout=None):
        """
        Returns the result when it arrives. If timeout is not None and
        the result does not arrive within timeout seconds then
        TimeoutError is raised. If the remote call raised an exception
        then that exception will be reraised by get().
        """
        if not self.wait(timeout):
            raise TimeoutError("Result not available within %fs" % timeout)
        if self._success:
            return self._data
        if sys.version_info[0] == 3:
            raise self._data[0](self._data[1]).with_traceback(self._data[2])
        else:
            exec("raise self._data[0], self._data[1], self._data[2]")

    def wait(self, timeout=None):
        """Waits until the result is available or until timeout
        seconds pass."""
        self._event.wait(timeout)
        return self._event.isSet()

    def ready(self):
        """Returns whether the call has completed."""
        return self._event.isSet()

    def successful(self):
        """Returns whether the call completed without raising an
        exception. Will raise AssertionError if the result is not
        ready."""
        assert self.ready()
        return self._success

    def _set_value(self, value):
        """Called by a Job object to tell the result is ready, and
        provides the value of this result. The object will become
        ready and successful. The collector's notify_ready() method
        will be called, and the callback method too"""
        assert not self.ready()
        self._data = value
        self._success = True
        self._event.set()
        if self._collector is not None:
            self._collector.notify_ready(self)
        if self._callback is not None:
            try:
                self._callback(value)
            except:
                traceback.print_exc()

    def _set_exception(self):
        """Called by a Job object to tell that an exception occurred
        during the processing of the function. The object will become
        ready but not successful. The collector's notify_ready()
        method will be called, but NOT the callback method"""
        # traceback.print_exc()
        assert not self.ready()
        self._data = sys.exc_info()
        self._success = False
        self._event.set()
        if self._collector is not None:
            self._collector.notify_ready(self)


class AbstractResultCollector(object):
    """ABC to define the interface of a ResultCollector object. It is
    basically an object which knows whuich results it's waiting for,
    and which is able to get notify when they get available. It is
    also able to provide an iterator over the results when they are
    available"""

    def __init__(self, to_notify):
        """
        \param to_notify ApplyResult object to notify when all the
        results we're waiting for become available. Can be None.
        """
        self._to_notify = to_notify

    def register_result(self, apply_result):
        """Used to identify which results we're waiting for. Will
        always be called BEFORE the Jobs get submitted to the work
        queue, and BEFORE the __iter__ and _get_result() methods can
        be called
        \param apply_result ApplyResult object to add in our collection
        """
        raise NotImplementedError("Children classes must implement it")

    def notify_ready(self, apply_result):
        """Called by the ApplyResult object (already registered via
        register_result()) that it is now ready (ie. the Job's result
        is available or an exception has been raised).
        \param apply_result ApplyResult object telling us that the job
        has been processed
        """
        raise NotImplementedError("Children classes must implement it")

    def _get_result(self, idx, timeout=None):
        """Called by the CollectorIterator object to retrieve the
        result's values one after another (order defined by the
        implementation)
        \param idx The index of the result we want, wrt collector's order
        \param timeout integer telling how long to wait (in seconds)
        for the result at index idx to be available, or None (wait
        forever)
        """
        raise NotImplementedError("Children classes must implement it")

    def __iter__(self):
        """Return a new CollectorIterator object for this collector"""
        return CollectorIterator(self)


class CollectorIterator(object):
    """An iterator that allows to iterate over the result values
    available in the given collector object. Equipped with an extended
    next() method accepting a timeout argument. Created by the
    AbstractResultCollector::__iter__() method"""

    def __init__(self, collector):
        """\param AbstractResultCollector instance"""
        self._collector = collector
        self._idx = 0

    def __iter__(self):
        return self

    def next(self, timeout=None):
        """Return the next result value in the sequence. Raise
        StopIteration at the end. Can raise the exception raised by
        the Job"""
        try:
            apply_result = self._collector._get_result(self._idx, timeout)
        except IndexError:
            # Reset for next time
            self._idx = 0
            raise StopIteration
        except:
            self._idx = 0
            raise
        self._idx += 1
        assert apply_result.ready()
        return apply_result.get(0)

    def __next__(self):
        return self.next()


class UnorderedResultCollector(AbstractResultCollector):
    """An AbstractResultCollector implementation that collects the
    values of the ApplyResult objects in the order they become ready. The
    CollectorIterator object returned by __iter__() will iterate over
    them in the order they become ready"""

    def __init__(self, to_notify=None):
        """
        \param to_notify ApplyResult object to notify when all the
        results we're waiting for become available. Can be None.
        """
        AbstractResultCollector.__init__(self, to_notify)
        self._cond = threading.Condition()
        self._collection = []
        self._expected = 0

    def register_result(self, apply_result):
        """Used to identify which results we're waiting for. Will
        always be called BEFORE the Jobs get submitted to the work
        queue, and BEFORE the __iter__ and _get_result() methods can
        be called
        \param apply_result ApplyResult object to add in our collection
        """
        self._expected += 1

    def _get_result(self, idx, timeout=None):
        """Called by the CollectorIterator object to retrieve the
        result's values one after another, in the order the results have
        become available.
        \param idx The index of the result we want, wrt collector's order
        \param timeout integer telling how long to wait (in seconds)
        for the result at index idx to be available, or None (wait
        forever)
        """
        self._cond.acquire()
        try:
            if idx >= self._expected:
                raise IndexError
            elif idx < len(self._collection):
                return self._collection[idx]
            elif idx != len(self._collection):
                # Violation of the sequence protocol
                raise IndexError()
            else:
                self._cond.wait(timeout=timeout)
                try:
                    return self._collection[idx]
                except IndexError:
                    # Still not added !
                    raise TimeoutError("Timeout while waiting for results")
        finally:
            self._cond.release()

    def notify_ready(self, apply_result=None):
        """Called by the ApplyResult object (already registered via
        register_result()) that it is now ready (ie. the Job's result
        is available or an exception has been raised).
        \param apply_result ApplyResult object telling us that the job
        has been processed
        """
        first_item = False
        self._cond.acquire()
        try:
            self._collection.append(apply_result)
            first_item = (len(self._collection) == 1)

            self._cond.notifyAll()
        finally:
            self._cond.release()

        if first_item and self._to_notify is not None:
            self._to_notify._set_value(iter(self))


class OrderedResultCollector(AbstractResultCollector):
    """An AbstractResultCollector implementation that collects the
    values of the ApplyResult objects in the order they have been
    submitted. The CollectorIterator object returned by __iter__()
    will iterate over them in the order they have been submitted"""

    def __init__(self, to_notify=None, as_iterator=True):
        """
        \param to_notify ApplyResult object to notify when all the
        results we're waiting for become available. Can be None.
        \param as_iterator boolean telling whether the result value
        set on to_notify should be an iterator (available as soon as 1
        result arrived) or a list (available only after the last
        result arrived)
        """
        AbstractResultCollector.__init__(self, to_notify)
        self._results = []
        self._lock = threading.Lock()
        self._remaining = 0
        self._as_iterator = as_iterator

    def register_result(self, apply_result):
        """Used to identify which results we're waiting for. Will
        always be called BEFORE the Jobs get submitted to the work
        queue, and BEFORE the __iter__ and _get_result() methods can
        be called
        \param apply_result ApplyResult object to add in our collection
        """
        self._results.append(apply_result)
        self._remaining += 1

    def _get_result(self, idx, timeout=None):
        """Called by the CollectorIterator object to retrieve the
        result's values one after another (order defined by the
        implementation)
        \param idx The index of the result we want, wrt collector's order
        \param timeout integer telling how long to wait (in seconds)
        for the result at index idx to be available, or None (wait
        forever)
        """
        res = self._results[idx]
        res.wait(timeout)
        return res

    def notify_ready(self, apply_result):
        """Called by the ApplyResult object (already registered via
        register_result()) that it is now ready (ie. the Job's result
        is available or an exception has been raised).
        \param apply_result ApplyResult object telling us that the job
        has been processed
        """
        got_first = False
        got_last = False
        self._lock.acquire()
        try:
            assert self._remaining > 0
            got_first = (len(self._results) == self._remaining)
            self._remaining -= 1
            got_last = (self._remaining == 0)
        finally:
            self._lock.release()

        if self._to_notify is not None:
            if self._as_iterator and got_first:
                self._to_notify._set_value(iter(self))
            elif not self._as_iterator and got_last:
                try:
                    lst = [r.get(0) for r in self._results]
                except:
                    self._to_notify._set_exception()
                else:
                    self._to_notify._set_value(lst)
