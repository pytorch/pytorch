import sys
import pickle
from io import BytesIO
if sys.version_info[0] >= 3:
    import copyreg

import torch

# The code below was copied from joblib (https://github.com/joblib/joblib)
#
# This software is OSI Certified Open Source Software. OSI Certified is a
# certification mark of the Open Source Initiative.
#
# Copyright (c) 2009-2011, joblib developpers All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of Gael Varoquaux. nor the names of other joblib
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# This software is provided by the copyright holders and contributors "as is"
# and any express or implied warranties, including, but not limited to, the
# implied warranties of merchantability and fitness for a particular purpose
# are disclaimed. In no event shall the copyright owner or contributors be
# liable for any direct, indirect, incidental, special, exemplary, or
# consequential damages (including, but not limited to, procurement of
# substitute goods or services; loss of use, data, or profits; or business
# interruption) however caused and on any theory of liability, whether in
# contract, strict liability, or tort (including negligence or otherwise)
# arising in any way out of the use of this software, even if advised of the
# possibility of such damage.


class CustomizablePickler(pickle.Pickler):
    """Pickler that accepts custom reducers.
    HIGHEST_PROTOCOL is selected by default as this pickler is used
    to pickle ephemeral datastructures for interprocess communication
    hence no backward compatibility is required.
    `reducers` is expected to be a dictionary with key/values
    being `(type, callable)` pairs where `callable` is a function that
    give an instance of `type` will return a tuple `(constructor,
    tuple_of_objects)` to rebuild an instance out of the pickled
    `tuple_of_objects` as would return a `__reduce__` method. See the
    standard library documentation on pickling for more details.
    """

    # We override the pure Python pickler as its the only way to be able to
    # customize the dispatch table without side effects in Python 2.6
    # to 3.2. For Python 3.3+ leverage the new dispatch_table
    # feature from http://bugs.python.org/issue14166 that makes it possible
    # to use the C implementation of the Pickler which is faster.

    def __init__(self, writer, reducers=None, protocol=pickle.HIGHEST_PROTOCOL):
        pickle.Pickler.__init__(self, writer, protocol=protocol)
        self.extended_init = set()
        if reducers is None:
            reducers = {}

        if hasattr(pickle.Pickler, 'dispatch'):
            # Make the dispatch registry an instance level attribute instead of
            # a reference to the class dictionary under Python 2
            self.dispatch = pickle.Pickler.dispatch.copy()
        else:
            # Under Python 3 initialize the dispatch table with a copy of the
            # default registry
            self.dispatch_table = copyreg.dispatch_table.copy()

        for type, reduce_func in reducers.items():
            self.register(type, reduce_func)

    def register(self, type, reduce_func):
        """Attach a reducer function to a given type in the dispatch table."""
        if hasattr(pickle.Pickler, 'dispatch'):
            # Python 2 pickler dispatching is not explicitly customizable.
            # Let us use a closure to workaround this limitation.
            def dispatcher(self, obj):
                reduced = reduce_func(self, obj)
                self.save_reduce(obj=obj, *reduced)
            self.dispatch[type] = dispatcher
        else:
            self.dispatch_table[type] = lambda obj: reduce_func(self, obj)


class ExtendedInitPickler(CustomizablePickler):

    def __init__(self, *args, **kwargs):
        CustomizablePickler.__init__(self, *args, **kwargs)
        self.extended_init = set()

    def register_extended_init(self, obj):
        self.extended_init.add(obj)

    def dump(self, obj):
        CustomizablePickler.dump(self, obj)
        CustomizablePickler.dump(self, self.extended_init)


class ExtendedInitUnpickler(pickle.Unpickler):

    def load(self):
        result = pickle.Unpickler.load(self)
        self.extended_init = pickle.Unpickler.load(self)
        return result


class CustomizablePicklingQueue(object):
    """Locked Pipe implementation that uses a customizable pickler.
    This class is an alternative to the multiprocessing implementation
    of SimpleQueue in order to make it possible to pass custom
    pickling reducers, for instance to avoid memory copy when passing
    memory mapped datastructures.
    `reducers` is expected to be a dict with key / values being
    `(type, callable)` pairs where `callable` is a function that, given an
    instance of `type`, will return a tuple `(constructor, tuple_of_objects)`
    to rebuild an instance out of the pickled `tuple_of_objects` as would
    return a `__reduce__` method.
    See the standard library documentation on pickling for more details.
    """

    def __init__(self, context, reducers=None):
        self._reducers = reducers
        self._reader, self._writer = context.Pipe(duplex=False)
        self._rlock = context.Lock()
        if sys.platform == 'win32':
            self._wlock = None
        else:
            self._wlock = context.Lock()
        self._make_methods()

    def __getstate__(self):
        # TODO: assert spawning
        return (self._reader, self._writer, self._rlock, self._wlock,
                self._reducers)

    def __setstate__(self, state):
        (self._reader, self._writer, self._rlock, self._wlock,
         self._reducers) = state
        self._make_methods()

    def empty(self):
        return not self._reader.poll()

    def _load(self):
        return self._reader.recv()

    def _recv(self):
        return self._load()

    def get(self):
        self._rlock.acquire()
        try:
            # TODO: unpickle outside of a read lock
            return self._load()
        finally:
            self._rlock.release()

    def _send(self, obj):
        buffer = BytesIO()
        CustomizablePickler(buffer, self._reducers).dump(obj)
        self._writer.send_bytes(buffer.getvalue())

    def put(self, obj):
        self._wlock.acquire()
        try:
            return self._send(obj)
        finally:
            self._wlock.release()

    def _make_methods(self):
        if self._wlock is None:
            # writes to a message oriented win32 pipe are atomic
            self.put = self.send
