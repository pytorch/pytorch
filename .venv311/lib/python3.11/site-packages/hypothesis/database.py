# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import abc
import errno
import json
import os
import struct
import sys
import tempfile
import warnings
import weakref
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from hashlib import sha384
from os import PathLike, getenv
from pathlib import Path, PurePath
from queue import Queue
from threading import Thread
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    TypeAlias,
    cast,
)
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

from hypothesis._settings import note_deprecation
from hypothesis.configuration import storage_directory
from hypothesis.errors import HypothesisException, HypothesisWarning
from hypothesis.internal.conjecture.choice import ChoiceT
from hypothesis.utils.conventions import UniqueIdentifier, not_set

__all__ = [
    "DirectoryBasedExampleDatabase",
    "ExampleDatabase",
    "GitHubArtifactDatabase",
    "InMemoryExampleDatabase",
    "MultiplexedDatabase",
    "ReadOnlyDatabase",
]

if TYPE_CHECKING:
    from watchdog.observers.api import BaseObserver

StrPathT: TypeAlias = str | PathLike[str]
SaveDataT: TypeAlias = tuple[bytes, bytes]  # key, value
DeleteDataT: TypeAlias = tuple[bytes, bytes | None]  # key, value
ListenerEventT: TypeAlias = (
    tuple[Literal["save"], SaveDataT] | tuple[Literal["delete"], DeleteDataT]
)
ListenerT: TypeAlias = Callable[[ListenerEventT], Any]


def _usable_dir(path: StrPathT) -> bool:
    """
    Returns True if the desired path can be used as database path because
    either the directory exists and can be used, or its root directory can
    be used and we can make the directory as needed.
    """
    path = Path(path)
    try:
        while not path.exists():
            # Loop terminates because the root dir ('/' on unix) always exists.
            path = path.parent
        return path.is_dir() and os.access(path, os.R_OK | os.W_OK | os.X_OK)
    except PermissionError:
        return False


def _db_for_path(
    path: StrPathT | UniqueIdentifier | Literal[":memory:"] | None = None,
) -> "ExampleDatabase":
    if path is not_set:
        if os.getenv("HYPOTHESIS_DATABASE_FILE") is not None:  # pragma: no cover
            raise HypothesisException(
                "The $HYPOTHESIS_DATABASE_FILE environment variable no longer has any "
                "effect.  Configure your database location via a settings profile instead.\n"
                "https://hypothesis.readthedocs.io/en/latest/settings.html#settings-profiles"
            )

        path = storage_directory("examples", intent_to_write=False)
        if not _usable_dir(path):  # pragma: no cover
            warnings.warn(
                "The database setting is not configured, and the default "
                "location is unusable - falling back to an in-memory "
                f"database for this session.  {path=}",
                HypothesisWarning,
                stacklevel=3,
            )
            return InMemoryExampleDatabase()
    if path in (None, ":memory:"):
        return InMemoryExampleDatabase()
    path = cast(StrPathT, path)
    return DirectoryBasedExampleDatabase(path)


class _EDMeta(abc.ABCMeta):
    def __call__(self, *args: Any, **kwargs: Any) -> "ExampleDatabase":
        if self is ExampleDatabase:
            note_deprecation(
                "Creating a database using the abstract ExampleDatabase() class "
                "is deprecated. Prefer using a concrete subclass, like "
                "InMemoryExampleDatabase() or DirectoryBasedExampleDatabase(path). "
                'In particular, the special string ExampleDatabase(":memory:") '
                "should be replaced by InMemoryExampleDatabase().",
                since="2025-04-07",
                has_codemod=False,
            )
            return _db_for_path(*args, **kwargs)
        return super().__call__(*args, **kwargs)


# This __call__ method is picked up by Sphinx as the signature of all ExampleDatabase
# subclasses, which is accurate, reasonable, and unhelpful.  Fortunately Sphinx
# maintains a list of metaclass-call-methods to ignore, and while they would prefer
# not to maintain it upstream (https://github.com/sphinx-doc/sphinx/pull/8262) we
# can insert ourselves here.
#
# This code only runs if Sphinx has already been imported; and it would live in our
# docs/conf.py except that we would also like it to work for anyone documenting
# downstream ExampleDatabase subclasses too.
if "sphinx" in sys.modules:
    try:
        import sphinx.ext.autodoc

        signature = "hypothesis.database._EDMeta.__call__"
        # _METACLASS_CALL_BLACKLIST is a frozenset in later sphinx versions
        if isinstance(sphinx.ext.autodoc._METACLASS_CALL_BLACKLIST, frozenset):
            sphinx.ext.autodoc._METACLASS_CALL_BLACKLIST = (
                sphinx.ext.autodoc._METACLASS_CALL_BLACKLIST | {signature}
            )
        else:
            sphinx.ext.autodoc._METACLASS_CALL_BLACKLIST.append(signature)
    except Exception:
        pass


class ExampleDatabase(metaclass=_EDMeta):
    """
    A Hypothesis database, for use in |settings.database|.

    Hypothesis automatically saves failures to the database set in
    |settings.database|. The next time the test is run, Hypothesis will replay
    any failures from the database in |settings.database| for that test (in
    |Phase.reuse|).

    The database is best thought of as a cache that you never need to invalidate.
    Entries may be transparently dropped when upgrading your Hypothesis version
    or changing your test. Do not rely on the database for correctness; to ensure
    Hypothesis always tries an input, use |@example|.

    A Hypothesis database is a simple mapping of bytes to sets of bytes. Hypothesis
    provides several concrete database subclasses. To write your own database class,
    see :doc:`/how-to/custom-database`.

    Change listening
    ----------------

    An optional extension to |ExampleDatabase| is change listening. On databases
    which support change listening, calling |ExampleDatabase.add_listener| adds
    a function as a change listener, which will be called whenever a value is
    added, deleted, or moved inside the database. See |ExampleDatabase.add_listener|
    for details.

    All databases in Hypothesis support change listening. Custom database classes
    are not required to support change listening, though they will not be compatible
    with features that require change listening until they do so.

    .. note::

        While no Hypothesis features currently require change listening, change
        listening is required by `HypoFuzz <https://hypofuzz.com/>`_.

    Database methods
    ----------------

    Required methods:

    * |ExampleDatabase.save|
    * |ExampleDatabase.fetch|
    * |ExampleDatabase.delete|

    Optional methods:

    * |ExampleDatabase.move|

    Change listening methods:

    * |ExampleDatabase.add_listener|
    * |ExampleDatabase.remove_listener|
    * |ExampleDatabase.clear_listeners|
    * |ExampleDatabase._start_listening|
    * |ExampleDatabase._stop_listening|
    * |ExampleDatabase._broadcast_change|
    """

    def __init__(self) -> None:
        self._listeners: list[ListenerT] = []

    @abc.abstractmethod
    def save(self, key: bytes, value: bytes) -> None:
        """Save ``value`` under ``key``.

        If ``value`` is already present in ``key``, silently do nothing.
        """
        raise NotImplementedError(f"{type(self).__name__}.save")

    @abc.abstractmethod
    def fetch(self, key: bytes) -> Iterable[bytes]:
        """Return an iterable over all values matching this key."""
        raise NotImplementedError(f"{type(self).__name__}.fetch")

    @abc.abstractmethod
    def delete(self, key: bytes, value: bytes) -> None:
        """Remove ``value`` from ``key``.

        If ``value`` is not present in ``key``, silently do nothing.
        """
        raise NotImplementedError(f"{type(self).__name__}.delete")

    def move(self, src: bytes, dest: bytes, value: bytes) -> None:
        """
        Move ``value`` from key ``src`` to key ``dest``.

        Equivalent to ``delete(src, value)`` followed by ``save(src, value)``,
        but may have a more efficient implementation.

        Note that ``value`` will be inserted at ``dest`` regardless of whether
        it is currently present at ``src``.
        """
        if src == dest:
            self.save(src, value)
            return
        self.delete(src, value)
        self.save(dest, value)

    def add_listener(self, f: ListenerT, /) -> None:
        """
        Add a change listener. ``f`` will be called whenever a value is saved,
        deleted, or moved in the database.

        ``f`` can be called with two different event values:

        * ``("save", (key, value))``
        * ``("delete", (key, value))``

        where ``key`` and ``value`` are both ``bytes``.

        There is no ``move`` event. Instead, a move is broadcasted as a
        ``delete`` event followed by a ``save`` event.

        For the ``delete`` event, ``value`` may be ``None``. This might occur if
        the database knows that a deletion has occurred in ``key``, but does not
        know what value was deleted.
        """
        had_listeners = bool(self._listeners)
        self._listeners.append(f)
        if not had_listeners:
            self._start_listening()

    def remove_listener(self, f: ListenerT, /) -> None:
        """
        Removes ``f`` from the list of change listeners.

        If ``f`` is not in the list of change listeners, silently do nothing.
        """
        if f not in self._listeners:
            return
        self._listeners.remove(f)
        if not self._listeners:
            self._stop_listening()

    def clear_listeners(self) -> None:
        """Remove all change listeners."""
        had_listeners = bool(self._listeners)
        self._listeners.clear()
        if had_listeners:
            self._stop_listening()

    def _broadcast_change(self, event: ListenerEventT) -> None:
        """
        Called when a value has been either added to or deleted from a key in
        the underlying database store. The possible values for ``event`` are:

        * ``("save", (key, value))``
        * ``("delete", (key, value))``

        ``value`` may be ``None`` for the ``delete`` event, indicating we know
        that some value was deleted under this key, but not its exact value.

        Note that you should not assume your instance is the only reference to
        the underlying database store. For example, if two instances of
        |DirectoryBasedExampleDatabase| reference the same directory,
        _broadcast_change should be called whenever a file is added or removed
        from the directory, even if that database was not responsible for
        changing the file.
        """
        for listener in self._listeners:
            listener(event)

    def _start_listening(self) -> None:
        """
        Called when the database adds a change listener, and did not previously
        have any change listeners. Intended to allow databases to wait to start
        expensive listening operations until necessary.

        ``_start_listening`` and ``_stop_listening`` are guaranteed to alternate,
        so you do not need to handle the case of multiple consecutive
        ``_start_listening`` calls without an intermediate ``_stop_listening``
        call.
        """
        warnings.warn(
            f"{self.__class__} does not support listening for changes",
            HypothesisWarning,
            stacklevel=4,
        )

    def _stop_listening(self) -> None:
        """
        Called whenever no change listeners remain on the database.

        ``_stop_listening`` and ``_start_listening`` are guaranteed to alternate,
        so you do not need to handle the case of multiple consecutive
        ``_stop_listening`` calls without an intermediate ``_start_listening``
        call.
        """
        warnings.warn(
            f"{self.__class__} does not support stopping listening for changes",
            HypothesisWarning,
            stacklevel=4,
        )


class InMemoryExampleDatabase(ExampleDatabase):
    """A non-persistent example database, implemented in terms of an in-memory
    dictionary.

    This can be useful if you call a test function several times in a single
    session, or for testing other database implementations, but because it
    does not persist between runs we do not recommend it for general use.
    """

    def __init__(self) -> None:
        super().__init__()
        self.data: dict[bytes, set[bytes]] = {}

    def __repr__(self) -> str:
        return f"InMemoryExampleDatabase({self.data!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, InMemoryExampleDatabase) and self.data is other.data

    def fetch(self, key: bytes) -> Iterable[bytes]:
        yield from self.data.get(key, ())

    def save(self, key: bytes, value: bytes) -> None:
        value = bytes(value)
        values = self.data.setdefault(key, set())
        changed = value not in values
        values.add(value)

        if changed:
            self._broadcast_change(("save", (key, value)))

    def delete(self, key: bytes, value: bytes) -> None:
        value = bytes(value)
        values = self.data.get(key, set())
        changed = value in values
        values.discard(value)

        if changed:
            self._broadcast_change(("delete", (key, value)))

    def _start_listening(self) -> None:
        # declare compatibility with the listener api, but do the actual
        # implementation in .delete and .save, since we know we are the only
        # writer to .data.
        pass

    def _stop_listening(self) -> None:
        pass


def _hash(key: bytes) -> str:
    return sha384(key).hexdigest()[:16]


class DirectoryBasedExampleDatabase(ExampleDatabase):
    """Use a directory to store Hypothesis examples as files.

    Each test corresponds to a directory, and each example to a file within that
    directory.  While the contents are fairly opaque, a
    |DirectoryBasedExampleDatabase| can be shared by checking the directory
    into version control, for example with the following ``.gitignore``::

        # Ignore files cached by Hypothesis...
        .hypothesis/*
        # except for the examples directory
        !.hypothesis/examples/

    Note however that this only makes sense if you also pin to an exact version of
    Hypothesis, and we would usually recommend implementing a shared database with
    a network datastore - see |ExampleDatabase|, and the |MultiplexedDatabase| helper.
    """

    # we keep a database entry of the full values of all the database keys.
    # currently only used for inverse mapping of hash -> key in change listening.
    _metakeys_name: ClassVar[bytes] = b".hypothesis-keys"
    _metakeys_hash: ClassVar[str] = _hash(_metakeys_name)

    def __init__(self, path: StrPathT) -> None:
        super().__init__()
        self.path = Path(path)
        self.keypaths: dict[bytes, Path] = {}
        self._observer: BaseObserver | None = None

    def __repr__(self) -> str:
        return f"DirectoryBasedExampleDatabase({self.path!r})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, DirectoryBasedExampleDatabase) and self.path == other.path
        )

    def _key_path(self, key: bytes) -> Path:
        try:
            return self.keypaths[key]
        except KeyError:
            pass
        self.keypaths[key] = self.path / _hash(key)
        return self.keypaths[key]

    def _value_path(self, key: bytes, value: bytes) -> Path:
        return self._key_path(key) / _hash(value)

    def fetch(self, key: bytes) -> Iterable[bytes]:
        kp = self._key_path(key)
        if not kp.is_dir():
            return

        try:
            for path in os.listdir(kp):
                try:
                    yield (kp / path).read_bytes()
                except OSError:
                    pass
        except OSError:  # pragma: no cover
            # the `kp` directory might have been deleted in the meantime
            pass

    def save(self, key: bytes, value: bytes) -> None:
        key_path = self._key_path(key)
        if key_path.name != self._metakeys_hash:
            # add this key to our meta entry of all keys - taking care to avoid
            # infinite recursion.
            self.save(self._metakeys_name, key)

        # Note: we attempt to create the dir in question now. We
        # already checked for permissions, but there can still be other issues,
        # e.g. the disk is full, or permissions might have been changed.
        try:
            key_path.mkdir(exist_ok=True, parents=True)
            path = self._value_path(key, value)
            if not path.exists():
                # to mimic an atomic write, create and write in a temporary
                # directory, and only move to the final path after. This avoids
                # any intermediate state where the file is created (and empty)
                # but not yet written to.
                fd, tmpname = tempfile.mkstemp()
                tmppath = Path(tmpname)
                os.write(fd, value)
                os.close(fd)
                try:
                    tmppath.rename(path)
                except OSError as err:  # pragma: no cover
                    if err.errno == errno.EXDEV:
                        # Can't rename across filesystem boundaries, see e.g.
                        # https://github.com/HypothesisWorks/hypothesis/issues/4335
                        try:
                            path.write_bytes(tmppath.read_bytes())
                        except OSError:
                            pass
                    tmppath.unlink()
                assert not tmppath.exists()
        except OSError:  # pragma: no cover
            pass

    def move(self, src: bytes, dest: bytes, value: bytes) -> None:
        if src == dest:
            self.save(src, value)
            return

        src_path = self._value_path(src, value)
        dest_path = self._value_path(dest, value)
        # if the dest key path does not exist, os.renames will create it for us,
        # and we will never track its creation in the meta keys entry. Do so now.
        if not self._key_path(dest).exists():
            self.save(self._metakeys_name, dest)

        try:
            os.renames(src_path, dest_path)
        except OSError:
            self.delete(src, value)
            self.save(dest, value)

    def delete(self, key: bytes, value: bytes) -> None:
        try:
            self._value_path(key, value).unlink()
        except OSError:
            return

        # try deleting the key dir, which will only succeed if the dir is empty
        # (i.e. ``value`` was the last value in this key).
        try:
            self._key_path(key).rmdir()
        except OSError:
            pass
        else:
            # if the deletion succeeded, also delete this key entry from metakeys.
            # (if this key happens to be the metakey itself, this deletion will
            # fail; that's ok and faster than checking for this rare case.)
            self.delete(self._metakeys_name, key)

    def _start_listening(self) -> None:
        try:
            from watchdog.events import (
                DirCreatedEvent,
                DirDeletedEvent,
                DirMovedEvent,
                FileCreatedEvent,
                FileDeletedEvent,
                FileMovedEvent,
                FileSystemEventHandler,
            )
            from watchdog.observers import Observer
        except ImportError:
            warnings.warn(
                f"listening for changes in a {self.__class__.__name__} "
                "requires the watchdog library. To install, run "
                "`pip install hypothesis[watchdog]`",
                HypothesisWarning,
                stacklevel=4,
            )
            return

        hash_to_key = {_hash(key): key for key in self.fetch(self._metakeys_name)}
        _metakeys_hash = self._metakeys_hash
        _broadcast_change = self._broadcast_change

        class Handler(FileSystemEventHandler):
            def on_created(_self, event: FileCreatedEvent | DirCreatedEvent) -> None:
                # we only registered for the file creation event
                assert not isinstance(event, DirCreatedEvent)
                # watchdog events are only bytes if we passed a byte path to
                # .schedule
                assert isinstance(event.src_path, str)

                value_path = Path(event.src_path)
                # the parent dir represents the key, and its name is the key hash
                key_hash = value_path.parent.name

                if key_hash == _metakeys_hash:
                    try:
                        hash_to_key[value_path.name] = value_path.read_bytes()
                    except OSError:  # pragma: no cover
                        # this might occur if all the values in a key have been
                        # deleted and DirectoryBasedExampleDatabase removes its
                        # metakeys entry (which is `value_path` here`).
                        pass
                    return

                key = hash_to_key.get(key_hash)
                if key is None:  # pragma: no cover
                    # we didn't recognize this key. This shouldn't ever happen,
                    # but some race condition trickery might cause this.
                    return

                try:
                    value = value_path.read_bytes()
                except OSError:  # pragma: no cover
                    return

                _broadcast_change(("save", (key, value)))

            def on_deleted(self, event: FileDeletedEvent | DirDeletedEvent) -> None:
                assert not isinstance(event, DirDeletedEvent)
                assert isinstance(event.src_path, str)

                value_path = Path(event.src_path)
                key = hash_to_key.get(value_path.parent.name)
                if key is None:  # pragma: no cover
                    return

                _broadcast_change(("delete", (key, None)))

            def on_moved(self, event: FileMovedEvent | DirMovedEvent) -> None:
                assert not isinstance(event, DirMovedEvent)
                assert isinstance(event.src_path, str)
                assert isinstance(event.dest_path, str)

                src_path = Path(event.src_path)
                dest_path = Path(event.dest_path)
                k1 = hash_to_key.get(src_path.parent.name)
                k2 = hash_to_key.get(dest_path.parent.name)

                if k1 is None or k2 is None:  # pragma: no cover
                    return

                try:
                    value = dest_path.read_bytes()
                except OSError:  # pragma: no cover
                    return

                _broadcast_change(("delete", (k1, value)))
                _broadcast_change(("save", (k2, value)))

        # If we add a listener to a DirectoryBasedExampleDatabase whose database
        # directory doesn't yet exist, the watchdog observer will not fire any
        # events, even after the directory gets created.
        #
        # Ensure the directory exists before starting the observer.
        self.path.mkdir(exist_ok=True, parents=True)
        self._observer = Observer()
        self._observer.schedule(
            Handler(),
            # remove type: ignore when released
            # https://github.com/gorakhargosh/watchdog/pull/1096
            self.path,  # type: ignore
            recursive=True,
            event_filter=[FileCreatedEvent, FileDeletedEvent, FileMovedEvent],
        )
        self._observer.start()

    def _stop_listening(self) -> None:
        assert self._observer is not None
        self._observer.stop()
        self._observer.join()
        self._observer = None


class ReadOnlyDatabase(ExampleDatabase):
    """A wrapper to make the given database read-only.

    The implementation passes through ``fetch``, and turns ``save``, ``delete``, and
    ``move`` into silent no-ops.

    Note that this disables Hypothesis' automatic discarding of stale examples.
    It is designed to allow local machines to access a shared database (e.g. from CI
    servers), without propagating changes back from a local or in-development branch.
    """

    def __init__(self, db: ExampleDatabase) -> None:
        super().__init__()
        assert isinstance(db, ExampleDatabase)
        self._wrapped = db

    def __repr__(self) -> str:
        return f"ReadOnlyDatabase({self._wrapped!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ReadOnlyDatabase) and self._wrapped == other._wrapped

    def fetch(self, key: bytes) -> Iterable[bytes]:
        yield from self._wrapped.fetch(key)

    def save(self, key: bytes, value: bytes) -> None:
        pass

    def delete(self, key: bytes, value: bytes) -> None:
        pass

    def _start_listening(self) -> None:
        # we're read only, so there are no changes to broadcast.
        pass

    def _stop_listening(self) -> None:
        pass


class MultiplexedDatabase(ExampleDatabase):
    """A wrapper around multiple databases.

    Each ``save``, ``fetch``, ``move``, or ``delete`` operation will be run against
    all of the wrapped databases.  ``fetch`` does not yield duplicate values, even
    if the same value is present in two or more of the wrapped databases.

    This combines well with a :class:`ReadOnlyDatabase`, as follows:

    .. code-block:: python

        local = DirectoryBasedExampleDatabase("/tmp/hypothesis/examples/")
        shared = CustomNetworkDatabase()

        settings.register_profile("ci", database=shared)
        settings.register_profile(
            "dev", database=MultiplexedDatabase(local, ReadOnlyDatabase(shared))
        )
        settings.load_profile("ci" if os.environ.get("CI") else "dev")

    So your CI system or fuzzing runs can populate a central shared database;
    while local runs on development machines can reproduce any failures from CI
    but will only cache their own failures locally and cannot remove examples
    from the shared database.
    """

    def __init__(self, *dbs: ExampleDatabase) -> None:
        super().__init__()
        assert all(isinstance(db, ExampleDatabase) for db in dbs)
        self._wrapped = dbs

    def __repr__(self) -> str:
        return "MultiplexedDatabase({})".format(", ".join(map(repr, self._wrapped)))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, MultiplexedDatabase) and self._wrapped == other._wrapped
        )

    def fetch(self, key: bytes) -> Iterable[bytes]:
        seen = set()
        for db in self._wrapped:
            for value in db.fetch(key):
                if value not in seen:
                    yield value
                    seen.add(value)

    def save(self, key: bytes, value: bytes) -> None:
        for db in self._wrapped:
            db.save(key, value)

    def delete(self, key: bytes, value: bytes) -> None:
        for db in self._wrapped:
            db.delete(key, value)

    def move(self, src: bytes, dest: bytes, value: bytes) -> None:
        for db in self._wrapped:
            db.move(src, dest, value)

    def _start_listening(self) -> None:
        for db in self._wrapped:
            db.add_listener(self._broadcast_change)

    def _stop_listening(self) -> None:
        for db in self._wrapped:
            db.remove_listener(self._broadcast_change)


class GitHubArtifactDatabase(ExampleDatabase):
    """
    A file-based database loaded from a `GitHub Actions <https://docs.github.com/en/actions>`_ artifact.

    You can use this for sharing example databases between CI runs and developers, allowing
    the latter to get read-only access to the former. This is particularly useful for
    continuous fuzzing (i.e. with `HypoFuzz <https://hypofuzz.com/>`_),
    where the CI system can help find new failing examples through fuzzing,
    and developers can reproduce them locally without any manual effort.

    .. note::
        You must provide ``GITHUB_TOKEN`` as an environment variable. In CI, Github Actions provides
        this automatically, but it needs to be set manually for local usage. In a developer machine,
        this would usually be a `Personal Access Token <https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens>`_.
        If the repository is private, it's necessary for the token to have ``repo`` scope
        in the case of a classic token, or ``actions:read`` in the case of a fine-grained token.


    In most cases, this will be used
    through the :class:`~hypothesis.database.MultiplexedDatabase`,
    by combining a local directory-based database with this one. For example:

    .. code-block:: python

        local = DirectoryBasedExampleDatabase(".hypothesis/examples")
        shared = ReadOnlyDatabase(GitHubArtifactDatabase("user", "repo"))

        settings.register_profile("ci", database=local)
        settings.register_profile("dev", database=MultiplexedDatabase(local, shared))
        # We don't want to use the shared database in CI, only to populate its local one.
        # which the workflow should then upload as an artifact.
        settings.load_profile("ci" if os.environ.get("CI") else "dev")

    .. note::
        Because this database is read-only, you always need to wrap it with the
        :class:`ReadOnlyDatabase`.

    A setup like this can be paired with a GitHub Actions workflow including
    something like the following:

    .. code-block:: yaml

        - name: Download example database
          uses: dawidd6/action-download-artifact@v9
          with:
            name: hypothesis-example-db
            path: .hypothesis/examples
            if_no_artifact_found: warn
            workflow_conclusion: completed

        - name: Run tests
          run: pytest

        - name: Upload example database
          uses: actions/upload-artifact@v3
          if: always()
          with:
            name: hypothesis-example-db
            path: .hypothesis/examples

    In this workflow, we use `dawidd6/action-download-artifact <https://github.com/dawidd6/action-download-artifact>`_
    to download the latest artifact given that the official `actions/download-artifact <https://github.com/actions/download-artifact>`_
    does not support downloading artifacts from previous workflow runs.

    The database automatically implements a simple file-based cache with a default expiration period
    of 1 day. You can adjust this through the ``cache_timeout`` property.

    For mono-repo support, you can provide a unique ``artifact_name`` (e.g. ``hypofuzz-example-db-frontend``).
    """

    def __init__(
        self,
        owner: str,
        repo: str,
        artifact_name: str = "hypothesis-example-db",
        cache_timeout: timedelta = timedelta(days=1),
        path: StrPathT | None = None,
    ):
        super().__init__()
        self.owner = owner
        self.repo = repo
        self.artifact_name = artifact_name
        self.cache_timeout = cache_timeout

        # Get the GitHub token from the environment
        # It's unnecessary to use a token if the repo is public
        self.token: str | None = getenv("GITHUB_TOKEN")

        if path is None:
            self.path: Path = Path(
                storage_directory(f"github-artifacts/{self.artifact_name}/")
            )
        else:
            self.path = Path(path)

        # We don't want to initialize the cache until we need to
        self._initialized: bool = False
        self._disabled: bool = False

        # This is the path to the artifact in usage
        # .hypothesis/github-artifacts/<artifact-name>/<modified_isoformat>.zip
        self._artifact: Path | None = None
        # This caches the artifact structure
        self._access_cache: dict[PurePath, set[PurePath]] | None = None

        # Message to display if user doesn't wrap around ReadOnlyDatabase
        self._read_only_message = (
            "This database is read-only. "
            "Please wrap this class with ReadOnlyDatabase"
            "i.e. ReadOnlyDatabase(GitHubArtifactDatabase(...))."
        )

    def __repr__(self) -> str:
        return (
            f"GitHubArtifactDatabase(owner={self.owner!r}, "
            f"repo={self.repo!r}, artifact_name={self.artifact_name!r})"
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, GitHubArtifactDatabase)
            and self.owner == other.owner
            and self.repo == other.repo
            and self.artifact_name == other.artifact_name
            and self.path == other.path
        )

    def _prepare_for_io(self) -> None:
        assert self._artifact is not None, "Artifact not loaded."

        if self._initialized:  # pragma: no cover
            return

        # Test that the artifact is valid
        try:
            with ZipFile(self._artifact) as f:
                if f.testzip():  # pragma: no cover
                    raise BadZipFile

            # Turns out that testzip() doesn't work quite well
            # doing the cache initialization here instead
            # will give us more coverage of the artifact.

            # Cache the files inside each keypath
            self._access_cache = {}
            with ZipFile(self._artifact) as zf:
                namelist = zf.namelist()
                # Iterate over files in the artifact
                for filename in namelist:
                    fileinfo = zf.getinfo(filename)
                    if fileinfo.is_dir():
                        self._access_cache[PurePath(filename)] = set()
                    else:
                        # Get the keypath from the filename
                        keypath = PurePath(filename).parent
                        # Add the file to the keypath
                        self._access_cache[keypath].add(PurePath(filename))
        except BadZipFile:
            warnings.warn(
                "The downloaded artifact from GitHub is invalid. "
                "This could be because the artifact was corrupted, "
                "or because the artifact was not created by Hypothesis. ",
                HypothesisWarning,
                stacklevel=3,
            )
            self._disabled = True

        self._initialized = True

    def _initialize_db(self) -> None:
        # Trigger warning that we suppressed earlier by intent_to_write=False
        storage_directory(self.path.name)
        # Create the cache directory if it doesn't exist
        self.path.mkdir(exist_ok=True, parents=True)

        # Get all artifacts
        cached_artifacts = sorted(
            self.path.glob("*.zip"),
            key=lambda a: datetime.fromisoformat(a.stem.replace("_", ":")),
        )

        # Remove all but the latest artifact
        for artifact in cached_artifacts[:-1]:
            artifact.unlink()

        try:
            found_artifact = cached_artifacts[-1]
        except IndexError:
            found_artifact = None

        # Check if the latest artifact is a cache hit
        if found_artifact is not None and (
            datetime.now(timezone.utc)
            - datetime.fromisoformat(found_artifact.stem.replace("_", ":"))
            < self.cache_timeout
        ):
            self._artifact = found_artifact
        else:
            # Download the latest artifact from GitHub
            new_artifact = self._fetch_artifact()

            if new_artifact:
                if found_artifact is not None:
                    found_artifact.unlink()
                self._artifact = new_artifact
            elif found_artifact is not None:
                warnings.warn(
                    "Using an expired artifact as a fallback for the database: "
                    f"{found_artifact}",
                    HypothesisWarning,
                    stacklevel=2,
                )
                self._artifact = found_artifact
            else:
                warnings.warn(
                    "Couldn't acquire a new or existing artifact. Disabling database.",
                    HypothesisWarning,
                    stacklevel=2,
                )
                self._disabled = True
                return

        self._prepare_for_io()

    def _get_bytes(self, url: str) -> bytes | None:  # pragma: no cover
        request = Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28 ",
                "Authorization": f"Bearer {self.token}",
            },
        )
        warning_message = None
        response_bytes: bytes | None = None
        try:
            with urlopen(request) as response:
                response_bytes = response.read()
        except HTTPError as e:
            if e.code == 401:
                warning_message = (
                    "Authorization failed when trying to download artifact from GitHub. "
                    "Check that you have a valid GITHUB_TOKEN set in your environment."
                )
            else:
                warning_message = (
                    "Could not get the latest artifact from GitHub. "
                    "This could be because because the repository "
                    "or artifact does not exist. "
                )
            # see https://github.com/python/cpython/issues/128734
            e.close()
        except URLError:
            warning_message = "Could not connect to GitHub to get the latest artifact. "
        except TimeoutError:
            warning_message = (
                "Could not connect to GitHub to get the latest artifact "
                "(connection timed out)."
            )

        if warning_message is not None:
            warnings.warn(warning_message, HypothesisWarning, stacklevel=4)
            return None

        return response_bytes

    def _fetch_artifact(self) -> Path | None:  # pragma: no cover
        # Get the list of artifacts from GitHub
        url = f"https://api.github.com/repos/{self.owner}/{self.repo}/actions/artifacts"
        response_bytes = self._get_bytes(url)
        if response_bytes is None:
            return None

        artifacts = json.loads(response_bytes)["artifacts"]
        artifacts = [a for a in artifacts if a["name"] == self.artifact_name]

        if not artifacts:
            return None

        # Get the latest artifact from the list
        artifact = max(artifacts, key=lambda a: a["created_at"])
        url = artifact["archive_download_url"]

        # Download the artifact
        artifact_bytes = self._get_bytes(url)
        if artifact_bytes is None:
            return None

        # Save the artifact to the cache
        # We replace ":" with "_" to ensure the filenames are compatible
        # with Windows filesystems
        timestamp = datetime.now(timezone.utc).isoformat().replace(":", "_")
        artifact_path = self.path / f"{timestamp}.zip"
        try:
            artifact_path.write_bytes(artifact_bytes)
        except OSError:
            warnings.warn(
                "Could not save the latest artifact from GitHub. ",
                HypothesisWarning,
                stacklevel=3,
            )
            return None

        return artifact_path

    @staticmethod
    @lru_cache
    def _key_path(key: bytes) -> PurePath:
        return PurePath(_hash(key) + "/")

    def fetch(self, key: bytes) -> Iterable[bytes]:
        if self._disabled:
            return

        if not self._initialized:
            self._initialize_db()
            if self._disabled:
                return

        assert self._artifact is not None
        assert self._access_cache is not None

        kp = self._key_path(key)

        with ZipFile(self._artifact) as zf:
            # Get the all files in the the kp from the cache
            filenames = self._access_cache.get(kp, ())
            for filename in filenames:
                with zf.open(filename.as_posix()) as f:
                    yield f.read()

    # Read-only interface
    def save(self, key: bytes, value: bytes) -> None:
        raise RuntimeError(self._read_only_message)

    def move(self, src: bytes, dest: bytes, value: bytes) -> None:
        raise RuntimeError(self._read_only_message)

    def delete(self, key: bytes, value: bytes) -> None:
        raise RuntimeError(self._read_only_message)


class BackgroundWriteDatabase(ExampleDatabase):
    """A wrapper which defers writes on the given database to a background thread.

    Calls to :meth:`~hypothesis.database.ExampleDatabase.fetch` wait for any
    enqueued writes to finish before fetching from the database.
    """

    def __init__(self, db: ExampleDatabase) -> None:
        super().__init__()
        self._db = db
        self._queue: Queue[tuple[str, tuple[bytes, ...]]] = Queue()
        self._thread: Thread | None = None

    def _ensure_thread(self):
        if self._thread is None:
            self._thread = Thread(target=self._worker, daemon=True)
            self._thread.start()
            # avoid an unbounded timeout during gc. 0.1 should be plenty for most
            # use cases.
            weakref.finalize(self, self._join, 0.1)

    def __repr__(self) -> str:
        return f"BackgroundWriteDatabase({self._db!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BackgroundWriteDatabase) and self._db == other._db

    def _worker(self) -> None:
        while True:
            method, args = self._queue.get()
            getattr(self._db, method)(*args)
            self._queue.task_done()

    def _join(self, timeout: float | None = None) -> None:
        # copy of Queue.join with a timeout. https://bugs.python.org/issue9634
        with self._queue.all_tasks_done:
            while self._queue.unfinished_tasks:
                self._queue.all_tasks_done.wait(timeout)

    def fetch(self, key: bytes) -> Iterable[bytes]:
        self._join()
        return self._db.fetch(key)

    def save(self, key: bytes, value: bytes) -> None:
        self._ensure_thread()
        self._queue.put(("save", (key, value)))

    def delete(self, key: bytes, value: bytes) -> None:
        self._ensure_thread()
        self._queue.put(("delete", (key, value)))

    def move(self, src: bytes, dest: bytes, value: bytes) -> None:
        self._ensure_thread()
        self._queue.put(("move", (src, dest, value)))

    def _start_listening(self) -> None:
        self._db.add_listener(self._broadcast_change)

    def _stop_listening(self) -> None:
        self._db.remove_listener(self._broadcast_change)


def _pack_uleb128(value: int) -> bytes:
    """
    Serialize an integer into variable-length bytes. For each byte, the first 7
    bits represent (part of) the integer, while the last bit indicates whether the
    integer continues into the next byte.

    https://en.wikipedia.org/wiki/LEB128
    """
    parts = bytearray()
    assert value >= 0
    while True:
        # chop off 7 bits
        byte = value & ((1 << 7) - 1)
        value >>= 7
        # set the continuation bit if we have more left
        if value:
            byte |= 1 << 7

        parts.append(byte)
        if not value:
            break
    return bytes(parts)


def _unpack_uleb128(buffer: bytes) -> tuple[int, int]:
    """
    Inverts _pack_uleb128, and also returns the index at which at which we stopped
    reading.
    """
    value = 0
    for i, byte in enumerate(buffer):
        n = byte & ((1 << 7) - 1)
        value |= n << (i * 7)

        if not byte >> 7:
            break
    return (i + 1, value)


def choices_to_bytes(choices: Iterable[ChoiceT], /) -> bytes:
    """Serialize a list of choices to a bytestring.  Inverts choices_from_bytes."""
    # We use a custom serialization format for this, which might seem crazy - but our
    # data is a flat sequence of elements, and standard tools like protobuf or msgpack
    # don't deal well with e.g. nonstandard bit-pattern-NaNs, or invalid-utf8 unicode.
    #
    # We simply encode each element with a metadata byte, if needed a uint16 size, and
    # then the payload bytes.  For booleans, the payload is inlined into the metadata.
    parts = []
    for choice in choices:
        if isinstance(choice, bool):
            # `000_0000v` - tag zero, low bit payload.
            parts.append(b"\1" if choice else b"\0")
            continue

        # `tag_ssss [uint16 size?] [payload]`
        if isinstance(choice, float):
            tag = 1 << 5
            choice = struct.pack("!d", choice)
        elif isinstance(choice, int):
            tag = 2 << 5
            choice = choice.to_bytes(1 + choice.bit_length() // 8, "big", signed=True)
        elif isinstance(choice, bytes):
            tag = 3 << 5
        else:
            assert isinstance(choice, str)
            tag = 4 << 5
            choice = choice.encode(errors="surrogatepass")

        size = len(choice)
        if size < 0b11111:
            parts.append((tag | size).to_bytes(1, "big"))
        else:
            parts.append((tag | 0b11111).to_bytes(1, "big"))
            parts.append(_pack_uleb128(size))
        parts.append(choice)

    return b"".join(parts)


def _choices_from_bytes(buffer: bytes, /) -> tuple[ChoiceT, ...]:
    # See above for an explanation of the format.
    parts: list[ChoiceT] = []
    idx = 0
    while idx < len(buffer):
        tag = buffer[idx] >> 5
        size = buffer[idx] & 0b11111
        idx += 1

        if tag == 0:
            parts.append(bool(size))
            continue
        if size == 0b11111:
            (offset, size) = _unpack_uleb128(buffer[idx:])
            idx += offset
        chunk = buffer[idx : idx + size]
        idx += size

        if tag == 1:
            assert size == 8, "expected float64"
            parts.extend(struct.unpack("!d", chunk))
        elif tag == 2:
            parts.append(int.from_bytes(chunk, "big", signed=True))
        elif tag == 3:
            parts.append(chunk)
        else:
            assert tag == 4
            parts.append(chunk.decode(errors="surrogatepass"))
    return tuple(parts)


def choices_from_bytes(buffer: bytes, /) -> tuple[ChoiceT, ...] | None:
    """
    Deserialize a bytestring to a tuple of choices. Inverts choices_to_bytes.

    Returns None if the given bytestring is not a valid serialization of choice
    sequences.
    """
    try:
        return _choices_from_bytes(buffer)
    except Exception:
        # deserialization error, eg because our format changed or someone put junk
        # data in the db.
        return None
