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
import binascii
import json
import os
import sys
import warnings
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from hashlib import sha384
from os import getenv
from pathlib import Path, PurePath
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

from hypothesis.configuration import storage_directory
from hypothesis.errors import HypothesisException, HypothesisWarning
from hypothesis.utils.conventions import not_set

__all__ = [
    "DirectoryBasedExampleDatabase",
    "ExampleDatabase",
    "InMemoryExampleDatabase",
    "MultiplexedDatabase",
    "ReadOnlyDatabase",
    "GitHubArtifactDatabase",
]


def _usable_dir(path: os.PathLike) -> bool:
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


def _db_for_path(path=None):
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
    return DirectoryBasedExampleDatabase(path)


class _EDMeta(abc.ABCMeta):
    def __call__(self, *args, **kwargs):
        if self is ExampleDatabase:
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
        from sphinx.ext.autodoc import _METACLASS_CALL_BLACKLIST

        _METACLASS_CALL_BLACKLIST.append("hypothesis.database._EDMeta.__call__")
    except Exception:
        pass


class ExampleDatabase(metaclass=_EDMeta):
    """An abstract base class for storing examples in Hypothesis' internal format.

    An ExampleDatabase maps each ``bytes`` key to many distinct ``bytes``
    values, like a ``Mapping[bytes, AbstractSet[bytes]]``.
    """

    @abc.abstractmethod
    def save(self, key: bytes, value: bytes) -> None:
        """Save ``value`` under ``key``.

        If this value is already present for this key, silently do nothing.
        """
        raise NotImplementedError(f"{type(self).__name__}.save")

    @abc.abstractmethod
    def fetch(self, key: bytes) -> Iterable[bytes]:
        """Return an iterable over all values matching this key."""
        raise NotImplementedError(f"{type(self).__name__}.fetch")

    @abc.abstractmethod
    def delete(self, key: bytes, value: bytes) -> None:
        """Remove this value from this key.

        If this value is not present, silently do nothing.
        """
        raise NotImplementedError(f"{type(self).__name__}.delete")

    def move(self, src: bytes, dest: bytes, value: bytes) -> None:
        """Move ``value`` from key ``src`` to key ``dest``. Equivalent to
        ``delete(src, value)`` followed by ``save(src, value)``, but may
        have a more efficient implementation.

        Note that ``value`` will be inserted at ``dest`` regardless of whether
        it is currently present at ``src``.
        """
        if src == dest:
            self.save(src, value)
            return
        self.delete(src, value)
        self.save(dest, value)


class InMemoryExampleDatabase(ExampleDatabase):
    """A non-persistent example database, implemented in terms of a dict of sets.

    This can be useful if you call a test function several times in a single
    session, or for testing other database implementations, but because it
    does not persist between runs we do not recommend it for general use.
    """

    def __init__(self):
        self.data = {}

    def __repr__(self) -> str:
        return f"InMemoryExampleDatabase({self.data!r})"

    def fetch(self, key: bytes) -> Iterable[bytes]:
        yield from self.data.get(key, ())

    def save(self, key: bytes, value: bytes) -> None:
        self.data.setdefault(key, set()).add(bytes(value))

    def delete(self, key: bytes, value: bytes) -> None:
        self.data.get(key, set()).discard(bytes(value))


def _hash(key):
    return sha384(key).hexdigest()[:16]


class DirectoryBasedExampleDatabase(ExampleDatabase):
    """Use a directory to store Hypothesis examples as files.

    Each test corresponds to a directory, and each example to a file within that
    directory.  While the contents are fairly opaque, a
    ``DirectoryBasedExampleDatabase`` can be shared by checking the directory
    into version control, for example with the following ``.gitignore``::

        # Ignore files cached by Hypothesis...
        .hypothesis/*
        # except for the examples directory
        !.hypothesis/examples/

    Note however that this only makes sense if you also pin to an exact version of
    Hypothesis, and we would usually recommend implementing a shared database with
    a network datastore - see :class:`~hypothesis.database.ExampleDatabase`, and
    the :class:`~hypothesis.database.MultiplexedDatabase` helper.
    """

    def __init__(self, path: os.PathLike) -> None:
        self.path = Path(path)
        self.keypaths: dict[bytes, Path] = {}

    def __repr__(self) -> str:
        return f"DirectoryBasedExampleDatabase({self.path!r})"

    def _key_path(self, key: bytes) -> Path:
        try:
            return self.keypaths[key]
        except KeyError:
            pass
        self.keypaths[key] = self.path / _hash(key)
        return self.keypaths[key]

    def _value_path(self, key, value):
        return self._key_path(key) / _hash(value)

    def fetch(self, key: bytes) -> Iterable[bytes]:
        kp = self._key_path(key)
        if not kp.is_dir():
            return
        for path in os.listdir(kp):
            try:
                yield (kp / path).read_bytes()
            except OSError:
                pass

    def save(self, key: bytes, value: bytes) -> None:
        # Note: we attempt to create the dir in question now. We
        # already checked for permissions, but there can still be other issues,
        # e.g. the disk is full, or permissions might have been changed.
        try:
            self._key_path(key).mkdir(exist_ok=True, parents=True)
            path = self._value_path(key, value)
            if not path.exists():
                suffix = binascii.hexlify(os.urandom(16)).decode("ascii")
                tmpname = path.with_suffix(f"{path.suffix}.{suffix}")
                tmpname.write_bytes(value)
                try:
                    tmpname.rename(path)
                except OSError:  # pragma: no cover
                    tmpname.unlink()
                assert not tmpname.exists()
        except OSError:  # pragma: no cover
            pass

    def move(self, src: bytes, dest: bytes, value: bytes) -> None:
        if src == dest:
            self.save(src, value)
            return
        try:
            os.renames(
                self._value_path(src, value),
                self._value_path(dest, value),
            )
        except OSError:
            self.delete(src, value)
            self.save(dest, value)

    def delete(self, key: bytes, value: bytes) -> None:
        try:
            self._value_path(key, value).unlink()
        except OSError:
            pass


class ReadOnlyDatabase(ExampleDatabase):
    """A wrapper to make the given database read-only.

    The implementation passes through ``fetch``, and turns ``save``, ``delete``, and
    ``move`` into silent no-ops.

    Note that this disables Hypothesis' automatic discarding of stale examples.
    It is designed to allow local machines to access a shared database (e.g. from CI
    servers), without propagating changes back from a local or in-development branch.
    """

    def __init__(self, db: ExampleDatabase) -> None:
        assert isinstance(db, ExampleDatabase)
        self._wrapped = db

    def __repr__(self) -> str:
        return f"ReadOnlyDatabase({self._wrapped!r})"

    def fetch(self, key: bytes) -> Iterable[bytes]:
        yield from self._wrapped.fetch(key)

    def save(self, key: bytes, value: bytes) -> None:
        pass

    def delete(self, key: bytes, value: bytes) -> None:
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
        assert all(isinstance(db, ExampleDatabase) for db in dbs)
        self._wrapped = dbs

    def __repr__(self) -> str:
        return "MultiplexedDatabase({})".format(", ".join(map(repr, self._wrapped)))

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
        this would usually be a `Personal Access Token <https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token>`_.
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
          uses: dawidd6/action-download-artifact@v2.24.3
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
        path: Optional[os.PathLike] = None,
    ):
        self.owner = owner
        self.repo = repo
        self.artifact_name = artifact_name
        self.cache_timeout = cache_timeout

        # Get the GitHub token from the environment
        # It's unnecessary to use a token if the repo is public
        self.token: Optional[str] = getenv("GITHUB_TOKEN")

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
        self._artifact: Optional[Path] = None
        # This caches the artifact structure
        self._access_cache: Optional[dict[PurePath, set[PurePath]]] = None

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

    def _get_bytes(self, url: str) -> Optional[bytes]:  # pragma: no cover
        request = Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28 ",
                "Authorization": f"Bearer {self.token}",
            },
        )
        warning_message = None
        response_bytes: Optional[bytes] = None
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

    def _fetch_artifact(self) -> Optional[Path]:  # pragma: no cover
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
