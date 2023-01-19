from __future__ import absolute_import

import hashlib

from pip._vendor.six import iteritems, iterkeys, itervalues

from pip._internal.exceptions import (
    HashMismatch,
    HashMissing,
    InstallationError,
)
from pip._internal.utils.misc import read_chunks
from pip._internal.utils.typing import MYPY_CHECK_RUNNING

if MYPY_CHECK_RUNNING:
    from typing import (
        Dict, List, BinaryIO, NoReturn, Iterator
    )
    from pip._vendor.six import PY3
    if PY3:
        from hashlib import _Hash
    else:
        from hashlib import _hash as _Hash


# The recommended hash algo of the moment. Change this whenever the state of
# the art changes; it won't hurt backward compatibility.
FAVORITE_HASH = 'sha256'


# Names of hashlib algorithms allowed by the --hash option and ``pip hash``
# Currently, those are the ones at least as collision-resistant as sha256.
STRONG_HASHES = ['sha256', 'sha384', 'sha512']


class Hashes(object):
    """A wrapper that builds multiple hashes at once and checks them against
    known-good values

    """
    def __init__(self, hashes=None):
        # type: (Dict[str, List[str]]) -> None
        """
        :param hashes: A dict of algorithm names pointing to lists of allowed
            hex digests
        """
        self._allowed = {} if hashes is None else hashes

    def __or__(self, other):
        # type: (Hashes) -> Hashes
        if not isinstance(other, Hashes):
            return NotImplemented
        new = self._allowed.copy()
        for alg, values in iteritems(other._allowed):
            try:
                new[alg] += values
            except KeyError:
                new[alg] = values
        return Hashes(new)

    @property
    def digest_count(self):
        # type: () -> int
        return sum(len(digests) for digests in self._allowed.values())

    def is_hash_allowed(
        self,
        hash_name,   # type: str
        hex_digest,  # type: str
    ):
        # type: (...) -> bool
        """Return whether the given hex digest is allowed."""
        return hex_digest in self._allowed.get(hash_name, [])

    def check_against_chunks(self, chunks):
        # type: (Iterator[bytes]) -> None
        """Check good hashes against ones built from iterable of chunks of
        data.

        Raise HashMismatch if none match.

        """
        gots = {}
        for hash_name in iterkeys(self._allowed):
            try:
                gots[hash_name] = hashlib.new(hash_name)
            except (ValueError, TypeError):
                raise InstallationError(
                    'Unknown hash name: {}'.format(hash_name)
                )

        for chunk in chunks:
            for hash in itervalues(gots):
                hash.update(chunk)

        for hash_name, got in iteritems(gots):
            if got.hexdigest() in self._allowed[hash_name]:
                return
        self._raise(gots)

    def _raise(self, gots):
        # type: (Dict[str, _Hash]) -> NoReturn
        raise HashMismatch(self._allowed, gots)

    def check_against_file(self, file):
        # type: (BinaryIO) -> None
        """Check good hashes against a file-like object

        Raise HashMismatch if none match.

        """
        return self.check_against_chunks(read_chunks(file))

    def check_against_path(self, path):
        # type: (str) -> None
        with open(path, 'rb') as file:
            return self.check_against_file(file)

    def __nonzero__(self):
        # type: () -> bool
        """Return whether I know any known-good hashes."""
        return bool(self._allowed)

    def __bool__(self):
        # type: () -> bool
        return self.__nonzero__()


class MissingHashes(Hashes):
    """A workalike for Hashes used when we're missing a hash for a requirement

    It computes the actual hash of the requirement and raises a HashMissing
    exception showing it to the user.

    """
    def __init__(self):
        # type: () -> None
        """Don't offer the ``hashes`` kwarg."""
        # Pass our favorite hash in to generate a "gotten hash". With the
        # empty list, it will never match, so an error will always raise.
        super(MissingHashes, self).__init__(hashes={FAVORITE_HASH: []})

    def _raise(self, gots):
        # type: (Dict[str, _Hash]) -> NoReturn
        raise HashMissing(gots[FAVORITE_HASH].hexdigest())
