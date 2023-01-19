""" PEP 610 """
import json
import re

from pip._vendor import six
from pip._vendor.six.moves.urllib import parse as urllib_parse

from pip._internal.utils.typing import MYPY_CHECK_RUNNING

if MYPY_CHECK_RUNNING:
    from typing import (
        Any, Dict, Iterable, Optional, Type, TypeVar, Union
    )

    T = TypeVar("T")


DIRECT_URL_METADATA_NAME = "direct_url.json"
ENV_VAR_RE = re.compile(r"^\$\{[A-Za-z0-9-_]+\}(:\$\{[A-Za-z0-9-_]+\})?$")

__all__ = [
    "DirectUrl",
    "DirectUrlValidationError",
    "DirInfo",
    "ArchiveInfo",
    "VcsInfo",
]


class DirectUrlValidationError(Exception):
    pass


def _get(d, expected_type, key, default=None):
    # type: (Dict[str, Any], Type[T], str, Optional[T]) -> Optional[T]
    """Get value from dictionary and verify expected type."""
    if key not in d:
        return default
    value = d[key]
    if six.PY2 and expected_type is str:
        expected_type = six.string_types  # type: ignore
    if not isinstance(value, expected_type):
        raise DirectUrlValidationError(
            "{!r} has unexpected type for {} (expected {})".format(
                value, key, expected_type
            )
        )
    return value


def _get_required(d, expected_type, key, default=None):
    # type: (Dict[str, Any], Type[T], str, Optional[T]) -> T
    value = _get(d, expected_type, key, default)
    if value is None:
        raise DirectUrlValidationError("{} must have a value".format(key))
    return value


def _exactly_one_of(infos):
    # type: (Iterable[Optional[InfoType]]) -> InfoType
    infos = [info for info in infos if info is not None]
    if not infos:
        raise DirectUrlValidationError(
            "missing one of archive_info, dir_info, vcs_info"
        )
    if len(infos) > 1:
        raise DirectUrlValidationError(
            "more than one of archive_info, dir_info, vcs_info"
        )
    assert infos[0] is not None
    return infos[0]


def _filter_none(**kwargs):
    # type: (Any) -> Dict[str, Any]
    """Make dict excluding None values."""
    return {k: v for k, v in kwargs.items() if v is not None}


class VcsInfo(object):
    name = "vcs_info"

    def __init__(
        self,
        vcs,  # type: str
        commit_id,  # type: str
        requested_revision=None,  # type: Optional[str]
        resolved_revision=None,  # type: Optional[str]
        resolved_revision_type=None,  # type: Optional[str]
    ):
        self.vcs = vcs
        self.requested_revision = requested_revision
        self.commit_id = commit_id
        self.resolved_revision = resolved_revision
        self.resolved_revision_type = resolved_revision_type

    @classmethod
    def _from_dict(cls, d):
        # type: (Optional[Dict[str, Any]]) -> Optional[VcsInfo]
        if d is None:
            return None
        return cls(
            vcs=_get_required(d, str, "vcs"),
            commit_id=_get_required(d, str, "commit_id"),
            requested_revision=_get(d, str, "requested_revision"),
            resolved_revision=_get(d, str, "resolved_revision"),
            resolved_revision_type=_get(d, str, "resolved_revision_type"),
        )

    def _to_dict(self):
        # type: () -> Dict[str, Any]
        return _filter_none(
            vcs=self.vcs,
            requested_revision=self.requested_revision,
            commit_id=self.commit_id,
            resolved_revision=self.resolved_revision,
            resolved_revision_type=self.resolved_revision_type,
        )


class ArchiveInfo(object):
    name = "archive_info"

    def __init__(
        self,
        hash=None,  # type: Optional[str]
    ):
        self.hash = hash

    @classmethod
    def _from_dict(cls, d):
        # type: (Optional[Dict[str, Any]]) -> Optional[ArchiveInfo]
        if d is None:
            return None
        return cls(hash=_get(d, str, "hash"))

    def _to_dict(self):
        # type: () -> Dict[str, Any]
        return _filter_none(hash=self.hash)


class DirInfo(object):
    name = "dir_info"

    def __init__(
        self,
        editable=False,  # type: bool
    ):
        self.editable = editable

    @classmethod
    def _from_dict(cls, d):
        # type: (Optional[Dict[str, Any]]) -> Optional[DirInfo]
        if d is None:
            return None
        return cls(
            editable=_get_required(d, bool, "editable", default=False)
        )

    def _to_dict(self):
        # type: () -> Dict[str, Any]
        return _filter_none(editable=self.editable or None)


if MYPY_CHECK_RUNNING:
    InfoType = Union[ArchiveInfo, DirInfo, VcsInfo]


class DirectUrl(object):

    def __init__(
        self,
        url,  # type: str
        info,  # type: InfoType
        subdirectory=None,  # type: Optional[str]
    ):
        self.url = url
        self.info = info
        self.subdirectory = subdirectory

    def _remove_auth_from_netloc(self, netloc):
        # type: (str) -> str
        if "@" not in netloc:
            return netloc
        user_pass, netloc_no_user_pass = netloc.split("@", 1)
        if (
            isinstance(self.info, VcsInfo) and
            self.info.vcs == "git" and
            user_pass == "git"
        ):
            return netloc
        if ENV_VAR_RE.match(user_pass):
            return netloc
        return netloc_no_user_pass

    @property
    def redacted_url(self):
        # type: () -> str
        """url with user:password part removed unless it is formed with
        environment variables as specified in PEP 610, or it is ``git``
        in the case of a git URL.
        """
        purl = urllib_parse.urlsplit(self.url)
        netloc = self._remove_auth_from_netloc(purl.netloc)
        surl = urllib_parse.urlunsplit(
            (purl.scheme, netloc, purl.path, purl.query, purl.fragment)
        )
        return surl

    def validate(self):
        # type: () -> None
        self.from_dict(self.to_dict())

    @classmethod
    def from_dict(cls, d):
        # type: (Dict[str, Any]) -> DirectUrl
        return DirectUrl(
            url=_get_required(d, str, "url"),
            subdirectory=_get(d, str, "subdirectory"),
            info=_exactly_one_of(
                [
                    ArchiveInfo._from_dict(_get(d, dict, "archive_info")),
                    DirInfo._from_dict(_get(d, dict, "dir_info")),
                    VcsInfo._from_dict(_get(d, dict, "vcs_info")),
                ]
            ),
        )

    def to_dict(self):
        # type: () -> Dict[str, Any]
        res = _filter_none(
            url=self.redacted_url,
            subdirectory=self.subdirectory,
        )
        res[self.info.name] = self.info._to_dict()
        return res

    @classmethod
    def from_json(cls, s):
        # type: (str) -> DirectUrl
        return cls.from_dict(json.loads(s))

    def to_json(self):
        # type: () -> str
        return json.dumps(self.to_dict(), sort_keys=True)
