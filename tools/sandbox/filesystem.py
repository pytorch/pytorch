import hashlib
import os
import re
import socket
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import modal
from modal._object import _get_environment_name
from modal._utils.async_utils import synchronize_api
from modal.client import _Client
from modal_proto import api_pb2

from tools.sandbox.common import (
    delete_image,
    exit_snapshot,
    FILESYSTEM_TAG,
    image_exists,
    sandbox_infos,
    SNAPSHOT_TIMEOUT_SECONDS,
    timestamp,
    username,
)


LOCKS_NAME = "pytorch-filesystem-locks"


def _image_prefix() -> str:
    owner = hashlib.sha256(username().encode()).hexdigest()[:16]
    return f"pytorch-fs-{owner}-"


class Filesystem:
    """A stable name for a saved root filesystem. Mutating methods require lock()."""

    def __init__(self, filesystem_id: str) -> None:
        if (
            len(filesystem_id) > 35
            or re.fullmatch(r"(?:fs|im)-[a-zA-Z0-9-]+", filesystem_id) is None
        ):
            raise RuntimeError(f"invalid filesystem id: {filesystem_id}")
        self.id = filesystem_id
        self.name = _image_prefix() + filesystem_id

    @classmethod
    def new(cls) -> "Filesystem":
        return cls(f"fs-{uuid.uuid4().hex}")

    @property
    def sandbox_name(self) -> str:
        return self.name

    @contextmanager
    def lock(self) -> Iterator[None]:
        locks = modal.Dict.from_name(LOCKS_NAME, create_if_missing=True)
        owner = {
            "token": uuid.uuid4().hex,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "started_at": time.time(),
        }
        if not locks.put(self.name, owner, skip_if_exists=True):
            raise RuntimeError(
                f"filesystem {self.id} has another operation in progress: {locks.get(self.name)}. "
                f"If its client was killed, remove its lock from Modal Dict {LOCKS_NAME} "
                "only after confirming that client has exited."
            )
        try:
            yield
        finally:
            if locks.get(self.name) == owner:
                locks.pop(self.name)

    def image(self) -> modal.Image:
        history = sandbox_infos(include_finished=True, filesystem=self.id)
        self._check_unused(history)
        published_id = current_image(self.name)
        image_id = published_id
        if not history and image_id is None:
            raise RuntimeError(f"no such filesystem: {self.id}")
        if history:
            snapshot_id = exit_snapshot(history[0].id, SNAPSHOT_TIMEOUT_SECONDS)
            if snapshot_id is not None:
                image_id = snapshot_id
            elif (
                self.id.startswith("im-")
                and image_id in (None, self.id)
                and image_exists(self.id)
            ):
                image_id = self.id
            else:
                raise RuntimeError(
                    f"the final filesystem of sandbox {history[0].id} is unavailable; "
                    "refusing to restore an older version"
                )
        if image_id is None:
            raise RuntimeError(f"filesystem {self.id} has no saved image")
        image = modal.Image.from_id(image_id)
        if published_id != image_id:
            image.publish(self.name)
        self._cleanup(history, keep=image_id)
        return image

    def save(self, sandbox: modal.Sandbox) -> str:
        history = sandbox_infos(include_finished=True, filesystem=self.id)
        if not history or history[0].id != sandbox.object_id:
            raise RuntimeError(
                f"sandbox {sandbox.object_id} is not the latest sandbox on filesystem {self.id}"
            )
        sandbox.terminate(wait=True)
        snapshot_id = exit_snapshot(sandbox.object_id, SNAPSHOT_TIMEOUT_SECONDS)
        if snapshot_id is None:
            raise RuntimeError(
                f"sandbox {sandbox.object_id} exited without an available filesystem snapshot"
            )
        if current_image(self.name) != snapshot_id:
            modal.Image.from_id(snapshot_id).publish(self.name)
        self._cleanup(history, keep=snapshot_id)
        return snapshot_id

    def delete(self) -> list[str]:
        history = sandbox_infos(include_finished=True, filesystem=self.id)
        self._check_unused(history)
        current = current_image(self.name)
        image_ids = self._images(history)
        if not history and not image_ids and current is None:
            raise RuntimeError(f"no such filesystem: {self.id}")
        if current is not None:
            image_ids.discard(current)
        self._delete_images(image_ids)
        if current is not None:
            self._delete_images({current})
        sandbox_ids = []
        for info in history:
            tags = {tag.tag_name: tag.tag_value for tag in info.tags}
            tags.pop(FILESYSTEM_TAG, None)
            modal.Sandbox.from_id(info.id).set_tags(tags)
            sandbox_ids.append(info.id)
        return sandbox_ids

    def _check_unused(self, history: list[api_pb2.SandboxInfo]) -> None:
        running = [info.id for info in history if not info.task_info.finished_at]
        if running:
            raise RuntimeError(
                f"filesystem {self.id} is used by {', '.join(running)}; stop it first"
            )

    def _images(self, history: list[api_pb2.SandboxInfo]) -> set[str]:
        image_ids = set(image_revisions(self.name))
        if self.id.startswith("im-") and history:
            image_ids.add(self.id)
        for info in history:
            try:
                snapshot_id = exit_snapshot(info.id, SNAPSHOT_TIMEOUT_SECONDS)
            except modal.exception.SnapshotCreationError:
                continue
            if snapshot_id is not None:
                image_ids.add(snapshot_id)
        return image_ids

    def _cleanup(self, history: list[api_pb2.SandboxInfo], *, keep: str) -> None:
        image_ids = self._images(history)
        image_ids.discard(keep)
        self._delete_images(image_ids)

    def _delete_images(self, image_ids: set[str]) -> None:
        failures = {}
        for image_id in sorted(image_ids):
            error = delete_image(image_id)
            if error:
                failures[image_id] = error
        if failures:
            raise RuntimeError(
                f"filesystem {self.id} still has images awaiting deletion: {failures}; retry the command"
            )


def current_image(name: str) -> str | None:
    return synchronize_api(_current_image)(name)


async def _current_image(name: str) -> str | None:
    client = await _Client.from_env()
    try:
        response = await client._stub.ImageGetByTag(
            api_pb2.ImageGetByTagRequest(
                tag=name, environment_name=_get_environment_name()
            )
        )
    except modal.exception.NotFoundError:
        return None
    return response.image_id


def image_revisions(name: str) -> list[str]:
    return synchronize_api(_image_revisions)(name)


async def _image_revisions(name: str) -> list[str]:
    client = await _Client.from_env()
    image_ids = []
    page_token = ""
    while True:
        try:
            response = await client._stub.ImageTagRevisions(
                api_pb2.ImageTagRevisionsRequest(
                    tag=name,
                    environment_name=_get_environment_name(),
                    page_token=page_token,
                )
            )
        except modal.exception.NotFoundError:
            return image_ids
        image_ids.extend(item.image_id for item in response.items)
        page_token = response.next_page_token
        if not page_token:
            return image_ids


def named_images() -> list[api_pb2.ImageListTagsItem]:
    return synchronize_api(_named_images)(_image_prefix())


async def _named_images(prefix: str) -> list[api_pb2.ImageListTagsItem]:
    client = await _Client.from_env()
    images = []
    page_token = ""
    while True:
        response = await client._stub.ImageListTags(
            api_pb2.ImageListTagsRequest(
                environment_name=_get_environment_name(),
                tag_prefix=prefix,
                page_token=page_token,
            )
        )
        images.extend(response.items)
        page_token = response.next_page_token
        if not page_token:
            return images


def filesystem_entries(infos: list[api_pb2.SandboxInfo]) -> list[dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    prefix = _image_prefix()
    for image in named_images():
        name = image.tag.rsplit("/", 1)[-1].removesuffix(":latest")
        if not name.startswith(prefix) or not image_exists(image.image_id):
            continue
        filesystem_id = name.removeprefix(prefix)
        entries[filesystem_id] = {
            "filesystem_id": filesystem_id,
            "in_use_by": [],
            "created_at": image.created_at,
            "updated_at": image.updated_at,
        }
    for info in infos:
        tags = {tag.tag_name: tag.tag_value for tag in info.tags}
        filesystem_id = tags.get(FILESYSTEM_TAG)
        if filesystem_id is None:
            continue
        entry = entries.setdefault(
            filesystem_id,
            {
                "filesystem_id": filesystem_id,
                "in_use_by": [],
                "created_at": info.created_at,
                "updated_at": info.created_at,
            },
        )
        entry["created_at"] = min(entry["created_at"], info.created_at)
        entry["updated_at"] = max(
            entry["updated_at"], info.task_info.finished_at or info.created_at
        )
        if not info.task_info.finished_at:
            entry["in_use_by"].append(info.id)
    ordered = sorted(
        entries.values(), key=lambda entry: entry["updated_at"], reverse=True
    )
    return [
        {
            **entry,
            "created_at": timestamp(entry["created_at"]),
            "updated_at": timestamp(entry["updated_at"]),
        }
        for entry in ordered
    ]
