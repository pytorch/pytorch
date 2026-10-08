# Owner(s): ["module: ci"]

from __future__ import annotations

import argparse
import io
import json
import sys
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


try:
    import modal
    from modal_proto import api_pb2

except ImportError:
    HAS_MODAL = False
else:
    HAS_MODAL = True

if HAS_MODAL:
    from tools.sandbox import common, filesystem, flavor, sandbox
    from tools.sandbox.cmd import create, stop


@unittest.skipIf(not HAS_MODAL, "requires the optional Modal SDK")
class TestSandboxHardware(TestCase):
    @parametrize(
        "name,rate_name",
        [
            ("a10:2", "a10g"),
            ("a100-40gb:2", "a100_40gb"),
            ("a100-80gb:2", "a100_80gb"),
            ("rtx-pro-6000:1", "rtx6000"),
        ],
    )
    def test_gpu_pricing_aliases(self, name: str, rate_name: str) -> None:
        rates = {
            f"gpu_hour_cost_{rate_name}": 1.0,
            "cpu_hour_cost_sandbox": 0.1,
            "mem_gib_hour_cost_sandbox": 0.01,
        }
        with patch.object(flavor, "billing_rates", return_value=rates):
            hardware = flavor.parse_flavor(name)
            self.assertEqual(flavor.hourly_rate(hardware), 3.24 * hardware.gpu_count)

    @parametrize("name", ["mi355x:1", "mi355x:8", "a10:5", "rtx-pro-6000:2"])
    def test_rejects_unsupported_flavor_before_remote_work(self, name: str) -> None:
        stderr = io.StringIO()
        argv = ["tools.sandbox", "create", "--flavor", name]
        with (
            patch.object(sys, "argv", argv),
            redirect_stderr(stderr),
            patch.object(sandbox, "username") as username,
            patch.object(create, "base_image") as base_image,
            patch.object(modal.Sandbox, "create") as start_sandbox,
        ):
            with self.assertRaisesRegex(SystemExit, "^2$"):
                sandbox.main()
        self.assertIn("invalid choice", stderr.getvalue())
        username.assert_not_called()
        base_image.assert_not_called()
        start_sandbox.assert_not_called()


@unittest.skipIf(not HAS_MODAL, "requires the optional Modal SDK")
class TestSandboxFilesystem(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.history: list[api_pb2.SandboxInfo] = []
        self.current: str | None = "im-current"
        self.revisions = ["im-current", "im-old"]
        self.exits: dict[str, str | Exception | None] = {}
        self.deleted: set[str] = set()
        self.deletion_errors: dict[str, str] = {}
        self.events: list[tuple[str, str]] = []
        self.handles: dict[str, Mock] = {}
        self.locks: dict[str, object] = {}
        self.lock_store = Mock(spec=["put", "get", "pop"])
        self.lock_store.put.side_effect = self._put_lock
        self.lock_store.get.side_effect = self.locks.get
        self.lock_store.pop.side_effect = self.locks.pop
        patches = ExitStack()
        self.addCleanup(patches.close)
        patches.enter_context(
            patch.object(filesystem, "username", return_value="tester")
        )
        patches.enter_context(
            patch.object(
                filesystem, "sandbox_infos", side_effect=lambda **_: self.history
            )
        )
        patches.enter_context(
            patch.object(
                filesystem, "current_image", side_effect=lambda _: self.current
            )
        )
        patches.enter_context(
            patch.object(
                filesystem,
                "image_revisions",
                side_effect=lambda _: self.revisions.copy(),
            )
        )
        patches.enter_context(
            patch.object(filesystem, "exit_snapshot", side_effect=self._exit_snapshot)
        )
        patches.enter_context(
            patch.object(filesystem, "delete_image", side_effect=self._delete_image)
        )
        patches.enter_context(
            patch.object(
                filesystem,
                "image_exists",
                side_effect=lambda image: image not in self.deleted,
            )
        )
        patches.enter_context(
            patch.object(modal.Image, "from_id", side_effect=self._image)
        )
        patches.enter_context(
            patch.object(modal.Sandbox, "from_id", side_effect=self.handles.__getitem__)
        )
        patches.enter_context(
            patch.object(modal.Dict, "from_name", return_value=self.lock_store)
        )
        self.filesystem = filesystem.Filesystem("fs-test")

    def _put_lock(
        self, key: str, value: object, *, skip_if_exists: bool = False
    ) -> bool:
        if skip_if_exists and key in self.locks:
            return False
        self.locks[key] = value
        return True

    def _image(self, image_id: str) -> Mock:
        image = Mock(spec=["object_id", "publish"])
        image.object_id = image_id

        def publish(name: str) -> None:
            self.events.append(("publish", image_id))
            self.current = image_id
            if image_id not in self.revisions:
                self.revisions.insert(0, image_id)

        image.publish.side_effect = publish
        return image

    def _exit_snapshot(self, sandbox_id: str, timeout: float) -> str | None:
        self.events.append(("exit", sandbox_id))
        image = self.exits[sandbox_id]
        if isinstance(image, Exception):
            raise image
        return None if image in self.deleted else image

    def _delete_image(self, image_id: str) -> str | None:
        self.events.append(("delete", image_id))
        if image_id in self.deletion_errors:
            return self.deletion_errors[image_id]
        self.deleted.add(image_id)
        if self.current == image_id:
            self.current = None
        return None

    def _sandbox(
        self,
        sandbox_id: str,
        *,
        created_at: float,
        running: bool = False,
        image: str = "im-final",
    ) -> Mock:
        info = api_pb2.SandboxInfo(id=sandbox_id, created_at=created_at)
        info.task_info.finished_at = 0 if running else created_at + 10
        info.tags.extend(
            [
                api_pb2.SandboxTag(tag_name="username", tag_value="tester"),
                api_pb2.SandboxTag(tag_name="filesystem", tag_value=self.filesystem.id),
            ]
        )
        self.history.append(info)
        self.history.sort(key=lambda entry: entry.created_at, reverse=True)
        self.exits[sandbox_id] = image
        sandbox = Mock(spec=["object_id", "terminate", "poll", "get_tags", "set_tags"])
        sandbox.object_id = sandbox_id
        sandbox.poll.side_effect = lambda: 0 if info.task_info.finished_at else None
        sandbox.get_tags.side_effect = lambda: {
            tag.tag_name: tag.tag_value for tag in info.tags
        }

        def terminate(*, wait: bool = True) -> None:
            self.events.append(("terminate", sandbox_id))
            info.task_info.finished_at = created_at + 10

        def set_tags(tags: dict[str, str]) -> None:
            del info.tags[:]
            info.tags.extend(
                api_pb2.SandboxTag(tag_name=key, tag_value=value)
                for key, value in tags.items()
            )

        sandbox.terminate.side_effect = terminate
        sandbox.set_tags.side_effect = set_tags
        self.handles[sandbox_id] = sandbox
        return sandbox

    def test_lock_excludes_another_client(self) -> None:
        other = filesystem.Filesystem(self.filesystem.id)
        with self.filesystem.lock():
            held_locks = self.locks.copy()
            with self.assertRaisesRegex(RuntimeError, "busy|progress|locked|operation"):
                with other.lock():
                    self.fail("another client entered the filesystem lock")
            self.assertEqual(self.locks, held_locks)
        self.assertEqual(self.locks, {})
        self.assertTrue(
            all(
                call.kwargs["skip_if_exists"]
                for call in self.lock_store.put.call_args_list
            )
        )

    def test_image_rejects_an_active_sandbox(self) -> None:
        self._sandbox("sb-active", created_at=100, running=True)
        with self.assertRaisesRegex(RuntimeError, "use|running|active"):
            self.filesystem.image()
        self.assertEqual(self.deleted, set())
        self.assertEqual(self.current, "im-current")

    @parametrize("running", [False, True])
    def test_save_keeps_the_final_exit_snapshot(self, running: bool) -> None:
        sandbox = self._sandbox("sb-latest", created_at=200, running=running)
        self._sandbox("sb-older", created_at=100, image="im-older-exit")
        image_id = self.filesystem.save(sandbox)
        self.assertEqual(image_id, "im-final")
        self.assertEqual(self.current, "im-final")
        self.assertEqual(self.deleted, {"im-current", "im-old", "im-older-exit"})
        self.assertLess(
            self.events.index(("terminate", "sb-latest")),
            self.events.index(("exit", "sb-latest")),
        )
        self.assertLess(
            self.events.index(("publish", "im-final")),
            self.events.index(("delete", "im-current")),
        )

    def test_old_stop_cannot_replace_a_newer_filesystem(self) -> None:
        old = self._sandbox("sb-old", created_at=100, image="im-old-exit")
        self._sandbox("sb-new", created_at=200, image="im-final")
        with self.assertRaisesRegex(RuntimeError, "latest|newer|current"):
            self.filesystem.save(old)
        old.terminate.assert_not_called()
        self.assertEqual(self.current, "im-current")
        self.assertEqual(self.deleted, set())

    def test_failed_cleanup_preserves_current_and_can_be_retried(self) -> None:
        sandbox = self._sandbox("sb-latest", created_at=200)
        self.deletion_errors["im-current"] = "temporary deletion failure"
        with self.assertRaisesRegex(RuntimeError, "im-current"):
            self.filesystem.save(sandbox)
        self.assertEqual(self.current, "im-final")
        self.assertNotIn("im-current", self.deleted)
        self.assertNotIn("im-final", self.deleted)
        self.assertIn("im-current", self.revisions)
        self.deletion_errors.clear()
        self.assertEqual(self.filesystem.save(sandbox), "im-final")
        self.assertIn("im-current", self.deleted)
        self.assertNotIn("im-final", self.deleted)

    def test_image_recovers_the_latest_exit(self) -> None:
        self._sandbox("sb-latest", created_at=200)
        self.assertEqual(self.filesystem.image().object_id, "im-final")
        self.assertEqual(self.current, "im-final")
        self.assertEqual(self.deleted, {"im-current", "im-old"})

    def test_missing_final_snapshot_cannot_restore_old_data(self) -> None:
        self._sandbox("sb-latest", created_at=200)
        self.exits["sb-latest"] = None
        with self.assertRaisesRegex(RuntimeError, "unavailable|older"):
            self.filesystem.image()
        self.assertEqual(self.current, "im-current")
        self.assertEqual(self.deleted, set())

    @parametrize("error_name", ["TimeoutError", "SnapshotCreationError"])
    def test_recovery_failure_cannot_fall_back_to_old_data(
        self, error_name: str
    ) -> None:
        self._sandbox("sb-latest", created_at=200)
        error_type = getattr(modal.exception, error_name)
        self.exits["sb-latest"] = error_type("snapshot unavailable")
        with self.assertRaisesRegex(error_type, "snapshot unavailable"):
            self.filesystem.image()
        self.assertEqual(self.current, "im-current")
        self.assertEqual(self.deleted, set())

    def test_delete_removes_every_published_and_exit_snapshot(self) -> None:
        self._sandbox("sb-latest", created_at=200)
        self._sandbox("sb-older", created_at=100, image="im-older-exit")
        deleted_sandboxes = self.filesystem.delete()
        self.assertEqual(set(deleted_sandboxes), {"sb-latest", "sb-older"})
        self.assertEqual(
            self.deleted, {"im-current", "im-old", "im-final", "im-older-exit"}
        )
        self.assertEqual(self.events[-1], ("delete", "im-current"))
        self.assertTrue(
            all(
                "filesystem" not in sandbox.get_tags()
                for sandbox in self.handles.values()
            )
        )

    def test_delete_failure_retains_tracking_for_retry(self) -> None:
        sandbox = self._sandbox("sb-latest", created_at=200)
        self.deletion_errors["im-old"] = "temporary deletion failure"
        with self.assertRaisesRegex(RuntimeError, "im-old"):
            self.filesystem.delete()
        self.assertEqual(sandbox.get_tags()["filesystem"], self.filesystem.id)
        self.assertNotIn("im-current", self.deleted)
        self.deletion_errors.clear()
        self.filesystem.delete()
        self.assertEqual(self.deleted, {"im-current", "im-old", "im-final"})
        self.assertNotIn("filesystem", sandbox.get_tags())

    def test_delete_waits_for_pending_exit_snapshot(self) -> None:
        sandbox = self._sandbox("sb-latest", created_at=200)
        self.exits["sb-latest"] = modal.exception.TimeoutError("snapshot still pending")
        with self.assertRaisesRegex(
            modal.exception.TimeoutError, "snapshot still pending"
        ):
            self.filesystem.delete()
        self.assertEqual(self.deleted, set())
        self.assertEqual(sandbox.get_tags()["filesystem"], self.filesystem.id)

    @parametrize("existing", [False, True])
    @parametrize(
        "flavor_name,gpu",
        [("cpu", None), ("a100-80gb:2", "A100-80GB:2"), ("h100:1", "H100:1")],
    )
    def test_create_claims_a_native_sandbox_name(
        self, existing: bool, flavor_name: str, gpu: str | None
    ) -> None:
        self._sandbox("sb-latest", created_at=200)
        args = argparse.Namespace(
            flavor=flavor_name, filesystem=self.filesystem.id if existing else None
        )

        def reject_duplicate(*args: str, **kwargs: Any) -> None:
            self.assertIn(self.filesystem.name, self.locks)
            self.assertEqual(kwargs["name"], self.filesystem.sandbox_name)
            self.assertEqual(kwargs["tags"]["filesystem"], self.filesystem.id)
            self.assertEqual(kwargs["gpu"], gpu)
            if not existing:
                base_image.return_value.build.assert_called_once_with(
                    app=application.return_value
                )
                self.assertIs(kwargs["image"], base_image.return_value)
            raise modal.exception.AlreadyExistsError("sandbox name already in use")

        with (
            patch.object(create, "Filesystem", return_value=self.filesystem) as manager,
            patch.object(create, "username", return_value="tester"),
            patch.object(create, "github_keys", return_value="ssh-ed25519 test"),
            patch.object(create, "app") as application,
            patch.object(create, "base_image") as base_image,
            patch.object(modal.Sandbox, "create", side_effect=reject_duplicate),
            patch.object(create, "output") as output,
        ):
            manager.new.return_value = self.filesystem
            with self.assertRaisesRegex(SystemExit, "^1$"):
                create.run(args)
        output.assert_not_called()
        if existing:
            base_image.assert_not_called()
        else:
            base_image.assert_called_once_with()
        self.assertEqual(self.locks, {})

    def test_stop_retry_saves_an_already_exited_sandbox(self) -> None:
        sandbox = self._sandbox("sb-latest", created_at=200)
        with (
            patch.object(stop, "owned_sandbox", return_value=sandbox),
            patch.object(stop, "output") as output,
        ):
            stop.run(argparse.Namespace(sandbox_id=sandbox.object_id))
        self.assertEqual(self.current, "im-final")
        self.assertEqual(self.locks, {})
        output.assert_called_once_with(
            {"sandbox_id": sandbox.object_id, "filesystem": self.filesystem.id}
        )

    def test_create_keeps_build_logs_out_of_json(self) -> None:
        stdout, stderr = io.StringIO(), io.StringIO()
        args = argparse.Namespace(flavor="h100:1", filesystem=self.filesystem.id)
        entry = {"connect": {"ssh": {"port": 22, "user": "root", "host": "host"}}}

        def start_sandbox(*args: str, **kwargs: Any) -> Mock:
            sys.stdout.write("image build progress\n")
            return Mock(object_id="sb-created")

        with (
            redirect_stdout(stdout),
            redirect_stderr(stderr),
            patch.object(create, "Filesystem", return_value=self.filesystem),
            patch.object(create, "username", return_value="tester"),
            patch.object(create, "github_keys", return_value="ssh-ed25519 test"),
            patch.object(create, "app"),
            patch.object(modal.Sandbox, "create", side_effect=start_sandbox),
            patch.object(create, "sandbox_info"),
            patch.object(create, "sandbox_entry", return_value=entry),
        ):
            create.run(args)
        self.assertEqual(json.loads(stdout.getvalue()), entry)
        self.assertIn("image build progress", stderr.getvalue())

    def test_listing_matches_filesystem_names_for_long_usernames(self) -> None:
        with patch.object(filesystem, "username", return_value="u" * 39):
            self.filesystem = filesystem.Filesystem.new()
            self._sandbox("sb-current", created_at=100, running=True)
            image = api_pb2.ImageListTagsItem(
                tag=f"main/{self.filesystem.name}:latest",
                image_id="im-current",
                created_at=50,
                updated_at=100,
            )
            with patch.object(filesystem, "named_images", return_value=[image]):
                entries = filesystem.filesystem_entries(self.history)
        self.assertLessEqual(len(self.filesystem.name), 64)
        self.assertLessEqual(len(self.filesystem.sandbox_name), 64)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["filesystem_id"], self.filesystem.id)
        self.assertEqual(entries[0]["in_use_by"], ["sb-current"])


@unittest.skipIf(not HAS_MODAL, "requires the optional Modal SDK")
class TestSandboxImageDeletion(TestCase):
    def test_image_history_reads_every_page(self) -> None:
        client = Mock()
        client._stub.ImageTagRevisions = AsyncMock(
            side_effect=[
                api_pb2.ImageTagRevisionsResponse(
                    items=[api_pb2.ImageTagRevisionsItem(image_id="im-current")],
                    next_page_token="page-2",
                ),
                api_pb2.ImageTagRevisionsResponse(
                    items=[api_pb2.ImageTagRevisionsItem(image_id="im-older")],
                ),
            ]
        )
        with (
            patch.object(
                filesystem._Client, "from_env", new=AsyncMock(return_value=client)
            ),
            patch.object(filesystem, "_get_environment_name", return_value="main"),
        ):
            images = filesystem.image_revisions("pytorch-fs-tester-fs-test")
        self.assertEqual(images, ["im-current", "im-older"])
        requests = [
            call.args[0] for call in client._stub.ImageTagRevisions.call_args_list
        ]
        self.assertEqual([request.page_token for request in requests], ["", "page-2"])

    def test_image_lookup_not_found(self) -> None:
        client = Mock()
        client._stub.ImageFromId = AsyncMock(
            side_effect=modal.exception.NotFoundError("missing image")
        )
        with patch.object(
            common._Client, "from_env", new=AsyncMock(return_value=client)
        ):
            self.assertFalse(common.image_exists("im-missing"))
        request = client._stub.ImageFromId.call_args.args[0]
        self.assertEqual(request.image_id, "im-missing")

    @parametrize("error_name", ["PermissionDeniedError", "ServiceError"])
    def test_image_lookup_does_not_hide_rpc_errors(self, error_name: str) -> None:
        error_type = getattr(modal.exception, error_name)
        client = Mock()
        client._stub.ImageFromId = AsyncMock(side_effect=error_type("lookup failed"))
        with patch.object(
            common._Client, "from_env", new=AsyncMock(return_value=client)
        ):
            with self.assertRaisesRegex(error_type, "lookup failed"):
                common.image_exists("im-current")

    @parametrize("verification", ["still_exists", "lookup_failed"])
    def test_delete_reports_unsuccessful_verification(self, verification: str) -> None:
        error = (
            modal.exception.ServiceError("verification unavailable")
            if verification == "lookup_failed"
            else None
        )
        with (
            patch.object(modal.experimental, "image_delete") as delete,
            patch.object(common, "image_exists", return_value=True, side_effect=error),
        ):
            result = common.delete_image("im-current")
        self.assertIsInstance(result, str)
        self.assertTrue(result)
        delete.assert_called_once_with("im-current")


instantiate_parametrized_tests(TestSandboxHardware)
instantiate_parametrized_tests(TestSandboxFilesystem)
instantiate_parametrized_tests(TestSandboxImageDeletion)


if __name__ == "__main__":
    run_tests()
