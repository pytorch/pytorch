from __future__ import annotations

import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

from tools import ramjet
from tools.ramjet_schedule import load_history, update_history
from tools.test.test_ramjet import FakeImageBuildError, worker_result

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


FIRST = "test/test_first.py::Suite::test_first"
SECOND = "test/test_second.py::Suite::test_second"


@instantiate_parametrized_tests
class TestRamjetScheduleMain(TestCase):
    @parametrize(
        "case", ["success", "failure", "invalid_junit", "wrong_identity", "collection"]
    )
    @parametrize("write_error", [False, True])
    def test_history_orders_inputs_and_learns_only_validated_results(
        self, case, write_error
    ):
        modal = mock.MagicMock()
        modal.exception.ImageBuildError = FakeImageBuildError
        app = modal.App.return_value
        app.app_id = "ap-ramjet-schedule-test"
        app.run.return_value.__exit__.return_value = False
        functions = {
            "collection_targets": mock.Mock(remote=mock.Mock(return_value=[["test"]])),
            "collect_file": mock.Mock(
                map=mock.Mock(
                    return_value=[
                        {
                            "markers": {
                                node: [ramjet.CPU_PLATFORM] for node in (FIRST, SECOND)
                            }
                        }
                    ]
                )
            ),
            "run_tests": mock.Mock(),
        }
        app.function.side_effect = lambda **options: lambda function: functions[
            function.__name__
        ]
        if case == "collection":
            functions["collect_file"].map.return_value = [
                RuntimeError("broken collection")
            ]

        workers = {
            node: worker_result(
                [node],
                platform=ramjet.CPU_PLATFORM,
                started_at=100,
                finished_at=100 + seconds,
            )
            for node, seconds in ((FIRST, 11), (SECOND, 3))
        }
        if case == "failure":
            report = ET.fromstring(workers[SECOND]["junit"])
            testcase = report.find(".//testcase")
            if testcase is None:
                raise AssertionError("Expected a test case in the worker fixture")
            ET.SubElement(testcase, "failure", message="unrelated assertion failure")
            workers[SECOND]["junit"] = ET.tostring(report, encoding="unicode")
        elif case == "invalid_junit":
            workers[SECOND]["junit"] = "{not XML"
        elif case == "wrong_identity":
            workers[SECOND]["junit"] = workers[FIRST]["junit"]

        def prepare(root, snapshots, base_image, volume, plan, clean):
            plan.update(
                build_cache_hit=True, source_delta_bytes=0, build_image_id="im-cached"
            )
            return mock.MagicMock()

        def run_shards(test, shards, jobs, gpu_jobs, output):
            self.assertEqual(shards, {"cpu": [[SECOND], [FIRST]]})
            return [workers[batch[0]] for batch in shards["cpu"]]

        with tempfile.TemporaryDirectory(prefix="ramjet-schedule-main-") as directory:
            project = Path(directory)
            output = project / "run-current"
            output.mkdir()
            previous = project / "run-previous"
            path = project / "timings.json"
            update_history(
                path,
                [
                    worker_result(
                        [node],
                        platform=ramjet.CPU_PLATFORM,
                        started_at=100,
                        finished_at=end,
                    )
                    for node, end in ((FIRST, 105), (SECOND, 150))
                ],
                previous,
            )
            before = path.read_bytes()
            snapshots = {
                "build": (project / "build.json", "build-hash"),
                "tests": (project / "tests.tar.gz", "test-hash"),
                "bootstrap": (project / "bootstrap", "bootstrap-hash"),
            }
            with (
                mock.patch.dict(sys.modules, {"modal": modal}),
                mock.patch.dict(ramjet.os.environ, {}, clear=True),
                mock.patch.object(sys, "argv", ["ramjet", "--platform", "cpu", "test"]),
                mock.patch.object(ramjet, "normalize_targets", return_value=["test"]),
                mock.patch.object(ramjet, "create_run_directory", return_value=output),
                mock.patch.object(ramjet, "snapshot_source", return_value=snapshots),
                mock.patch.object(ramjet, "prepare_image", side_effect=prepare),
                mock.patch.object(
                    ramjet, "prepare_fixtures", side_effect=lambda image, *args: image
                ),
                mock.patch.object(
                    ramjet,
                    "prepare_dependencies",
                    side_effect=lambda image, *args: image,
                ),
                mock.patch.object(
                    ramjet, "prepare_cli_tools", side_effect=lambda image, *args: image
                ),
                mock.patch.object(ramjet, "save_build"),
                mock.patch.object(ramjet, "run_shards", side_effect=run_shards) as run,
                mock.patch.object(
                    ramjet, "update_history", wraps=update_history
                ) as update,
                mock.patch("builtins.print"),
            ):
                if write_error:
                    update.side_effect = OSError("timing cache is read-only")
                if case == "collection":
                    with self.assertRaisesRegex(RuntimeError, "collection failed"):
                        ramjet.main()
                else:
                    self.assertEqual(ramjet.main(), int(case != "success"))

            result = json.loads(output.joinpath("result.json").read_text())
            plan = json.loads(output.joinpath("plan.json").read_text())
            self.assertEqual(result["returncode"], int(case != "success"))
            self.assertEqual(plan["timing_history"], str(path))
            if case == "collection":
                run.assert_not_called()
                update.assert_not_called()
                self.assertEqual(path.read_bytes(), before)
                return
            self.assertNotIn("error", result)
            run.assert_called_once()
            update.assert_called_once()
            self.assertEqual(plan["shards"], {"cpu": [[SECOND], [FIRST]]})
            self.assertEqual(plan["scheduling"]["cpu"]["sources"], [str(previous)])
            self.assertEqual(plan["scheduling"]["cpu"]["exact_batches"], 2)
            self.assertEqual(
                [record["nodes"] for record in result["shards"]], [[SECOND], [FIRST]]
            )
            self.assertEqual(result["shards"][0]["returncode"], int(case != "success"))
            self.assertEqual(workers[SECOND]["returncode"], 0)
            if write_error:
                self.assertEqual(
                    result["timing_history_error"], "timing cache is read-only"
                )
                self.assertEqual(path.read_bytes(), before)
            else:
                self.assertEqual(
                    result["timing_history_updated_batches"],
                    2 if case == "success" else 1,
                )
                files = load_history(path)["platforms"][ramjet.CPU_PLATFORM]["files"]
                self.assertEqual(files["test/test_first.py"]["source"], str(output))
                self.assertEqual(files["test/test_first.py"]["seconds"], 11)
                expected_source = output if case == "success" else previous
                self.assertEqual(
                    files["test/test_second.py"]["source"], str(expected_source)
                )
                self.assertEqual(
                    files["test/test_second.py"]["seconds"],
                    3 if case == "success" else 50,
                )


if __name__ == "__main__":
    run_tests()
