# Owner(s): ["module: ci"]

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterable


_SCHEMA_VERSION = 4


def _load_output(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as input_file:
        output = json.load(input_file)
    if output.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(f"Unsupported TD tracer schema in {path}")
    required_types = {
        "run_id": str,
        "complete": bool,
        "successful": bool,
        "usable": bool,
        "running_participants": list,
        "participants": list,
        "environments": list,
        "coverage_by_test": dict,
    }
    for key, expected_type in required_types.items():
        if type(output.get(key)) is not expected_type:
            raise ValueError(f"Invalid TD tracer {key} in {path}")
    if not output["run_id"]:
        raise ValueError(f"TD tracer output has no run ID in {path}")
    if output["usable"] != (output["complete"] and output["successful"]):
        raise ValueError(f"Inconsistent TD tracer status in {path}")
    if output["complete"] and output["running_participants"]:
        raise ValueError(
            f"Completed TD tracer output has running participants in {path}"
        )
    if not all(isinstance(value, str) for value in output["running_participants"]):
        raise ValueError(f"Invalid TD tracer running participants in {path}")
    participant_types = {
        "complete": bool,
        "exit_status": int,
        "participant_id": str,
        "pid": int,
        "session_id": str,
        "worker_id": str,
    }
    for participant in output["participants"]:
        if not isinstance(participant, dict) or any(
            type(participant.get(key)) is not expected_type
            for key, expected_type in participant_types.items()
        ):
            raise ValueError(f"Invalid TD tracer participant in {path}")
    if not all(
        isinstance(environment, dict)
        and (
            environment.get("revision") is None
            or isinstance(environment.get("revision"), str)
        )
        for environment in output["environments"]
    ):
        raise ValueError(f"Invalid TD tracer environments in {path}")
    if not all(
        isinstance(test, str)
        and isinstance(dependencies, list)
        and all(isinstance(dependency, str) for dependency in dependencies)
        for test, dependencies in output["coverage_by_test"].items()
    ):
        raise ValueError(f"Invalid TD tracer coverage in {path}")
    return output


def merge_td_tracer_outputs(
    paths: Iterable[Path],
    run_id: str,
    workflow_run_id: str,
    workflow_run_attempt: int,
    expected_shards: Iterable[str],
) -> dict[str, Any]:
    if not run_id:
        raise ValueError("TD tracer run ID must not be empty")
    if not workflow_run_id:
        raise ValueError("Workflow run ID must not be empty")
    if workflow_run_attempt <= 0:
        raise ValueError("Workflow run attempt must be positive")
    expected_shard_list = list(expected_shards)
    expected_shard_set = set(expected_shard_list)
    if not expected_shard_set:
        raise ValueError("Expected shards must not be empty")
    if len(expected_shard_set) != len(expected_shard_list):
        raise ValueError("Expected shards must be unique")

    source_run_ids = {
        f"{workflow_run_id}-{attempt}-{shard}": (attempt, shard)
        for attempt in range(1, workflow_run_attempt + 1)
        for shard in expected_shard_set
    }
    coverage_by_test: dict[str, set[str]] = collections.defaultdict(set)
    environments: dict[str, dict[str, Any]] = {}
    revisions: set[str] = set()
    selected_outputs: dict[str, tuple[int, dict[str, Any]]] = {}
    seen_source_run_ids: set[str] = set()
    for path in sorted(paths):
        output = _load_output(path)
        source_run_id = output["run_id"]
        if source_run_id in seen_source_run_ids:
            raise ValueError(f"Duplicate TD tracer run ID: {source_run_id}")
        seen_source_run_ids.add(source_run_id)
        source = source_run_ids.get(source_run_id)
        if source is None:
            raise ValueError(f"Unexpected TD tracer run ID: {source_run_id}")
        for test, dependencies in output.pop("coverage_by_test").items():
            coverage_by_test[test].update(dependencies)
        for environment in output["environments"]:
            environments[json.dumps(environment, sort_keys=True)] = environment
            revision = environment.get("revision")
            if revision is not None:
                revisions.add(revision)
        attempt, shard = source
        selected = selected_outputs.get(shard)
        if selected is None or attempt > selected[0]:
            selected_outputs[shard] = (attempt, output)

    outputs = [selected_outputs[shard][1] for shard in sorted(selected_outputs)]
    participants: list[dict[str, Any]] = []
    running_participants: set[str] = set()

    for output in outputs:
        participants.extend(output["participants"])
        running_participants.update(output["running_participants"])

    has_all_shards = set(selected_outputs) == expected_shard_set
    complete = (
        has_all_shards
        and not running_participants
        and len(revisions) <= 1
        and all(output["complete"] for output in outputs)
    )
    successful = has_all_shards and all(output["successful"] for output in outputs)
    return {
        "schema_version": _SCHEMA_VERSION,
        "run_id": run_id,
        "complete": complete,
        "successful": successful,
        "usable": complete and successful,
        "running_participants": sorted(running_participants),
        "participants": sorted(
            participants, key=lambda participant: participant["participant_id"]
        ),
        "environments": [environments[key] for key in sorted(environments)],
        "coverage_by_test": {
            test: sorted(dependencies)
            for test, dependencies in sorted(coverage_by_test.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge TD tracer outputs produced by separate CI shards."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workflow-run-id", required=True)
    parser.add_argument("--workflow-run-attempt", type=int, required=True)
    parser.add_argument("--expected-shard", action="append", required=True)
    args = parser.parse_args()

    paths = list(args.input_dir.glob("**/td_result.json"))
    merged = merge_td_tracer_outputs(
        paths,
        args.run_id,
        args.workflow_run_id,
        args.workflow_run_attempt,
        args.expected_shard,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(merged, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    print(
        f"Processed {len(paths)} TD tracer artifacts for "
        f"{len(args.expected_shard)} expected shards"
    )


if __name__ == "__main__":
    main()
