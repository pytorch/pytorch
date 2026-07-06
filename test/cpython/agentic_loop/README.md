# CPython Dynamo Agentic Loop

This directory is the durable home for the CPython Dynamo agentic-loop plan.
Use this file as the maintenance runbook. Treat `coverage.md` as a generated or
regenerable planning artifact.

Main files:

- `START_HERE.md`: fresh-context bootstrap. Point a new agent at this file.
- `coverage.md`: current gate plan, regenerated from the relevance CSV.
- `agent_manager.md`: manager and review-agent workflow instructions.
- `CPYTHON_MIRRORING.md`: CPython object-protocol implementation guidance.
- `cpu_fast_ci_baseline.md`: fast CPU validation baseline and expected failures.
- `cpython_dynamo_expected_failure_relevance.csv`: ranked actionable backlog.
- `cpython_dynamo_expected_failure_relevance_minimal.csv`: import-friendly
  two-column ranking.

## Check Whether Top Tests Still Fail

The top-ranked tests are expected to be present in
`test/dynamo_expected_failures/`. With the sentinel present:

- an underlying Dynamo failure is reported as `SKIPPED`;
- an underlying pass is reported as an unexpected-success failure and the
  sentinel should be removed.

Use this scratch command to check the current top 10 from the ranking CSV:

```bash
python - <<'PY'
import csv
import re
import subprocess
from pathlib import Path

csv_path = Path("test/cpython/agentic_loop/cpython_dynamo_expected_failure_relevance.csv")

def key_to_nodeid(key: str) -> str:
    m = re.fullmatch(r"CPython313-(.+?)-([^.]+)\.(.+)", key)
    if m is None:
        raise RuntimeError(f"bad key: {key}")
    module, cls, test = m.groups()
    if module == "test_assertions":
        path = "test/cpython/v3_13/test_unittest/test_assertions.py"
    else:
        path = f"test/cpython/v3_13/{module}.py"
    return f"{path}::{cls}::{test}"

with csv_path.open(newline="") as f:
    rows = list(csv.DictReader(f))[:10]

for row in rows:
    key = row["testname"]
    nodeid = key_to_nodeid(key)
    cmd = [
        "pytest",
        "-q",
        "--tb=short",
        nodeid,
    ]
    env = {
        "CUDA_VISIBLE_DEVICES": "",
        "PYTORCH_TESTING_DEVICE_ONLY_FOR": "cpu",
        "PYTORCH_TEST_WITH_DYNAMO": "1",
    }
    result = subprocess.run(
        cmd,
        env={**__import__("os").environ, **env},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    status = "still_expected_failure"
    if "Unexpected success" in result.stdout:
        status = "passes_remove_sentinel"
    elif result.returncode not in (0, 5):
        status = "real_failure_or_infra_failure"
    print(f"{status}: {key}")
PY
```

If a top-10 test reports `passes_remove_sentinel`, verify it directly, remove
the corresponding sentinel, update the relevance CSVs, and regenerate
`coverage.md`.

## Update Relevance CSVs

The relevance CSVs are the source for gate ordering. They should contain tests
that are still actionable expected failures.

When a test starts passing:

1. Remove its sentinel from `test/dynamo_expected_failures/`.
2. Remove the row from `cpython_dynamo_expected_failure_relevance.csv`.
3. Remove the row from `cpython_dynamo_expected_failure_relevance_minimal.csv`.
4. Re-rank both files so ranks are contiguous and sorted by descending
   `relevance_score`.
5. Regenerate `coverage.md` so the gates are the new top 10 rows.

Use this scratch command after deleting rows to repair ranks and the minimal
CSV:

```bash
python - <<'PY'
import csv
from pathlib import Path

full = Path("test/cpython/agentic_loop/cpython_dynamo_expected_failure_relevance.csv")
minimal = Path("test/cpython/agentic_loop/cpython_dynamo_expected_failure_relevance_minimal.csv")

with full.open(newline="") as f:
    rows = list(csv.DictReader(f))

rows.sort(key=lambda r: (-float(r["relevance_score"]), r["testname"]))
for i, row in enumerate(rows, 1):
    row["rank"] = str(i)

with full.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

with minimal.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["testname", "relevance_score"])
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "testname": row["testname"],
                "relevance_score": row["relevance_score"],
            }
        )
PY
```

## Regenerate `coverage.md`

Regenerate `coverage.md` whenever the top 10 ranked rows change. The regenerated
file should preserve:

- the ground rules;
- the CPU fast validation command;
- the one-gate-per-test structure;
- the active status marker;
- the "Proposed Gate Changes Awaiting Human Approval" section.

For each of the top 10 rows, create one gate with:

- gate number and short title;
- target sentinel key;
- target pytest nodeid;
- relevance score;
- baseline failure kind;
- likely source areas;
- exit criteria requiring sentinel removal, focused regression coverage, target
  test pass, affected CPython file run, CPU fast validation, and exactly one
  gate commit.

Use the current top 10 rows:

```bash
python - <<'PY'
import csv
from pathlib import Path

path = Path("test/cpython/agentic_loop/cpython_dynamo_expected_failure_relevance.csv")
with path.open(newline="") as f:
    for row in list(csv.DictReader(f))[:10]:
        print(
            row["rank"],
            row["testname"],
            row["relevance_score"],
            row["failure_kind"],
            sep=" | ",
        )
PY
```

After regenerating, run:

```bash
git diff --check -- test/cpython/agentic_loop
```

Do not commit `agent_space/` scratch output.
