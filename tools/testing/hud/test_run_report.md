# Test run report format

PyTorch writes one JSON Lines file per pytest process. The first line describes
the run context; every later line describes one completed test run. Lines are
flushed as tests finish, there is no end marker, and readers skip a torn line.

## Path

`test/test-run-reports/<launched file>/<launched file>-<16 hex>.jsonl`

The folder is the launched file with path separators replaced by dots, for
example `test_type_info`, `distributed.test_c10d_nccl`, or `cpp.test_api`.
There is no launcher-level folder. Each process, including each `--subprocess`
child, gets a separate file.

## Upload

At job end, `.github/actions/upload-test-artifacts` zips the folder into
`test-run-reports-<file suffix>.zip`, with entries under `test/test-run-reports/`,
and uploads it to
`s3://gha-artifacts/<org>/<repo>/<run id>/<run attempt>/artifact/`. It is kept
apart from the junit zips, and the in-run uploader leaves it out, so only the
final copy is uploaded. When S3 is skipped (ROCm) or this upload fails, the
action uploads the GitHub artifact
`test-run-reports-runattempt<N>-<file suffix>.zip`, which
`tools/stats/upload_artifacts.py` copies to the same S3 key.

## Schema version 0.1

Every listed field is present and key order is stable. `properties` objects are
open-ended string-to-string maps.

### Report line

| Field | Type | Meaning |
|---|---|---|
| `type` | string | `"report"` |
| `schema_version` | string | `"0.1"` |
| `repo` | string | `GITHUB_REPOSITORY`, or `""` outside CI |
| `github_workflow_job_id` | int | Numeric `JOB_ID`, or `0` when unset or invalid |
| `environment` | object | Machine identity fields listed below |
| `flags` | object | Every registered `TestEnvironment` setting, sorted by name, with string values |
| `properties` | object | Nonempty report properties listed below |

`environment` contains these fields in order:

| Field | Type | Source |
|---|---|---|
| `os` | string | `linux`, `macos`, `windows`, or the platform name |
| `os_version` | string | OS version |
| `cpu_architecture` | string | Normalized machine architecture |
| `cpu_capability` | string | PyTorch CPU capability, including AMX detection |
| `python_version` | string | Python major.minor, with `t` for free-threaded builds |
| `cc_compiler` | string | `gcc`, `clang`, `msvc`, or `""` |
| `cc_compiler_version` | string | Compiler major version |
| `accelerator` | string | `cpu`, `cuda`, `rocm`, `xpu`, or `mps` |
| `accelerator_version` | string | Accelerator major.minor version |
| `device_count` | int | Visible accelerator count |
| `device_name` | string | Raw NVML, amdsmi, xpu-smi, or MPS device string; empty with no device |

`flags` includes every setting registered with `include_in_repro`. Flags use
`"1"` or `"0"`, other settings use their value, and unset settings use `""`.

The report `properties` writer emits these keys when their values are nonempty:
`torch_version`, `os_release`, `device_memory_mib`, `driver_version`,
`host_memory_mib`, `build_environment`, `test_config`, and `runner_name`.
`host_memory_mib` is the cgroup v2 memory limit when one is set, otherwise
physical RAM; it is omitted where neither is available.

### Run line

| Field | Type | Meaning |
|---|---|---|
| `type` | string | `"run"` |
| `schema_version` | string | `"0.1"` |
| `file` | string | Launched Python file, or `cpp/<binary>` |
| `suite` | string | Innermost Python class or gtest suite |
| `case_name` | string | Collected test name, including generated parameters |
| `declared_case_name` | string | Source test name before generated device, dtype, op, or parameter suffixes |
| `rerun_number` | int | Execution number within one process, starting at `0` |
| `outcome` | string | `passed`, `failed`, `error`, `skipped`, `xfailed`, `xpassed`, `crashed`, or `timed_out` |
| `outcome_summary` | string | Failure, skip, xfail, crash, or timeout summary; otherwise `""` |
| `started_at` | int | Setup start as Unix epoch milliseconds |
| `ended_at` | int | Last phase stop as Unix epoch milliseconds |
| `properties` | object | Empty in version 0.1 |

For pytest-cpp value-parameterized tests, `declared_case_name` drops the gtest
suffix (`Name/0` becomes `Name`).

## Crash recording

- `test/run_test.py` appends the in-flight test after a retry process crashes
  or times out.
- xdist reports worker crashes to the controller, including C++ tests and
  Python runs using `-n`.
- A `--subprocess` parent writes the final crashed or timed-out attempt to its
  own report file. A crash followed by a passing retry inside `retry_shell` is
  not recorded.

Fresh `run_test.py` retry processes write fresh files, so their numbering
starts at zero. Pytest reruns and flakefinder repeats within one process count
up in that process's file.

## Enablement

Reports default on in CI and off locally. Both `test/run_test.py` and tests using
`common_utils.run_tests` accept:

- `--save-test-run-reports [DIR]` to enable reports.
- `--no-save-test-run-reports` to disable reports explicitly.

For `test/run_test.py`, the default is `<repo>/test/test-run-reports`, a relative
`DIR` is resolved under `<repo>/test`, and an absolute path is unchanged. For
`common_utils.run_tests`, the default is `test-run-reports` relative to the
current working directory.

## Versioning

Readers compare `schema_version` exactly. While the format is 0.x, a breaking
change bumps the minor version (`0.1` to `0.2`); additive changes keep the
version. `1.0` will be the first stable format.

## Local check

From the repository root:

```bash
python test/run_test.py -i test_type_info --save-test-run-reports
ls test/test-run-reports/test_type_info/
head -1 test/test-run-reports/test_type_info/*.jsonl | python -m json.tool
```
