# Test run reports

`plugin.py` writes one JSON Lines file per test process for the `tests.*`
ClickHouse tables: a report line, then one line per completed test run. Lines are
flushed as tests finish; readers skip a torn last line.

## Path

`<dir>/<dir name>-<report_uuid>.jsonl`, for example
`test/torchci-reports/test_type_info/test_type_info-919108f7-52d1-4320-9bac-f847db4148a8.jsonl`.
Each process writes its own file under a new uuid4.

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
| `report_uuid` | string | The uuid4 in the file name |
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
| `device_name` | string | Normalized model of the first visible device (`l4`, `h100`, `mi350x`, `m2`, `max1100`); an unknown name lower-cased with dashes; empty with no device |

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
| `file` | string | Launched Python file |
| `suite` | string | Innermost Python class |
| `case_name` | string | Collected test name, including generated parameters |
| `language` | string | `python` |
| `declared_case_name` | string | Source test name before generated device, dtype, op, or parameter suffixes |
| `rerun_number` | int | Execution number within one process, starting at `0` |
| `outcome` | string | `passed`, `failed`, `error`, `skipped`, `xfailed`, `xpassed`, `crashed`, or `timed_out` |
| `outcome_summary` | string | Failure, skip, xfail, crash, or timeout summary; otherwise `""` |
| `started_at` | int | Setup start as Unix epoch milliseconds |
| `ended_at` | int | Last phase stop as Unix epoch milliseconds |
| `properties` | object | Empty in version 0.1 |

Pytest reruns and flakefinder repeats within one process count up
`rerun_number`. A test's subtests share its run line, and a failing subtest fails
it.

## Crash recording

A process that dies can't record the test it was running, so whoever sees it die
records that run as `crashed`, or `timed_out` after a timeout:

| Process | Recorded by |
|---|---|
| xdist worker | The xdist controller, from xdist's crash report |
| `run_test.py` retry process | `run_test.py`, from the in-flight run the writer publishes in the stepcurrent cache (`recovery.finish`) |
| `run_tests --subprocess` child | The parent, in a report of its own (`recovery.record_dead_subprocess`) |

Each retry process writes its own file, so its `rerun_number` starts at 0. A crash
followed by a passing retry inside `retry_shell` isn't recorded.

## Versioning

Readers compare `schema_version` exactly. While the format is 0.x, a breaking
change bumps the minor version (`"0.1"` to `"0.2"`); additive changes keep it.
`"1.0"` will be the first stable format.

## Enablement

Off by default. `test/run_test.py` and tests using `common_utils.run_tests`
accept `--save-test-run-reports [DIR]` and `--no-save-test-run-reports`. Each
test file gets its own folder, `<DIR>/<file>/`, with path separators replaced by
dots (`distributed.test_c10d_nccl`). `run_test.py` defaults to
`test/torchci-reports` and resolves a relative `DIR` under `test/`; `run_tests`
defaults to `torchci-reports` in the cwd.

## Local check

```bash
python test/run_test.py -i test_type_info --save-test-run-reports
head -1 test/torchci-reports/test_type_info/*.jsonl | python -m json.tool
```
