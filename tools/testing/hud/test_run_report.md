# Test run report format

PyTorch writes one JSON Lines file per pytest process. The first line describes
the run context; every later line describes one completed test run. Lines are
flushed as tests finish, there is no end marker, and readers skip a torn line.
`test/run_test.py` appends an in-flight test as `crashed` or `timed_out` when a
process exits abnormally.

## Path

`test/test-run-reports/<launched file>/<launched file>-<16 hex>.jsonl`

The folder is the launched file with path separators replaced by dots, for
example `test_type_info`, `distributed.test_c10d_nccl`, or `cpp.test_api`.
There is no launcher-level folder. Each process, including each `--subprocess`
child, gets a separate file.

## Schema version 1

Every listed field is present and key order is stable. `properties` objects are
open-ended string-to-string maps.

### Report line

| Field | Type | Meaning |
|---|---|---|
| `type` | string | `"report"` |
| `schema_version` | int | `1` |
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
| `device_name` | string | Raw NVML, amdsmi, xpu-smi, or sysctl device string; empty with no device |

`flags` includes every registered setting. Flags use `"1"` or `"0"`, other
settings use their value, and unset settings use `""`.

The report `properties` writer emits these keys when their values are nonempty:
`torch_version`, `os_release`, `device_memory_mib`, `driver_version`,
`host_memory_mib`, `build_environment`, `test_config`, and `runner_name`.

### Run line

| Field | Type | Meaning |
|---|---|---|
| `type` | string | `"run"` |
| `schema_version` | int | `1` |
| `file` | string | Launched Python file, or `cpp/<binary>` |
| `suite` | string | Innermost Python class or gtest suite |
| `case_name` | string | Collected test name, including generated parameters |
| `declared_case_name` | string | Source test name before generated device, dtype, op, or parameter suffixes |
| `rerun_number` | int | `0` for the first execution, then `1`, `2`, and so on |
| `outcome` | string | `passed`, `failed`, `error`, `skipped`, `xfailed`, `xpassed`, `crashed`, or `timed_out` |
| `outcome_summary` | string | Failure, skip, xfail, crash, or timeout summary; otherwise `""` |
| `started_at` | int | Setup start as Unix epoch milliseconds |
| `ended_at` | int | Last phase stop as Unix epoch milliseconds |
| `properties` | object | Empty in version 1 |

For pytest-cpp value-parameterized tests, `declared_case_name` drops the gtest
suffix (`Name/0` becomes `Name`).

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

Readers select behavior from `schema_version`. Breaking field changes require a
new version. Additive entries in open-ended `properties` or `flags` maps do not.

## Local check

From the repository root:

```bash
python test/run_test.py -i test_type_info --save-test-run-reports
ls test/test-run-reports/test_type_info/
head -1 test/test-run-reports/test_type_info/*.jsonl | python -m json.tool
```

After removing `test/test-run-reports`, the same `run_test.py` command without
`--save-test-run-reports` must not create that directory.
