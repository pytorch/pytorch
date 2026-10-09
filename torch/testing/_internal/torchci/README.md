# Test run reports

`plugin.py` writes one JSON Lines file per test process for the `tests.*`
ClickHouse tables: a report line, then one line per completed test run. Lines are
flushed as tests finish; readers skip a torn last line. Split lines on `\n`, not
with `str.splitlines()`, which also breaks at U+2028, U+2029 and U+0085 inside
strings.

## Path

`<prefix>-<report_uuid>.report.jsonl`, for example
`test/torchci-reports/test_type_info-919108f7-52d1-4320-9bac-f847db4148a8.report.jsonl`.
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
| `flags` | object | `TestEnvironment` flags and settings, described below, sorted by env var |
| `properties` | object | Nonempty report properties listed below |

`environment` contains these fields in order:

| Field | Type | Source |
|---|---|---|
| `os` | string | `linux`, `macos`, `windows`, or the platform name |
| `os_version` | string | OS version |
| `cpu_architecture` | string | Normalized machine architecture |
| `cpu_capability` | string | `torch.backends.cpu.get_cpu_capability()` lower-cased (`default`, `avx2`, `avx512`, `sve256`, ...), with `ATEN_CPU_CAPABILITY` applied; `amx` for `avx512` on CPUs with AMX tiles |
| `python_version` | string | Python major.minor, with `t` for free-threaded builds |
| `cc_compiler` | string | `gcc`, `clang`, `msvc`, or `""` |
| `cc_compiler_version` | string | Compiler major version |
| `accelerator` | string | `cpu`, `cuda`, `rocm`, `xpu`, or `mps` |
| `accelerator_version` | string | CUDA toolkit, ROCm release (not HIP), or SYCL compiler major.minor; `""` for CPU and MPS |
| `device_count` | int | Visible devices of `accelerator`, honoring `CUDA_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES`; `0` for CPU |
| `device_name` | string | The first visible device's name as its vendor reports it (`NVIDIA L4`, `AMD Instinct MI350X VF`, `Apple M2 Pro`, `Intel(R) Data Center GPU Max 1100`); empty with no device |

`flags` maps the env var of every flag and setting registered with
`include_in_repro` to its value as set, unparsed (`PYTORCH_CUDA_ALLOC_CONF` is
the allocator config string, such as `expandable_segments:True`), or `""` when
unset. An unset flag that is on by default or implication is `"1"`.

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
| `outcome` | string | `passed`, `failed`, `error`, `skipped`, `xfailed`, or `xpassed` |
| `outcome_summary` | string | Failure, skip, or xfail summary; otherwise `""` |
| `started_at` | int | Setup start as Unix epoch milliseconds |
| `ended_at` | int | Last phase stop as Unix epoch milliseconds |
| `properties` | object | Empty in version 0.1 |

Pytest reruns and flakefinder repeats within one process count up
`rerun_number`. A test's subtests share its run line, and a failing subtest fails
it.

A test aliased under a longer name (`test_x_dynamic = TestX.test_x`) reports the
original's `declared_case_name`, `test_x`.

## Versioning

Readers compare `schema_version` exactly. While the format is 0.x, a breaking
change bumps the minor version (`"0.1"` to `"0.2"`); additive changes keep it.
`"1.0"` will be the first stable format.

## Local check

With PyTorch built from this branch, from `test/`:

```bash
python -m pytest test_type_info.py -p torch.testing._internal.torchci.plugin \
    --torchci-report-prefix=/tmp/torchci-reports/test_type_info
cat /tmp/torchci-reports/test_type_info-*.report.jsonl | python -m json.tool --json-lines
```
