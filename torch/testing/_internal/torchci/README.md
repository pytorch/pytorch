# Test run reports

`plugin.py` writes one JSON Lines file per test process for the `tests.*`
ClickHouse tables: a report line, then one line per completed test run. A process
that ends without starting a test removes its file, which would hold only the
report line, unless an error turned the writer off first; one that started a test
keeps it (see Crashes and timeouts). Lines are flushed as tests finish; readers
skip a torn last line. Split lines on `\n`, not with `str.splitlines()`, which
also breaks at U+2028, U+2029 and U+0085 inside strings.

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
| `cc_compiler` | string | `gcc`, `clang`, `msvc`, or `""`, from `torch.__config__._cxx_compiler()`; `""` with a torch that predates it |
| `cc_compiler_version` | string | Compiler version as `torch.__config__._cxx_compiler()` gives it (`11.4`, `21.1.0`, or MSVC's `_MSC_FULL_VER`, `194134120`) |
| `accelerator` | string | `cpu`, `cuda`, `rocm`, `xpu`, or `mps` |
| `accelerator_version` | string | `torch.version.cuda`, `torch.version.rocm` (the ROCm release, not HIP), or `torch.version.xpu` (the SYCL compiler, `20250101`); `""` for CPU and MPS |
| `device_count` | int | Visible devices of `accelerator`, honoring `CUDA_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES`; `0` for CPU |
| `device_name` | string | The first visible device's name as its vendor reports it (`NVIDIA L4`, `AMD Instinct MI350X VF`, `Apple M2 Pro`, `Intel(R) Data Center GPU Max 1100`); empty with no device |

`flags` maps the env var of every flag and setting registered with
`include_in_repro` to its value as set, unparsed (`PYTORCH_CUDA_ALLOC_CONF` is
the allocator config string, such as `expandable_segments:True`), or `""` when
unset. An unset flag that is on by default or implication is `"1"`.

The report `properties` writer emits these keys when their values are nonempty:
`torch_version`, `os_release`, `device_memory_mib`, `driver_version`,
`host_memory_mib`, `build_environment`, and `runner_name`.
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
| `language` | string | `python` or `cpp` |
| `declared_case_name` | string | Source test name before generated device, dtype, op, or parameter suffixes |
| `rerun_number` | int | Execution number within one process, starting at `0` |
| `outcome` | string | `passed`, `failed`, `error`, `skipped`, `xfailed`, or `xpassed`; `crashed` or `timed_out` for a run recorded by `run_test.py` (below) |
| `outcome_summary` | string | Failure, skip, xfail, crash, or timeout summary; otherwise `""` |
| `started_at` | int | Setup start as Unix epoch milliseconds |
| `ended_at` | int | Last phase stop as Unix epoch milliseconds |
| `properties` | object | Empty in version 0.1 |

Pytest reruns and flakefinder repeats within one process count up
`rerun_number`. A test's subtests share its run line, and a failing subtest fails
it. For a value-parameterized gtest, `declared_case_name` drops the `/<param>`
suffix. A gtest whose process crashed is `failed`, with pytest-cpp's message, as in
junit.

A test aliased under a longer name (`test_x_dynamic = TestX.test_x`) reports the
original's `declared_case_name`, `test_x`.

## Crashes and timeouts

A process that dies can't record the test it was running. Under `run_test.py`'s
retry loop (`--sc`, `--scs` or `--rs` without `-n`), the writer publishes the run
in flight as `report_inflight` next to the stepcurrent plugin's `lastrun` in
`.pytest_cache/v/cache/stepcurrent/<key>/`, from each test's start until just
before its run line is written, and `null` in between. The record holds `path`
(the report's absolute path), `report_size` (its size when the run started),
`test` (the run line's identity fields), `rerun_number` and `started_at` (epoch
seconds). On a non-zero exit, `run_test.py` removes it and, unless the exit was
pytest's own (2 interrupted, 3 internal error, 4 usage error) or the report grew
since (the run did finish), appends the run: `timed_out` on `retry_shell`'s 124
(the process hit its time limit, whether it exited on the SIGINT or was killed),
otherwise `crashed`: a signal, an abort, or any other exit with the run still in
flight. `outcome_summary` carries the time limit or the exit code and signal, and
`ended_at` is when `run_test.py` recorded it.

Not covered: launches with `-n` (xdist, every C++ test, mpiexec), `run_tests
--subprocess` children, and rerun-disabled-tests mode.

## Versioning

Readers compare `schema_version` exactly. While the format is 0.x, a breaking
change bumps the minor version (`"0.1"` to `"0.2"`); additive changes keep it.
`"1.0"` will be the first stable format.

## Enablement

`test/run_test.py` and tests using `common_utils.run_tests` write reports when
given `--save-torchci-reports DIR`. A test file's reports are
`<DIR>/<file>-<report_uuid>.report.jsonl`, with path separators in `<file>`
replaced by dots (`distributed.test_c10d_nccl`). A relative `DIR` is relative to
the cwd.

In pytorch/pytorch's CI (`GITHUB_REPOSITORY`), `run_test.py` defaults to
`test/torchci-reports`, which is uploaded as described below, in every job except
on XPU builds, which write none until the XPU CI image has `pyzes`: with it, the
writer counts XPU devices with Level Zero Sysman instead of the XPU runtime.
`run_tests` has no default, so a test file run some other way writes no report.

## Upload

At job end, `.github/actions/upload-test-artifacts` uploads each report as its own
object:

```
s3://gha-artifacts/torchci-reports/<org>/<repo>/<run id>/<job id>/<file>-<report_uuid>.report.jsonl
```

The job id is `job.check_run_id`, which equals the job's id. When S3 is skipped
(ROCm) or the junit upload fails, the action uploads the GitHub artifact
`torchci-reports-runattempt<N>-<job id>.zip` instead, and
`tools/stats/upload_artifacts.py` uploads each report in it to the same key. A row in
the tests tables has its report's `github_workflow_job_id` and `report_uuid`: list
its job's folder and take the name that ends with the uuid.

## Local check

With PyTorch built from this branch, from the repo root:

```bash
python test/run_test.py -i test_type_info --save-torchci-reports /tmp/torchci-reports
cat /tmp/torchci-reports/test_type_info-*.report.jsonl | python -m json.tool --json-lines
```
