import glob
import json
import locale
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Any


_ORIGINAL_POPEN = subprocess.Popen
_ORIGINAL_CALL = subprocess.call
_ORIGINAL_RUN = subprocess.run
_INSTALLED = False
_LOG_LOCK = threading.Lock()
_MAX_TEXT = 1000


def _debug_dir() -> str:
    path = os.path.join(os.path.dirname(__file__), "test-reports", "subprocess-debug")
    os.makedirs(path, exist_ok=True)
    return path


def _log_path() -> str:
    return os.path.join(_debug_dir(), f"subprocess_{os.getpid()}.jsonl")


def _output_path(stream: str) -> str:
    fd, path = tempfile.mkstemp(
        dir=_debug_dir(), prefix=f"subprocess_{os.getpid()}_", suffix=f".{stream}"
    )
    os.close(fd)
    return path


def _is_devnull(value: Any) -> bool:
    if value == subprocess.DEVNULL:
        return True
    if isinstance(value, int) and value >= 0:
        try:
            return os.path.abspath(os.readlink(f"/proc/self/fd/{value}")) == os.path.abspath(
                os.devnull
            )
        except Exception:
            return False
    try:
        return os.path.abspath(value.name) == os.path.abspath(os.devnull)
    except Exception:
        return False


def _short_text(value: Any) -> str:
    if isinstance(value, bytes):
        text = os.fsdecode(value)
    else:
        text = str(value)
    if len(text) > _MAX_TEXT:
        return text[:_MAX_TEXT] + "...<truncated>"
    return text


def _command_text(args: Any) -> Any:
    if isinstance(args, (list, tuple)):
        return [_short_text(arg) for arg in args]
    return _short_text(args)


def _stdio_info(value: Any) -> Any:
    if value is None:
        return "inherit"
    if value == subprocess.PIPE:
        return "PIPE"
    if value == subprocess.STDOUT:
        return "STDOUT"
    if value == subprocess.DEVNULL:
        return "DEVNULL"
    name = getattr(value, "name", None)
    if name is not None:
        return {"file": _short_text(name)}
    return _short_text(value)


def _spawn_stack() -> list[str]:
    stack = traceback.extract_stack()[:-2]
    lines = []
    for frame in stack:
        filename = frame.filename
        if filename.endswith("subprocess.py") or filename.endswith("subprocess_debug.py"):
            continue
        try:
            filename = os.path.relpath(
                filename, os.path.dirname(os.path.dirname(__file__))
            )
        except ValueError:
            pass
        lines.append(f"{filename}:{frame.lineno} in {frame.name}")
    return lines[-8:]


def _write_record(record: dict[str, Any]) -> None:
    record["time"] = time.time()
    try:
        with _LOG_LOCK:
            with open(_log_path(), "a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
    except Exception:
        pass


class _TracedPopen(_ORIGINAL_POPEN):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        start_time = time.time()
        command = args[0] if args else kwargs.get("args")
        kwargs, spooled_files = _maybe_spool_devnull_outputs(kwargs)
        try:
            super().__init__(*args, **kwargs)
        except Exception as exc:
            _close_files(spooled_files)
            _write_record(
                {
                    "event": "spawn_error",
                    "parent_pid": os.getpid(),
                    "command": _command_text(command),
                    "cwd": _short_text(kwargs.get("cwd") or os.getcwd()),
                    "shell": bool(kwargs.get("shell", False)),
                    "error": repr(exc),
                    "stack": _spawn_stack(),
                }
            )
            raise

        self._pytorch_subprocess_debug_start = start_time
        self._pytorch_subprocess_debug_finished = False
        self._pytorch_subprocess_debug_trace_id = (
            f"{os.getpid()}:{self.pid}:{start_time}"
        )
        self._pytorch_subprocess_debug_spooled_files = spooled_files
        _write_record(
            {
                "event": "start",
                "trace_id": self._pytorch_subprocess_debug_trace_id,
                "parent_pid": os.getpid(),
                "child_pid": self.pid,
                "command": _command_text(command),
                "cwd": _short_text(kwargs.get("cwd") or os.getcwd()),
                "shell": bool(kwargs.get("shell", False)),
                "stdout": _stdio_info(kwargs.get("stdout")),
                "stderr": _stdio_info(kwargs.get("stderr")),
                "stack": _spawn_stack(),
            }
        )

    def _record_finish(self) -> None:
        if getattr(self, "_pytorch_subprocess_debug_finished", False):
            return
        if self.returncode is None:
            return
        self._pytorch_subprocess_debug_finished = True
        _close_files(getattr(self, "_pytorch_subprocess_debug_spooled_files", []))
        start_time = getattr(self, "_pytorch_subprocess_debug_start", None)
        elapsed = time.time() - start_time if start_time is not None else None
        _write_record(
            {
                "event": "finish",
                "trace_id": getattr(self, "_pytorch_subprocess_debug_trace_id", None),
                "parent_pid": os.getpid(),
                "child_pid": self.pid,
                "returncode": self.returncode,
                "elapsed_seconds": elapsed,
            }
        )

    def wait(self, *args: Any, **kwargs: Any) -> int:
        try:
            return super().wait(*args, **kwargs)
        finally:
            self._record_finish()

    def communicate(self, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        try:
            return super().communicate(*args, **kwargs)
        finally:
            self._record_finish()

    def poll(self) -> int | None:
        result = super().poll()
        self._record_finish()
        return result


def _maybe_spool_devnull_outputs(kwargs: dict[str, Any]) -> tuple[dict[str, Any], list[Any]]:
    spooled_kwargs = kwargs.copy()
    files = []
    stdout = spooled_kwargs.get("stdout")
    stderr = spooled_kwargs.get("stderr")

    if _is_devnull(stdout):
        path = _output_path("stdout")
        stdout_file = open(path, "w+b")
        files.append(stdout_file)
        spooled_kwargs["stdout"] = stdout_file
        if stderr is stdout:
            spooled_kwargs["stderr"] = stdout_file
            return spooled_kwargs, files

    if _is_devnull(stderr):
        path = _output_path("stderr")
        stderr_file = open(path, "w+b")
        files.append(stderr_file)
        spooled_kwargs["stderr"] = stderr_file

    return spooled_kwargs, files


def _text_mode(kwargs: dict[str, Any]) -> bool:
    return bool(
        kwargs.get("text")
        or kwargs.get("universal_newlines")
        or kwargs.get("encoding")
        or kwargs.get("errors")
    )


def _read_output_file(path: str, kwargs: dict[str, Any]) -> bytes | str:
    with open(path, "rb") as f:
        data = f.read()
    if not _text_mode(kwargs):
        return data
    encoding = kwargs.get("encoding") or locale.getpreferredencoding(False)
    errors = kwargs.get("errors") or "strict"
    return data.decode(encoding, errors)


def _maybe_spool_outputs(
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], list[Any], dict[str, str]]:
    spooled_kwargs = kwargs.copy()
    files = []
    paths = {}

    if spooled_kwargs.get("stdout") == subprocess.PIPE:
        paths["stdout"] = _output_path("stdout")
        stdout_file = open(paths["stdout"], "w+b")
        files.append(stdout_file)
        spooled_kwargs["stdout"] = stdout_file

    if spooled_kwargs.get("stderr") == subprocess.PIPE:
        paths["stderr"] = _output_path("stderr")
        stderr_file = open(paths["stderr"], "w+b")
        files.append(stderr_file)
        spooled_kwargs["stderr"] = stderr_file

    return spooled_kwargs, files, paths


def _close_files(files: list[Any]) -> None:
    for f in files:
        try:
            f.close()
        except Exception:
            pass


def _traced_run(
    *popenargs: Any,
    input: Any = None,
    capture_output: bool = False,
    timeout: float | None = None,
    check: bool = False,
    **kwargs: Any,
) -> Any:
    if input is not None:
        if kwargs.get("stdin") is not None:
            raise ValueError("stdin and input arguments may not both be used.")
        kwargs["stdin"] = subprocess.PIPE

    if capture_output:
        if kwargs.get("stdout") is not None or kwargs.get("stderr") is not None:
            raise ValueError(
                "stdout and stderr arguments may not be used with capture_output."
            )
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE

    captures_stdout = kwargs.get("stdout") == subprocess.PIPE
    captures_stderr = kwargs.get("stderr") == subprocess.PIPE
    if not captures_stdout and not captures_stderr:
        return _ORIGINAL_RUN(
            *popenargs,
            input=input,
            capture_output=False,
            timeout=timeout,
            check=check,
            **kwargs,
        )

    spooled_kwargs, files, paths = _maybe_spool_outputs(kwargs)
    try:
        result = _ORIGINAL_RUN(
            *popenargs,
            input=input,
            capture_output=False,
            timeout=timeout,
            check=False,
            **spooled_kwargs,
        )
        if captures_stdout:
            result.stdout = _read_output_file(paths["stdout"], kwargs)
        if captures_stderr:
            result.stderr = _read_output_file(paths["stderr"], kwargs)
        if check and result.returncode:
            raise subprocess.CalledProcessError(
                result.returncode,
                result.args,
                output=result.stdout,
                stderr=result.stderr,
            )
        return result
    except subprocess.TimeoutExpired as exc:
        if captures_stdout:
            exc.output = _read_output_file(paths["stdout"], kwargs)
            exc.stdout = exc.output
        if captures_stderr:
            exc.stderr = _read_output_file(paths["stderr"], kwargs)
        raise
    finally:
        _close_files(files)


def _traced_call(
    *popenargs: Any, timeout: float | None = None, **kwargs: Any
) -> int:
    captures_stdout = kwargs.get("stdout") == subprocess.PIPE
    captures_stderr = kwargs.get("stderr") == subprocess.PIPE
    if not captures_stdout and not captures_stderr:
        return _ORIGINAL_CALL(*popenargs, timeout=timeout, **kwargs)

    spooled_kwargs, files, _ = _maybe_spool_outputs(kwargs)
    try:
        return _ORIGINAL_CALL(*popenargs, timeout=timeout, **spooled_kwargs)
    finally:
        _close_files(files)


def install() -> None:
    global _INSTALLED
    if _INSTALLED or subprocess.Popen is _TracedPopen:
        return
    subprocess.Popen = _TracedPopen
    subprocess.run = _traced_run
    subprocess.call = _traced_call
    _INSTALLED = True
    _start_watchdog()


def _read_file(path: str) -> str:
    try:
        with open(path, errors="replace") as f:
            return f.read().strip()
    except Exception as exc:
        return f"<unavailable: {exc!r}>"


def _read_cmdline(pid: int) -> str:
    text = _read_file(f"/proc/{pid}/cmdline")
    return text.replace("\x00", " ").strip()


def _read_link(path: str) -> str:
    try:
        return os.readlink(path)
    except Exception as exc:
        return f"<unavailable: {exc!r}>"


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _children(pid: int) -> list[int]:
    children: list[int] = []
    for path in glob.glob(f"/proc/{pid}/task/*/children"):
        for child in _read_file(path).split():
            try:
                children.append(int(child))
            except ValueError:
                pass
    return sorted(set(children))


def _descendants(pid: int) -> list[int]:
    seen: set[int] = set()
    pending = [pid]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        pending.extend(_children(current))
    return [p for p in seen if p != pid]


def _proc_state_char(pid: int) -> str:
    # Single-char task state (R/S/D/Z/T) from /proc/<pid>/stat field 3. D = the
    # uninterruptible kernel wedge we are trying to confirm.
    try:
        data = _read_file(f"/proc/{pid}/stat")
        return data[data.rfind(")") + 2]
    except Exception:
        return "?"


def _gpu_fds(pid: int) -> list[str]:
    # fds pointing at the GPU driver nodes — the concrete link that makes a hang
    # "GPU related" even for a non-GPU process (it inherited them from a torch parent).
    found: list[str] = []
    try:
        for name in os.listdir(f"/proc/{pid}/fd"):
            tgt = _read_link(f"/proc/{pid}/fd/{name}")
            if "/dev/kfd" in tgt or "/dev/dri/render" in tgt or "/dev/dri/card" in tgt:
                found.append(f"{name} -> {tgt}")
    except Exception:
        pass
    return found


def _kernel_stacks(pid: int) -> str:
    # THE DECISIVE datum: per-thread kernel stack, readable even for a D-state task.
    # Needs CAP_SYS_ADMIN (root / --cap-add=SYS_ADMIN); best-effort otherwise (the
    # wchan/syscall/state above already classify the cause without it).
    lines: list[str] = []
    try:
        tids = sorted(os.listdir(f"/proc/{pid}/task"))
    except Exception:
        tids = [str(pid)]
    for tid in tids:
        lines.append(f"[tid {tid}] {_read_file(f'/proc/{pid}/task/{tid}/stack')}")
    return "\n".join(lines)


def _dump_proc_state(pid: int, out: Any) -> None:
    print(f"--- /proc state for pid={pid} ---", file=out)
    print(f"cmdline: {_read_cmdline(pid)}", file=out)
    print(f"cwd: {_read_link(f'/proc/{pid}/cwd')}", file=out)
    print(f"state: {_proc_state_char(pid)}", file=out)
    print(f"wchan: {_read_file(f'/proc/{pid}/wchan')}", file=out)
    print(f"syscall: {_read_file(f'/proc/{pid}/syscall')}", file=out)
    gpu = _gpu_fds(pid)
    print(f"gpu_fds: {gpu if gpu else 'none'}", file=out)
    print(f"kernel_stack:\n{_kernel_stacks(pid)}", file=out)
    print(_read_file(f"/proc/{pid}/status"), file=out)


def _dump_output_file(label: str, path: str, out: Any) -> None:
    try:
        with open(path, "rb") as f:
            content = f.read().decode("utf-8", "replace")
    except Exception as exc:
        content = f"<unavailable: {exc!r}>"
    print(f"--- captured {label} path={path} ---", file=out)
    print(content, file=out)


def _dump_record_output(record: dict[str, Any], out: Any) -> None:
    seen: set[str] = set()
    for label in ("stdout", "stderr"):
        info = record.get(label)
        if not isinstance(info, dict):
            continue
        path = info.get("file")
        if not isinstance(path, str) or path in seen:
            continue
        seen.add(path)
        _dump_output_file(label, path, out)


def _recent_records(max_age_seconds: float) -> list[dict[str, Any]]:
    cutoff = time.time() - max_age_seconds
    records: list[dict[str, Any]] = []
    try:
        paths = glob.glob(os.path.join(_debug_dir(), "subprocess_*.jsonl"))
    except Exception:
        return records
    for path in paths:
        try:
            if os.path.getmtime(path) < cutoff:
                continue
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("time", 0.0) >= cutoff:
                        records.append(record)
        except Exception:
            pass
    return records


def dump_recent_subprocess_traces(
    reason: str, max_age_seconds: float = 1800.0
) -> None:
    records = _recent_records(max_age_seconds)
    starts: dict[str, dict[str, Any]] = {}
    finished: set[str] = set()
    for record in records:
        trace_id = record.get("trace_id")
        child_pid = record.get("child_pid")
        if not isinstance(trace_id, str):
            continue
        if not isinstance(child_pid, int):
            continue
        if record.get("event") == "start":
            starts[trace_id] = record
        elif record.get("event") == "finish":
            finished.add(trace_id)

    active = [
        record
        for trace_id, record in starts.items()
        if trace_id not in finished and _process_exists(record["child_pid"])
    ]
    if not active:
        return

    out = sys.__stderr__ or sys.stderr
    print(
        f"\n===== PYTORCH SUBPROCESS DEBUG BEGIN reason={reason} =====",
        file=out,
        flush=True,
    )
    for record in sorted(active, key=lambda record: record["child_pid"]):
        pid = record["child_pid"]
        print(json.dumps(record, sort_keys=True), file=out)
        _dump_record_output(record, out)
        _dump_proc_state(pid, out)
        for child in _descendants(pid):
            _dump_proc_state(child, out)
    print("===== PYTORCH SUBPROCESS DEBUG END =====\n", file=out, flush=True)


# ---------------------------------------------------------------------------
# Kernel-level hang classification + proactive watchdog (added for MI355 triage)
#
# WHY: the Python-stack dump above (a) only ever shows Python frames, so it cannot
# say *why* a child is stuck, and (b) only fires when something sends SIGUSR1 / on a
# run_test.py per-file timeout -- but the worst MI355 hang
# (inductor/test_torchinductor_opinfo_properties) gets NO per-file timeout and nothing
# sends SIGUSR1, so it produced zero useful data and just burned the 270-min job.
#
# This watchdog PROACTIVELY detects a stuck/D-state traced child and captures the
# KERNEL evidence (task state + wchan + syscall + GPU fds + per-thread kernel stack
# + dmesg + rocm-smi) and emits a one-line VERDICT classifying the cause:
#   amdgpu/kfd driver D-state wedge  vs  memory/mmu-notifier reclaim  vs
#   IPC pipe / sccache socket  vs  userspace futex lock  vs  spinning.
# That verdict is what lets a CI hang be attributed to (or ruled out as) the
# amdgpu/kfd hypothesis. State/wchan/gpu-fds ARE readable as the non-root `jenkins`
# CI user and ALREADY classify the cause (e.g. wchan=amdgpu_fill_buffer / kfd_* /
# svm_range_* + state=D => CONFIRMED). The full kernel stack needs CAP_SYS_ADMIN
# *effective*, which the non-root container user does not have (CapEff=0 even with
# --cap-add), so it is best-effort here and is instead grabbed by the runner-host
# step in _rocm-test.yml (root); classification does NOT depend on it.
# ---------------------------------------------------------------------------

_HANG_CAPTURED: set[int] = set()
_DSTATE_SAMPLES: dict[int, int] = {}
_CPU_LAST: dict[int, tuple[int, float]] = {}
_WATCH_INTERVAL = float(os.environ.get("PYTORCH_HANG_WATCH_INTERVAL", "20"))
_WATCH_D_SAMPLES = int(os.environ.get("PYTORCH_HANG_D_SAMPLES", "2"))
_WATCH_STUCK_SECS = float(os.environ.get("PYTORCH_HANG_STUCK_SECS", "180"))

# First matching family wins. amdgpu/kfd first so a GPU wedge is never mis-bucketed.
_CLASSIFY_RULES = [
    ("amdgpu_kfd_driver_wedge",
     ("amdgpu", "kfd_", "kgd2kfd", "amdttm", "ttm_bo", "svm_range", "dma_fence",
      "drm_sched", "drm_gem", "amdgpu_mn")),
    ("memory_or_mmu_notifier_reclaim",
     ("mem_cgroup", "try_to_free_pages", "shrink_", "__alloc_pages",
      "mmu_notifier", "mn_invl", "do_swap_page", "handle_over_high")),
    ("ipc_pipe_or_sccache_socket",
     ("pipe_read", "pipe_write", "sk_wait_data", "tcp_recvmsg", "inet_recvmsg",
      "unix_stream_read", "do_select", "ep_poll", "do_poll")),
    ("userspace_futex_lock", ("futex",)),
    ("filesystem_io", ("wait_on_page", "folio_wait", "__lock_page", "jbd2", "nfs")),
]


def _cpu_ticks(pid: int) -> int:
    try:
        data = _read_file(f"/proc/{pid}/stat")
        parts = data[data.rfind(")") + 2:].split()
        return int(parts[11]) + int(parts[12])  # utime + stime
    except Exception:
        return -1


def classify_hang(pid: int) -> tuple[str, str]:
    """Return (verdict, evidence) for a suspected-stuck pid from kernel /proc state.

    Decision: D-state AND (amdgpu/kfd/ttm/svm in wchan|kernel-stack, OR blocked in an
    ioctl while holding a /dev/kfd|renderD fd) => the claimed amdgpu/kfd driver wedge.
    """
    state = _proc_state_char(pid)
    wchan = _read_file(f"/proc/{pid}/wchan")
    stack = _kernel_stacks(pid)
    syscall = _read_file(f"/proc/{pid}/syscall")
    gpu_fds = _gpu_fds(pid)
    blob = f"{wchan}\n{stack}".lower()

    bucket = None
    for name, needles in _CLASSIFY_RULES:
        if any(n in blob for n in needles):
            bucket = name
            break
    # Fallback when wchan/stack are unhelpful: a D-state task blocked in ioctl(2)
    # while holding a /dev/kfd|renderD fd is the amdgpu/kfd wedge signature.
    # /proc/<pid>/syscall reports the syscall NUMBER (16 == ioctl on x86_64), not its name.
    sc = syscall.strip()
    blocked_in_ioctl = sc.startswith("16 ") or "ioctl" in sc.lower()
    if bucket is None and state == "D" and gpu_fds and blocked_in_ioctl:
        bucket = "amdgpu_kfd_driver_wedge"
    if bucket is None:
        bucket = "spinning" if state == "R" else f"unknown(state={state})"

    confirmed = state == "D" and bucket == "amdgpu_kfd_driver_wedge"
    verdict = ("CONFIRMED amdgpu/kfd driver D-state wedge" if confirmed
               else f"NOT-this-cause: {bucket} (state={state})")
    evidence = (f"state={state} wchan={wchan!r} syscall={syscall!r} "
                f"gpu_fds={gpu_fds or 'none'} bucket={bucket}")
    return verdict, evidence


def capture_hang(pid: int, reason: str) -> None:
    """Dump full kernel evidence + a cause VERDICT for a stuck pid (once per pid)."""
    if pid in _HANG_CAPTURED:
        return
    _HANG_CAPTURED.add(pid)
    verdict, evidence = classify_hang(pid)
    path = os.path.join(_debug_dir(), f"HANG_CAPTURE_{pid}_{int(time.time())}.txt")
    try:
        f: Any = open(path, "w", encoding="utf-8")
        opened = True
    except Exception:
        f = sys.__stderr__ or sys.stderr
        opened = False
    try:
        print(f"===== PYTORCH HANG CAPTURE pid={pid} reason={reason} =====", file=f)
        print(f"VERDICT: {verdict}", file=f)
        print(f"evidence: {evidence}\n", file=f)
        _dump_proc_state(pid, f)
        for child in _descendants(pid):
            _dump_proc_state(child, f)
        for label, cmd in (("dmesg", ["dmesg", "--ctime"]),
                           ("rocm-smi", ["rocm-smi", "--showpids", "--showuse"])):
            try:
                res = _ORIGINAL_RUN(cmd, capture_output=True, text=True, timeout=15)
                text = res.stdout
                if label == "dmesg":
                    text = "\n".join(
                        ln for ln in text.splitlines()
                        if any(k in ln.lower() for k in
                               ("amdgpu", "kfd", "svm_range", "userptr", "hung_task",
                                "blocked for more than", "gpu reset", "ring", "watchdog"))
                    )[-8000:]
                print(f"\n--- {label} ---\n{text}", file=f)
            except Exception as exc:
                print(f"\n--- {label} unavailable: {exc!r} ---", file=f)
    finally:
        if opened:
            try:
                f.close()
            except Exception:
                pass
    err = sys.__stderr__ or sys.stderr
    print(f"[hang-watchdog] pid={pid} {verdict} | {evidence} | full={path}",
          file=err, flush=True)


def _watchdog_active_pids() -> list[int]:
    # Children spawned by THIS process (its own trace log), still alive & unfinished.
    starts: dict[str, int] = {}
    finished: set[str] = set()
    try:
        with open(_log_path(), encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tid, cpid = r.get("trace_id"), r.get("child_pid")
                if not isinstance(tid, str) or not isinstance(cpid, int):
                    continue
                if r.get("event") == "start":
                    starts[tid] = cpid
                elif r.get("event") == "finish":
                    finished.add(tid)
    except Exception:
        return []
    return [c for t, c in starts.items() if t not in finished and _process_exists(c)]


def _watchdog_loop() -> None:
    while True:
        time.sleep(_WATCH_INTERVAL)
        try:
            now = time.time()
            live = set(_watchdog_active_pids())
            for pid in live:
                if pid in _HANG_CAPTURED:
                    continue
                state = _proc_state_char(pid)
                if state == "D":
                    _DSTATE_SAMPLES[pid] = _DSTATE_SAMPLES.get(pid, 0) + 1
                    if _DSTATE_SAMPLES[pid] >= _WATCH_D_SAMPLES:
                        capture_hang(pid, f"Dstate_x{_DSTATE_SAMPLES[pid]}")
                        continue
                else:
                    _DSTATE_SAMPLES.pop(pid, None)
                cpu = _cpu_ticks(pid)
                last = _CPU_LAST.get(pid)
                if last is None or last[0] != cpu:
                    _CPU_LAST[pid] = (cpu, now)
                elif now - last[1] >= _WATCH_STUCK_SECS:
                    capture_hang(pid, f"flatCPU_{int(now - last[1])}s")
            for d in list(_DSTATE_SAMPLES):
                if d not in live:
                    _DSTATE_SAMPLES.pop(d, None)
            for d in list(_CPU_LAST):
                if d not in live:
                    _CPU_LAST.pop(d, None)
        except Exception:
            pass


def _start_watchdog() -> None:
    if os.environ.get("PYTORCH_HANG_WATCHDOG", "1") == "0":
        return
    try:
        threading.Thread(target=_watchdog_loop, name="pytorch-hang-watchdog",
                         daemon=True).start()
    except Exception:
        pass
