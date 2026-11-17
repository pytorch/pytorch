"""pytest runner with resume + hang/crash handling
python3 /any/where/mulit_run_case.py test/test_ops.py
Optional --crash-list <file>: case skip list; listed cases are deselected, new CRASH/HANG cases are appended.
Optional --failed-list <file>: on exit, append FAILED/ERROR/CRASH/HANG cases as `nodeid<Tab>status`.
"""

import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------
CHILD_FLAG = "--resilient-child"
CHILD_ROOT_FLAG = "--resilient-repo-root"
ROOT_FLAG = "--repo-root"
CRASH_LIST_FLAG = "--crash-list"
FAILED_LIST_FLAG = "--failed-list"
CRASH_LIST_ENV = "RESILIENT_CRASH_LIST"
PLUGIN_MODULE = os.path.splitext(os.path.basename(__file__))[0]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_CWD = tempfile.gettempdir()
POLL_SECONDS = 5
OK_STATUSES = frozenset({"PASSED", "SKIPPED"})


def _parse_cli(argv):
    opts, rest = {}, []
    i = 0
    while i < len(argv):
        a = argv[i]
        name, eq, value = a.partition("=")
        if name in (ROOT_FLAG, CRASH_LIST_FLAG, FAILED_LIST_FLAG):
            if not eq:
                if i + 1 >= len(argv):
                    raise ValueError(f"{a} requires a value")
                i += 1
                value = argv[i]
            opts[name] = value
        else:
            rest.append(a)
        i += 1
    return opts, rest


@dataclass(frozen=True)
class Config:

    repo_root: str         
    progress_file: str      
    per_test_timeout: int
    pytest_args: tuple = ()
    crash_list: str = ""
    failed_list: str = ""

    @classmethod
    def from_env(cls, argv, repo_root, progress_file, crash_list=None,
                 failed_list=None):
        crash_list = crash_list or os.environ.get(CRASH_LIST_ENV, "")
        return cls(
            repo_root=os.path.abspath(repo_root or os.getcwd()),
            progress_file=progress_file,
            per_test_timeout=int(os.environ.get("PER_TEST_TIMEOUT", "180")),
            pytest_args=tuple(argv),
            crash_list=os.path.abspath(crash_list) if crash_list else "",
            failed_list=os.path.abspath(failed_list) if failed_list else "",
        )


# ---------------------------------------------------------------------------
# 2. Progress
# ---------------------------------------------------------------------------

class ProgressLog:

    START, RESULT, TOTAL = "START", "RESULT", "TOTAL"

    def __init__(self, path: str):
        self.path = path

    def record_start(self, nodeid: str) -> None:
        self._append(self.START, nodeid)

    def record_result(self, nodeid: str, status: str) -> None:
        self._append(self.RESULT, nodeid, status)

    def record_total(self, total: int) -> None:
        self._append(self.TOTAL, str(total))

    def _append(self, *fields: str) -> None:
        with open(self.path, "a") as f:
            f.write("\t".join(fields) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _lines(self):
        if not os.path.exists(self.path):
            return
        with open(self.path) as f:
            for line in f:
                yield line.rstrip("\n").split("\t")

    def done_nodeids(self) -> set:
        return {p[1] for p in self._lines()
                if p[0] == self.RESULT and len(p) >= 3}

    def stuck_nodeid(self):
        running = None
        for p in self._lines():
            if p[0] == self.START:
                running = p[1]
            elif p[0] == self.RESULT and len(p) >= 3 and running == p[1]:
                running = None
        return running

    def total_cases(self):
        total = None
        for p in self._lines():
            if p[0] == self.TOTAL and len(p) >= 2:
                total = int(p[1])
        return total

    def results(self) -> dict:
        """First recorded result per nodeid, consistent with status_counts."""
        results = {}
        for p in self._lines():
            if p[0] == self.RESULT and len(p) >= 3:
                results.setdefault(p[1], p[2])
        return results

    def status_counts(self) -> dict:
        counts, seen = {}, set()
        for p in self._lines():
            if p[0] == self.RESULT and len(p) >= 3 and p[1] not in seen:
                seen.add(p[1])
                counts[p[2]] = counts.get(p[2], 0) + 1
        return counts

    def remove(self) -> None:
        if os.path.exists(self.path):
            os.remove(self.path)


class CrashList:

    def __init__(self, path: str):
        self.path = path

    def nodeids(self) -> set:
        if not self.path or not os.path.exists(self.path):
            return set()
        with open(self.path) as f:
            return {line.split("\t", 1)[0].strip() for line in f
                    if line.strip() and not line.lstrip().startswith("#")}

    def matching_nodeids(self, items) -> set:
        entries = self.nodeids()
        exact = {entry for entry in entries if "::" in entry}
        patterns = tuple(entry for entry in entries if "::" not in entry)
        return {item.nodeid for item in items
                if item.nodeid in exact
                or any(pattern in item.nodeid for pattern in patterns)}

    def add(self, nodeid: str, status: str) -> bool:
        """Append an entry; return False if no path is set or the full nodeid already exists."""
        if not self.path or nodeid in self.nodeids():
            return False
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(f"{nodeid}\t{status}\n")
            f.flush()
            os.fsync(f.fileno())
        return True


# ---------------------------------------------------------------------------
# 3. pytest plugin (child process loads these hooks via -p <this module>)
# ---------------------------------------------------------------------------

class _RepoSourceGuard:

    GUARDED = ("torch", "torchgen", "functorch")

    def __init__(self, repo_root: str):
        self._repo_root = os.path.abspath(repo_root)

    def find_spec(self, fullname, path=None, target=None):
        from importlib.machinery import PathFinder
        root, name = fullname, PathFinder
        if root not in self.GUARDED or path is not None:
            return None
        spec_repo = name.find_spec(root, path)
        if spec_repo is None or not self._inside_repo(spec_repo):
            return None 
        clean = [p for p in sys.path
                 if p and os.path.abspath(p) != self._repo_root]
        spec_installed = name.find_spec(root, clean)
        if spec_installed is None:
            return spec_repo 
        if self._is_dev_build(spec_repo):
            return spec_repo
        print(f"[resilient] guard: import {root!r} redirected to installed "
              f"package ({spec_installed.origin}); repo source copy "
              f"{spec_repo.origin} would shadow it", file=sys.stderr)
        return spec_installed

    def _inside_repo(self, spec) -> bool:
        loc = spec.origin or (spec.submodule_search_locations or [""])[0]
        try:
            return os.path.commonpath(
                [os.path.abspath(loc), self._repo_root]) == self._repo_root
        except ValueError:
            return False

    @staticmethod
    def _is_dev_build(spec) -> bool:

        import glob
        pkg_dir = os.path.dirname(spec.origin or "")
        if not pkg_dir:
            return False
        return any(
            os.path.isfile(p)
            for pat in ("_C*.so", "_C*.pyd")
            for p in glob.glob(os.path.join(pkg_dir, pat))
        )


_PLUGIN_LOG = None


def _plugin_log() -> ProgressLog:
    global _PLUGIN_LOG
    if _PLUGIN_LOG is None:
        _PLUGIN_LOG = ProgressLog(
            os.environ.get("RESILIENT_PROGRESS_FILE", "test_progress.log")
        )
    return _PLUGIN_LOG


def pytest_collection_modifyitems(config, items):
    done = _plugin_log().done_nodeids()
    crash_list = CrashList(os.environ.get(CRASH_LIST_ENV, ""))
    listed = crash_list.matching_nodeids(items)
    skip = done | listed
    if not skip:
        return
    deselected = [it for it in items if it.nodeid in skip]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = [it for it in items if it.nodeid not in skip]
    # Only notify on the first round (no progress yet) to avoid repeating every round
    if listed and not done:
        print(f"\n[resilient] crash list: deselected {len(listed)} listed "
              f"case(s) from {crash_list.path}", flush=True)


def pytest_collection_finish(session):
    log = _plugin_log()
    done = log.done_nodeids()
    total = len(done) + len(session.items)
    log.record_total(total)
    print(f"\n[resilient] cumulative progress: done={len(done)}/{total}, "
          f"remaining={len(session.items)}", flush=True)


def pytest_runtest_logstart(nodeid, location):
    _plugin_log().record_start(nodeid)


def pytest_runtest_logreport(report):
    log = _plugin_log()
    if report.when == "call":
        log.record_result(report.nodeid, report.outcome.upper())
    elif report.when == "setup" and report.outcome != "passed":
        status = "SKIPPED" if report.skipped else "ERROR"
        log.record_result(report.nodeid, status)


# ---------------------------------------------------------------------------
# 4. Monitor
# ---------------------------------------------------------------------------

class Watchdog:

    def __init__(self, proc, progress: ProgressLog, cfg: Config):
        self._proc = proc
        self._progress = progress
        self._cfg = cfg
        self._stop = threading.Event()
        self._thread = None
        self.killed = False

    def start(self) -> "Watchdog":
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        return self

    def wait(self) -> int:
        rc = self._proc.wait()
        self._stop.set()
        self._thread.join(timeout=POLL_SECONDS + 2)
        return rc

    def _watch(self) -> None:
        last_size, last_change = -1, time.time()
        while not self._stop.wait(POLL_SECONDS):
            try:
                size = os.path.getsize(self._progress.path)
            except OSError:
                size = -1
            if size != last_size:
                last_size, last_change = size, time.time()
            elif time.time() - last_change > self._cfg.per_test_timeout:
                self.killed = True
                self._kill_group()
                return

    def _kill_group(self) -> None:
        try:
            pgid = os.getpgid(self._proc.pid)
            os.killpg(pgid, signal.SIGINT) 
            time.sleep(10)
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass 


# ---------------------------------------------------------------------------
# 5. orchestrator
# ---------------------------------------------------------------------------

def _abs_arg(arg: str, cfg: Config) -> str:
    if arg.startswith("-"):
        return arg
    path, sep, rest = arg.partition("::")
    candidate = os.path.join(cfg.repo_root, path)
    if os.path.exists(candidate) or path.endswith(".py") or os.sep in path:
        return os.path.abspath(candidate) + sep + rest
    return arg


def _child_env(cfg: Config) -> dict:
    env = dict(os.environ)
    clean = [
        p for p in env.get("PYTHONPATH", "").split(os.pathsep)
        if p and os.path.abspath(p) not in (cfg.repo_root, SCRIPT_DIR)
    ]
    env["PYTHONPATH"] = os.pathsep.join([SCRIPT_DIR, *clean])
    env["PYTHONSAFEPATH"] = "1"
    if cfg.crash_list:
        env[CRASH_LIST_ENV] = cfg.crash_list
    else:
        env.pop(CRASH_LIST_ENV, None)
    env["RESILIENT_PROGRESS_FILE"] = cfg.progress_file 
    return env


def _has_verbosity_flag(args) -> bool:
    for a in args:
        if a in ("--verbose", "--quiet"):
            return True
        if a.startswith("-") and not a.startswith("--"):
            body = a[1:]
            if "v" in body or "q" in body:
                return True
    return False


def _spawn_child(cfg: Config) -> subprocess.Popen:
    user_args = [_abs_arg(a, cfg) for a in cfg.pytest_args]
    if not _has_verbosity_flag(user_args):
        user_args = ["-v", *user_args]
    cmd = [
        sys.executable, os.path.abspath(__file__), CHILD_FLAG,
        CHILD_ROOT_FLAG, cfg.repo_root,
        "-p", PLUGIN_MODULE,
        "-p", "no:cacheprovider",
        *user_args,
    ]

    return subprocess.Popen(
        cmd, env=_child_env(cfg), cwd=RUN_CWD, preexec_fn=os.setsid
    )


def _print_summary(log: ProgressLog) -> dict:
    counts = log.status_counts()
    if not counts:
        print("no progress recorded")
        return counts
    total = sum(counts.values())
    print("\n================ SUMMARY ================")
    print(f"total finished cases: {total}")
    for status in sorted(counts):
        rate = counts[status] / total * 100
        print(f"  {status:<16} {counts[status]:>6}  ({rate:5.1f}%)")
    print("=========================================")
    return counts


def _write_failed_list(cfg: Config, log: ProgressLog, aborted_rc=None) -> None:
    if not cfg.failed_list:
        return
    existing = CrashList(cfg.failed_list).nodeids()
    new = [(nodeid, status) for nodeid, status in log.results().items()
           if status not in OK_STATUSES and nodeid not in existing]
    if not new and aborted_rc is None:
        return
    directory = os.path.dirname(cfg.failed_list)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(cfg.failed_list, "a") as f:
        f.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')}  "
                f"{' '.join(cfg.pytest_args)}\n")
        # No nodeid on a no-progress abort; write a comment line only so it isn't taken as a whole-file skip entry
        if aborted_rc is not None:
            f.write(f"# ABORTED rc={aborted_rc} with no progress\n")
        for nodeid, status in new:
            f.write(f"{nodeid}\t{status}\n")
        f.flush()
        os.fsync(f.fileno())
    print(f"appended {len(new)} failed case(s) to failed list {cfg.failed_list}")


def orchestrate(cfg: Config) -> int:
    try:
        import pytest  # noqa: F401
    except ImportError:
        print(f"ERROR: pytest not found for {sys.executable}; "
              f"run with a python that has pytest installed.")
        return 2

    log = ProgressLog(cfg.progress_file)
    crash_list = CrashList(cfg.crash_list)
    if cfg.crash_list:
        print(f"crash list: {cfg.crash_list} "
              f"({len(crash_list.nodeids())} exclusion entries)")
    newly_listed = []
    aborted = False
    rc = None
    rounds = 0
    while True:
        rounds += 1
        prev_done = log.done_nodeids()
        total = log.total_cases()
        progress = (
            f"done={len(prev_done)}/{total}"
            if total is not None else f"done={len(prev_done)}/?"
        )
        print(f"\n===== round {rounds}: {progress} =====")

        watchdog = Watchdog(_spawn_child(cfg), log, cfg).start()
        rc, killed = watchdog.wait(), watchdog.killed

        stuck = log.stuck_nodeid()
        if stuck:
            status = "HANG" if killed else "CRASH"
            log.record_result(stuck, status)
            print(f"  detected stuck case -> marked {status}: {stuck}")
            if crash_list.add(stuck, status):
                newly_listed.append((stuck, status))
            continue 

        if rc in (0, 1, 5):
            print(f"  pytest exited normally (rc={rc}); all remaining cases done.")
            break

        if len(log.done_nodeids()) <= len(prev_done):
            print(f"  WARN: abnormal exit rc={rc} with no progress; aborting to "
                  f"avoid an infinite loop.")
            aborted = True
            break

    counts = _print_summary(log)
    if newly_listed:
        print(f"appended {len(newly_listed)} new case(s) to crash list "
              f"{cfg.crash_list} (deselected from next run on):")
        for nodeid, status in newly_listed:
            print(f"  {status:<6} {nodeid}")
    _write_failed_list(cfg, log, aborted_rc=rc if aborted else None)
    if aborted:
        return 1
    return 1 if set(counts) - OK_STATUSES else 0


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------

def _run_as_child(args, repo_root) -> int:
    repo_root = os.path.abspath(repo_root)
    sys.path[:] = [p for p in sys.path if p and os.path.abspath(p) != repo_root]

    if SCRIPT_DIR not in sys.path:
        sys.path.append(SCRIPT_DIR)
    sys.meta_path.insert(0, _RepoSourceGuard(repo_root))
    import pytest
    return pytest.main(list(args))


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    if CHILD_FLAG in argv:
        rest, repo_root = [], os.getcwd()
        for i, a in enumerate(argv):
            if a == CHILD_ROOT_FLAG:
                repo_root = argv[i + 1]
            elif a != CHILD_FLAG and argv[i - 1] != CHILD_ROOT_FLAG:
                rest.append(a)
        return _run_as_child(rest, repo_root)

    try:
        opts, user_args = _parse_cli(argv)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 2

    fd, progress_file = tempfile.mkstemp(prefix="resilient_progress_", suffix=".log")
    os.close(fd)
    try:
        cfg = Config.from_env(user_args or ["test/test_ops.py"],
                              repo_root=opts.get(ROOT_FLAG),
                              progress_file=progress_file,
                              crash_list=opts.get(CRASH_LIST_FLAG),
                              failed_list=opts.get(FAILED_LIST_FLAG))
        return orchestrate(cfg)
    finally:
        ProgressLog(progress_file).remove()


if __name__ == "__main__":
    sys.exit(main())
