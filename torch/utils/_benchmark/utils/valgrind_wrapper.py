import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap

import torch


class _ValgrindWrapper(object):
    def __init__(self):
        self._valgrind_available = not subprocess.run(
            ["which", "valgrind"], capture_output=True).returncode

        self._build_type = None
        build_search = re.search("BUILD_TYPE=(.+),", torch.__config__.show())
        if build_search is not None:
            self._build_type = build_search.groups()[0].split(",")[0]

        self._baseline_cache = {}

    def _validate(self):
        if not self._valgrind_available:
            raise OSError("Could not find `valgrind`")

        if self._build_type is None:
            raise OSError("Could not determine BUILD_TYPE")

        if self._build_type != "RelWithDebInfo":
            raise OSError(
                "PyTorch sould be built with REL_WITH_DEB_INFO=1 "
                "(debug symbols) in order to produce a useful C++ call graph, "
                f"got BUILD_TYPE={self._build_type} instead"
            )

    def collect_callgrind(self, stmt: str, setup: str, number: int, num_threads: int):
        self._validate()
        if number not in self._baseline_cache:
            self._baseline_cache[(number, num_threads)] = self._invoke(
                stmt="pass", setup="pass", number=number, num_threads=num_threads)

        # We include `pass` so it matches the baseline and reduces the instruction diff.
        fn_counts = self._invoke(
            stmt=f"pass;{stmt}", setup=setup, number=number, num_threads=num_threads)

        count_dict = {fn: count for count, fn in fn_counts}
        for count, fn in self._baseline_cache[(number, num_threads)]:
            count_dict.setdefault(fn, 0)
            count_dict[fn] -= count
            if not count_dict[fn]:
                count_dict.pop(fn)

        fn_diff_counts = []
        for fn, count in sorted(count_dict.items(), key=lambda x: x[1], reverse=True):
            if count < 0:
                # The profiling process is not 100% deterministic. For instance
                # the subprocess calls to control callgrind poll resulting in
                # a non-deterministic number of instructions. This delta is
                # generally small.
                # FIXME: Validate that differences are small and expected, and warn otherwise.
                break
            fn_diff_counts.append((count, fn))
        return fn_diff_counts

    def _invoke(self, stmt: str, setup: str, number: int, num_threads: int):
        working_dir = tempfile.mkdtemp()

        _, script_file = tempfile.mkstemp(prefix="timer_callgrind_", suffix=".py", dir=working_dir)
        _, callgrind_out = tempfile.mkstemp(prefix="callgrind.", suffix=".out", dir=working_dir)
        _, error_log = tempfile.mkstemp(prefix="error_", suffix=".txt", dir=working_dir)

        try:
            with open(script_file, "wt") as f:
                f.write(self._construct_script(
                    stmt=stmt, setup=setup, number=number,
                    num_threads=num_threads, error_log=error_log))

            valgrind_invocation = subprocess.run(
                [
                    "valgrind",
                    "--tool=callgrind",
                    f"--callgrind-out-file={callgrind_out}",
                    "--dump-line=yes",
                    "--dump-instr=yes",
                    "--collect-jumps=yes",
                    "--instr-atstart=no",
                    "python",
                    script_file,
                ],
                capture_output=True
            )
            if valgrind_invocation.returncode:
                with open(error_log, "rt") as f:
                    error_report = f.read()
                if not error_report:
                    error_report = "Unknown error."
                    error_report += "\n" + valgrind_invocation.stdout.decode("utf-8")
                    error_report += "\n" + valgrind_invocation.stderr.decode("utf-8")
                raise OSError(f"Failed to collect callgrind profile:\n{error_report}")

            annotate_invocation = subprocess.run(
                [
                    "callgrind_annotate",
                    "--inclusive=no",
                    callgrind_out
                ],
                capture_output=True,
                check=True,
            )

            begin_collecting = False
            fn_counts = []
            for l in annotate_invocation.stdout.decode("utf-8").splitlines(keepends=False):
                if not begin_collecting and re.match(r"Ir\s+file:function", l):
                    begin_collecting = True
                    continue

                count_match = re.match(r"^\s*([0-9,]+)\s+(.+:.+)$", l)
                if count_match:
                    ir, file_function = count_match.groups()
                    ir = int(ir.replace(",", ""))
                    fn_counts.append((ir, file_function))
                    continue

                if begin_collecting and re.match(r"-+", l):
                    continue

                begin_collecting = False

            return fn_counts
        finally:
            shutil.rmtree(working_dir)

    @staticmethod
    def _construct_script(stmt: str, setup: str, number: int, num_threads: int, error_log: str):
        return textwrap.dedent(r"""
            import os
            import subprocess
            import sys

            import torch
            torch.set_num_threads({num_threads})

            PID = os.getpid()

            def log_failure(msg):
                with open("{error_log}", "wt") as f:
                    f.write(msg)
                sys.exit(1)

            def check_result(completed_process):
                if completed_process.returncode:
                    log_failure(f"Command failed: {{' '.join(completed_process.args)}}")
                return completed_process

            # =============================================================================
            # == Check that subprocess matches parent =====================================
            # =============================================================================
            if sys.executable != "{parent_interpreter}":
                log_failure(
                    "Interpreter mismatch:\n"
                    f"  {{sys.executable}}\n    vs.\n  {parent_interpreter}"
                )

            if torch.__file__ != "{torch_file}":
                log_failure(
                    "PyTorch does not match expected file:\n"
                    f"  {{torch.__file__}}\n    vs.\n  {torch_file}"
                )

            # =============================================================================
            # == User specified setup =====================================================
            # =============================================================================
            {setup}

            for _ in range({warmup_number}):
                {stmt}

            # =============================================================================
            # == Callgrind management =====================================================
            # =============================================================================
            callgrind_stat = check_result(subprocess.run(
                ["callgrind_control", "--stat"],
                capture_output=True,
            ))

            stat_lines = callgrind_stat.stdout.decode("utf-8").splitlines()
            if f"PID {{PID}}: python {{__file__}}" not in stat_lines:
                log_failure("Process does not appear to be running callgrind.")

            # Disable and zero counters to ensure a clean run.
            for cmd in ["--instr=off", "--zero", "--instr=on"]:
                check_result(subprocess.run(
                    ["callgrind_control", cmd, str(PID)],
                    capture_output=True,
                ))

            # =============================================================================
            # == User code block ==========================================================
            # =============================================================================
            for _ in range({number}):
                {stmt}

            # Ensure that cleanup / shutdown code is not instrumented.
            check_result(subprocess.run(
                ["callgrind_control", "--instr=off", str(PID)],
                capture_output=True,
            ))
        """).strip().format(
            stmt=stmt,
            setup=setup,
            number=number,
            warmup_number=min(number, 10),
            num_threads=num_threads,
            error_log=error_log,
            parent_interpreter=sys.executable,
            torch_file=torch.__file__,
        )

CALLGRIND_SINGLETON = None
def wrapper_singleton():
    global CALLGRIND_SINGLETON
    if CALLGRIND_SINGLETON is None:
        CALLGRIND_SINGLETON = _ValgrindWrapper()
    return CALLGRIND_SINGLETON
