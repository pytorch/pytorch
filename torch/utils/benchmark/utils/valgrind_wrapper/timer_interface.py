"""Intermediate layer between `Timer` and `valgrind`."""
import collections
import dataclasses
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from typing import Any, DefaultDict, Dict, List, NamedTuple, Optional, Tuple

import torch


FunctionCount = NamedTuple("FunctionCount", [("count", int), ("function", str)])


@dataclasses.dataclass(repr=False, eq=False, frozen=True)
class CallgrindStats(object):
    stmt: str
    setup: str
    number_per_run: int
    num_threads: int
    built_with_debug_symbols: bool
    baseline_inclusive_stats: Tuple[FunctionCount, ...]
    baseline_exclusive_stats: Tuple[FunctionCount, ...]
    stmt_inclusive_stats: Tuple[FunctionCount, ...]
    stmt_exclusive_stats: Tuple[FunctionCount, ...]

    def __repr__(self) -> str:
        newline = "\n"  # `\` cannot appear in fstring code section.
        base_stats = self.baseline_exclusive_stats
        self_stats = self.stmt_exclusive_stats
        output = textwrap.dedent(f"""
        {super().__repr__()}
          stmt:  {self.stmt.replace(newline, newline + ' ' * 9)}
          setup: {self.setup.replace(newline, newline + ' ' * 9)}
          {self.num_threads} thread{'s' if self.num_threads > 1 else ''}
        {'':>25}All{'':>10}Noisy symbols removed
          Instructions: {self._counts(self_stats, True):>12}{'':>15}{self._counts(self_stats, False):>12}
          Baseline:     {self._counts(base_stats, True):>12}{'':>15}{self._counts(base_stats, False):>12}
        """).strip()
        if not self.built_with_debug_symbols:
            output += textwrap.dedent("""
            Warning: PyTorch was not built with debug symbols.
                     Source information may be limited. Rebuild with
                     REL_WITH_DEB_INFO=1 for more detailed results.""")
        return output

    def stats(self, inclusive: bool = False) -> Tuple[FunctionCount, ...]:
        """Returns stats as a tuple of (count, function)

        `inclusive` matches the semantics of callgrind. If True, the counts
        include instructions executed by children. `inclusive=True` is useful
        for identifying hot spots in code; `inclusive=False` is useful for
        identifying reducing noise when diffing counts from two different
        runs. (See CallgrindStats.delta(...) for more details)
        """
        if inclusive:
            first, second = self.stmt_inclusive_stats, self.baseline_inclusive_stats
        else:
            first, second = self.stmt_exclusive_stats, self.baseline_exclusive_stats
        return self._diff(first, second)

    def counts(self, *, include_lookdict_unicode: bool = True) -> int:
        """Returns the total number of instructions executed.

        Several instructions in the CPython interpreter are rather noisy. These
        instructions involve unicode to dictionary lookups which Python uses to
        map variable names. By default these are included, but setting
        `include_lookdict_unicode=False` will exclude them and generally lead
        to less noisy counts.
        """
        return self._counts(self.stmt_exclusive_stats, include_lookdict_unicode)

    # FIXME: Once 3.7 is the minimum version, type annotate `other` per PEP 563
    def delta(
        self,
        other,  # type: CallgrindStats
        inclusive: bool = False,
        subtract_baselines: bool = True
    ) -> Tuple[FunctionCount, ...]:
        """Diff two sets of counts.

        One common reason to collect instruction counts is to determine the
        the effect that a particular change will have on the number of instructions
        needed to perform some unit of work. If a change increases that number, the
        next logical question is "why". This generally involves looking at what part
        if the code increased in instruction count. This function automates that
        process so that one can easily diff counts on both an inclusive and
        exclusive basis. The `subtract_baselines` argument allows one to disable
        baseline correction, though in most cases it shouldn't matter as the
        baselines are expected to more or less cancel out.
        """
        if subtract_baselines:
            first = self.stats(inclusive=inclusive)
            second = other.stats(inclusive=inclusive)
        else:
            if inclusive:
                first, second = self.stmt_inclusive_stats, other.stmt_inclusive_stats
            else:
                first, second = self.stmt_exclusive_stats, other.stmt_exclusive_stats
        return self._diff(first, second)

    def as_standardized(self) -> "CallgrindStats":
        """Strip library names and some prefixes from function strings.

        When comparing two different sets of instruction counts, on stumbling
        block can be path prefixes. Callgrind includes the full filepath
        when reporting a function (as it should). However, this can cause
        issues when diffing profiles. If a key component such as Python
        or PyTorch was built in separate locations in the two profiles, which
        can result in something resembling:
            23234231 /tmp/first_build_dir/thing.c:foo(...)
             9823794 /tmp/first_build_dir/thing.c:bar(...)
              ...
               53453 .../aten/src/Aten/...:function_that_actually_changed(...)
              ...
             -9823794 /tmp/second_build_dir/thing.c:bar(...)
            -23234231 /tmp/second_build_dir/thing.c:foo(...)

        Stripping prefixes can ameliorate this issue by regularizing the
        strings and causing better cancellation of equivilent call sites
        when diffing.
        """
        def strip(stats: Tuple[FunctionCount, ...]) -> Tuple[FunctionCount, ...]:
            counts: DefaultDict[str, int] = collections.defaultdict(int)

            # "Python" and "Objects" come from CPython.
            prefix_truncations = ("build/aten/", "Python/", "Objects/")
            for c, fn in stats:
                fn = re.sub(r"^.+build/\.\./", "build/../", fn)
                for new_prefix in prefix_truncations:
                    fn = re.sub(r"^.+/" + re.escape(new_prefix), new_prefix, fn)

                # Strip library name. e.g. `libtorch.so`
                fn = re.sub(r"\s\[.+\]$", "", fn)
                counts[fn] += c
            return tuple(sorted([
                FunctionCount(c, fn) for fn, c in counts.items() if c
            ], reverse=True))

        return CallgrindStats(
            stmt=self.stmt,
            setup=self.setup,
            number_per_run=self.number_per_run,
            num_threads=self.num_threads,
            built_with_debug_symbols=self.built_with_debug_symbols,
            baseline_inclusive_stats=strip(self.baseline_inclusive_stats),
            baseline_exclusive_stats=strip(self.baseline_exclusive_stats),
            stmt_inclusive_stats=strip(self.stmt_inclusive_stats),
            stmt_exclusive_stats=strip(self.stmt_exclusive_stats),
        )

    @staticmethod
    def _counts(stats: Tuple[FunctionCount, ...], include_lookdict_unicode: bool) -> int:
        return sum(
            c for c, fn in stats
            if include_lookdict_unicode
            or "dictobject.c:lookdict_unicode" not in fn
        )

    @staticmethod
    def _diff(first: Tuple[FunctionCount, ...], second: Tuple[FunctionCount, ...]) -> Tuple[FunctionCount, ...]:
        counts = collections.defaultdict(int, {fn: c for c, fn in first})
        assert len(counts) == len(first)
        for c, fn in second:
            counts[fn] -= c

        return tuple(sorted([
            FunctionCount(c, fn) for fn, c in counts.items() if c
        ], reverse=True))


class _ValgrindWrapper(object):
    def __init__(self) -> None:
        self._commands_available: Dict[str, bool] = {}
        if torch._C._valgrind_supported_platform():
            # Only bother checking on supported platforms.
            for cmd in ("valgrind", "callgrind_control", "callgrind_annotate"):
                self._commands_available[cmd] = not subprocess.run(
                    ["which", cmd],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                ).returncode

        self._build_type: Optional[str] = None
        build_search = re.search("BUILD_TYPE=(.+),", torch.__config__.show())
        if build_search is not None:
            self._build_type = build_search.groups()[0].split(",")[0]

        self._baseline_cache: Dict[Tuple[int, int], Tuple[Tuple[FunctionCount, ...], Tuple[FunctionCount, ...]]] = {}

    def _validate(self) -> None:
        if not torch._C._valgrind_supported_platform():
            raise OSError("Valgrind is not supported on this platform.")

        missing_cmds = [cmd for cmd, available in self._commands_available.items() if not available]
        if missing_cmds:
            raise OSError("Missing: " + ", ".join(missing_cmds))

    def collect_callgrind(
        self,
        stmt: str,
        setup: str,
        number: int,
        num_threads: int,
        collect_baseline: bool
    ) -> CallgrindStats:
        """Collect stats, and attach a reference run which can be used to filter interpreter overhead."""
        self._validate()
        baseline_inclusive_stats: Tuple[FunctionCount, ...] = ()
        baseline_exclusive_stats: Tuple[FunctionCount, ...] = ()
        if collect_baseline:
            cache_key = (number, num_threads)
            if cache_key not in self._baseline_cache:
                self._baseline_cache[cache_key] = self._invoke(
                    stmt="pass", setup="pass", number=number, num_threads=num_threads)
            baseline_inclusive_stats, baseline_exclusive_stats = \
                self._baseline_cache[cache_key]

        stmt_inclusive_stats, stmt_exclusive_stats = self._invoke(
            stmt=stmt,
            setup=setup,
            number=number,
            num_threads=num_threads
        )

        return CallgrindStats(
            stmt=stmt,
            setup=setup,
            number_per_run=number,
            num_threads=num_threads,
            built_with_debug_symbols=self._build_type == "RelWithDebInfo",
            baseline_inclusive_stats=baseline_inclusive_stats,
            baseline_exclusive_stats=baseline_exclusive_stats,
            stmt_inclusive_stats=stmt_inclusive_stats,
            stmt_exclusive_stats=stmt_exclusive_stats,
        )

    def _invoke(
        self,
        stmt: str,
        setup: str,
        number: int,
        num_threads: int
    ) -> Tuple[Tuple[FunctionCount, ...], Tuple[FunctionCount, ...]]:
        """Core invocation method for Callgrind collection.

        Valgrind operates by effectively replacing the CPU with an emulated
        version which allows it to instrument any code at the cost of severe
        performance degradation. This has the practical effect that in order
        to collect Callgrind statistics, a new process has to be created
        running under `valgrind`. The steps for this process are:

        1) Create a scratch directory.
        2) Codegen a run script. (_ValgrindWrapper._construct_script)
            Inside the run script:
                * Validate that Python and torch match the parent process
                * Validate that it is indeed running under valgrind
                * Execute `setup` and warm up `stmt`
                * Begin collecting stats
                * Run the `stmt` loop
                * Stop collecting stats
        3) Parse the run results.
        4) Cleanup the scratch directory.
        """
        working_dir = tempfile.mkdtemp()
        script_file = os.path.join(working_dir, "timer_callgrind.py")
        callgrind_out = os.path.join(working_dir, "callgrind.out")
        error_log = os.path.join(working_dir, "error.txt")
        stat_log = os.path.join(working_dir, "callgrind_stat.txt")
        stdout_stderr_log = os.path.join(working_dir, "stdout_stderr.log")

        def run(args: List[str], **kwargs: Any) -> Tuple[subprocess.CompletedProcess, str]:
            # https://thraxil.org/users/anders/posts/2008/03/13/Subprocess-Hanging-PIPE-is-your-enemy/
            f_stdout_stderr = open(stdout_stderr_log, "wb")
            try:
                invocation = subprocess.run(
                    args,
                    stdout=f_stdout_stderr,
                    stderr=subprocess.STDOUT,
                    **kwargs,
                )
                with open(stdout_stderr_log, "rt") as f:
                    return invocation, f.read()
            finally:
                f_stdout_stderr.close()

        try:
            with open(script_file, "wt") as f:
                f.write(self._construct_script(
                    stmt=stmt, setup=setup, number=number,
                    num_threads=num_threads, error_log=error_log,
                    stat_log=stat_log))

            valgrind_invocation, valgrind_invocation_output = run([
                "valgrind",
                "--tool=callgrind",
                f"--callgrind-out-file={callgrind_out}",
                "--dump-line=yes",
                "--dump-instr=yes",
                "--instr-atstart=yes",
                "--collect-atstart=no",
                "python",
                script_file,
            ])

            if valgrind_invocation.returncode:
                error_report = ""
                if os.path.exists(error_log):
                    with open(error_log, "rt") as f:
                        error_report = f.read()
                if not error_report:
                    error_report = "Unknown error.\n" + valgrind_invocation_output

                raise OSError(f"Failed to collect callgrind profile:\n{error_report}")

            def parse_output(inclusive: bool) -> Tuple[FunctionCount, ...]:
                annotate_invocation, annotate_invocation_output = run([
                    "callgrind_annotate",
                    f"--inclusive={'yes' if inclusive else 'no'}",
                    callgrind_out
                ], check=True)

                begin_collecting = False
                fn_counts = []
                for l in annotate_invocation_output.splitlines(keepends=False):
                    if not begin_collecting and re.match(r"Ir\s+file:function", l):
                        begin_collecting = True
                        continue

                    count_match = re.match(r"^\s*([0-9,]+)\s+(.+:.+)$", l)
                    if count_match:
                        ir_str, file_function = count_match.groups()
                        ir = int(ir_str.replace(",", ""))
                        fn_counts.append(FunctionCount(ir, file_function))
                        continue

                    if begin_collecting and re.match(r"-+", l):
                        continue

                    begin_collecting = False

                return tuple(fn_counts)
            return parse_output(inclusive=True), parse_output(inclusive=False)
        finally:
            shutil.rmtree(working_dir)

    @staticmethod
    def _construct_script(
        stmt: str,
        setup: str,
        number: int,
        num_threads: int,
        error_log: str,
        stat_log: str
    ) -> str:
        # The naive template looks something like:
        #   "for _ in range({number}): {stmt}"
        # However a loop in Python is surprisingly expensive, and significantly
        # increases the number of background Python instructions. So instead we
        # partially unroll the loops, with a block size of 100 chosen to keep
        # the instruction overhead from `range` low while also not ballooning
        # the size of the generated file.
        block_size = 100
        loop_count = number // block_size
        remainder = number - block_size * loop_count
        blocked_stmt = ""
        if loop_count:
            unrolled_stmts = textwrap.indent("\n".join([stmt] * block_size), " " * 4)
            blocked_stmt += f"for _ in range({loop_count}):\n{unrolled_stmts}\n"
        if remainder:
            blocked_stmt += "\n".join([stmt] * remainder)

        return textwrap.dedent(r"""
            import gc
            import os
            import subprocess
            import sys
            import time

            import torch
            torch.set_num_threads({num_threads})

            PID = os.getpid()

            def log_failure(msg):
                with open({error_log_repr}, "wt") as f:
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
            {indented_stmt}

            # =============================================================================
            # == Callgrind management =====================================================
            # =============================================================================
            with open("{stat_log}", "wb") as stat_file:
                # If many instances of callgrind are running at once, the output of
                # `callgrind_control` may exceed 16kb which would cause `subprocess.PIPE`
                # to deadlock. So instead we use a file.
                callgrind_stat = check_result(subprocess.run(
                    ["callgrind_control", "--stat"],
                    stdout=stat_file,
                    stderr=subprocess.STDOUT,
                ))

            with open("{stat_log}", "rt") as stat_file:
                stat_lines = stat_file.read().splitlines()

            if f"PID {{PID}}: python {{__file__}}" not in stat_lines:
                log_failure("Process does not appear to be running callgrind.")

            gc.collect()
            time.sleep(0.01)

            # =============================================================================
            # == User code block ==========================================================
            # =============================================================================
            torch._C._valgrind_toggle()
            {blocked_stmt}

            # Sleep is to allow the interpreter to catch up before we stop collecting in
            # order to reduce jitter.
            time.sleep(0.01)
            torch._C._valgrind_toggle()
        """).strip().format(
            indented_stmt=textwrap.indent(stmt, " " * 4),
            blocked_stmt=blocked_stmt,
            number=number,
            setup=setup,
            warmup_number=min(number, 10),
            num_threads=num_threads,
            error_log_repr=repr(error_log),
            stat_log=stat_log,
            parent_interpreter=sys.executable,
            torch_file=torch.__file__,
        )


CALLGRIND_SINGLETON: Optional[_ValgrindWrapper] = None
def wrapper_singleton() -> _ValgrindWrapper:
    global CALLGRIND_SINGLETON
    if CALLGRIND_SINGLETON is None:
        CALLGRIND_SINGLETON = _ValgrindWrapper()
    return CALLGRIND_SINGLETON
