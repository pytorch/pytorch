"""Run CPython's own test modules with and without torch._rust's CFG executor
and report the tests whose outcome differs.

Needs a CPython checkout at the tag of the running interpreter, for Lib/test:

    git clone --depth 1 --filter=blob:none -b v3.12.11 https://github.com/python/cpython

Example:

    python torch/_rust/tools/compare_cpython.py --cpython ../cpython [test_gc test_scope ...]

With no modules it runs every module in cpython_test_modules.txt (about 30
minutes on 16 workers). These environment variables are passed on to the
executor runs: CFG_OPTIMIZE, CFG_AUTOGRAD, CFG_ELIDE, CFG_OUTLINE and
CFG_RECORD=0. CMP_JOBS (default 16) sets the number of modules run at once,
CMP_TIMEOUT (default 600) the seconds per module run, and CMP_SKIP a
comma-separated list of extra test ids to skip.

A test counts as differing when its outcome isn't the same in both runs,
except that two failures (or two errors) with different messages don't count.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from typing import Any


HERE = os.path.dirname(os.path.abspath(__file__))
MARKER = "@@RESULT@@"

# Tests that turn on sys.settrace or sys.monitoring, which the executor
# refuses by design. They're skipped in both runs.
SKIP = {
    "test.test_scope.ScopeTests.testInteractionWithTraceFunc",
    "test.test_scope.ScopeTests.testLocalsClass_WithTrace",
    "test.test_import.ImportTests.test_concurrency",
    "test.test_code.CodeTest.test_code_equal_with_instrumentation",
    "test.test_peepholer.TestMarkingVariablesAsUnKnown.test_setting_lineno_no_undefined",
    "test.test_peepholer.TestMarkingVariablesAsUnKnown.test_setting_lineno_one_undefined",
    "test.test_peepholer.TestMarkingVariablesAsUnKnown.test_setting_lineno_two_undefined",
    "test.test_peepholer.TestMarkingVariablesAsUnKnown.test_deleting_local_warns_and_assigns_none",
    "test.test_peepholer.TestMarkingVariablesAsUnKnown.test_modifying_local_does_not_add_check",
    "test.test_peepholer.TestMarkingVariablesAsUnKnown.test_initializing_local_does_not_add_check",
    "test.test_frame.TestIncompleteFrameAreInvisible.test_sneaky_frame_object",
}


def run_module(mode, name):
    # Runs in a child process whose cwd holds a `test` link to CPython's
    # Lib/test, so only the tests come from the checkout, not the stdlib.
    # Prints {test id: outcome} as JSON after MARKER.
    sys.path.insert(0, os.getcwd())
    import torch

    tr: Any = torch._rust

    skip = SKIP | set(os.environ.get("CMP_SKIP", "").split(","))
    outcomes = {}

    def last_line(result, err, test):
        return result._exc_info_to_string(err, test).strip().splitlines()[-1][:300]

    class Result(unittest.TestResult):
        def addSuccess(self, t):
            outcomes[t.id()] = "ok"

        def addFailure(self, t, err):
            outcomes[t.id()] = "FAIL: " + last_line(self, err, t)

        def addError(self, t, err):
            outcomes[t.id()] = "ERROR: " + last_line(self, err, t)

        def addSkip(self, t, reason):
            outcomes[t.id()] = "skip"

        def addExpectedFailure(self, t, err):
            outcomes[t.id()] = "xfail"

        def addUnexpectedSuccess(self, t):
            outcomes[t.id()] = "FAIL: unexpected success"

        def addSubTest(self, test, sub, err):
            if err is not None:
                kind = "FAIL" if issubclass(err[0], test.failureException) else "ERROR"
                outcomes[sub.id()] = f"{kind}: {last_line(self, err, test)}"

    def filtered(suite):
        out = unittest.TestSuite()
        for t in suite:
            if isinstance(t, unittest.TestSuite):
                out.addTest(filtered(t))
            elif t.id() in skip:
                outcomes[t.id()] = "skip"
            else:
                out.addTest(t)
        return out

    def go():
        mod = __import__(f"test.{name}", fromlist=["_"])
        filtered(unittest.TestLoader().loadTestsFromModule(mod)).run(Result())

    try:
        if os.environ.get("CFG_OUTLINE"):
            tr._set_cfg_outline(os.environ["CFG_OUTLINE"])
        if os.environ.get("CFG_RECORD") == "0":
            tr._set_cfg_record(False)
        if os.environ.get("CFG_OPTIMIZE"):
            import torch._dynamo.trace_rules

            tr._set_cfg_optimize(True)
        if os.environ.get("CFG_AUTOGRAD"):
            tr._set_cfg_autograd(True)
        if os.environ.get("CFG_ELIDE"):
            tr._set_cfg_elide(True)
        if mode == "cfg":
            tr.call_with_cfg_execution(go)
        else:
            go()
    except BaseException as e:
        outcomes["<module>"] = f"ERROR: {type(e).__name__}: {str(e)[:300]}"
    sys.stdout.flush()
    print(f"\n{MARKER}{json.dumps(outcomes)}")


def run(lib_test, mode, name, timeout):
    start = time.time()
    with tempfile.TemporaryDirectory() as cwd:
        os.symlink(lib_test, os.path.join(cwd, "test"))
        cmd = [sys.executable, os.path.abspath(__file__), "--run-module", mode, name]
        try:
            p = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd
            )
        except subprocess.TimeoutExpired:
            return {"<module>": "TIMEOUT"}, time.time() - start
    at = p.stdout.rfind(MARKER)
    if at < 0:
        tail = " | ".join((p.stderr or p.stdout).strip().splitlines()[-3:])
        return {
            "<module>": f"CRASH rc={p.returncode}: {tail[:400]}"
        }, time.time() - start
    return json.loads(p.stdout[at + len(MARKER) :]), time.time() - start


def differs(base, cfg):
    return base != cfg and not any(
        base.startswith(k) and cfg.startswith(k) for k in ("FAIL", "ERROR")
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cpython", help="a CPython checkout, for Lib/test")
    parser.add_argument(
        "--run-module", nargs=2, metavar=("MODE", "NAME"), help=argparse.SUPPRESS
    )
    parser.add_argument(
        "modules",
        nargs="*",
        help="test modules to run, e.g. test_gc (default: all in cpython_test_modules.txt)",
    )
    args = parser.parse_args()
    if args.run_module:
        run_module(*args.run_module)
        return
    if not args.cpython:
        parser.error("--cpython is required")
    lib_test = os.path.abspath(os.path.join(args.cpython, "Lib", "test"))
    if not os.path.isdir(lib_test):
        parser.error(f"{lib_test} doesn't exist")
    modules = args.modules
    if not modules:
        with open(os.path.join(HERE, "cpython_test_modules.txt")) as f:
            modules = f.read().split()
    timeout = int(os.environ.get("CMP_TIMEOUT", "600"))

    def both(name):
        return (
            name,
            run(lib_test, "base", name, timeout),
            run(lib_test, "cfg", name, timeout),
        )

    total = 0
    with ThreadPoolExecutor(int(os.environ.get("CMP_JOBS", "16"))) as pool:
        for name, (base, tb), (cfg, tc) in pool.map(both, modules):
            diffs = []
            for k in sorted(set(base) | set(cfg)):
                b, c = base.get(k, "missing"), cfg.get(k, "missing")
                if differs(b, c):
                    diffs.append((k, b, c))
            nok = sum(v == "ok" for v in base.values())
            print(
                f"{name}: {len(base)} tests ({nok} ok in base), {len(diffs)} differ  [{tb:.0f}s / {tc:.0f}s]",
                flush=True,
            )
            for k, b, c in diffs[:40]:
                print(f"    {k}\n        base: {b}\n        cfg:  {c}")
            if len(diffs) > 40:
                print(f"    ... {len(diffs) - 40} more")
            total += len(diffs)
    print("TOTAL DIFFERENCES:", total)


if __name__ == "__main__":
    main()
