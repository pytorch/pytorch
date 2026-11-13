"""Runs modules of CPython's own test suite through torch.cfg_compile.

Every test method of a module is compiled, and what it calls is compiled in
turn, so a test passes only if the frontend and the VM behave like the
interpreter for everything the test does. The outcome of each test is compared
with its outcome under the plain interpreter; the two runs are separate
processes, so neither sees state the other left behind.

    python benchmarks/cfg_compile/cpython_suite.py test_grammar test_scope
    python benchmarks/cfg_compile/cpython_suite.py --lib ~/cpython/Lib --verbose test_dict

``--lib`` names the ``Lib`` directory of a CPython source tree of the running
version, for interpreters that are installed without the ``test`` package.
"""

import argparse
import importlib
import io
import json
import os
import subprocess
import sys
import time
import types
import unittest
import warnings


MARKER = "CFGSUITE "
KINDS = ("failures", "errors", "skipped", "expectedFailures", "unexpectedSuccesses")


def test_methods(suite):
    """The functions behind the tests of a suite, each with the class that
    defines it: also those a test class inherits from another module."""
    seen = set()
    stack = [suite]
    while stack:
        item = stack.pop()
        if isinstance(item, unittest.TestSuite):
            stack.extend(item)
            continue
        name = getattr(item, "_testMethodName", None)
        for cls in type(item).__mro__ if name else ():
            fn = vars(cls).get(name)
            if fn is None:
                continue
            if isinstance(fn, types.FunctionType) and (cls, name) not in seen:
                seen.add((cls, name))
                yield cls, name, fn
            break


def run_one(name, compiled):
    """Runs one test module in this process and prints its outcomes as JSON."""
    # Both runs import torch, so that all they differ in is who runs the tests:
    # an imported torch is visible to tests that look at sys.modules.
    import torch

    warnings.simplefilter("ignore")
    report = {"compiled": 0, "rejected": 0, "reasons": {}}
    try:
        module = importlib.import_module("test." + name)
    except BaseException as e:
        print(MARKER + json.dumps({"import_error": f"{type(e).__name__}: {e}"[:80]}))
        return
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    if compiled:
        for cls, method, fn in test_methods(suite):
            try:
                setattr(cls, method, torch.cfg_compile(fn, fallback=False))
                report["compiled"] += 1
            except torch.cfg_compile.Unsupported as e:
                report["rejected"] += 1
                reason = str(e).split(": ", 1)[-1]
                reason = " ".join(
                    w for w in reason.split() if not w.rstrip(")").isdigit()
                )
                report["reasons"][reason] = report["reasons"].get(reason, 0) + 1
    result = unittest.TextTestResult(io.StringIO(), False, 0)
    start = time.time()
    suite.run(result)
    report["seconds"] = time.time() - start
    report["ran"] = result.testsRun
    report["outcomes"] = {}
    for kind in KINDS:
        for item in getattr(result, kind):
            test, detail = item if isinstance(item, tuple) else (item, "")
            report["outcomes"][test.id()] = (kind, detail)
    print(MARKER + json.dumps(report))


def run_module(name, args, compiled):
    cmd = [sys.executable, os.path.abspath(__file__), "--one", name]
    if args.lib:
        cmd += ["--lib", args.lib]
    if compiled:
        cmd.append("--compiled")
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=args.timeout, cwd=args.cwd
        )
    except subprocess.TimeoutExpired:
        return {"problem": "timed out"}
    for line in reversed(r.stdout.splitlines()):
        if line.startswith(MARKER):
            report = json.loads(line[len(MARKER) :])
            if "import_error" in report:
                return {"problem": "import failed: " + report["import_error"]}
            return report
    tail = (r.stderr.strip().splitlines() or ["?"])[-1][:100]
    return {"problem": f"crashed (exit {r.returncode}): {tail}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--lib", help="the Lib directory of a CPython source tree")
    parser.add_argument(
        "--verbose", action="store_true", help="list what differs and why"
    )
    parser.add_argument(
        "--timeout", type=int, default=600, help="seconds per run of a module"
    )
    parser.add_argument("--cwd", default=os.getcwd(), help="directory the tests run in")
    parser.add_argument("--json", help="also write the raw results to this file")
    parser.add_argument("--one", help=argparse.SUPPRESS)
    parser.add_argument("--compiled", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("modules", nargs="*")
    args = parser.parse_args()
    if args.one:
        if args.lib:
            sys.path.insert(0, args.lib)
        run_one(args.one, args.compiled)
        return

    print(
        "| module | tests | compiled test methods | not compiled | outcomes that differ | seconds |"
    )
    print("|---|---:|---:|---:|---:|---:|")
    totals = [0, 0, 0, 0]
    raw = {}
    for name in args.modules:
        baseline = run_module(name, args, compiled=False)
        report = (
            baseline if "problem" in baseline else run_module(name, args, compiled=True)
        )
        raw[name] = {"baseline": baseline, "compiled": report}
        if "problem" in report:
            print(f"| {name} | {report['problem']} | | | | |")
            sys.stdout.flush()
            continue
        before, after = baseline["outcomes"], report["outcomes"]
        differ = sorted(
            t
            for t in set(before) | set(after)
            if before.get(t, ["ok"])[0] != after.get(t, ["ok"])[0]
        )
        row = [report["ran"], report["compiled"], report["rejected"], len(differ)]
        totals = [a + b for a, b in zip(totals, row)]
        print(
            f"| {name} | {row[0]} | {row[1]} | {row[2]} | {row[3]} | {report['seconds']:.1f} |"
        )
        if args.verbose:
            for reason, n in sorted(report["reasons"].items(), key=lambda kv: -kv[1])[
                :8
            ]:
                print(f"    not compiled x{n}: {reason[:110]}")
            for t in differ[:12]:
                print(
                    f"    differs: {t}: {before.get(t, ['ok'])[0]} -> {after.get(t, ['ok'])[0]}"
                )
                for line in after.get(t, ("", ""))[1].strip().splitlines()[-4:]:
                    print(f"        {line[:170]}")
        sys.stdout.flush()
    print(f"| **total** | {totals[0]} | {totals[1]} | {totals[2]} | {totals[3]} | |")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(raw, f)


if __name__ == "__main__":
    main()
