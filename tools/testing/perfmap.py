# perfmap: map code changes to operator benchmarks

import glob
import json
import os
import re
import subprocess
import sys

import tree_sitter_cpp
import tree_sitter_python
import yaml
from tree_sitter import Language, Parser

CPP = Language(tree_sitter_cpp.language())
PY = Language(tree_sitter_python.language())
CPP_EXTS = {"cpp", "cu", "h", "cuh", "cc", "mm"}
PERFMAP_DIR = ".perfmap"

def detect(base):
    diff = subprocess.run(
        ["git", "diff", "-p", "-U0", f"{base}...HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout

    # Parse diff to get changed line ranges per file
    files = {}
    current = None
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current = line[6:]
        elif line.startswith("@@") and current:
            m = re.search(r"-(\d+)(?:,(\d+))?", line)
            if m:
                start = int(m.group(1))
                count = int(m.group(2)) if m.group(2) else 1
                end = start + count - 1 if count > 0 else start
                files.setdefault(current, []).append((start, end))

    if not files:
        print(f"No changes found between {base} and HEAD.")
        return []

    all_changed_functions = set()

    for path, ranges in files.items():
        if path.startswith("test/"):
            continue
        ext = path.rsplit(".", 1)[-1] if "." in path else ""
        if ext in CPP_EXTS:
            lang = CPP
        elif ext == "py":
            lang = PY
        else:
            continue

        try:
            source = subprocess.run(
                ["git", "show", f"{base}:{path}"],
                capture_output=True, check=True,
            ).stdout
        except subprocess.CalledProcessError:
            continue

        tree = Parser(lang).parse(source)

        # Find all function definitions and their line ranges in the file
        funcs = []
        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            if node.type == "function_definition":
                name = None
                for child in node.children:
                    if lang == PY and child.type == "identifier":
                        name = child.text.decode()
                        break
                    if lang == CPP and "declarator" in child.type:
                        t = child.text.decode()
                        p = t.find("(")
                        if p != -1:
                            prefix = t[:p]
                            for macro in ("TORCH_META_FUNC", "TORCH_IMPL_FUNC"):
                                if macro in prefix:
                                    inner = t[t.find("(") + 1 : t.find(")")]
                                    name = inner.split(",")[0].strip()
                                    break
                            else:
                                name = prefix.split()[-1].lstrip("&*")
                        break
                if name:
                    funcs.append((name, node.start_point[0] + 1, node.end_point[0] + 1))
            else:
                stack.extend(node.children)

        # Find functions whose line ranges overlap with changed lines
        matched = set()
        for name, fs, fe in funcs:
            for cs, ce in ranges:
                if fs <= ce and cs <= fe:
                    matched.add(name)
                    break

        print(f"\n{path}")
        if matched:
            for f in sorted(matched):
                print(f"  {f}")
        else:
            print(f"  (no functions found, file-level change)")
        all_changed_functions.update(matched)

    # Map function names to operator names via native_functions.yaml
    with open("aten/src/ATen/native/native_functions.yaml") as f:
        ops = yaml.safe_load(f)

    kernel_to_op = {}
    op_names = set()
    for entry in ops:
        if "func" not in entry:
            continue
        func_str = entry["func"].split("(")[0].split(".")[0]
        op_names.add(func_str)
        if "dispatch" in entry:
            for kernel in entry["dispatch"].values():
                kernel_to_op[kernel] = func_str

    print("\n--- Operator mapping ---")
    matched_ops = set()
    for func in sorted(all_changed_functions):
        if func in kernel_to_op:
            op = kernel_to_op[func]
            print(f"  {func} -> {op} (kernel match)")
            matched_ops.add(op)
        elif func in op_names:
            print(f"  {func} -> {func} (direct match)")
            matched_ops.add(func)
        else:
            print(f"  {func} -> (no operator match)")

    # Map operators to benchmark tests
    benchmark_modules = {}
    for f in glob.glob("benchmarks/operator_benchmark/pt/*_test.py"):
        with open(f) as fh:
            content = fh.read()
            for m in re.finditer(r'set_module_name\("(\w+)"\)', content):
                benchmark_modules.setdefault(m.group(1).lower(), set()).add(f.split("/")[-1])
            for line in content.splitlines():
                if line.lstrip().startswith("#"):
                    continue
                for m in re.finditer(r'\["(\w+)",\s*torch\.', line):
                    benchmark_modules.setdefault(m.group(1).lower(), set()).add(f.split("/")[-1])

    print("\n--- Benchmark mapping ---")
    matched_tests = set()
    for op in sorted(matched_ops):
        if op in benchmark_modules:
            print(f"  {op} -> {benchmark_modules[op]}")
            matched_tests.update(benchmark_modules[op])
        else:
            best = ""
            for mod in benchmark_modules:
                if op.startswith(mod) and (len(op) == len(mod) or op[len(mod)] == "_"):
                    if len(mod) > len(best):
                        best = mod
            if best:
                print(f"  {op} -> {benchmark_modules[best]} (via {best})")
                matched_tests.update(benchmark_modules[best])
            else:
                print(f"  {op} -> NO BENCHMARK")

    test_names = sorted(t.replace("_test.py", "") for t in matched_tests)
    print(f"\nOP_BENCHMARK_TESTS={' '.join(test_names)}")
    return test_names


def run(label, base):
    label_dir = os.path.join(PERFMAP_DIR, label)
    os.makedirs(label_dir, exist_ok=True)
    tests_file = os.path.join(label_dir, "tests.txt")

    # First run: detect tests and save. Later runs: reuse saved list.
    if not os.path.exists(tests_file):
        test_names = detect(base)
        if not test_names:
            print("No benchmarks found. Run on a branch with operator changes first.")
            return
        with open(tests_file, "w") as f:
            f.write("\n".join(test_names))
    else:
        with open(tests_file) as f:
            test_names = [l.strip() for l in f if l.strip()]
        if not test_names:
            print("Saved test list is empty. Delete .perfmap/{label}/ and run on a branch with changes.")
            return
        print(f"Using saved test list: {' '.join(test_names)}")

    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()

    run_dir = os.path.join(label_dir, f"{branch}_{commit}")
    os.makedirs(run_dir, exist_ok=True)

    for test in test_names:
        out_file = os.path.join(run_dir, f"{test}.json")
        print(f"Running {test}...")
        subprocess.run(
            [sys.executable, "-m", f"pt.{test}_test",
             "--tag-filter", "long", "--output-json", os.path.abspath(out_file)],
            cwd="benchmarks/operator_benchmark",
        )

    print(f"\nResults saved to {run_dir}")


def compare(label):
    label_dir = os.path.join(PERFMAP_DIR, label)
    if not os.path.exists(label_dir):
        print(f"No runs found for label '{label}'. Run benchmarks first with: perfmap.py run --label {label}")
        return

    runs = sorted(
        d for d in os.listdir(label_dir)
        if os.path.isdir(os.path.join(label_dir, d))
    )
    if len(runs) < 2:
        print(f"Need at least 2 runs to compare, found {len(runs)}. Run benchmarks on another branch.")
        return

    # Load latencies: {test_name: latency}
    def load_run(run_tag):
        run_dir = os.path.join(label_dir, run_tag)
        latencies = {}
        for f in sorted(os.listdir(run_dir)):
            if not f.endswith(".json"):
                continue
            with open(os.path.join(run_dir, f)) as fh:
                for entry in json.load(fh):
                    key = entry.get("test_name", "unknown")
                    latency = entry.get("latency", 0)
                    if latency > 0:
                        latencies[key] = latency
        return latencies

    baseline_tag, branch_tag = runs[0], runs[1]
    baseline = load_run(baseline_tag)
    branch = load_run(branch_tag)

    all_tests = sorted(set(baseline.keys()) | set(branch.keys()))
    if not all_tests:
        print("No benchmark results found in either run.")
        return

    lines = []
    lines.append(f"## Benchmark Comparison: `{baseline_tag}` vs `{branch_tag}`\n")
    lines.append(f"| Test | {baseline_tag} (us) | {branch_tag} (us) | Diff (us) | Change |")
    lines.append("|------|------|------|------|--------|")
    for test in all_tests:
        b = baseline.get(test)
        r = branch.get(test)
        if b is None or r is None:
            continue
        diff = r - b
        pct = (diff / b) * 100 if b > 0 else 0
        lines.append(f"| {test} | {b:.2f} | {r:.2f} | {diff:+.2f} | {pct:+.1f}% |")

    output = "\n".join(lines)
    print(f"\n{output}")

    out_file = os.path.join(label_dir, "comparison.md")
    with open(out_file, "w") as f:
        f.write(output + "\n")
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args or args[0] not in ("detect", "run", "compare"):
        base = args[0] if args else "upstream/viable/strict"
        detect(base)

    elif args[0] == "detect":
        base = args[1] if len(args) > 1 else "upstream/viable/strict"
        detect(base)

    elif args[0] == "run":
        label, base = None, "upstream/viable/strict"
        i = 1
        while i < len(args):
            if args[i] == "--label" and i + 1 < len(args):
                label = args[i + 1]
                i += 2
            elif args[i] == "--base" and i + 1 < len(args):
                base = args[i + 1]
                i += 2
            else:
                i += 1
        if not label:
            print("Usage: perfmap.py run --label <name>")
            sys.exit(1)
        run(label, base)

    elif args[0] == "compare":
        label = None
        i = 1
        while i < len(args):
            if args[i] == "--label" and i + 1 < len(args):
                label = args[i + 1]
                i += 2
            else:
                i += 1
        if not label:
            print("Usage: perfmap.py compare --label <name>")
            sys.exit(1)
        compare(label)
