# perfmap: map code changes to operator benchmarks

import subprocess
import sys

# Get diff against base ref (default: upstream/viable/strict)
base = sys.argv[1] if len(sys.argv) > 1 else "upstream/viable/strict"
diff = subprocess.run(
    ["git", "diff", "-p", "-U0", f"{base}...HEAD"],
    capture_output=True, text=True, check=True,
).stdout

import re

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

import tree_sitter_cpp
import tree_sitter_python
from tree_sitter import Language, Parser

CPP = Language(tree_sitter_cpp.language())
PY = Language(tree_sitter_python.language())

CPP_EXTS = {"cpp", "cu", "h", "cuh", "cc", "mm"}

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
import yaml

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

# Look up each changed function
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
import glob

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

# Map operators to benchmark test names
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

# Output for CI: space-separated test names (strip _test.py suffix)
test_names = sorted(t.replace("_test.py", "") for t in matched_tests)
print(f"\nOP_BENCHMARK_TESTS={' '.join(test_names)}")
