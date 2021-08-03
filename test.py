import sys
import subprocess
import re
import os
from collections import defaultdict


def add_ignore(ids_by_class, filename):
    with open(filename) as f:
        lines = f.readlines()

    for class_name, tests in ids_by_class.items():
        insert_line = None
        for index, line in enumerate(lines):
            if line.startswith(f"class {class_name}"):
                insert_line = index
                break
        if insert_line is None:
            print("unable to find class", class_name)
        else:
            items = "\n".join([f'        "{class_name}.{test}",' for test in tests])
            to_add = f"    _ignore_error_on_print_allowlist = {{\n{items}\n    }}\n\n"
            lines = lines[: insert_line + 1] + [to_add] + lines[insert_line + 1 :]
            with open(filename, "w") as f:
                f.write("".join(lines))


test_file = sys.argv[1]
env = os.environ.copy()
env["PYTORCH_ERROR_ON_TEST_PRINT"] = "1"
proc = subprocess.run([sys.executable, test_file], env=env, stderr=subprocess.PIPE)
stderr = proc.stderr.decode()

print(stderr)

m = re.findall(
    r"=======\nERROR: ([a-zA-Z0-9]+) \(__main__.([a-zA-Z0-9]+)\)", stderr, re.MULTILINE
)

all = defaultdict(list)

for test_name, test_class in m:
    all[test_class].append(test_name)

all = dict(all)
print(all)
add_ignore(all, test_file)
