import sys

log_file_path = sys.argv[1]

with open(log_file_path) as f:
    lines = f.readlines()

for line in lines:
    # Ignore errors from CPU instruction set testing
    if 'CMakeFiles/CMakeTmp/src.c' not in line:
        print(line)
