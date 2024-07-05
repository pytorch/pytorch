import sys

log_file_path = sys.argv[1]

with open(log_file_path) as f:
    lines = f.readlines()

for line in lines:
    # Ignore errors from CPU instruction set, symbol existing testing,
    # or compilation error formatting
    ignored_keywords = [
        "src.c",
        "CheckSymbolExists.c",
        "test_compilation_error_formatting",
    ]
    if all(keyword not in line for keyword in ignored_keywords):
        print(line)
