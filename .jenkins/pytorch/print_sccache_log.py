import sys

log_file_path = sys.argv[1]

with open(log_file_path) as f:
    lines = f.readlines()

for line in lines:
    # Ignore errors from CPU instruction set or symbol existing testing
    keywords = ['src.c', 'CheckSymbolExists.c']
    if all([keyword not in line for keyword in keywords]):
        print(line)
