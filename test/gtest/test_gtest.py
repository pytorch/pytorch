"""
This is a wrapper for our C++ tests
"""
from typing import List
import subprocess
from pathlib import Path

def strip_comment(s: str) -> str:
    return s.split("#")[0].rstrip()


def parse_gtest_list_tests(output: str) -> List[str]:
    lines = output.split("\n")
    curr = "unknown"
    tests = {}
    for line in lines:
        if curr not in tests:
            tests[curr] = []
        if len(line.strip()) == 0:
            continue
        if line[0] == ' ':
            tests[curr].append(strip_comment(line.strip()))
        else:
            curr = strip_comment(line.strip())
        
    return tests


if __name__ == "__main__":
    total = 0
    for binary in Path("build/bin").glob("*test*"):
        subprocess.run(f"{binary} --gtest_list_tests > tests.log", shell=True, check=True)
        with open("tests.log") as f:
            tests = parse_gtest_list_tests(f.read())
        
        # binary = "./build/bin/blob_test"
        filters = []
        for suite, test_names in tests.items():
            for test_name in test_names:
                filters.append(f"{suite}{test_name}")
                # print(suite, test_name)
        
        # for filter in filters:
        #     subprocess.run([binary, f"--gtest_filter={filter}"])
        # print(len(filters), "tests")
        print(binary, len(filters))
        total += len(filters)
    
    print("TOTAL", total)

"""
102 tests, runtime basically doubles == 0.16 seconds of overhead per test?

$ /bin/time -p ./build/bin/blob_test
...
real 17.11
user 16.64
sys 6.04


$ /bin/time -p python test/gtest/test_gtest.py
...
real 34.65
user 27.27
sys 14.00


        net_test (22 tests)
    before
real 4.17
user 0.11
sys 0.09
    after
real 6.90
user 2.35
sys 0.60

expected overhead = 22 * 0.16 = 3.52
actual overhead   = 6.90 - 4.17 = 2.73


        test_api (972 tests)
    before
real 40.84
user 1157.60
sys 0.75
    after
real 163.83
user 1523.23
sys 28.15

expected overhead = 972 * 0.16 = 155.52
actual overhead   = 123


4333 tests total
est overhead = 11 minutes
"""