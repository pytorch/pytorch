"""
This is a wrapper for our C++ tests
"""
from typing import List, Union, Dict, Optional, Any, NamedTuple
import os
import shlex
import shutil
import subprocess
import asyncio
import multiprocessing
from pathlib import Path
import jinja2


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TEST_BIN_DIR = REPO_ROOT / "build" / "bin"
GTEST_DIR = REPO_ROOT / "test" / "gtest"
GTEST_GENERATED_DIR = GTEST_DIR / "generated"


class CommandResult(NamedTuple):
    passed: bool
    stdout: str
    stderr: str


async def shell_cmd(
    cmd: List[str],
    env: Optional[Dict[str, Any]] = None,
    redirect: bool = True,
) -> CommandResult:
    cmd_str = " ".join(shlex.quote(arg) for arg in cmd)

    proc = await asyncio.create_subprocess_shell(
        cmd_str,
        shell=True,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE if redirect else None,
        stderr=subprocess.PIPE if redirect else None,
        executable=shutil.which("bash"),
    )
    stdout, stderr = await proc.communicate()

    passed = proc.returncode == 0
    if not redirect:
        return CommandResult(passed, "", "")

    return CommandResult(passed, stdout.decode().strip(), stderr.decode().strip())


async def gather_with_concurrency(
    n: int, tasks: List[Any], return_exceptions: bool = True
) -> Any:
    semaphore = asyncio.Semaphore(n)

    async def sem_task(task: Any) -> Any:
        async with semaphore:
            return await task

    return await asyncio.gather(
        *(sem_task(task) for task in tasks), return_exceptions=return_exceptions
    )


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
        if line[0] == " ":
            tests[curr].append(strip_comment(line.strip()))
        else:
            curr = strip_comment(line.strip()).rstrip(".")

    skip_keys = [
        "unknown",
        "Running main() from ../third_party/googletest/googletest/src/gtest_main.cc",
    ]
    for key in skip_keys:
        if key in tests:
            del tests[key]

    return tests


async def generate_tests():
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(GTEST_DIR)),
        undefined=jinja2.StrictUndefined,
    )
    template = env.get_template("template.py.j2")
    async def handle_binary(binary):
        result = await shell_cmd([str(binary), "--gtest_list_tests"])
        if not result.passed:
            raise RuntimeError(f"Failed to list tests for {binary}")


        filename = GTEST_GENERATED_DIR / f"test_{binary.name}_generated.py"
        tests = parse_gtest_list_tests(result.stdout)

        suites = []
        for suite, cases in tests.items():
            cases = [case for case in cases if len(case) > 0]
            if len(cases) == 0:
                continue
            suites.append(
                {
                    "name": suite,
                    "tests": cases,
                }
            )
        # import json
        # print(json.dumps(suites, indent=2))
        content = template.render(
            binary=str(binary.relative_to(REPO_ROOT)), suites=suites
        )
        with open(filename, "w") as f:
            f.write(content)

        black_exe = shutil.which("black")
        if black_exe is not None:
            result = await shell_cmd([black_exe, str(filename)])
            if not result.passed:
                raise RuntimeError(f"Failed to format {filename}:\n{result}")
            # subprocess.run()


        # return binary, parse_gtest_list_tests(result.stdout)

    # coros = [handle_binary(b) for b in TEST_BIN_DIR.glob("*test_tensorexpr*")]
    coros = [handle_binary(b) for b in TEST_BIN_DIR.glob("*test*")]
    # coros = [handle_binary(b) for b in [list(TEST_BIN_DIR.glob("*test*"))[1]]]
    return await gather_with_concurrency(
        multiprocessing.cpu_count(), coros, return_exceptions=False
    )



if __name__ == "__main__":
    if not TEST_BIN_DIR.exists():
        raise RuntimeError(
            f"{TEST_BIN_DIR} does not exist, build PyTorch before running this file"
        )

    existing_workflows = GTEST_DIR.glob("generated/*.py")
    for w in existing_workflows:
        try:
            os.remove(w)
        except Exception as e:
            print(f"Error occurred when deleting file {w}: {e}")

    if not GTEST_GENERATED_DIR.exists():
        os.mkdir(GTEST_GENERATED_DIR)

    asyncio.run(generate_tests())

    # total = 0
    # for binary in Path("build/bin").glob("*test*"):
    #     subprocess.run(f"{binary} --gtest_list_tests > tests.log", shell=True, check=True)
    #     with open("tests.log") as f:
    #         tests = parse_gtest_list_tests(f.read())

    #     # binary = "./build/bin/blob_test"
    #     filters = []
    #     for suite, test_names in tests.items():
    #         for test_name in test_names:
    #             filters.append(f"{suite}{test_name}")
    #             # print(suite, test_name)

    #     # for filter in filters:
    #     #     subprocess.run([binary, f"--gtest_filter={filter}"])
    #     # print(len(filters), "tests")
    #     print(binary, len(filters))
    #     total += len(filters)

    # print("TOTAL", total)

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
