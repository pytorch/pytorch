import sys
import textwrap
from pathlib import Path


def check(path):
    """Check a test file for common issues with pytest->pytorch conversion."""
    print(path.name)
    print("=" * len(path.name), "\n")

    src = path.read_text().split("\n")
    for num, line in enumerate(src):
        if is_comment(line):
            continue

        # module level test functions
        if line.startswith("def test"):
            report_violation(line, num, header="Module-level test function")

        # test classes must inherit from TestCase
        if line.startswith("class Test") and "TestCase" not in line:
            report_violation(
                line, num, header="Test class does not inherit from TestCase"
            )

        # last vestiges of pytest-specific stuff
        if "pytest.mark" in line:
            report_violation(line, num, header="pytest.mark.something")

        for part in ["pytest.xfail", "pytest.skip", "pytest.param"]:
            if part in line:
                report_violation(line, num, header=f"stray {part}")

        if textwrap.dedent(line).startswith("@parametrize"):
            # backtrack to check
            nn = num
            for nn in range(num, -1, -1):
                ln = src[nn]
                if "class Test" in ln:
                    # hack: large indent => likely an inner class
                    if len(ln) - len(ln.lstrip()) < 8:
                        break
            else:
                report_violation(line, num, "off-class parametrize")
            if not src[nn - 1].startswith("@instantiate_parametrized_tests"):
                # breakpoint()
                report_violation(
                    line, num, f"missing instantiation of parametrized tests in {ln}?"
                )


def is_comment(line):
    return textwrap.dedent(line).startswith("#")


def report_violation(line, lineno, header):
    print(f">>>> line {lineno} : {header}\n {line}\n")


if __name__ == "__main__":
    argv = sys.argv
    if len(argv) != 2:
        raise ValueError("Usage : python check_tests_conform path/to/file/or/dir")

    path = Path(argv[1])

    if path.is_dir():
        # run for all files in the directory (no subdirs)
        for this_path in path.glob("test*.py"):
            #   breakpoint()
            check(this_path)
    else:
        check(path)
