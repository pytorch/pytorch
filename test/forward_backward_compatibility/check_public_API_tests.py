import os
import subprocess
import sys
import tempfile


BAD_PUBLIC_FUNC = b"""
def new_public_func():
    pass

# valid public API functions have __module__ set correctly
new_public_func.__module__ = None
"""

# Step 1. Make sure the public API test "test_correct_module_names" fails when a new file
# introduces an invalid public API function.
with tempfile.NamedTemporaryFile(dir="torch", suffix=".py") as f:
    # Create a bad file underneath torch/ that introduces a new public API function incorrectly.
    f.write(BAD_PUBLIC_FUNC)
    f.seek(0)
    mname = os.path.basename(f.name).split(".")[0]
    breaking_name = f"torch.{mname}.new_public_func"
    print(f"Creating an invalid public API function: {breaking_name}...")

    # Run the public API test and make sure it fails.
    res = subprocess.run(
        [
            sys.executable,
            "test/test_public_bindings.py",
            "-k",
            "test_correct_module_names",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    # Fail the check if the public API test succeeded after introducing a bad file.
    print("Running public API test 'test_correct_module_names'...")
    if res.returncode == 0:
        print(
            "Expected the public API test 'test_correct_module_names' to fail after introducing "
            "a bad file, but it succeeded! Check test/test_public_bindings.py for any changes "
            "that may have broken the test."
        )
        sys.exit(1)

    if breaking_name not in res.stderr:
        print(
            "Expected the public API test 'test_correct_module_names' to identify an invalid "
            "public API, but it didn't! It's possible the test may not have run. Check "
            "test/test_public_bindings.py for any changes that may have broken the test."
        )
        sys.exit(1)

    print(
        "Success! 'test_correct_module_names' identified a new, invalid public API function."
    )

# Step 2. Make sure that the public API test "test_modules_can_be_imported" fails when an existing
# file is modified to introduce an invalid public API function.
EXISTING_FILEPATH = "torch/nn/functional.py"
with open(EXISTING_FILEPATH, "a+b") as f:
    print(
        f"Appending an invalid public API function to existing file: {EXISTING_FILEPATH}..."
    )
    pos = f.tell()
    f.write(BAD_PUBLIC_FUNC)
    f.seek(0)

    # Run the public API test and make sure it fails.
    try:
        print("Running public API test 'test_modules_can_be_imported'...")
        res = subprocess.run(
            [
                sys.executable,
                "test/test_public_bindings.py",
                "-k",
                "test_modules_can_be_imported",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        # undo file write
        f.seek(pos, os.SEEK_SET)
        f.truncate(pos)

    if res.returncode == 0:
        print(
            "Expected the public API test 'test_modules_can_be_imported' to fail after introducing "
            f"an invalid public API function to existing file {EXISTING_FILEPATH}, but it "
            "succeeded! Check test/test_public_bindings.py for any changes that may have broken "
            "the test."
        )
        sys.exit(1)

    print(
        "Success! 'test_modules_can_be_imported' identified the invalid public API function "
        f"appended to the existing file {EXISTING_FILEPATH}."
    )
