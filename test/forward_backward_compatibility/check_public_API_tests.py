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


def run_public_api_test(name):
    return subprocess.run(
        [
            sys.executable,
            "test/test_public_bindings.py",
            "-k",
            name,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def check_test_fails_properly(test_name, invalid_item_name, invalid_item_desc):
    # Run the public API test and make sure it fails.
    print(f"Running public API test '{test_name}'...")
    res = run_public_api_test(test_name)

    # Fail the check if the public API test succeeded after introducing a bad file.
    if res.returncode == 0:
        print(
            f"Expected the public API test '{test_name}' to fail after introducing "
            f"{invalid_item_desc}, but it succeeded! Check "
            "test/test_public_bindings.py for any changes that may have broken the test."
        )
        sys.exit(1)

    if invalid_item_name not in res.stderr:
        print(
            f"Expected the public API test '{test_name}' to identify "
            f"{invalid_item_desc}, but it didn't! It's possible the test may not have run. "
            "Check test/test_public_bindings.py for any changes that may have broken the test."
        )
        sys.exit(1)

    print(f"Success! '{test_name}' identified {invalid_item_desc}.")


# Step 1. Make sure the public API test "test_correct_module_names" fails when a new file
# introduces an invalid public API function.
with tempfile.NamedTemporaryFile(dir="torch", suffix=".py") as f:
    # Create a bad file underneath torch/ that introduces a new public API function incorrectly.
    f.write(BAD_PUBLIC_FUNC)
    f.seek(0)
    mname = os.path.basename(f.name).split(".")[0]
    invalid_public_api_name = f"torch.{mname}.new_public_func"
    print(f"Creating an invalid public API function: {invalid_public_api_name}...")

    # Run the public API test and make sure it fails.
    check_test_fails_properly(
        "test_correct_module_names",
        invalid_public_api_name,
        "an invalid public API function",
    )

# Step 2. Make sure that the public API test "test_correct_module_names" fails when an existing
# file is modified to introduce an invalid public API function.
EXISTING_FILEPATH = "torch/nn/parameter.py"
with open(EXISTING_FILEPATH, "a+b") as f:
    invalid_public_api_name = "torch.nn.parameter.new_public_func"
    print(
        f"Appending an invalid public API function {invalid_public_api_name} to existing "
        f"file: {EXISTING_FILEPATH}..."
    )
    pos = f.tell()
    f.write(BAD_PUBLIC_FUNC)
    f.seek(0)

    try:
        # Run the public API test and make sure it fails.
        check_test_fails_properly(
            "test_correct_module_names",
            invalid_public_api_name,
            "an invalid public API function",
        )
    finally:
        # undo file write
        f.seek(pos, os.SEEK_SET)
        f.truncate(pos)

# Step 3. Make sure that the public API test "test_modules_can_be_imported" fails when a module
# cannot be imported.
with tempfile.TemporaryDirectory(dir="torch") as d:
    print("Creating a non-importable module...")
    with open(os.path.join(d, "__init__.py"), "wb") as f:
        f.write(b"invalid syntax garbage")
        f.seek(0)

    invalid_module_name = d.replace("/", ".")
    check_test_fails_properly(
        "test_modules_can_be_imported", invalid_module_name, "a non-importable module"
    )
