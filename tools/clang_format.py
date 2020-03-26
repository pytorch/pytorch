#!/usr/bin/env python
"""
A script that runs clang-format on all C/C++ files in CLANG_FORMAT_WHITELIST. There is
also a diff mode which simply checks if clang-format would make any changes, which is useful for 
CI purposes.

If clang-format is not available, the script also downloads a platform-appropriate binary from
and S3 bucket and verifies it against a precommited set of blessed binary hashes.
"""
import argparse
import asyncio
import hashlib
import os
import platform
import stat
import re
import sys
import urllib.request
import urllib.error


# String representing the host platform (e.g. Linux, Darwin).
HOST_PLATFORM = platform.system()

# PyTorch directory root, derived from the location of this file.
PYTORCH_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# This dictionary maps each platform to the S3 object URL for its clang-format binary.
PLATFORM_TO_CF_URL = {
    "Darwin": "https://oss-clang-format.s3.us-east-2.amazonaws.com/mac/clang-format-mojave",
    "Linux": "https://oss-clang-format.s3.us-east-2.amazonaws.com/linux64/clang-format-linux64",
}

# This dictionary maps each platform to a relative path to a file containing its reference hash.
PLATFORM_TO_HASH = {
    "Darwin": os.path.join("tools", "clang_format_hash", "mac", "clang-format-mojave"),
    "Linux": os.path.join("tools", "clang_format_hash", "linux64", "clang-format-linux64"),
}

# Directory and file paths for the clang-format binary.
CLANG_FORMAT_DIR = os.path.join(PYTORCH_ROOT, ".clang-format-bin")
CLANG_FORMAT_PATH = os.path.join(CLANG_FORMAT_DIR, "clang-format")

# Whitelist of directories to check. All files that in that directory
# (recursively) will be checked.
CLANG_FORMAT_WHITELIST = ["torch/csrc/jit/", "test/cpp/jit/"]

# Only files with names matching this regex will be formatted.
CPP_FILE_REGEX = re.compile(".*\\.(h|cpp|cc|c|hpp)$")


def compute_file_sha1(path):
    """Compute the SHA1 hash of a file and return it as a hex string."""
    # If the file doesn't exist, return an empty string.
    if not os.path.exists(path):
        return ""

    hash = hashlib.sha1()

    # Open the file in binary mode and hash it.
    with open(path, "rb") as f:
        for b in f:
            hash.update(b)

    # Return the hash as a hexadecimal string.
    return hash.hexdigest()


def get_whitelisted_files():
    """
    Parse CLANG_FORMAT_WHITELIST and resolve all directories.
    Returns the set of whitelist cpp source files.
    """
    matches = []
    for dir in CLANG_FORMAT_WHITELIST:
        for root, dirnames, filenames in os.walk(dir):
            for filename in filenames:
                if CPP_FILE_REGEX.match(filename):
                    matches.append(os.path.join(root, filename))
    return set(matches)


async def run_clang_format_on_file(filename, semaphore, verbose=False):
    """
    Run clang-format on the provided file.
    """
    # -style=file picks up the closest .clang-format, -i formats the files inplace.
    cmd = "{} -style=file -i {}".format(CLANG_FORMAT_PATH, filename)
    async with semaphore:
        proc = await asyncio.create_subprocess_shell(cmd)
        _ = await proc.wait()
    if verbose:
        print("Formatted {}".format(filename))


async def file_clang_formatted_correctly(filename, semaphore, verbose=False):
    """
    Checks if a file is formatted correctly and returns True if so.
    """
    ok = True
    # -style=file picks up the closest .clang-format
    cmd = "{} -style=file {}".format(CLANG_FORMAT_PATH, filename)

    async with semaphore:
        proc = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE)
        # Read back the formatted file.
        stdout, _ = await proc.communicate()

    formatted_contents = stdout.decode()
    # Compare the formatted file to the original file.
    with open(filename) as orig:
        orig_contents = orig.read()
        if formatted_contents != orig_contents:
            ok = False
            if verbose:
                print("{} is not formatted correctly".format(filename))

    return ok


async def run_clang_format(max_processes, diff=False, verbose=False):
    """
    Run clang-format to all files in CLANG_FORMAT_WHITELIST that match CPP_FILE_REGEX.
    """
    # Check to make sure the clang-format binary exists.
    if not os.path.exists(CLANG_FORMAT_PATH):
        print("clang-format binary not found")
        return False

    # Gather command-line options for clang-format.
    args = [CLANG_FORMAT_PATH, "-style=file"]

    if not diff:
        args.append("-i")

    ok = True

    # Semaphore to bound the number of subprocesses that can be created at once to format files.
    semaphore = asyncio.Semaphore(max_processes)

    # Format files in parallel.
    if diff:
        for f in asyncio.as_completed([file_clang_formatted_correctly(f, semaphore, verbose) for f in get_whitelisted_files()]):
            ok &= await f

        if ok:
            print("All files formatted correctly")
        else:
            print("Some files not formatted correctly")
    else:
        await asyncio.gather(*[run_clang_format_on_file(f, semaphore, verbose) for f in get_whitelisted_files()])


def report_download_progress(chunk_number, chunk_size, file_size):
    """
    Pretty printer for file download progress.
    """
    if file_size != -1:
        percent = min(1, (chunk_number * chunk_size) / file_size)
        bar = "#" * int(64 * percent)
        sys.stdout.write("\r0% |{:<64}| {}%".format(bar, int(percent * 100)))


def download_clang_format(path):
    """
    Downloads a clang-format binary appropriate for the host platform and stores it at the given location.
    """
    if HOST_PLATFORM not in PLATFORM_TO_CF_URL:
        print("Unsupported platform: {}".format(HOST_PLATFORM))
        return False

    cf_url = PLATFORM_TO_CF_URL[HOST_PLATFORM]
    filename = os.path.join(path, "clang-format")

    # Try to download clang-format.
    print("Downloading clang-format to {}".format(path))
    try:
        urllib.request.urlretrieve(
            cf_url, filename, reporthook=report_download_progress
        )
    except urllib.error.URLError as e:
        print("Error downloading {}: {}".format(filename, str(e)))
        return False
    finally:
        print()

    return True


def get_and_check_clang_format(verbose=False):
    """
    Download a platform-appropriate clang-format binary if one doesn't already exist at the expected location and verify
    that it is the right binary by checking its SHA1 hash against the expected hash.
    """
    if not os.path.exists(CLANG_FORMAT_DIR):
        # If the directory doesn't exist, try to create it.
        try:
            os.mkdir(CLANG_FORMAT_DIR)
        except os.OSError as e:
            print("Unable to create directory for clang-format binary: {}".format(CLANG_FORMAT_DIR))
            return False
        finally:
            if verbose:
                print("Created directory {} for clang-format binary".format(CLANG_FORMAT_DIR))

        # If the directory didn't exist, neither did the binary, so download it.
        ok = download_clang_format(CLANG_FORMAT_DIR)

        if not ok:
            return False
    else:
        # If the directory exists but the binary doesn't, download it.
        if not os.path.exists(CLANG_FORMAT_PATH):
            ok = download_clang_format(CLANG_FORMAT_DIR)

            if not ok:
                return False
        else:
            if verbose:
                print("Found pre-existing clang-format binary, skipping download")

    # Now that the binary is where it should be, hash it.
    actual_bin_hash = compute_file_sha1(CLANG_FORMAT_PATH)

    # If the host platform is not in PLATFORM_TO_HASH, it is unsupported.
    if HOST_PLATFORM not in PLATFORM_TO_HASH:
        print("Unsupported platform: {}".format(HOST_PLATFORM))
        return False

    # This is the path to the file containing the reference hash.
    hashpath = os.path.join(PYTORCH_ROOT, PLATFORM_TO_HASH[HOST_PLATFORM])

    if not os.path.exists(hashpath):
        print("Unable to find reference binary hash")
        return False

    # Load the reference hash and compare the actual hash to it.
    with open(hashpath, "r") as f:
        reference_bin_hash = f.readline()

        if verbose:
            print("Reference Hash: {}".format(reference_bin_hash))
            print("Actual Hash: {}".format(actual_bin_hash))

        if reference_bin_hash != actual_bin_hash:
            print("The downloaded binary is not what was expected!")

            # Err on the side of caution and try to delete the downloaded binary.
            try:
                os.unlink(CLANG_FORMAT_PATH)
            except os.OSError as e:
                print("Failed to delete binary: {}".format(str(e)))
                print("Delete this binary as soon as possible and do not execute it!")

            return False
        else:
            # Make sure the binary is executable.
            mode = os.stat(CLANG_FORMAT_PATH).st_mode
            mode |= stat.S_IXUSR
            os.chmod(CLANG_FORMAT_PATH, mode)
            print("Using clang-format located at {}".format(CLANG_FORMAT_PATH))

    return True


def parse_args(args):
    """
    Parse and return command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Execute clang-format on your working copy changes."
    )
    parser.add_argument(
        "-d",
        "--diff",
        action="store_true",
        default=False,
        help="Determine whether running clang-format would produce changes",
    )
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    parser.add_argument("--max-processes", type=int, default=50,
                        help="Maximum number of subprocesses to create to format files in parallel")
    return parser.parse_args(args)


def main(args):
    # Parse arguments.
    options = parse_args(args)
    # Get clang-format and make sure it is the right binary and it is in the right place.
    ok = get_and_check_clang_format(options.verbose)
    # Invoke clang-format on all files in the directories in the whitelist.
    if ok:
        ok = asyncio.run(run_clang_format(options.max_processes, options.diff, options.verbose))

    # We have to invert because False -> 0, which is the code to be returned if everything is okay.
    return not ok


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
