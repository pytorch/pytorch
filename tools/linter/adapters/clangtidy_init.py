import platform
import argparse
import sys
import stat
import hashlib
import subprocess
import os
import urllib.request
import urllib.error
import pathlib

from typing import Dict

# String representing the host platform (e.g. Linux, Darwin).
HOST_PLATFORM = platform.system()

# PyTorch directory root
result = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE, check=True,
)
PYTORCH_ROOT = result.stdout.decode("utf-8").strip()

HASH_PATH = pathlib.Path(PYTORCH_ROOT) / "tools" / "linter" / "install" / "hashes"


def compute_file_sha256(path: str) -> str:
    """Compute the SHA256 hash of a file and return it as a hex string."""
    # If the file doesn't exist, return an empty string.
    if not os.path.exists(path):
        return ""

    hash = hashlib.sha256()

    # Open the file in binary mode and hash it.
    with open(path, "rb") as f:
        for b in f:
            hash.update(b)

    # Return the hash as a hexadecimal string.
    return hash.hexdigest()


def report_download_progress(
    chunk_number: int, chunk_size: int, file_size: int
) -> None:
    """
    Pretty printer for file download progress.
    """
    if file_size != -1:
        percent = min(1, (chunk_number * chunk_size) / file_size)
        bar = "#" * int(64 * percent)
        sys.stdout.write("\r0% |{:<64}| {}%".format(bar, int(percent * 100)))


def download_bin(name: str, output_dir: str, platform_to_url: Dict[str, str], dry_run: bool) -> bool:
    """
    Downloads the binary appropriate for the host platform and stores it in the given output directory.
    """
    if HOST_PLATFORM not in platform_to_url:
        print(f"Unsupported platform: {HOST_PLATFORM}")
        return False

    url = platform_to_url[HOST_PLATFORM]
    filename = os.path.join(output_dir, name)
    if dry_run:
        print(f"DRY RUN: Would download {url} to {filename}")
        return True

    # Try to download binary.
    print(f"Downloading {name} to {output_dir}")
    try:
        urllib.request.urlretrieve(
            url,
            filename,
            reporthook=report_download_progress if sys.stdout.isatty() else None,
        )
    except urllib.error.URLError as e:
        print(f"Error downloading {filename}: {e}")
        return False
    finally:
        print()

    return True


def download(
    name: str,
    output_dir: str,
    platform_to_url: Dict[str, str],
    platform_to_hash: Dict[str, str],
    dry_run: bool,
) -> bool:
    """
    Download a platform-appropriate binary if one doesn't already exist at the expected location and verifies
    that it is the right binary by checking its SHA256 hash against the expected hash.
    """

    output_path = os.path.join(output_dir, name)
    if not os.path.exists(output_dir):
        # If the directory doesn't exist, try to create it.
        if dry_run:
            print(f"DRY RUN: would create directory for {name} binary: {output_dir}")
        else:
            try:
                os.mkdir(output_dir)
            except OSError as e:
                print(f"Unable to create directory for {name} binary: {output_dir}")
                return False
            finally:
                print(f"Created directory {output_dir} for {name} binary")

        # If the directory didn't exist, neither did the binary, so download it.
        ok = download_bin(name, output_dir, platform_to_url, dry_run)

        if not ok:
            return False
    else:
        # If the directory exists but the binary doesn't, download it.
        if not os.path.exists(output_path):
            ok = download_bin(name, output_dir, platform_to_url, dry_run)

            if not ok:
                return False
        else:
            print(f"Found pre-existing {name} binary, skipping download")

    # Now that the binary is where it should be, hash it.
    actual_bin_hash = compute_file_sha256(output_path)

    # If the host platform is not in platform_to_hash, it is unsupported.
    if HOST_PLATFORM not in platform_to_hash:
        print(f"Unsupported platform: {HOST_PLATFORM}")
        return False

    # This is the path to the file containing the reference hash.
    hashpath = os.path.join(PYTORCH_ROOT, platform_to_hash[HOST_PLATFORM])

    if not os.path.exists(hashpath):
        print("Unable to find reference binary hash")
        return False

    # Load the reference hash and compare the actual hash to it.
    if dry_run:
        # We didn't download anything, just bail
        return True

    with open(hashpath, "r") as f:
        reference_bin_hash = f.readline().strip()

        print(f"Reference Hash: {reference_bin_hash}")
        print(f"Actual Hash: {repr(actual_bin_hash)}")

        if reference_bin_hash != actual_bin_hash:
            print("The downloaded binary is not what was expected!")
            print(f"Downloaded hash: {repr(actual_bin_hash)} vs expected {reference_bin_hash}")

            # Err on the side of caution and try to delete the downloaded binary.
            try:
                os.unlink(output_path)
                print("The binary has been deleted just to be safe")
            except OSError as e:
                print(f"Failed to delete binary: {e}")
                print("Delete this binary as soon as possible and do not execute it!")

            return False
        else:
            # Make sure the binary is executable.
            mode = os.stat(output_path).st_mode
            mode |= stat.S_IXUSR
            os.chmod(output_path, mode)
            print(f"Using {name} located at {output_path}")

    return True



PLATFORM_TO_URL = {
    "Linux": "https://oss-clang-format.s3.us-east-2.amazonaws.com/linux64/clang-tidy",
    "Darwin": "https://oss-clang-format.s3.us-east-2.amazonaws.com/macos/clang-tidy",
}

PLATFORM_TO_HASH = {
    "Linux": os.path.join(HASH_PATH, "clang-tidy-linux64"),
    "Darwin": os.path.join(HASH_PATH, "clang-tidy-macos"),
}

OUTPUT_DIR = os.path.join(PYTORCH_ROOT, ".clang-tidy-bin")
INSTALLATION_PATH = os.path.join(OUTPUT_DIR, "clang-tidy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="downloads clang-tidy",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="place to put the binary",
    )
    parser.add_argument(
        "--output_name",
        required=True,
        help="name of binary",
    )
    parser.add_argument(
        "--dry_run",
        default=False,
        help="do not download, just print what would be done",
    )

    args = parser.parse_args()
    if args.dry_run == "0":
        args.dry_run = False
    else:
        args.dry_run = True

    ok = download(args.output_name, args.output_dir, PLATFORM_TO_URL, PLATFORM_TO_HASH, args.dry_run)
    if not ok:
        print(f"Failed to download clang-tidy binary from {PLATFORM_TO_URL}")
        exit(1)
