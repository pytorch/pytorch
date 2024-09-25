#!/usr/bin/env python3

import sys
import os
import subprocess
import urllib.request
import tempfile
import shutil
import argparse
from pathlib import Path

def parse_arguments():
    """
    Parses command-line arguments using argparse.

    Returns:
        argparse.Namespace: The parsed arguments containing the PR number.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Download and apply a Pull Request (PR) patch from the PyTorch GitHub repository "
            "to your local PyTorch installation.\n\n"
            "Best Practice: Since this script involves hot-patching PyTorch, it's recommended to use "
            "a disposable environment like a Docker container or a dedicated Python virtual environment (venv). "
            "This ensures that if the patching fails, you can easily recover by resetting the environment."
        ),
        epilog=(
            "Example:\n"
            "  python nightly_hotpatch.py 12345\n\n"
            "This command will download the patch for PR #12345 and apply it to your local "
            "PyTorch installation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'PR_NUMBER',
        type=int,
        help='The number of the Pull Request (PR) from the PyTorch GitHub repository to download and apply as a patch.'
    )

    return parser.parse_args()

def get_pytorch_path():
    """
    Retrieves the installation path of PyTorch in the current environment.

    Returns:
        str: The parent directory of the PyTorch installation.

    Exits:
        If PyTorch is not installed in the current Python environment, the script will exit.
    """
    try:
        import torch
        torch_path = torch.__path__[0]
        parent_path = os.path.dirname(torch_path)
        print(f"PyTorch is installed at: {torch_path}")
        print(f"Parent directory for patching: {parent_path}")
        return parent_path
    except ImportError:
        print("Error: PyTorch is not installed in the current Python environment.")
        sys.exit(1)

def download_patch(pr_number, repo_url, download_dir):
    """
    Downloads the patch file for a given PR from the specified GitHub repository.

    Args:
        pr_number (int): The pull request number.
        repo_url (str): The URL of the repository where the PR is hosted.
        download_dir (str): The directory to store the downloaded patch.

    Returns:
        str: The path to the downloaded patch file.

    Exits:
        If the download fails, the script will exit.
    """
    patch_url = f"{repo_url}/pull/{pr_number}.diff"
    patch_file = os.path.join(download_dir, f"pr-{pr_number}.patch")
    print(f"Downloading PR #{pr_number} patch from {patch_url}...")
    try:
        with urllib.request.urlopen(patch_url) as response, open(patch_file, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        if not os.path.isfile(patch_file):
            print(f"Failed to download patch for PR #{pr_number}")
            sys.exit(1)
        print(f"Patch downloaded to {patch_file}")
        return patch_file
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} when downloading patch for PR #{pr_number}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while downloading the patch: {e}")
        sys.exit(1)

def apply_patch(patch_file, target_dir):
    """
    Applies the downloaded patch to the specified directory.

    Args:
        patch_file (str): The path to the patch file.
        target_dir (str): The directory to apply the patch to.

    Exits:
        If the patch command fails or the 'patch' utility is not available, the script will exit.
    """
    print("Applying patch...")
    try:
        original_cwd = os.getcwd()
        os.chdir(target_dir)

        result = subprocess.run(['patch', '-p1', '-i', patch_file], capture_output=True, text=True)

        if result.returncode != 0:
            print("Failed to apply patch.")
            print("Patch output:")
            print(result.stdout)
            print(result.stderr)
            sys.exit(1)
        else:
            print("Patch applied successfully.")
    except FileNotFoundError:
        print("Error: The 'patch' utility is not installed or not found in PATH.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while applying the patch: {e}")
        sys.exit(1)
    finally:
        os.chdir(original_cwd)

def main():
    """
    Main function to orchestrate the patch download and application process.

    Steps:
        1. Parse command-line arguments to get the PR number.
        2. Retrieve the local PyTorch installation path.
        3. Download the patch for the provided PR number.
        4. Apply the patch to the local PyTorch installation.
    """
    args = parse_arguments()
    pr_number = args.PR_NUMBER

    target_dir = get_pytorch_path()

    repo_url = "https://github.com/pytorch/pytorch"

    with tempfile.TemporaryDirectory() as tmpdirname:
        patch_file = download_patch(pr_number, repo_url, tmpdirname)

        apply_patch(patch_file, target_dir)


if __name__ == "__main__":
    main()
