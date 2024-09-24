#!/usr/bin/env python3

import sys
import os
import subprocess
import urllib.request
import tempfile
import shutil
from pathlib import Path

def print_usage():
    print(f"Usage: {sys.argv[0]} <PR_NUMBER>")
    sys.exit(1)

def get_pytorch_path():
    try:
        import torch
        torch_path = torch.__path__[0]
        parent_path = os.path.dirname(torch_path)
        print(f"PyTorch is installed at: {torch_path}")
        print(f"Parent directory for patching: {parent_path}")
        return parent_path
    except ImportError:
        print("PyTorch is not installed in the current Python environment.")
        sys.exit(1)

def download_patch(pr_number, repo_url, download_dir):
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
    print("Applying patch...")
    try:
        # Change to the target directory
        original_cwd = os.getcwd()
        os.chdir(target_dir)
        
        # Execute the patch command
        result = subprocess.run(['patch', '-p1', '-i', patch_file], capture_output=True, text=True)
        
        # Check if the patch was applied successfully
        if result.returncode != 0:
            print("Failed to apply patch.")
            print("Patch output:")
            print(result.stdout)
            print(result.stderr)
            sys.exit(1)
        else:
            print("Patch applied successfully.")
    except FileNotFoundError:
        print("The 'patch' utility is not installed or not found in PATH.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while applying the patch: {e}")
        sys.exit(1)
    finally:
        # Change back to the original directory
        os.chdir(original_cwd)

def main():
    # 1. Check if PR number is provided
    if len(sys.argv) != 2:
        print_usage()
    
    pr_number = sys.argv[1]
    
    # 2. Get the path where PyTorch is installed
    target_dir = get_pytorch_path()
    
    # 3. Set the repository URL
    repo_url = "https://github.com/pytorch/pytorch"
    
    # 4. Download the PR patch from GitHub
    with tempfile.TemporaryDirectory() as tmpdirname:
        patch_file = download_patch(pr_number, repo_url, tmpdirname)
        
        # 5. Apply the patch to the PyTorch installation
        apply_patch(patch_file, target_dir)
        
        # 6. Clean up is handled automatically by TemporaryDirectory

if __name__ == "__main__":
    main()
