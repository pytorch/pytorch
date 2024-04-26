import os
import zipfile
import argparse
import tempfile
import shutil

# Function to extract specific subfolders from a pytorch wheel file into a
# temporary directory and copy them to the output directory which will look like libtorch

def extract_and_copy(whl_path, output_dir):
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Directories to copy from the .whl file
        torch_dirs = ["torch/bin", "torch/share", "torch/lib", "torch/include"]

        # Extract all contents to the temporary directory
        with zipfile.ZipFile(whl_path, 'r') as whl_file:
            whl_file.extractall(temp_dir)

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Copy only the specified directories from the temporary directory to the output directory
        for torch_dir in torch_dirs:
            temp_path = os.path.join(temp_dir, torch_dir)
            # we don't want to copy the torch/ directory itself, so we remove the first 5 characters
            output_path = os.path.join(output_dir, torch_dir[6:])

            # If the subfolder exists in the extracted content, copy it to the output directory
            if os.path.exists(temp_path):
                shutil.copytree(temp_path, output_path, dirs_exist_ok=True)

        # # add build-version and build-hash
        # with open(os.path.join(output_dir, "build-version"), "w") as f:
        #     f.write(args.build_version)
        # with open(os.path.join(output_dir, "build-hash"), "w") as f:
        #     f.write(args.build_hash)

        # # zip up file
        # with zipfile.ZipFile(output_dir + ".zip", 'w') as zip_file:
        #     for root, dirs, files in os.walk(output_dir):
        #         for file in files:
        #             zip_file.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), output_dir))

# Command-line argument setup
def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract specific subfolders from a .whl file to an output directory")
    parser.add_argument("--whl", required=True, help="Path to the .whl file")
    parser.add_argument("--out", required=True, help="Output directory to copy the specified subfolders")
    parser.add_argument("--pytorch_root", required=True, help="Path to the pytorch root")
    # parser.add_argument("--build_version", required=True, help="Build version of PyTorch")
    # parser.add_argument("--build_hash", required=True, help="Build hash of PyTorch")
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Call the function to extract and copy specific subfolders
    extract_and_copy(args.whl, args.out)
