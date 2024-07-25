import os
import subprocess
import sys


def main(output_file_name):
    # Set the path to the folder containing the Python programs
    folder_path = os.path.dirname(__file__) + "/benchmarks"

    # Open the output file for writing
    with open(output_file_name, "w") as output_file:
        # Loop through all files in the folder
        for filename in os.listdir(folder_path):
            # Check if the file is a Python program
            if filename.endswith(".py"):
                print("running benchmark:", filename)
                # Run the Python program and capture the output
                output = subprocess.check_output(
                    ["python", os.path.join(folder_path, filename)]
                )
                # Write the output to the output file
                output_file.write(f"{filename}:\n{output.decode()}\n")


if __name__ == "__main__":
    # Take the folder path as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python program.py <output_file_name>")
        sys.exit(1)
    output_file_name = sys.argv[1]
    main(output_file_name)
