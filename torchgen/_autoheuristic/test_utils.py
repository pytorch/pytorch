import subprocess


def read_file_to_string(file_path: str) -> str:
    with open(file_path) as file:
        return file.read()


def run_bash(bash_script_path: str) -> None:
    try:
        print("Executing: ", bash_script_path)
        result = subprocess.run(
            ["bash", bash_script_path], capture_output=True, text=True, check=True
        )
        # Print the output
        print(f"Output of {bash_script_path}: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred executing {bash_script_path}: {e}")
        print("Error output:", e.stderr)
