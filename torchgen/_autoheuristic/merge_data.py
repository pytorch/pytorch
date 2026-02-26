import sys


def merge_txt_files(file_list: list[str], output_file: str) -> None:
    if not file_list:
        print("No input files provided.")
        return

    metadata: list[str] = []
    content: list[str] = []

    # Read metadata and content from all files
    for file_path in file_list:
        try:
            with open(file_path) as file:
                lines = file.readlines()
                if len(lines) < 2:
                    print(
                        f"Error: {file_path} does not have enough lines for metadata."
                    )
                    return

                file_metadata = lines[:2]
                file_content = lines[2:]

                if not metadata:
                    metadata = file_metadata
                elif metadata != file_metadata:
                    print(f"Error: Metadata mismatch in {file_path}")
                    print("Expected metadata:")
                    print("".join(metadata))
                    print(f"Metadata in {file_path}:")
                    print("".join(file_metadata))
                    return

                content.extend(file_content)
        except OSError as e:
            print(f"Error reading file {file_path}: {e}")
            return

    # Write merged content to output file
    try:
        with open(output_file, "w") as outfile:
            outfile.writelines(metadata)
            outfile.writelines(content)
        print(f"Successfully merged files into {output_file}")
    except OSError as e:
        print(f"Error writing to output file {output_file}: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python script.py output_file.txt input_file1.txt input_file2.txt ..."
        )
    else:
        output_file = sys.argv[1]
        input_files = sys.argv[2:]
        merge_txt_files(input_files, output_file)
