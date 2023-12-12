import argparse
import os
import os.path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file")
    parser.add_argument("--output-file")
    parser.add_argument("--install-dir", "--install_dir")
    parser.add_argument("--replace", action="append", nargs=2)
    options = parser.parse_args()

    with open(options.input_file) as f:
        contents = f.read()

    output_file = os.path.join(options.install_dir, options.output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    for old, new in options.replace:
        contents = contents.replace(old, new)

    with open(output_file, "w") as f:
        f.write(contents)


if __name__ == "__main__":
    main()  # pragma: no cover
