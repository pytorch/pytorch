import argparse

parser = argparse.ArgumentParser(description='Add type ignore comments to files automatically.')
parser.add_argument('--log', help='mypy log file (mypy 2>&1 > log)')
args = parser.parse_args()

with open(args.log, "r") as f:
    for line in f.readlines():
        if line.startswith("Found "):
            # Summary line, ignore it
            continue
        line = line.strip()
        filename = line.split(":")[0]
        lineno = int(line.split(":")[1])
        error_code = line.split(" ")[-1].replace("[", "").replace("]", "")

        content = open(filename, "r").readlines()
        output = []
        for index, content_line in enumerate(content):
            if content_line[-1] == "\n":
                # remove trailing newline
                content_line = content_line[:-1]

            if index + 1 == lineno:
                if "type: ignore" not in content_line:
                    content_line = content_line + f"  # type: ignore[{error_code}]"
            output.append(content_line)

        open(filename, "w").write("\n".join(output) + "\n")
