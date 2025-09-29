import glob


requires_files = glob.glob("requirements/*.txt")
requires_files += ["pyproject.toml"]
for file in requires_files:
    print(f">>> cleaning {file}")
    with open(file) as f:
        lines = f.readlines()
    if "torch" in "".join(lines).lower():
        print("removed:")
        with open(file, "w") as f:
            for line in lines:
                if "torch" not in line.lower():
                    f.write(line)
    print(f"<<< done cleaning {file}")
    print()
