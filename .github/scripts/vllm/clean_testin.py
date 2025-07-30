
import glob

# Explicitly define the files to clean
requires_files = [
    "requirements/test.in",
]

# Keywords to match exactly
keywords_to_remove = ['torch==', 'torchaudio==', 'torchvision==']

for file in requires_files:
    print(f">>> cleaning {file}")
    with open(file) as f:
        lines = f.readlines()

    cleaned_lines = []
    for line in lines:
        line_lower = line.strip().lower()
        if any(line_lower.startswith(kw) for kw in keywords_to_remove):
            print("removed:", line.strip())
        else:
            cleaned_lines.append(line)
    print(f"<<< done cleaning {file}\n")
