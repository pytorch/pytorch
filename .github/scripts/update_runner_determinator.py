#!/usr/bin/env python3

import re


# Read the contents of runner_determinator.py
with open(".github/scripts/runner_determinator.py") as script_file:
    script_content = script_file.read()

# Indent the script content by 10 spaces to match destination indentation
indented_script_content = "\n".join(
    [" " * 10 + line if line else line for line in script_content.splitlines()]
)

# Read the contents of _runner-determinator.yml
with open(".github/workflows/_runner-determinator.yml") as yml_file:
    yml_content = yml_file.read()

# Replace the content between the markers
new_yml_content = re.sub(
    r"(cat <<EOF > runner_determinator.py\n)(.*?)(\n\s+EOF)",
    lambda match: match.group(1) + indented_script_content + match.group(3),
    yml_content,
    flags=re.DOTALL,
)

# Save the modified content back to _runner-determinator.yml
with open(".github/workflows/_runner-determinator.yml", "w") as yml_file:
    yml_file.write(new_yml_content)

print("Updated _runner-determinator.yml with the contents of runner_determinator.py")
