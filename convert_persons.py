import re

with open("docs/source/community/persons_of_interest.md", "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []

if lines[0].strip() == ":orphan:":
    new_lines.append("---\n")
    new_lines.append("orphan: true\n")
    new_lines.append("---\n")
    lines = lines[1:]

i = 0
while i < len(lines):
    line = lines[i]
    if i + 1 < len(lines):
        next_line = lines[i+1].strip()
        if len(next_line) > 0 and set(next_line) == {'='} and len(next_line) >= 3:
            new_lines.append(f"# {line.strip()}\n")
            i += 2
            continue
        elif len(next_line) > 0 and set(next_line) == {'-'} and len(next_line) >= 3:
            new_lines.append(f"## {line.strip()}\n")
            i += 2
            continue
        elif len(next_line) > 0 and set(next_line) == {'~'} and len(next_line) >= 3:
            new_lines.append(f"### {line.strip()}\n")
            i += 2
            continue
            
    # replace links `text <url>`__ -> [text](url)
    line = re.sub(r'`([^`]+?)\s*<([^>]+?)>`__', r'[\1](\2)', line)
    new_lines.append(line)
    i += 1

with open("docs/source/community/persons_of_interest.md", "w", encoding="utf-8") as f:
    f.writelines(new_lines)
