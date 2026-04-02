#!/usr/bin/env python3
"""Convert RST files to MyST Markdown."""

import re
from pathlib import Path

SOURCE_DIR = Path(__file__).parent / "source"


def detect_heading_levels(lines: list[str]) -> dict[str, int]:
    """Map underline characters to heading levels by order of appearance."""
    char_to_level = {}
    level = 1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or len(stripped) < 2:
            continue
        if len(set(stripped)) == 1 and stripped[0] in "=-^~*+#":
            char = stripped[0]
            if i > 0 and lines[i - 1].strip():
                if char not in char_to_level:
                    char_to_level[char] = level
                    level += 1
    return char_to_level


def convert_inline_markup(text: str) -> str:
    """Convert RST inline markup to MyST Markdown."""
    # Convert RST links: `text <url>`_ -> [text](url)
    text = re.sub(r"`([^<]+?)\s*<([^>]+)>`_", r"[\1](\2)", text)
    # Convert RST inline code: ``code`` -> `code`
    text = re.sub(r"``(.+?)``", r"`\1`", text)
    # Convert domain-qualified RST roles: :c:macro:`X` -> {c:macro}`X`
    text = re.sub(r":(\w+:\w+):`([^`]+)`", r"{\1}`\2`", text)
    # Convert RST roles: :role:`text` -> {role}`text`
    # (but not :param, :type which are directive options)
    text = re.sub(r":(\w+):`([^`]+)`", r"{\1}`\2`", text)
    return text


def get_indent(line: str) -> int:
    """Return number of leading spaces."""
    return len(line) - len(line.lstrip())


def collect_indented_block(lines: list[str], start: int, min_indent: int) -> tuple[list[str], int]:
    """Collect lines indented at least min_indent spaces (or blank).

    Returns (collected_lines_with_indent_stripped, next_index).
    """
    collected = []
    i = start
    while i < len(lines):
        if not lines[i].strip():
            # Blank line - include if more indented content follows
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and get_indent(lines[j]) >= min_indent:
                for _ in range(i, j):
                    collected.append("")
                i = j
                continue
            else:
                break
        elif get_indent(lines[i]) >= min_indent:
            collected.append(lines[i][min_indent:])
            i += 1
        else:
            break
    while collected and not collected[-1].strip():
        collected.pop()
    return collected, i


def convert_block(lines: list[str], heading_levels: dict[str, int],
                  fence_depth: int = 3) -> list[str]:
    """Convert a block of RST lines to MyST Markdown.

    fence_depth: number of backticks for directives at this nesting level.
    """
    output: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # --- Heading detection ---
        if (
            i + 1 < len(lines)
            and line.strip()
            and not line.strip().startswith("..")
            and lines[i + 1].strip()
            and len(lines[i + 1].strip()) >= 2
            and len(set(lines[i + 1].strip())) == 1
            and lines[i + 1].strip()[0] in heading_levels
        ):
            char = lines[i + 1].strip()[0]
            level = heading_levels[char]
            heading_text = convert_inline_markup(line.strip())
            output.append("")
            output.append(f"{'#' * level} {heading_text}")
            output.append("")
            i += 2
            continue

        # --- RST directive ---
        directive_match = re.match(r"^(\s*)\.\.\s+(\S+)::\s*(.*)", line)
        if directive_match:
            base_indent_str = directive_match.group(1)
            directive_name = directive_match.group(2)
            directive_arg = directive_match.group(3).strip()
            content_indent = len(base_indent_str) + 3

            i += 1
            # Skip blank lines after directive to find actual content indent
            peek = i
            while peek < len(lines) and not lines[peek].strip():
                peek += 1
            if peek < len(lines) and get_indent(lines[peek]) > len(base_indent_str):
                actual_indent = get_indent(lines[peek])
                content_indent = min(content_indent, actual_indent)
            block, i = collect_indented_block(lines, i, content_indent)

            # Split block into options and body
            options = []
            body_start = 0
            for j, bl in enumerate(block):
                opt_match = re.match(r"^:(\w[\w-]*):\s*(.*)", bl)
                if opt_match:
                    options.append(f":{opt_match.group(1)}: {opt_match.group(2)}".rstrip())
                    body_start = j + 1
                elif bl.strip() == "":
                    body_start = j + 1
                else:
                    body_start = j
                    break
            body = block[body_start:]
            while body and not body[0].strip():
                body = body[1:]

            # --- code-block: use plain fenced code ---
            if directive_name == "code-block":
                lang = directive_arg or "cpp"
                output.append(f"{base_indent_str}```{lang}")
                for bl in body:
                    output.append(f"{base_indent_str}{bl}" if bl else "")
                output.append(f"{base_indent_str}```")
                continue

            # Determine fence depth: must be more than any inner fences.
            # Inner code blocks use ``` (3 ticks), nested directives use more.
            # If body contains code-blocks, directives, or literal block shorthand (::),
            # the outer fence needs at least 4 ticks.
            body_has_fences = any(
                re.match(r"^\s*\.\.\s+\S+::", bl) or bl.rstrip().endswith("::")
                for bl in body
            )
            actual_depth = max(fence_depth, 4) if body_has_fences else fence_depth
            backticks = "`" * actual_depth
            fence = f"{base_indent_str}{backticks}"

            if directive_arg:
                output.append(f"{fence}{{{directive_name}}} {directive_arg}")
            else:
                output.append(f"{fence}{{{directive_name}}}")

            for opt in options:
                output.append(f"{base_indent_str}{opt}")

            if body:
                output.append("")
                # Recursively convert body (may contain nested directives)
                body_converted = convert_block(body, heading_levels, actual_depth + 1)
                for bc in body_converted:
                    output.append(f"{base_indent_str}{bc}" if bc.strip() else "")
            output.append(fence)
            continue

        # --- Literal block shorthand (line ending with ::) ---
        if line.rstrip().endswith("::") and line.strip() != "::" and not line.strip().startswith(".."):
            text_part = line.rstrip()[:-1]  # Remove one colon
            output.append(convert_inline_markup(text_part))
            output.append("")
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i < len(lines) and get_indent(lines[i]) > get_indent(line):
                code_indent = get_indent(lines[i])
                code_lines, i = collect_indented_block(lines, i, code_indent)
                output.append("```cpp")
                for cl in code_lines:
                    output.append(cl if cl else "")
                output.append("```")
            continue

        # --- Grid table conversion ---
        if line.strip().startswith("+") and re.match(r"^\+[-=+]+\+$", line.strip()):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith(("+", "|")):
                table_lines.append(lines[i])
                i += 1

            rows = []
            current_row = None
            for tl in table_lines:
                stripped = tl.strip()
                if stripped.startswith("+"):
                    if current_row is not None:
                        rows.append(current_row)
                    current_row = None
                elif stripped.startswith("|"):
                    cells = [c.strip() for c in stripped.split("|")[1:-1]]
                    if current_row is None:
                        current_row = cells
                    else:
                        for j, cell in enumerate(cells):
                            if j < len(current_row) and cell:
                                current_row[j] += " " + cell
            if current_row is not None:
                rows.append(current_row)

            if rows:
                for j, row in enumerate(rows):
                    rows[j] = [convert_inline_markup(cell) for cell in row]
                output.append("| " + " | ".join(rows[0]) + " |")
                output.append("| " + " | ".join("---" for _ in rows[0]) + " |")
                for row in rows[1:]:
                    output.append("| " + " | ".join(row) + " |")
            continue

        # --- Regular line ---
        output.append(convert_inline_markup(line))
        i += 1

    return output


def convert_file(rst_path: Path) -> None:
    """Convert a single RST file to MyST Markdown."""
    lines = rst_path.read_text().splitlines()
    heading_levels = detect_heading_levels(lines)
    output = convert_block(lines, heading_levels)

    # Clean up multiple consecutive blank lines
    cleaned = []
    prev_blank = False
    for line in output:
        if line.strip() == "":
            if not prev_blank:
                cleaned.append("")
            prev_blank = True
        else:
            prev_blank = False
            cleaned.append(line)

    while cleaned and not cleaned[0].strip():
        cleaned.pop(0)

    md_path = rst_path.with_suffix(".md")
    md_path.write_text("\n".join(cleaned) + "\n")
    rst_path.unlink()
    print(f"  {rst_path.relative_to(SOURCE_DIR)} -> {md_path.relative_to(SOURCE_DIR)}")


def main():
    rst_files = sorted(SOURCE_DIR.rglob("*.rst"))
    if not rst_files:
        print("No RST files found. Already converted?")
        return
    print(f"Converting {len(rst_files)} RST files to MyST Markdown...")
    for rst_file in rst_files:
        convert_file(rst_file)
    print("Done.")


if __name__ == "__main__":
    main()
