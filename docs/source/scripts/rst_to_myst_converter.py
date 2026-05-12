#!/usr/bin/env python3
"""
RST to MyST Markdown Converter
Based on https://github.com/svekars/sphinx-read-thedocs-test/blob/main/rst_to_myst_md_converter.py
"""

import argparse
import re
import sys
from pathlib import Path


class RSTToMarkdownConverter:
    def __init__(self):
        self.is_first_chunk = False
        self.is_last_chunk = False

    def set_chunk_position(self, is_first: bool, is_last: bool):
        self.is_first_chunk = is_first
        self.is_last_chunk = is_last

    def convert_links(self, content: str) -> str:
        """Convert RST links to Markdown format."""
        # Convert external links with text: `text <url>`_ -> [text](url)
        content = re.sub(
            r"`([^`]+)\s+<([^>]+)>`_",
            lambda m: f"[{m.group(1)}]({m.group(2)})",
            content,
        )

        # Convert simple links: `text`_ -> [text](text)
        content = re.sub(
            r"`([^`]+)`_", lambda m: f"[{m.group(1)}]({m.group(1)})", content
        )

        return content

    def convert_heading(self, content: str) -> str:
        """Convert RST headings to Markdown headings based on order of appearance."""
        lines = content.splitlines()
        converted_lines = []

        # Dictionary to store the mapping of underline style to heading level
        style_to_level = {}

        # First pass: determine hierarchy
        i = 0
        while i < len(lines) - 1:
            line = lines[i]
            next_line = lines[i + 1]

            # Check if this is a heading (text followed by underline)
            if (
                line.strip()  # Non-empty line
                and re.match(r"^[-=^~_]+$", next_line.strip())  # Underline line
                and len(next_line.strip()) >= len(line.strip())
            ):  # Underline at least as long as text

                underline_char = next_line.strip()[0]

                # If this is the first heading we've seen
                if not style_to_level:
                    if underline_char == "=" or self.is_first_chunk:
                        style_to_level[underline_char] = 1
                    else:
                        style_to_level[underline_char] = 2
                # For subsequent headings
                elif underline_char not in style_to_level:
                    style_to_level[underline_char] = len(style_to_level) + (
                        1 if self.is_first_chunk else 2
                    )

            i += 1

        # Second pass: convert headings
        i = 0
        while i < len(lines):
            line = lines[i]

            # Check for underline-style headers
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if re.match(r"^[-=^~_]+$", next_line.strip()) and len(
                    next_line.strip()
                ) >= len(line.strip()):

                    underline_char = next_line.strip()[0]
                    if underline_char in style_to_level:
                        level = style_to_level[underline_char]
                        converted_lines.append("#" * level + " " + line)
                        i += 2
                        continue

            converted_lines.append(line)
            i += 1

        print(f"Style to level mapping: {style_to_level}")
        return "\n".join(converted_lines)

    def remove_rst_labels(self, content: str) -> str:
        """Remove RST cross-reference labels."""
        return re.sub(r"\.\. _[\w-]+:\n\n", "", content)

    def convert_automodule_directive(self, content: str) -> str:
        """Convert automodule directives to MyST format."""

        def process_automodule(match):
            full_block = match.group(0).rstrip()
            return f"```{{eval-rst}}\n{full_block}\n```\n"

        automodule_pattern = r"\.\.?\s+automodule::\s*[^\n]+(?:\n\s+:[^\n]+)*"
        content = re.sub(
            automodule_pattern, process_automodule, content, flags=re.MULTILINE
        )
        return content

    def convert_currentmodule_directive(self, content: str) -> str:
        """Convert currentmodule directives to MyST format."""

        def process_currentmodule(match):
            full_block = match.group(0).rstrip()
            return f"```{{eval-rst}}\n{full_block}\n```\n"

        currentmodule_pattern = r"\.\.?\s+currentmodule::\s*[^\n]+"
        content = re.sub(
            currentmodule_pattern, process_currentmodule, content, flags=re.MULTILINE
        )
        return content

    def convert_autofunction_directive(self, content: str) -> str:
        """Convert autofunction directives to MyST format."""

        def process_autofunction(match):
            full_block = match.group(0).rstrip()
            return f"```{{eval-rst}}\n{full_block}\n```\n"

        autofunction_pattern = r"\.\.?\s+autofunction::\s*[^\n]+(?:\n\s+:[^\n]+)*"
        content = re.sub(
            autofunction_pattern, process_autofunction, content, flags=re.MULTILINE
        )
        return content

    def convert_autoclass_directive(self, content: str) -> str:
        """Convert autoclass directives to MyST format."""

        def process_autoclass(match):
            full_block = match.group(0).rstrip()
            return f"```{{eval-rst}}\n{full_block}\n```\n"

        autoclass_pattern = r"\.\.?\s+autoclass::\s*[^\n]+(?:\n\s+:[^\n]+)*"
        content = re.sub(
            autoclass_pattern, process_autoclass, content, flags=re.MULTILINE
        )
        return content

    def convert_cross_references(self, content: str) -> str:
        """Convert cross-references to MyST format."""
        ref_patterns = {
            r":py:class:`([^`]+)`": r"{py:class}`\1`",
            r":py:func:`([^`]+)`": r"{py:func}`\1`",
            r":py:meth:`([^`]+)`": r"{py:meth}`\1`",
            r":py:mod:`([^`]+)`": r"{py:mod}`\1`",
            r":py:attr:`([^`]+)`": r"{py:attr}`\1`",
            r":py:data:`([^`]+)`": r"{py:data}`\1`",
            r":py:exc:`([^`]+)`": r"{py:exc}`\1`",
            r":py:obj:`([^`]+)`": r"{py:obj}`\1`",
            r":doc:`([^`]+)`": r"{doc}`\1`",
            r":ref:`([^`]+)`": r"{ref}`\1`",
            r":title:`([^`]+)`": r"{title}`\1`",
            r":numref:`([^`]+)`": r"{numref}`\1`",
            r":cpp:class:`([^`]+)`": r"{cpp:class}`\1`",
            r":cpp:func:`([^`]+)`": r"{cpp:func}`\1`",
        }

        for pattern, replacement in ref_patterns.items():
            content = re.sub(pattern, replacement, content)
        return content

    def convert_comments(self, content: str) -> str:
        """Convert RST comments to HTML comments, but ignore autodoc directives."""

        def is_special_directive(line):
            special_patterns = [
                r"\.\.?\s+automodule::",
                r"\.\.?\s+autoclass::",
                r"\.\.?\s+autofunction::",
                r"\.\.?\s+automethod::",
                r"\.\.?\s+autoattribute::",
                r"\.\.?\s+autodata::",
                r"\.\.?\s+autoexception::",
                r"\.\.?\s+autoproperty::",
                r"\.\.?\s+autosummary::",
                r"\.\.?\s+currentmodule::",
                r"\.\.?\s+role::",
                r"\s+:toctree:",
                r"\s+:nosignatures:",
                r"\s+:template:",
                r"\s+:class:",
                r"\s+~[\w\.]+",
            ]
            return any(re.match(pattern, line) for pattern in special_patterns)

        lines = content.splitlines()
        converted_lines = []

        i = 0
        while i < len(lines):
            line = lines[i]
            if is_special_directive(line):
                converted_lines.append(line)
                i += 1
                continue

            if line.strip().startswith(".."):
                # Check if it's a single-line comment
                if not any(
                    line.strip().startswith(".. " + x)
                    for x in [
                        "figure::",
                        "image::",
                        "math::",
                        "include::",
                        "toctree::",
                        "contents::",
                        "role::",
                    ]
                ):
                    converted_lines.append(f"<!-- {line.strip()[3:]} -->")
                else:
                    converted_lines.append(line)
            else:
                converted_lines.append(line)
            i += 1

        return "\n".join(converted_lines)

    def convert(self, content: str) -> str:
        """Apply all conversion rules in sequence."""
        content = self.convert_heading(content)
        content = self.remove_rst_labels(content)
        content = self.convert_automodule_directive(content)
        content = self.convert_currentmodule_directive(content)
        content = self.convert_autofunction_directive(content)
        content = self.convert_autoclass_directive(content)
        content = self.convert_cross_references(content)
        content = self.convert_links(content)
        content = self.convert_comments(content)
        return content


def convert_rst_to_markdown(rst_content: str) -> str:
    """Convert RST content to Markdown using Python-based converter"""
    # Initialize converter
    converter = RSTToMarkdownConverter()
    converter.set_chunk_position(is_first=True, is_last=True)

    # Convert content
    try:
        converted_content = converter.convert(rst_content)
        return converted_content
    except Exception as e:
        print(f"Error in conversion: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Convert RST files to Markdown"
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Input RST file to convert"
    )
    parser.add_argument(
        "output_file",
        type=Path,
        nargs="?",
        help="Output Markdown file (default: input file with .md extension)"
    )
    parser.add_argument(
        "--show-content",
        action="store_true",
        help="Show sample of original and converted content",
    )

    args = parser.parse_args()

    # Set output file if not provided
    if not args.output_file:
        args.output_file = args.input_file.with_suffix(".md")

    try:
        # Read RST content
        with open(args.input_file, encoding="utf-8") as f:
            rst_content = f.read()
        if args.show_content:
            print("\nOriginal RST content:")
            print(rst_content)
            print("-" * 80)
        # Convert content
        markdown_content = convert_rst_to_markdown(rst_content)
        if args.show_content:
            print("\nConverted Markdown content:")
            print(markdown_content)
            print("-" * 80)
        # Save Markdown content
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        print(f"✓ Converted {args.input_file} to {args.output_file}")
    except Exception as e:
        print(f"✗ Error converting {args.input_file}: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
