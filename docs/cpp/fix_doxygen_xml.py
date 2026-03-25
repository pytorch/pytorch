#!/usr/bin/env python3
"""Fix Doxygen 1.11.x XML output where \\verbatim/\\endverbatim pairs used
through aliases (\\rst/\\endrst) produce malformed verbatim blocks.

Two bugs are fixed:

1. Text between two \\rst/\\endrst blocks gets incorrectly placed inside a
   <verbatim> element that doesn't start with "embed:rst". Breathe renders
   this as a literal block instead of normal documentation text.

2. Valid embed:rst:leading-slashes verbatim blocks retain the /// prefixes
   in Doxygen 1.11.x output. Breathe's parser in some configurations fails
   to strip these, causing raw RST directives to appear in the HTML output.
   We strip the prefixes and the embed:rst header directly in the XML.
"""

import glob
import re
import sys
import textwrap
import xml.etree.ElementTree as ET


def fix_file(path: str) -> bool:
    """Fix broken verbatim blocks in a single XML file. Returns True if modified."""
    tree = ET.parse(path)
    root = tree.getroot()
    modified = False

    for parent in root.iter():
        for child in list(parent):
            if child.tag != "verbatim":
                continue
            if not child.text:
                continue

            text = child.text

            if not text.strip().startswith("embed:rst"):
                # Bug 1: verbatim block contains regular text that should be
                # a <para> element. Strip any nested \verbatim directives.
                text = re.sub(r"\\verbatim\s+embed:rst\S*\s*", "", text)
                text = text.strip()
                if text:
                    new_para = ET.Element("para")
                    new_para.text = text
                    new_para.tail = child.tail
                    idx = list(parent).index(child)
                    parent.remove(child)
                    parent.insert(idx, new_para)
                else:
                    parent.remove(child)
                modified = True

            elif text.strip().startswith("embed:rst:leading-slashes"):
                # Bug 2: strip /// prefixes and convert to plain
                # embed:rst so Breathe processes it correctly.
                lines = text.splitlines()
                # Remove the first line ("embed:rst:leading-slashes")
                lines = lines[1:]
                # Strip /// prefix from each line
                stripped = []
                for line in lines:
                    stripped.append(re.sub(r"^/// ?", "   ", line))
                rst_text = textwrap.dedent("\n".join(stripped))
                child.text = "embed:rst\n" + rst_text
                modified = True

    if modified:
        tree.write(path, encoding="unicode", xml_declaration=True)
    return modified


def main():
    xml_dir = sys.argv[1] if len(sys.argv) > 1 else "build/xml"
    patterns = [f"{xml_dir}/class*.xml", f"{xml_dir}/struct*.xml"]
    fixed = 0
    for pattern in patterns:
        for path in glob.glob(pattern):
            if fix_file(path):
                fixed += 1
                print(f"  Fixed: {path.split('/')[-1]}")
    print(f"Fixed {fixed} files with broken verbatim blocks.")


if __name__ == "__main__":
    main()
