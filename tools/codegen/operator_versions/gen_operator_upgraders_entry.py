#!/usr/bin/env python3
import argparse
import os
import sys
import yaml
from tools.codegen.code_template import CodeTemplate
from collections import defaultdict

UPGRADER_CPP_SRC = """#pragma once
#include <string>
#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {

struct UpgraderEntry {
  int version_bump;
  std::string upgrader_name;
  std::string old_schema;
};

static std::unordered_map<std::string, std::vector<UpgraderEntry>> operator_version_map(
    $version_entry
);

} // namespace jit
} // namespace torch
"""

def load_yaml(version_entry_path):
    with open(version_entry_path, "rb") as yaml_file:
        return yaml.safe_load(yaml_file)

def write_header(header_path, entry_map):
    file_path = os.path.join(header_path, "version_map.h")
    print("Writing file to : ", file_path)
    ct = CodeTemplate(UPGRADER_CPP_SRC)
    file_content = ct.substitute(version_entry=entry_map)

    with open(file_path, "w") as out_file:
        out_file.write(file_content)

def format_version_entry(version_entry_path):
    entries = load_yaml(version_entry_path)
    entry_dict = defaultdict(list)
    for entry in entries:
        op_name = entry["func"]
        op_version = entry["version"]
        op_upgrader_name = entry["upgrader"]
        op_old_schema = entry["old_schema"]
        upgrader_entry = [op_version, op_upgrader_name, op_old_schema]
        entry_dict[op_name].append(upgrader_entry)
    # to make the formatting easier
    entry_list = []
    for key, val in entry_dict.items():
        entry_list.append([key, val])

    formatted_str = str(entry_list).replace("[", "{").replace("]", "}").replace("\'", "\"")
    return formatted_str

def get_parser_options(parser: argparse.ArgumentParser) -> argparse.Namespace:
    parser.add_argument(
        "--input_yaml",
        type=str,
        help="A path to input yaml file.",
        required=True,
    )
    parser.add_argument(
        "--output_path",
        help="Output the version entry map header file.",
        required=True,
    )
    options = parser.parse_args()
    return options


def main(argv):
    parser = argparse.ArgumentParser(description="Generate used operators YAML")
    options = get_parser_options(parser)

    yaml_path = options.input_yaml
    output_path = options.output_path

    entry_map = format_version_entry(yaml_path)
    write_header(options.output_path, entry_map)


if __name__ == "__main__":
    main(sys.argv)
