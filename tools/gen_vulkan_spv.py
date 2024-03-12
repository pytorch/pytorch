#!/usr/bin/env python3

import argparse
import array
import codecs
import copy
import glob
import io
import os
import re
import sys
from itertools import product

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import subprocess
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader  # type: ignore[assignment, misc]

CPP_H_NAME = "spv.h"
CPP_SRC_NAME = "spv.cpp"
DEFAULT_ENV = {
    "PRECISION": "highp",
    "FLOAT_IMAGE_FORMAT": "rgba16f",
    "INT_IMAGE_FORMAT": "rgba32i",
    "UINT_IMAGE_FORMAT": "rgba32ui",
}


def extract_filename(path: str, keep_ext: bool = True) -> Any:
    if keep_ext:
        return os.path.basename(path)
    else:
        return os.path.basename(path).split(".")[0]


############################
#  SPIR-V Code Generation  #
############################


# https://gist.github.com/pypt/94d747fe5180851196eb
class UniqueKeyLoader(Loader):
    def construct_mapping(self, node, deep=False):  # type: ignore[no-untyped-def]
        if not isinstance(node, MappingNode):
            raise ConstructorError(
                None,
                None,
                f"expected a mapping node, but found {node.id}",
                node.start_mark,
            )
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)  # type: ignore[no-untyped-call]
            try:
                hash(key)
            except TypeError as e:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found unacceptable key ",
                    key_node.start_mark,
                ) from e
            # check for duplicate keys
            if key in mapping:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found duplicate key",
                    key_node.start_mark,
                )
            value = self.construct_object(value_node, deep=deep)  # type: ignore[no-untyped-call]
            mapping[key] = value
        return mapping


# https://github.com/google/XNNPACK/blob/master/tools/xngen.py
def extract_leading_whitespace(line: str) -> str:
    match = re.match(r"\s*", line)
    return match.group(0) if match else ""


# https://github.com/google/XNNPACK/blob/master/tools/xngen.py
def escape(line: str) -> str:
    output_parts = []
    while "${" in line:
        start_pos = line.index("${")
        end_pos = line.index("}", start_pos + 2)
        if start_pos != 0:
            output_parts.append('"' + line[:start_pos].replace('"', '\\"') + '"')
        output_parts.append("str(" + line[start_pos + 2 : end_pos] + ")")
        line = line[end_pos + 1 :]
    if line:
        output_parts.append('"' + line.replace('"', '\\"') + '"')
    return " + ".join(output_parts)


# https://github.com/google/XNNPACK/blob/master/tools/xngen.py
def preprocess(
    input_text: str, variables: Dict[str, Any], input_path: str = "codegen"
) -> str:
    input_lines = input_text.splitlines()
    python_lines = []

    blank_lines = 0

    last_indent = ""

    # List of tuples (total_index, python_indent)
    indent_stack = [("", "")]

    # Indicates whether this is the first line inside Python
    # code block (i.e. for, while, if, elif, else)
    python_block_start = True
    for i, input_line in enumerate(input_lines):
        if input_line == "":
            blank_lines += 1
            continue
        # Skip lint markers.
        if "LINT" in input_line:
            continue

        input_indent = extract_leading_whitespace(input_line)
        if python_block_start:
            assert input_indent.startswith(last_indent)
            extra_python_indent = input_indent[len(last_indent) :]
            python_indent = indent_stack[-1][1] + extra_python_indent
            indent_stack.append((input_indent, python_indent))
            assert input_indent.startswith(indent_stack[-1][0])
        else:
            while not input_indent.startswith(indent_stack[-1][0]):
                del indent_stack[-1]
        python_block_start = False

        python_indent = indent_stack[-1][1]
        stripped_input_line = input_line.strip()
        if stripped_input_line.startswith("$") and not stripped_input_line.startswith(
            "${"
        ):
            if stripped_input_line.endswith(":"):
                python_block_start = True
            while blank_lines != 0:
                python_lines.append(python_indent + "print(file=OUT_STREAM)")
                blank_lines -= 1
            python_lines.append(python_indent + stripped_input_line.replace("$", ""))
        else:
            assert input_line.startswith(python_indent)
            while blank_lines != 0:
                python_lines.append(python_indent + "print(file=OUT_STREAM)")
                blank_lines -= 1
            python_lines.append(
                python_indent
                + "print(%s, file=OUT_STREAM)"
                % escape(input_line[len(python_indent) :])
            )
        last_indent = input_indent

    while blank_lines != 0:
        python_lines.append(python_indent + "print(file=OUT_STREAM)")
        blank_lines -= 1

    exec_globals = dict(variables)
    output_stream = io.StringIO()
    exec_globals["OUT_STREAM"] = output_stream

    python_bytecode = compile("\n".join(python_lines), input_path, "exec")
    exec(python_bytecode, exec_globals)

    return output_stream.getvalue()


class SPVGenerator:
    def __init__(
        self,
        src_dir_paths: Union[str, List[str]],
        env: Dict[Any, Any],
        glslc_path: Optional[str],
    ) -> None:
        if isinstance(src_dir_paths, str):
            self.src_dir_paths = [src_dir_paths]
        else:
            self.src_dir_paths = src_dir_paths

        self.env = env
        self.glslc_path = glslc_path

        self.glsl_src_files: Dict[str, str] = {}
        self.template_yaml_files: List[str] = []

        self.addSrcAndYamlFiles(self.src_dir_paths)
        self.shader_template_params: Dict[Any, Any] = {}
        for yaml_file in self.template_yaml_files:
            self.parseTemplateYaml(yaml_file)

        self.output_shader_map: Dict[str, Tuple[str, Dict[str, str]]] = {}
        self.constructOutputMap()

    def addSrcAndYamlFiles(self, src_dir_paths: List[str]) -> None:
        for src_path in src_dir_paths:
            # Collect glsl source files
            glsl_files = glob.glob(
                os.path.join(src_path, "**", "*.glsl*"), recursive=True
            )
            for file in glsl_files:
                if len(file) > 1:
                    self.glsl_src_files[extract_filename(file, keep_ext=False)] = file
            # Collect template yaml files
            yaml_files = glob.glob(
                os.path.join(src_path, "**", "*.yaml"), recursive=True
            )
            for file in yaml_files:
                if len(file) > 1:
                    self.template_yaml_files.append(file)

    def generateVariantCombinations(
        self,
        iterated_params: Dict[str, Any],
        exclude_params: Optional[Set[str]] = None,
    ) -> List[Any]:
        if exclude_params is None:
            exclude_params = set()
        all_iterated_params = []
        for param_name, value_list in iterated_params.items():
            if param_name not in exclude_params:
                param_values = []
                for value in value_list:
                    suffix = value.get("SUFFIX", value["VALUE"])
                    param_values.append((param_name, suffix, value["VALUE"]))
                all_iterated_params.append(param_values)

        return list(product(*all_iterated_params))

    def parseTemplateYaml(self, yaml_file: str) -> None:
        with open(yaml_file) as f:
            contents = yaml.load(f, Loader=UniqueKeyLoader)
            for template_name, params_dict in contents.items():
                if template_name in self.shader_template_params:
                    raise KeyError(f"{template_name} params file is defined twice")

                default_params = params_dict["parameter_names_with_default_values"]
                params_names = set(default_params.keys()).union({"NAME"})

                self.shader_template_params[template_name] = []

                default_iterated_params = params_dict.get(
                    "generate_variant_forall", None
                )

                for variant in params_dict["shader_variants"]:
                    variant_params_names = set(variant.keys())
                    invalid_keys = (
                        variant_params_names
                        - params_names
                        - {"generate_variant_forall"}
                    )
                    assert len(invalid_keys) == 0

                    iterated_params = variant.get(
                        "generate_variant_forall", default_iterated_params
                    )

                    if iterated_params is not None:
                        variant_combinations = self.generateVariantCombinations(
                            iterated_params, variant_params_names
                        )

                        for combination in variant_combinations:
                            default_params_copy = copy.deepcopy(default_params)
                            for key in variant:
                                if key != "generate_variant_forall":
                                    default_params_copy[key] = variant[key]

                            variant_name = variant["NAME"]
                            for param_value in combination:
                                default_params_copy[param_value[0]] = param_value[2]
                                if len(param_value[1]) > 0:
                                    variant_name = f"{variant_name}_{param_value[1]}"

                            default_params_copy["NAME"] = variant_name

                            self.shader_template_params[template_name].append(
                                default_params_copy
                            )
                    else:
                        default_params_copy = copy.deepcopy(default_params)
                        for key in variant:
                            default_params_copy[key] = variant[key]

                        self.shader_template_params[template_name].append(
                            default_params_copy
                        )

    def create_shader_params(
        self, variant_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        if variant_params is None:
            variant_params = {}
        shader_params = copy.deepcopy(self.env)
        for key, value in variant_params.items():
            shader_params[key] = value

        shader_dtype = shader_params.get("DTYPE", "float")

        if shader_dtype == "int":
            shader_params["FORMAT"] = self.env["INT_IMAGE_FORMAT"]
        elif shader_dtype == "uint":
            shader_params["FORMAT"] = self.env["UINT_IMAGE_FORMAT"]
        elif shader_dtype == "int32":
            shader_params["FORMAT"] = "rgba32i"
        elif shader_dtype == "uint32":
            shader_params["FORMAT"] = "rgba32ui"
        elif shader_dtype == "int8":
            shader_params["FORMAT"] = "rgba8i"
        elif shader_dtype == "uint8":
            shader_params["FORMAT"] = "rgba8ui"
        elif shader_dtype == "float32":
            shader_params["FORMAT"] = "rgba32f"
        # Assume float by default
        else:
            shader_params["FORMAT"] = self.env["FLOAT_IMAGE_FORMAT"]

        return shader_params

    def constructOutputMap(self) -> None:
        for shader_name, params in self.shader_template_params.items():
            for variant in params:
                source_glsl = self.glsl_src_files[shader_name]

                self.output_shader_map[variant["NAME"]] = (
                    source_glsl,
                    self.create_shader_params(variant),
                )

        for shader_name, source_glsl in self.glsl_src_files.items():
            if shader_name not in self.shader_template_params:
                self.output_shader_map[shader_name] = (
                    source_glsl,
                    self.create_shader_params(),
                )

    def generateSPV(self, output_dir: str) -> Dict[str, str]:
        output_file_map = {}
        for shader_name in self.output_shader_map:
            source_glsl = self.output_shader_map[shader_name][0]
            shader_params = self.output_shader_map[shader_name][1]

            with codecs.open(source_glsl, "r", encoding="utf-8") as input_file:
                input_text = input_file.read()
                output_text = preprocess(input_text, shader_params)

            glsl_out_path = os.path.join(output_dir, f"{shader_name}.glsl")
            with codecs.open(glsl_out_path, "w", encoding="utf-8") as output_file:
                output_file.write(output_text)

            # If no GLSL compiler is specified, then only write out the generated GLSL shaders.
            # This is mainly for testing purposes.
            if self.glslc_path is not None:
                spv_out_path = os.path.join(output_dir, f"{shader_name}.spv")

                cmd = [
                    self.glslc_path,
                    "-fshader-stage=compute",
                    glsl_out_path,
                    "-o",
                    spv_out_path,
                    "--target-env=vulkan1.0",
                    "-Werror",
                ] + [
                    arg
                    for src_dir_path in self.src_dir_paths
                    for arg in ["-I", src_dir_path]
                ]

                print("glslc cmd:", cmd)
                subprocess.check_call(cmd)

                output_file_map[spv_out_path] = glsl_out_path

        return output_file_map


##############################################
#  Shader Info and Shader Registry Handling  #
##############################################


@dataclass
class ShaderInfo:
    tile_size: List[int]
    layouts: List[str]
    weight_storage_type: str = ""
    bias_storage_type: str = ""
    register_for: Optional[Tuple[str, List[str]]] = None


def getName(filePath: str) -> str:
    return os.path.basename(filePath).replace("/", "_").replace(".", "_")


def isDescriptorLine(lineStr: str) -> bool:
    descriptorLineId = r"^layout\(set"
    return re.search(descriptorLineId, lineStr) is not None


def isTileSizeLine(lineStr: str) -> bool:
    tile_size_id = r"^ \* TILE_SIZE = \("
    return re.search(tile_size_id, lineStr) is not None


def findTileSizes(lineStr: str) -> List[int]:
    tile_size_id = r"^ \* TILE_SIZE = \(([0-9]+), ([0-9]+), ([0-9]+)\)"
    matches = re.search(tile_size_id, lineStr)
    if matches is None:
        raise AssertionError("matches is None in findTileSizes")
    return [int(matches.group(1)), int(matches.group(2)), int(matches.group(3))]


def isWeightStorageTypeLine(lineStr: str) -> bool:
    weight_storage_id = r"^ \* WEIGHT_STORAGE = "
    return re.search(weight_storage_id, lineStr) is not None


def getWeightStorageType(lineStr: str) -> str:
    weight_storage_id = r"^ \* WEIGHT_STORAGE = ([a-zA-Z]+_\dD)"
    matches = re.search(weight_storage_id, lineStr)
    if matches is None:
        raise AssertionError("matches is None in getWeightStorageType")
    return matches.group(1)


def isBiasStorageTypeLine(lineStr: str) -> bool:
    weight_storage_id = r"^ \* BIAS_STORAGE = "
    return re.search(weight_storage_id, lineStr) is not None


def getBiasStorageType(lineStr: str) -> str:
    weight_storage_id = r"^ \* BIAS_STORAGE = ([a-zA-Z]+_\dD)"
    matches = re.search(weight_storage_id, lineStr)
    if matches is None:
        raise AssertionError("matches is None in getBiasStorageType")
    return matches.group(1)


def isRegisterForLine(lineStr: str) -> bool:
    # Check for Shader Name and a list of at least one Registry Key
    register_for_id = (
        r"^ \* REGISTER_FOR = \('([A-Za-z0-9_]+)'\s*,\s*\['([A-Za-z0-9_]+)'.*\]\)"
    )
    return re.search(register_for_id, lineStr) is not None


def findRegisterFor(lineStr: str) -> Tuple[str, List[str]]:
    register_for_pattern = r"'([A-Za-z0-9_]+)'"
    matches = re.findall(register_for_pattern, lineStr)
    if matches is None:
        raise AssertionError("matches is None in getBiasStorageType")
    matches_list = list(matches)
    return (matches_list[0], matches_list[1:])


typeIdMapping = {
    r"image[123]D\b": "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE",
    r"sampler[123]D\b": "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER",
    r"\bbuffer\b": "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
    r"\buniform\b": "VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER",
}

storageTypeToEnum = {
    "TEXTURE_2D": "api::StorageType::TEXTURE_2D",
    "TEXTURE_3D": "api::StorageType::TEXTURE_3D",
    "BUFFER": "api::StorageType::BUFFER",
    "": "api::StorageType::UNKNOWN",
}


def determineDescriptorType(lineStr: str) -> str:
    for identifier, typeNum in typeIdMapping.items():
        if re.search(identifier, lineStr):
            return typeNum
    raise AssertionError(
        "No matching descriptor type for " + lineStr + " in determineDescriptorType"
    )


def getShaderInfo(srcFilePath: str) -> ShaderInfo:
    shader_info = ShaderInfo([], [], "")
    with open(srcFilePath) as srcFile:
        for line in srcFile:
            if isDescriptorLine(line):
                shader_info.layouts.append(determineDescriptorType(line))
            if isTileSizeLine(line):
                shader_info.tile_size = findTileSizes(line)
            if isWeightStorageTypeLine(line):
                shader_info.weight_storage_type = getWeightStorageType(line)
            if isBiasStorageTypeLine(line):
                shader_info.bias_storage_type = getBiasStorageType(line)
            if isRegisterForLine(line):
                shader_info.register_for = findRegisterFor(line)

    return shader_info


##########################
#  C++ File Generation  #
#########################

cpp_template = """
#include <ATen/native/vulkan/api/ShaderRegistry.h>
#include <stdint.h>
#include <vector>

using namespace at::native::vulkan;

namespace at {{
namespace native {{
namespace vulkan {{

namespace {{

{spv_bin_arrays}

}}

static void register_fn() {{

{register_shader_infos}

{shader_info_registry}

}}

static const api::ShaderRegisterInit register_shaders(&register_fn);

}}
}}
}}

"""


def generateSpvBinStr(spvPath: str, name: str) -> Tuple[int, str]:
    with open(spvPath, "rb") as fr:
        next_bin = array.array("I", fr.read())
        sizeBytes = 4 * len(next_bin)
        spv_bin_str = "const uint32_t {}_bin[] = {{\n{}\n}};".format(
            name,
            textwrap.indent(",\n".join(str(x) for x in next_bin), "  "),
        )

    return sizeBytes, spv_bin_str


def generateShaderInfoStr(shader_info: ShaderInfo, name: str, sizeBytes: int) -> str:
    tile_size = (
        f"{{{', '.join(str(x) for x in shader_info.tile_size)}}}"
        if (len(shader_info.tile_size) > 0)
        else "std::vector<uint32_t>()"
    )

    shader_info_layouts = "{{{}}}".format(",\n ".join(shader_info.layouts))

    shader_info_args = [
        f'"{name}"',
        f"{name}_bin",
        str(sizeBytes),
        shader_info_layouts,
        tile_size,
        storageTypeToEnum[shader_info.weight_storage_type],
        storageTypeToEnum[shader_info.bias_storage_type],
    ]

    shader_info_str = textwrap.indent(
        "api::shader_registry().register_shader(\n  api::ShaderInfo(\n{args}));\n".format(
            args=textwrap.indent(",\n".join(shader_info_args), "     "),
        ),
        "    ",
    )

    return shader_info_str


def generateShaderDispatchStr(shader_info: ShaderInfo, name: str) -> str:
    if shader_info.register_for is None:
        return ""

    (op_name, registry_keys) = shader_info.register_for
    for registry_key in registry_keys:
        shader_dispatch_str = textwrap.indent(
            f'api::shader_registry().register_op_dispatch("{op_name}", api::DispatchKey::{registry_key.upper()}, "{name}");',
            "    ",
        )

    return shader_dispatch_str


def genCppFiles(
    spv_files: Dict[str, str], cpp_header_path: str, cpp_src_file_path: str
) -> None:
    spv_bin_strs = []
    register_shader_info_strs = []
    shader_registry_strs = []

    for spvPath, srcPath in spv_files.items():
        name = getName(spvPath).replace("_spv", "")

        sizeBytes, spv_bin_str = generateSpvBinStr(spvPath, name)
        spv_bin_strs.append(spv_bin_str)

        shader_info = getShaderInfo(srcPath)

        register_shader_info_strs.append(
            generateShaderInfoStr(shader_info, name, sizeBytes)
        )

        if shader_info.register_for is not None:
            shader_registry_strs.append(generateShaderDispatchStr(shader_info, name))

    spv_bin_arrays = "\n".join(spv_bin_strs)
    register_shader_infos = "\n".join(register_shader_info_strs)
    shader_info_registry = "\n".join(shader_registry_strs)

    cpp = cpp_template.format(
        spv_bin_arrays=spv_bin_arrays,
        register_shader_infos=register_shader_infos,
        shader_info_registry=shader_info_registry,
    )

    with open(cpp_src_file_path, "w") as fw:
        fw.write(cpp)


##########
#  Main  #
##########


def parse_arg_env(items: Dict[Any, Any]) -> Dict[Any, Any]:
    d = {}
    if items:
        for item in items:
            tokens = item.split("=")
            key = tokens[0].strip()
            value = tokens[1].strip()
            d[key] = value
    return d


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-i",
        "--glsl-paths",
        nargs="+",
        help='List of paths to look for GLSL source files, separated by spaces. Ex: --glsl-paths "path1 path2 path3"',
        default=["."],
    )
    parser.add_argument("-c", "--glslc-path", required=True, help="")
    parser.add_argument("-t", "--tmp-dir-path", required=True, help="/tmp")
    parser.add_argument("-o", "--output-path", required=True, help="")
    parser.add_argument(
        "--env", metavar="KEY=VALUE", nargs="*", help="Set a number of key-value pairs"
    )
    options = parser.parse_args()

    env = DEFAULT_ENV
    for key, value in parse_arg_env(options.env).items():
        env[key] = value

    if not os.path.exists(options.output_path):
        os.makedirs(options.output_path)

    if not os.path.exists(options.tmp_dir_path):
        os.makedirs(options.tmp_dir_path)

    shader_generator = SPVGenerator(options.glsl_paths, env, options.glslc_path)
    output_spv_files = shader_generator.generateSPV(options.tmp_dir_path)

    genCppFiles(
        output_spv_files,
        f"{options.output_path}/{CPP_H_NAME}",
        f"{options.output_path}/{CPP_SRC_NAME}",
    )

    return 0


def invoke_main() -> None:
    sys.exit(main(sys.argv))


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
