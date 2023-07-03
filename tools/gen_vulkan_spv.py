#!/usr/bin/env python3

import argparse
import array
import copy
import glob
import os
import re
import sys
import subprocess
import textwrap
import yaml
from collections import OrderedDict
from torchgen.code_template import CodeTemplate
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader  # type: ignore[misc]

H_NAME = "spv.h"
CPP_NAME = "spv.cpp"
DEFAULT_ENV = {"precision": "highp", "format": "rgba32f"}

# https://gist.github.com/pypt/94d747fe5180851196eb
class UniqueKeyLoader(Loader):
    def construct_mapping(self, node, deep=False):  # type: ignore[no-untyped-def]
        if not isinstance(node, MappingNode):
            raise ConstructorError(
                None,
                None,
                "expected a mapping node, but found %s" % node.id,
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


class VulkanShaderGenerator:
    standard_header = """
#version 450 core
#define PRECISION $precision
#define FORMAT $format

"""

    def __init__(self: "VulkanShaderGenerator") -> None:
        self.ops_template_params: Dict[Any, Any] = {}

    def add_params_yaml(self, parameters_yaml_file):  # type: ignore[no-untyped-def]
        all_template_params = OrderedDict()
        with open(parameters_yaml_file, "r") as f:
            contents = yaml.load(f, Loader=UniqueKeyLoader)
            for key in contents:
                all_template_params[key] = contents[key]
        self.validate_and_construct_op_params(all_template_params)  # type: ignore[no-untyped-call]

    def validate_and_construct_op_params(self, all_template_params):  # type: ignore[no-untyped-def]
        for op in all_template_params:
            if op in self.ops_template_params:
                raise KeyError(f"{op} params file has already been parsed")
            op_params_default_vals = all_template_params[op][
                "parameter_names_with_default_values"
            ]
            template_params_set = set(op_params_default_vals.keys())
            self.ops_template_params[op] = []
            self.ops_template_params[op].append(op_params_default_vals)
            op_template_params_values = all_template_params[op]["parameter_values"]
            for param_vals in op_template_params_values:
                param_vals_set = set(param_vals.keys())
                missing_keys = template_params_set - param_vals_set
                invalid_keys = param_vals_set - template_params_set
                if (len(invalid_keys)) > 0:
                    raise KeyError(f"Invalid keys {invalid_keys} are found")
                param_vals_copy = copy.deepcopy(op_params_default_vals)
                for key in param_vals:
                    param_vals_copy[key] = param_vals[key]
                self.ops_template_params[op].append(param_vals_copy)

    def generate(self, glsl_template_in, out_dir):  # type: ignore[no-untyped-def]
        glsl_template_name = os.path.basename(glsl_template_in)
        op_name, extension_name = glsl_template_name.split(".")
        if extension_name != "glslt":
            raise TypeError(f"invalid file type for glsl template {extension_name}")
        if op_name not in self.ops_template_params:
            raise KeyError(f"{op_name} params have not been populated")
        code_template = CodeTemplate.from_file(glsl_template_in)
        for template_params in self.ops_template_params[op_name]:
            content = VulkanShaderGenerator.standard_header
            param_vals_string = "x".join([str(i) for (k, i) in template_params.items() if k != "REGISTER_FOR"])
            output_file_name = op_name + "_" + param_vals_string + ".glsl"
            content += code_template.substitute(template_params)
            output_file = os.path.join(out_dir, output_file_name)
            with open(output_file, "w") as f:
                f.write(content)


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
    register_for_id = r"^ \* REGISTER_FOR = \('([A-Za-z0-9_]+)'\s*,\s*\['([A-Za-z0-9_]+)'.*\]\)"
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
    "TEXTURE_2D" : "api::StorageType::TEXTURE_2D",
    "TEXTURE_3D" : "api::StorageType::TEXTURE_3D",
    "BUFFER" : "api::StorageType::BUFFER",
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
    with open(srcFilePath, 'r') as srcFile:
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

def genGLSLFromGLSLT(src_dir_path: str, tmp_dir_path: str) -> None:
    template_dir_path = os.path.join(src_dir_path, "templates")
    vexs = glob.glob(os.path.join(template_dir_path, '**', '*.yaml'), recursive=True)
    parameter_yaml_files = []
    for f in vexs:
        if len(f) > 1:
            parameter_yaml_files.append(f)
    generator = VulkanShaderGenerator()
    for params_yaml in parameter_yaml_files:
        generator.add_params_yaml(params_yaml)  # type: ignore[no-untyped-call]

    vexs = glob.glob(os.path.join(src_dir_path, '**', '*.glslt'), recursive=True)
    templateSrcPaths = []
    for f in vexs:
        if len(f) > 1:
            templateSrcPaths.append(f)
            templateSrcPaths.sort()
    for glslt in templateSrcPaths:
        generator.generate(glslt, tmp_dir_path)  # type: ignore[no-untyped-call]


def genCppH(
    hFilePath: str,
    cppFilePath: str,
    srcDirPaths: str,
    glslcPath: str,
    tmpDirPath: str,
    env: Dict[Any, Any],
) -> None:
    print(
        "hFilePath:{} cppFilePath:{} srcDirPaths:{} glslcPath:{} tmpDirPath:{}".format(
            hFilePath, cppFilePath, srcDirPaths, glslcPath, tmpDirPath
        )
    )

    templateSrcPaths = []

    for srcDirPath in srcDirPaths:
        vexs = glob.glob(os.path.join(srcDirPath, '**', '*.glsl'), recursive=True)
        for f in vexs:
            if len(f) > 1:
                templateSrcPaths.append(f)
                templateSrcPaths.sort()

        # Now add glsl files that are generated from templates
        genGLSLFromGLSLT(srcDirPath, tmpDirPath)

    vexs = glob.glob(os.path.join(tmpDirPath, '**', '*.glsl'), recursive=True)
    for f in vexs:
        if len(f) > 1:
            templateSrcPaths.append(f)
            templateSrcPaths.sort()
    print("templateSrcPaths:{}".format(templateSrcPaths))

    spvPaths = {}
    for templateSrcPath in templateSrcPaths:
        print("templateSrcPath {}".format(templateSrcPath))
        name = getName(templateSrcPath).replace("_glsl", "")
        print("name {}".format(name))

        codeTemplate = CodeTemplate.from_file(templateSrcPath)
        srcPath = tmpDirPath + "/" + name + ".glsl"
        content = codeTemplate.substitute(env)
        with open(srcPath, 'w') as fw:
            fw.write(content)

        spvPath = tmpDirPath + "/" + name + ".spv"
        print("spvPath {}".format(spvPath))

        cmd = [
            glslcPath, "-fshader-stage=compute",
            srcPath, "-o", spvPath,
            "--target-env=vulkan1.0",
            "-Werror"
        ] + [arg for srcDirPath in srcDirPaths for arg in ["-I", srcDirPath]]

        print("\nglslc cmd:", cmd)

        subprocess.check_call(cmd)
        spvPaths[spvPath] = templateSrcPath

    h = "#pragma once\n"
    h += "#include <ATen/native/vulkan/api/Types.h>\n"
    h += "#include <ATen/native/vulkan/api/vk_api.h>\n"
    h += "#include <c10/util/flat_hash_map.h>\n"
    h += "#include <string>\n"

    nsbegin = "namespace at {\nnamespace native {\nnamespace vulkan {\n"
    nsend = "} // namespace vulkan\n} // namespace native\n} // namespace at\n"

    anon_ns_begin = "namespace {\n"
    anon_ns_end = "} // namespace\n"

    h += nsbegin

    # Forward declaration of ShaderInfo
    h += "namespace api {\nstruct ShaderInfo;\n} // namespace api\n"
    h += "typedef ska::flat_hash_map<std::string, api::ShaderInfo> ShaderListing;\n"
    h += "typedef ska::flat_hash_map<std::string, std::string> RegistryKeyMap;\n"
    h += "typedef ska::flat_hash_map<std::string, RegistryKeyMap> ShaderRegistry;\n"
    h += "extern const ShaderListing shader_infos;\n"
    h += "extern ShaderRegistry shader_registry;\n"
    h += "inline const ShaderListing& get_shader_infos() {\n  return shader_infos;\n}\n"
    h += "inline ShaderRegistry& get_shader_registry() {\n  return shader_registry;\n}\n"

    h += nsend

    cpp = "#include <ATen/native/vulkan/api/Shader.h>\n"
    cpp += "#include <ATen/native/vulkan/{}>\n".format(H_NAME)
    cpp += "#include <stdint.h>\n"
    cpp += "#include <vector>\n"
    cpp += nsbegin

    shader_info_bin_code = []
    shader_info_cpp_code = []
    shader_info_registry_code = []

    for spvPath, srcPath in spvPaths.items():
        name = getName(spvPath).replace("_spv", "")

        print("spvPath:{}".format(spvPath))
        with open(spvPath, 'rb') as fr:
            next_bin = array.array('I', fr.read())
            sizeBytes = 4 * len(next_bin)
            shader_info_bin_code.append(
                "const uint32_t {}_bin[] = {{\n{}\n}};".format(
                    name,
                    textwrap.indent(",\n".join(str(x) for x in next_bin), "  "),
                ),
            )

        shader_info = getShaderInfo(srcPath)

        tile_size = (
            "{{{}}}".format(", ".join(str(x) for x in shader_info.tile_size))
            if (len(shader_info.tile_size) > 0)
            else "std::vector<uint32_t>()"
        )

        shader_info_layouts = "{{{}}}".format(",\n ".join(shader_info.layouts))

        shader_info_args = [
            "\"vulkan.{}\"".format(name),
            "{}_bin".format(name),
            str(sizeBytes),
            shader_info_layouts,
            tile_size,
            storageTypeToEnum[shader_info.weight_storage_type],
            storageTypeToEnum[shader_info.bias_storage_type],
        ]

        shader_info_cpp_code.append(
            textwrap.indent(
                "{{\"{}\",\n api::ShaderInfo(\n{})}}".format(
                    name,
                    textwrap.indent(",\n".join(shader_info_args), "     "),
                ),
                "    ",
            ),
        )

        if shader_info.register_for is not None:
            (op_name, registry_keys) = shader_info.register_for
            for registry_key in registry_keys:
                shader_info_registry_code.append(
                    textwrap.indent(
                        "{{\"{}\", {{{{\"{}\", \"{}\"}}}}}}".format(
                            op_name,
                            registry_key,
                            name,
                        ),
                        "        ",
                    ),
                )

    cpp += anon_ns_begin
    cpp += "\n".join(shader_info_bin_code) + "\n"
    cpp += anon_ns_end

    cpp += "const ShaderListing shader_infos = {{\n{}}};\n".format(
        ",\n".join(shader_info_cpp_code),
    )
    cpp += "ShaderRegistry shader_registry = {{\n{}}};\n".format(
        ",\n".join(shader_info_registry_code),
    )
    cpp += nsend

    with open(hFilePath, "w") as fw:
        fw.write(h)
    with open(cppFilePath, "w") as fw:
        fw.write(cpp)


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
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '-i',
        '--glsl-paths',
        nargs='+',
        help='List of paths to look for GLSL source files, separated by spaces. Ex: --glsl-paths "path1 path2 path3"',
        default=['.'],
    )
    parser.add_argument(
        '-c',
        '--glslc-path',
        required=True,
        help='')
    parser.add_argument(
        '-t',
        '--tmp-dir-path',
        required=True,
        help='/tmp')
    parser.add_argument(
        '-o',
        '--output-path',
        required=True,
        help='')
    parser.add_argument(
        "--env",
        metavar="KEY=VALUE",
        nargs='*',
        help="Set a number of key-value pairs")
    options = parser.parse_args()
    env = DEFAULT_ENV
    for key, value in parse_arg_env(options.env).items():
        env[key] = value

    if not os.path.exists(options.output_path):
        os.makedirs(options.output_path)

    if not os.path.exists(options.tmp_dir_path):
        os.makedirs(options.tmp_dir_path)

    genCppH(
        hFilePath=options.output_path + "/spv.h",
        cppFilePath=options.output_path + "/spv.cpp",
        srcDirPaths=options.glsl_paths,
        glslcPath=options.glslc_path,
        tmpDirPath=options.tmp_dir_path,
        env=env)

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
