#!/usr/bin/env python3

import argparse
import array
import glob
import os
import re
import sys
import subprocess
from torchgen.code_template import CodeTemplate

H_NAME = "spv.h"
CPP_NAME = "spv.cpp"
DEFAULT_ENV = {"precision": "highp", "format": "rgba32f"}

def getName(filePath):
    return os.path.basename(filePath).replace("/", "_").replace(".", "_")

def isDescriptorLine(lineStr):
    descriptorLineId = r"^layout\(set"
    return re.search(descriptorLineId, lineStr)

typeIdMapping = {
    r"image[123]D\b": "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE",
    r"sampler[123]D\b": "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER",
    r"\bbuffer\b": "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
    r"\buniform\b.*\bBlock\b": "VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER",
}

def determineDescriptorType(lineStr):
    for identifier, typeNum in typeIdMapping.items():
        if re.search(identifier, lineStr):
            return typeNum

    raise Exception("Could not identify descriptor type of line: {}".format(lineStr))

def getLayout(srcFilePath):
    layout = []
    with open(srcFilePath, 'r') as srcFile:
        for line in srcFile:
            if isDescriptorLine(line):
                layout.append(determineDescriptorType(line))

    return layout

def genCppH(hFilePath, cppFilePath, srcDirPath, glslcPath, tmpDirPath, env):
    print("hFilePath:{} cppFilePath:{} srcDirPath:{} glslcPath:{} tmpDirPath:{}".format(
        hFilePath, cppFilePath, srcDirPath, glslcPath, tmpDirPath))

    vexs = glob.glob(os.path.join(srcDirPath, '**', '*.glsl'), recursive=True)
    templateSrcPaths = []
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
        with open(srcPath, 'w') as f:
            f.write(content)

        spvPath = tmpDirPath + "/" + name + ".spv"
        print("spvPath {}".format(spvPath))

        cmd = [
            glslcPath, "-fshader-stage=compute",
            srcPath, "-o", spvPath,
            "--target-env=vulkan1.0",
            "-Werror"
        ]

        print("\nglslc cmd:", cmd)

        subprocess.check_call(cmd)
        spvPaths[spvPath] = templateSrcPath

    h = "#pragma once\n"
    h += "#include <stdint.h>\n"
    h += "#include <vector>\n"
    h += "#include <ATen/native/vulkan/api/vk_api.h>"

    nsbegin = "\nnamespace at {\nnamespace native {\nnamespace vulkan {\n"
    nsend = "\n}\n}\n} //namespace at::native::vulkan\n"

    h += nsbegin

    cpp = "#include <ATen/native/vulkan/{}>".format(H_NAME)
    cpp += nsbegin

    for spvPath, srcPath in spvPaths.items():
        name = getName(spvPath)
        name_len = name + "_len"
        h += "extern const uint32_t {}[];\n".format(name)
        h += "extern const uint32_t {};\n".format(name_len)

        layout = getLayout(srcPath)
        name_layout = name + "_layout"
        h += "extern const std::vector<VkDescriptorType> {};\n".format(name_layout)

        cpp += "const uint32_t " + name + "[] = {\n"
        sizeBytes = 0
        print("spvPath:{}".format(spvPath))
        with open(spvPath, 'rb') as f:
            for word in array.array('I', f.read()):
                cpp += "{},\n".format(word)
                sizeBytes += 4
            cpp += "};\n"
        cpp += "const uint32_t {} = {};\n".format(name_len, sizeBytes)

        # Add layout
        cpp += "const std::vector<VkDescriptorType> {} = {{\n".format(name_layout)
        for descriptor in layout:
            cpp += "  {},\n".format(descriptor)
        cpp += "};\n"

    cpp += nsend
    h += nsend

    with open(hFilePath, "w") as f:
        f.write(h)
    with open(cppFilePath, "w") as f:
        f.write(cpp)


def parse_arg_env(items):
    d = {}
    if items:
        for item in items:
            tokens = item.split("=")
            key = tokens[0].strip()
            value = tokens[1].strip()
            d[key] = value
    return d


def main(argv):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '-i',
        '--glsl-path',
        help='',
        default='.')
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
        srcDirPath=options.glsl_path,
        glslcPath=options.glslc_path,
        tmpDirPath=options.tmp_dir_path,
        env=env)

if __name__ == '__main__':
    sys.exit(main(sys.argv))
