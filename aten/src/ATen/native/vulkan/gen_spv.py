#!/usr/bin/env python3

import argparse
import array
import os
import sys
import subprocess

H_NAME = "spv.h"
CPP_NAME = "spv.cpp"

def getName(filePath):
    return os.path.basename(filePath).replace("/", "_").replace(".", "_")

def genCppH(hFilePath, cppFilePath, srcDirPath, glslcPath, tmpDirPath):
    print("hFilePath:{} cppFilePath:{} srcDirPath:{} glslcPath:{} tmpDirPath:{}".format(
        hFilePath, cppFilePath, srcDirPath, glslcPath, tmpDirPath))

    cmd = "find " + srcDirPath + " -name \"*.glsl\""
    vexs = os.popen(cmd).read().split('\n')
    srcPaths = []
    for f in vexs:
        if len(f) > 1:
            srcPaths.append(f)
            srcPaths.sort()
    print("srcPaths:{}".format(srcPaths))

    spvPaths = []
    for srcPath in srcPaths:
        print("srcPath {}".format(srcPath))
        name = getName(srcPath).replace("_glsl", "")
        print("name {}".format(name))

        spvPath = tmpDirPath + "/" + name + ".spv"
        print("spvPath {}".format(spvPath))

        cmd = [glslcPath, "-fshader-stage=compute", srcPath, "-o", spvPath, "--target-env=vulkan1.0"]
        print("\nglslc cmd:", cmd)

        subprocess.check_call(cmd)
        spvPaths.append(spvPath)

    h = "#pragma once\n"
    h += "#include <stdint.h>\n"
    nsbegin = "\nnamespace at { namespace native { namespace vulkan { \n"
    nsend = "\n} } } //namespace at::native::vulkan\n"

    h += nsbegin

    cpp = "#include <ATen/native/vulkan/{}>".format(H_NAME)
    cpp += nsbegin

    for spvPath in spvPaths:
        name = getName(spvPath)
        name_len = name + "_len"
        h += "extern const uint32_t {}[];\n".format(name)
        h += "extern const uint32_t {};\n".format(name_len)

        cpp += "const uint32_t " + name + "[] = {\n"
        sizeBytes = 0
        print("spvPath:{}".format(spvPath))
        with open(spvPath, 'rb') as f:
            for word in array.array('I', f.read()):
                cpp += "{},\n".format(word)
                sizeBytes += 4
            cpp += "};\n"
        cpp += "const uint32_t {} = {};\n".format(name_len, sizeBytes)

    cpp += nsend
    h += nsend

    with open(hFilePath, "w") as f:
        f.write(h)
    with open(cppFilePath, "w") as f:
        f.write(cpp)

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
        '--tmp-spv-path',
        required=True,
        help='/tmp')
    parser.add_argument(
        '-o',
        '--output-path',
        required=True,
        help='')
    options = parser.parse_args()

    if not os.path.exists(options.output_path):
        os.makedirs(options.output_path)

    if not os.path.exists(options.tmp_spv_path):
        os.makedirs(options.tmp_spv_path)

    genCppH(
        hFilePath=options.output_path + "/spv.h",
        cppFilePath=options.output_path + "/spv.cpp",
        srcDirPath=options.glsl_path,
        glslcPath=options.glslc_path,
        tmpDirPath=options.tmp_spv_path)

if __name__ == '__main__':
    sys.exit(main(sys.argv))
