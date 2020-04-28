
import argparse
import sys
import os
import re
import subprocess

H_NAME = "spv.h"
CPP_NAME = "spv.cpp"

def getName(filePath):
    dirPath, fileName = filePath.rsplit('/', 1)
    return fileName.replace("/", "_").replace(".", "_")

def genCppH(hFilePath, cppFilePath, srcDirPath, glslcPath, tmpDirPath):
    print("hFilePath:{} cppFilePath:{} srcDirPath:{} glslcPath:{} tmpDirPath:{}".format(
        hFilePath, cppFilePath, srcDirPath, glslcPath, tmpDirPath))

    cmd = "find " + srcDirPath + " -name \"vulkan*.glsl\""
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

        spvPath = tmpDirPath + "/" + name + ".spv";
        print("spvPath {}".format(spvPath))

        cmd = "{} -fshader-stage=compute {} -o {} --target-env=vulkan1.0".format(
            glslcPath, srcPath, spvPath)
        print("\nglslc cmd: {}".format(cmd))

        glslcProcess = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        glslcProcess.wait()
        print("returncode:{}\n".format(glslcProcess.returncode))
        if glslcProcess.returncode != 0:
          raise Exception("Error compiling glsl " + srcPath)
        spvPaths.append(spvPath)

    print("hFilePath:{}".format(hFilePath))
    print("cppFilePath:{}".format(cppFilePath))
    h = "#pragma once\n"
    nsbegin = "\nnamespace at { namespace native { namespace vulkan { \n"
    nsend = "\n} } } //namespace at::native::vulkan\n"

    h += nsbegin

    cpp = "#include <ATen/native/vulkan/{}>".format(H_NAME)
    cpp += nsbegin

    for spvPath in spvPaths:
        name = getName(spvPath)
        name_len = name + "_len"
        h += "extern const unsigned char {}[];\n".format(name)
        h += "extern unsigned int {};\n".format(name_len)

        cpp += "const unsigned char " + name + "[] = {\n"
        sizeBytes = 0
        with open(spvPath, 'rb') as f:
            line = ""
            n = 0
            while True:
                byte = f.read(1)
                if not byte:
                    break
                int_value = ord(byte)
                s = "0x{0:02X},".format(int_value)
                if n==16:
                    cpp += line + "\n"
                    line = ""
                    n = 0
                line += s
                n += 1
                sizeBytes += 1

            if n > 0:
                cpp += line + "\n"
                line = ""
                n = 0
            cpp += "};\n"
        cpp += "unsigned int {} = {};\n".format(name_len, sizeBytes)

    cpp += nsend
    h += nsend

    with open(hFilePath, "w") as f:
      f.write(h)
    with open(cppFilePath, "w") as f:
      f.write(cpp)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='')
  parser.add_argument(
      '-i',
      '--glsl-path',
      help='',
      default='.')
  parser.add_argument(
      '-c',
      '--glslc-path',
      help='')
  parser.add_argument(
      '-t',
      '--tmp-spv-path',
      help='/tmp')
  parser.add_argument(
      '-o',
      '--output-path',
      help='')
  options = parser.parse_args()

  GLSL_DIR_PATH = options.glsl_path
  GLSLC_PATH = options.glslc_path
  TMP_DIR_PATH = options.tmp_spv_path
  OUTPUT_DIR_PATH = options.output_path
  if GLSL_DIR_PATH is None: 
      raise Exception("")

  if GLSLC_PATH is None: 
      raise Exception("")

  if OUTPUT_DIR_PATH is None: 
      raise Exception("")

  if not os.path.exists(OUTPUT_DIR_PATH):
    os.makedirs(OUTPUT_DIR_PATH)

  if not os.path.exists(TMP_DIR_PATH):
    os.makedirs(TMP_DIR_PATH)

  genCppH(
      hFilePath=OUTPUT_DIR_PATH + "/spv.h", 
      cppFilePath=OUTPUT_DIR_PATH + "/spv.cpp", 
      srcDirPath=GLSL_DIR_PATH,
      glslcPath=GLSLC_PATH,
      tmpDirPath=TMP_DIR_PATH)
