import argparse
import sys
import os
import re

H_NAME = "glsl.h"
CPP_NAME = "glsl.cpp"

def findAllGlsls(path):
  cmd = "find " + path + " -name \"*.glsl\""
  vexs = os.popen(cmd).read().split('\n')
  output = []
  for f in vexs:
    if len(f) > 1:
      output.append(f)
  output.sort()
  return output

def getName(filePath):
    dirPath, fileName = filePath.rsplit('/', 1)
    return fileName.replace("/", "_").replace(".", "_")

def genCppH(hFilePath, cppFilePath, glsls):
  print("hFilePath:{}".format(hFilePath))
  print("cppFilePath:{}".format(cppFilePath))
  h = "#pragma once\n"
  nsbegin = "\nnamespace at { namespace native { namespace vulkan { \n"
  nsend = "\n} } } //namespace at::native::vulkan\n"

  h += nsbegin

  cpp = "#include <ATen/native/vulkan/{}>".format(H_NAME)
  cpp += nsbegin

  for s in glsls:
    name = getName(s)
    h += "extern const char* " + name + ";\n"
    cpp += "const char* " + name + " = \n"
    with open(s) as f:
      lines = f.read().split("\n")
      for l in lines:
        if (len(l) < 1):
          continue
        cpp += "\"" + l + "\\n\"\n"
    cpp += ";\n"

  cpp += nsend

  h += nsend

  with open(hFilePath, "w") as f:
    f.write(h)
  with open(cppFilePath, "w") as f:
    f.write(cpp)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Generate glsl.cpp and glsl.h containing glsl sources')
  parser.add_argument(
      '-i',
      '--glsl-path',
      help='path to directory with glsl to process',
      default='.')
  parser.add_argument(
      '-o',
      '--output-path',
      help='path to directory to generate glsl.h glsl.cpp (cpp namespace at::native::vulkan)')
  options = parser.parse_args()

  GLSL_DIR_PATH = options.glsl_path
  OUTPUT_DIR_PATH = options.output_path
  if GLSL_DIR_PATH is None: 
      raise Exception("--glsl-path was not specifie")

  if OUTPUT_DIR_PATH is None: 
      raise Exception("--output-path was not specifie")

  glsls = findAllGlsls(GLSL_DIR_PATH)
  genCppH(OUTPUT_DIR_PATH + "/" + H_NAME, OUTPUT_DIR_PATH + "/" + CPP_NAME, glsls)
