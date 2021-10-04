#!/usr/bin/env python3
## @package process
# Module doxygen.process
# Script to insert preamble for doxygen and regen API docs

import os
import shutil

# Module caffe2...caffe2.python.control_test
def insert(originalfile, first_line, description):
    with open(originalfile, 'r') as f:
        f1 = f.readline()
        if(f1.find(first_line) < 0):
            docs = first_line + description + f1
            with open('newfile.txt', 'w') as f2:
                f2.write(docs)
                f2.write(f.read())
            os.rename('newfile.txt', originalfile)
        else:
            print('already inserted')

# move up from /caffe2_root/doxygen
os.chdir("..")
os.system("git checkout caffe2/contrib/.")
os.system("git checkout caffe2/distributed/.")
os.system("git checkout caffe2/experiments/.")
os.system("git checkout caffe2/python/.")

for root, dirs, files in os.walk("."):
    for file in files:
        if (file.endswith(".py") and not file.endswith("_test.py") and not file.endswith("__.py")):
            filepath = os.path.join(root, file)
            print(("filepath: " + filepath))
            directory = os.path.dirname(filepath)[2:]
            directory = directory.replace("/", ".")
            print("directory: " + directory)
            name = os.path.splitext(file)[0]
            first_line = "## @package " + name
            description = "\n# Module " + directory + "." + name + "\n"
            print(first_line, description)
            insert(filepath, first_line, description)

if os.path.exists("doxygen/doxygen-python"):
    print("Looks like you ran this before, so we need to cleanup those old files...")
    shutil.rmtree("doxygen/doxygen-python")
else:
    os.makedirs("doxygen/doxygen-python")

if os.path.exists("doxygen/doxygen-c"):
    print("Looks like you ran this before, so we need to cleanup those old files...")
    shutil.rmtree("doxygen/doxygen-c")
else:
    os.makedirs("doxygen/doxygen-c")

os.system("doxygen .Doxyfile-python")
os.system("doxygen .Doxyfile-c")
