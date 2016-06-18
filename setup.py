from setuptools import setup, Extension
from os.path import expanduser
from tools.cwrap import cwrap
import platform

################################################################################
# Generate __init__.py from templates
################################################################################
in_init = "torch/__init__.py.in"
out_init = "torch/__init__.py"
templates = ["torch/Tensor.py", "torch/Storage.py"]
types = ['Double', 'Float', 'Long', 'Int', 'Short', 'Char', 'Byte']
generated = ''

for template in templates:
    with open(template, 'r') as f:
        template_content = f.read()
    for T in types:
        replacements = [
            ('RealTensorBase', T + 'TensorBase'),
            ('RealTensor', T + 'Tensor'),
            ('RealStorageBase', T + 'StorageBase'),
            ('RealStorage', T + 'Storage'),
        ]
        new_content = template_content
        for pattern, replacement in replacements:
            new_content = new_content.replace(pattern, replacement)
        generated += '\n{}\n'.format(new_content)

with open(in_init, 'r') as i:
    header, _, suffix = i.read().partition('# GENERATED CODE GOES HERE')

with open(out_init, 'w') as f:
    f.write("""
################################################################################
# WARNING
# This file is generated automatically. Do not edit it, as it will be
# regenerated during next build
################################################################################

""")
    f.write(header)
    f.write(generated)
    f.write(suffix)

cwrap_src = ['torch/csrc/generic/TensorMethods.cwrap.cpp']
for src in cwrap_src:
    print("Generating code for " + src)
    cwrap(src)

################################################################################
# Declare the package
################################################################################
extra_link_args = []

# TODO: remove and properly submodule TH in the repo itself
th_path = expanduser("~/torch/install/")
th_header_path = th_path + "include"
th_lib_path = th_path + "lib"
if platform.system() == 'Darwin':
    extra_link_args.append('-L' + th_lib_path)
    extra_link_args.append('-Wl,-rpath,' + th_lib_path)

sources = [
    "torch/csrc/Module.cpp",
    "torch/csrc/Generator.cpp",
    "torch/csrc/Tensor.cpp",
    "torch/csrc/Storage.cpp",
    "torch/csrc/utils.cpp",
]
C = Extension("torch.C",
              libraries=['TH'],
              sources=sources,
              language='c++',
              include_dirs=(["torch/csrc", th_header_path]),
              extra_link_args = extra_link_args,
)



setup(name="torch", version="0.1",
      ext_modules=[C],
      packages=['torch'],
)
