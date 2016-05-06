from setuptools import setup, Extension

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
    header = i.read()

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

################################################################################
# Declare the package
################################################################################
sources = [
    "torch/csrc/Module.cpp",
    "torch/csrc/Tensor.cpp",
    "torch/csrc/Storage.cpp",
    "torch/csrc/utils.cpp",
]
C = Extension("torch.C",
              libraries=['TH'],
              sources=sources,
              language='c++',
              include_dirs=["torch/csrc"])


setup(name="torch", version="0.1",
      ext_modules=[C],
      packages=['torch'])
