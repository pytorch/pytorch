from setuptools import setup, Extension

C = Extension("torch.C",
              libraries=['TH'],
              sources=["torch/csrc/Module.c", "torch/csrc/Tensor.c", "torch/csrc/Storage.c"],
              include_dirs=["torch/csrc"])

in_init = "torch/__init__.py.in"
out_init = "torch/__init__.py"
templates = ["torch/Tensor.py"]
types = ['Double', 'Float', 'Long', 'Int', 'Char', 'Byte']
to_append = ''
for template in templates:
    with open(template, 'r') as f:
        template_content = f.read()
    for T in types:
        t = T.lower()
        replacements = [
            ('RealTensorBase', T + 'TensorBase'),
            ('RealTensor', T + 'Tensor'),
        ]
        new_content = template_content
        for pattern, replacement in replacements:
            new_content = new_content.replace(pattern, replacement)
        to_append += '\n'
        to_append += new_content
        to_append += '\n'
with open(out_init, 'w') as f:
    with open(in_init, 'r') as i:
        header = i.read()
    f.write(header)
    f.write(to_append)

setup(name="torch", version="0.1",
      ext_modules=[C],
      packages=['torch'])
