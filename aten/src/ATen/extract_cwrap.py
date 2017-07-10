from optparse import OptionParser

parser = OptionParser()
parser.add_option('-o', '--output', help='where to write the result file.',
                  action='store', default='.')
options, _ = parser.parse_args()

files = [
    # '../../csrc/cudnn/cuDNN.cwrap',
    '../../csrc/generic/TensorMethods.cwrap',
    # '../../csrc/generic/methods/SparseTensor.cwrap',
    '../../csrc/generic/methods/Tensor.cwrap',
    '../../csrc/generic/methods/TensorApply.cwrap',
    '../../csrc/generic/methods/TensorCompare.cwrap',
    '../../csrc/generic/methods/TensorCuda.cwrap',
    '../../csrc/generic/methods/TensorMath.cwrap',
    '../../csrc/generic/methods/TensorRandom.cwrap',
    # '../../csrc/generic/methods/TensorSerialization.cwrap',
]

declaration_lines = []

for filename in files:
    with open(filename, 'r') as file:
        in_declaration = False
        for line in file.readlines():
            line = line.rstrip()
            if line == '[[':
                in_declaration = True
                declaration_lines.append(line)
            elif line == ']]':
                in_declaration = False
                declaration_lines.append(line)
            elif in_declaration:
                declaration_lines.append(line)

with open(options.output, 'w') as output:
    output.write('\n'.join(declaration_lines) + '\n')
