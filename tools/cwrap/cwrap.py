import math
from string import Template
from itertools import product

from .functions import make_function

def cwrap(filename):
    """Parses and generates code for a .cwrap file

       Assumes that filename ends with .cwrap.cpp and saves the result to
       .cpp file with the same prefix.
    """
    assert filename.endswith('.cwrap.cpp')
    with open(filename, 'r') as f:
        content = f.read()
    lines = content.split('\n')
    new_content = ''
    in_declaration = False
    for line in lines:
        if line == '[[':
            in_declaration = True
            func_lines = []
        elif line == ']]':
            in_declaration = False
            func_lines = remove_indentation(func_lines)
            new_content += make_function(func_lines, stateless=True).generate()
            new_content += make_function(func_lines, stateless=False).generate()
        elif in_declaration:
            func_lines.append(line)
        else:
            new_content += line + '\n'
    with open(filename.replace('.cwrap', ''), 'w') as f:
        f.write(new_content)

def remove_indentation(lines):
    """Removes 2 spaces from the left from each line.
       If anyone wants to use another indentation depth, please update
       this function first.
    """
    return [line[2:] for line in lines]
