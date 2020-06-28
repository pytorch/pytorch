import yaml
import copy

try:
    # use faster C loader if available
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

# follows similar logic to cwrap, ignores !inc, and just looks for [[]]


def parse(filename):
    with open(filename, 'r') as file:
        declaration_lines = []
        declarations = []
        in_declaration = False
        for line in file.readlines():
            line = line.rstrip()
            if line == '[[':
                declaration_lines = []
                in_declaration = True
            elif line == ']]':
                in_declaration = False
                declaration = yaml.load('\n'.join(declaration_lines), Loader=Loader)
                declarations.append(declaration)
            elif in_declaration:
                declaration_lines.append(line)
        declarations = [process_declaration(declaration) for declaration in declarations]
        return declarations

def process_declaration(declaration):
    declaration = copy.deepcopy(declaration)
    if "arguments" in declaration:
        declaration["schema_order_arguments"] = copy.deepcopy(declaration["arguments"])
    if "options" in declaration:
        declaration["options"] = [process_declaration(option) for option in declaration["options"]]
    return declaration
