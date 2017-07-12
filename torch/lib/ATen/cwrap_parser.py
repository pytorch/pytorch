import yaml

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
                declaration = yaml.load('\n'.join(declaration_lines))
                declarations.append(declaration)
            elif in_declaration:
                declaration_lines.append(line)
        return declarations
