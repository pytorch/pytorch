# finds raw declarations between [Declarations.yaml] and [/Declarations.yaml]

def parse(filename):
    with open(filename, 'r') as file:
        declaration_lines = []
        declarations = []
        in_declaration = False
        lstrip_chars = 0
        for line in file.readlines():
            line = line.rstrip()
            if '[Declarations.yaml]' in line:
                declaration_lines = []
                in_declaration = True
                lstrip_chars = len(line) - len(line.lstrip())
            elif '[/Declarations.yaml]' in line:
                in_declaration = False
                declarations.append('\n'.join(declaration_lines))
            elif in_declaration:
                if line[:lstrip_chars] == ' ' * lstrip_chars:
                    line = line[lstrip_chars:]
                declaration_lines.append(line)
        return declarations
