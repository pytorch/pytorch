def python_num(s):
    try:
        return int(s)
    except Exception:
        return float(s)


def parse(filename):
    with open(filename, 'r') as file:
        declarations = []
        in_declaration = False
        for line in file.readlines():
            if '[NativeFunction]' in line:
                in_declaration = True
                arguments = []
                declaration = {'mode': 'native'}
            elif '[/NativeFunction]' in line:
                in_declaration = False
                declaration['arguments'] = arguments
                declarations.append(declaration)
                if declaration.get('type_method_definition_level') != 'base':
                    raise RuntimeError("Native functions currently only support (and must be specified with) "
                                       "\'base\' type_method_definition_level")
            elif in_declaration:
                ls = line.strip().split(':', 1)
                key = ls[0].strip()
                value = ls[1].strip()
                if key == 'arg':
                    t, name = value.split(" ", 1)
                    if '=' in name:
                        ns = name.split("=", 1)
                        name, default = ns[0], python_num(ns[1])
                        arguments.append({'type': t, 'name': name, 'default': default})
                    else:
                        arguments.append({'type': t, 'name': name})
                else:
                    declaration[key] = value
        return declarations
