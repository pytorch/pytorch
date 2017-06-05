import common_with_cwrap

def exclude(declaration):
    return 'only_register' in declaration


def run(declarations):
    for declaration in declarations:
        common_with_cwrap.set_declaration_defaults(declaration)
        
    declarations = [d for d in declarations if not exclude(d)]
    return declarations
