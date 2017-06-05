import common_with_cwrap

def exclude(declaration):
    return 'only_register' in declaration


def run(declarations):
    for declaration in declarations:
        common_with_cwrap.set_declaration_defaults(declaration)
        common_with_cwrap.enumerate_options_due_to_default(declaration)
        common_with_cwrap.sort_by_number_of_options(declaration)
        
    declarations = [d for d in declarations if not exclude(d)]
    return declarations
