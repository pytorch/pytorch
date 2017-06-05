import common_with_cwrap

def exclude(declaration):
    return 'only_register' in declaration

def add_variants(declaration):
    only_stateless = declaration.get('only_stateless',False)
    with_stateless = declaration.get('with_stateless',False)
    # should we generate tensor.foo(...)
    declaration['as_method'] = not only_stateless
    # should we generate tlib::foo(tensor,...)
    declaration['as_function'] = with_stateless
    # regardless we always generate type().foo(...) because the above will
    # call it

def run(declarations):
    declarations = [d for d in declarations if not exclude(d)]
    for declaration in declarations:
        common_with_cwrap.set_declaration_defaults(declaration)
        common_with_cwrap.enumerate_options_due_to_default(declaration)
        common_with_cwrap.sort_by_number_of_options(declaration)
        add_variants(declaration)

    return declarations
