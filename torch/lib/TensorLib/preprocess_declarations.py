import common_with_cwrap

type_map = {
    'floating_point' : [
        'Float',
        'Double',
        'Half',
    ],
    'integral' : [
        'Byte',
        'Char',
        'Short',
        'Int',
        'Long'
    ],
}

all_types = type_map['floating_point'] + type_map['integral']
type_map['all'] = all_types

all_processors = ['CPU','CUDA']

def process_types_and_processors(option):
    # if specific pairs were not listed, then enumerate them
    # based on the processor and type attributes
    # if processor or type is not defined, it is assumed to be all of them
    if 'type_processor_pairs' not in option:
        processors = option.get('processors', all_processors)
        types = option.get('types', all_types)
        pairs = [ [p,t] for p in processors for t in types ]
    else:
        pairs = option['type_processor_pairs']

    # expand type alias (integral, floating_point, all)
    def expand(pair):
        p,t = pair
        assert(p in all_processors)
        if t in type_map:
            return [ (p,tt) for tt in type_map[t] ]
        assert(t in all_types)
        return [ (p,t) ]
    pairs = set(p for pair in pairs for p in expand(pair))

    # special case remove Half for cpu unless it is explicitly enabled
    if not option.get('cpu_half',False):
        pairs.discard(('CPU','Half'))

    # sort the result for easy reading
    option['type_processor_pairs'] = sorted([p for p in pairs])

def exclude(declaration):
    return 'only_register' in declaration

def add_variants(option):
    only_stateless = option.get('only_stateless', False)
    with_stateless = option.get('with_stateless', False)
    # should we generate tensor.foo(...)
    option['as_method'] = not only_stateless
    # should we generate tlib::foo(tensor,...)
    option['as_function'] = with_stateless
    # regardless we always generate type().foo(...) because the above will
    # call it

def run(declarations):
    declarations = [d for d in declarations if not exclude(d)]
    for declaration in declarations:
        common_with_cwrap.set_declaration_defaults(declaration)
        common_with_cwrap.enumerate_options_due_to_default(declaration)
        common_with_cwrap.sort_by_number_of_options(declaration)
        for option in declaration['options']:
            process_types_and_processors(option)
            add_variants(option)

    return declarations
