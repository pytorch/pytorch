import common_with_cwrap

cpu_floating_point = set([
    'float',
    'double',
])

cpu_integral = set([
    'byte',
    'char',
    'short',
    'int',
    'long'
])

cpu_types = cpu_floating_point | cpu_integral
cpu_type_map = {
    'floating_point': cpu_floating_point,
    'integral': cpu_integral,
    'all': cpu_types
}

cuda_floating_point = cpu_floating_point | set(['half'])
cuda_integral = cpu_integral
cuda_types = cuda_floating_point | cuda_integral
cuda_type_map = {
    'floating_point': cuda_floating_point,
    'integral': cuda_integral,
    'all': cuda_types
}

all_types = cpu_types | cuda_types

processor_types = set([
    'cpu',
    'cuda',
])

processor_type_map = {
    'cpu': cpu_type_map,
    'cuda': cuda_type_map,
}


def process_types_and_processors(option):
    # First, check if there are no types, processors, specifed. If so we assume
    # that the method/function operates on all types and processors
    if ('types' not in option and 'processors' not in option and
            'type_processor_pairs' not in option):
        return option

    # First, get the full set of types. If there are no  types specified, but a
    # processor is specified, assume  we meant all types for that processor
    processors = option['processors']
    types = option['types'] if 'types' in option else []

    if len(types) == 0:
        assert(len(processors) == 1)
        if processors[0] == 'cpu':
            types = list(cpu_types)
        elif processors[0] == 'cuda':
            types = list(cuda_types)
        else:
            assert(False)

    pairs = {}

    # generate pairs for all processors
    for processor in processors:
        assert(processor in processor_types)
        type_map = processor_type_map[processor]
        for tstr in types:
            # handle possible expansion
            type_list = type_map[tstr] if tstr in type_map else [tstr]
            for t in type_list:
                assert(t in type_map['all'])
                pairs[t] = processor

    # if there are any prespecified tuples, handle them now
    predefined_pairs = (option['type_processor_pairs'] if 'type_processor_pairs'
                        in option else {})
    for tstr in predefined_pairs:
        pr = predefined_pairs[tstr]
        assert(pr in processor_types)
        type_map = processor_type_map[processor]
        # handle possible expansion
        type_list = type_map[tstr] if tstr in type_map else [tstr]
        for t in type_list:
            assert(t in type_map['all'])
            pairs[t] = pr

    option['processor_type_pairs'] = pairs
    return option


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

    declarations = [d for d in declarations if not exclude(d)]
    declarations = [process_types_and_processors(d) for d in declarations]

    add_variants(declaration)

    return declarations
