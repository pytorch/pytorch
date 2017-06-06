import re
from copy import deepcopy

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
    option.setdefault('variants',['method'])

# if we have 'output' arguments, generate a variant where
# we mark oututs as allocate = True, and where the method variant
# is disabled...
def handle_outputs_taken_as_arguments(options,option):
    if any('output' in arg for arg in option['arguments']):
        new_option = deepcopy(option)
        if 'method' in new_option['variants']:
            new_option['variants'].remove('method')
        for arg in new_option['arguments']:
            if 'output' in arg:
                arg['allocate'] = True
        options.append(new_option)
def sanitize_return(option):
    ret = option['return']
    m = re.match('argument (\d+(,\d+)*)',ret)
    if m is not None:
        arguments = [ int(x) for x in m.group(1).split(',') ]
        option['return'] = { 'kind': 'arguments', 'arguments' : arguments }
    elif ret == 'self':
        option['return'] = { 'kind': 'arguments', 'arguments': []}
        for i,x in enumerate(option['arguments']):
            if x['name'] == 'self':
                option['return']['arguments'].append(i)
                break
    else:
        if option['return'] == 'THTensor*':
            print(option['cname'])
        option['return'] = { 'kind': 'type', 'type' : option['return'] }

def run(declarations):
    declarations = [d for d in declarations if not exclude(d)]
    for declaration in declarations:
        common_with_cwrap.set_declaration_defaults(declaration)
        common_with_cwrap.enumerate_options_due_to_default(declaration)
        common_with_cwrap.sort_by_number_of_options(declaration)
        new_options = []
        for option in declaration['options']:
            sanitize_return(option)
            process_types_and_processors(option)
            add_variants(option)
            handle_outputs_taken_as_arguments(new_options,option)
        declaration['options'] += new_options
    return declarations
