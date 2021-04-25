import ruamel.yaml
from ruamel.yaml.tokens import CommentToken
from ruamel.yaml.error import CommentMark
from tools.codegen.model import *  # noqa: F403

with open("aten/src/ATen/native/native_functions.yaml", "r") as f:
    contents = f.read()

yaml = ruamel.yaml.YAML()
yaml.preserve_quotes = True
yaml.width = 1000
yaml.boolean_representation = ['False', 'True']
r = yaml.load(contents)

convert = '''\
neg'''.split()

for e in r:
    f = NativeFunction.from_yaml(e, Location("", 0))
    if f.structured or f.structured_delegate is not None:
        continue
    n = f.func.name.name.base
    if n not in convert:
        continue
    # mutate e to make changes
    if f.func.kind() == SchemaKind.out:
        e.insert(1, 'structured', True)
        e.insert(2, 'structured_inherits', 'TensorIteratorBase')
    else:
        # TODO: The .out overload assumption is not sound in general
        e.insert(1, 'structured_delegate', f'{n}.out')

        e['dispatch'].pop('CPU', None)
        e['dispatch'].pop('CUDA', None)
        e['dispatch'].pop('CPU, CUDA', None)
        e['dispatch'].pop('CompositeExplicitAutograd', None)

        *_, last_k = e.keys()
        needs_fixup = False

        if not e['dispatch']:
            if last_k == 'dispatch':
                needs_fixup = True
            del e['dispatch']

        # Manually fix up newlines at the end, because ruamel
        # made some bad life choices about where to associate trailing
        # whitespace for nested dicts; see
        # https://stackoverflow.com/questions/42172399/modifying-yaml-using-ruamel-yaml-adds-extra-new-lines
        if needs_fixup:
            *_, last_k = e.keys()
            # post_key, pre_key, post_value, pre_value
            e.ca.items[last_k] = [None, None, CommentToken('\n\n', CommentMark(0), None), None]

with open("aten/src/ATen/native/native_functions.yaml.new", "w") as f:
    yaml.dump(r, f)
