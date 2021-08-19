import yaml
import csv
ops = yaml.load(open('/home/chilli/fb/pytorch/aten/src/ATen/native/native_functions.yaml', 'r').read())
# A list of all non-composite ops starting with aten::
noncomposite_ops = [i[6:].strip() for i in open('noncomposite.txt').readlines()]
# A list of all batching rules for ops starting with aten::
vmap_ops = [i[6:].strip() for i in open('vmap.txt').readlines()]
annotated_ops = {a.strip(): b.strip() for a,b in list(csv.reader(open('annotated_ops.txt')))}
from collections import defaultdict

uniq_ops = []
uniq_names = set()
overload_types = defaultdict(list)
cnt = 0
for op in ops:
    func_str = op['func']
    name = func_str[:func_str.index('(')]
    if '.' in name:
        uniq_name = name[:name.index('.')]
        overload_types[name[name.index('.') + 1:]].append(name)
    else:
        uniq_name = name
    op['name'] = uniq_name
    full_name = func_str[:func_str.index('(')]
    op['full_name'] = full_name
    ret_type = func_str[func_str.index('->') + 3:]
    op['ret_type'] = ret_type
    cnt += 1
    if uniq_name in uniq_names:
        continue
    uniq_names.add(uniq_name)
    uniq_ops.append(op)

def annotate_ops(ops, is_unique):
    categorization = defaultdict(int)
    for i in ops:
        old_tcnt = sum(categorization.values())
        if 'conv' in i['name']:
            categorization['conv'] += 1
            i['meta'] = 'conv'
            continue
        if not is_unique and 'out' in i['full_name']:
            categorization['out'] += 1
            i['meta'] = 'out'
            continue
        if 'pool' in i['name']:
            categorization['pool'] += 1
            i['meta'] = 'pool'
            continue
        if 'backward' in i['name']:
            categorization['backward'] += 1
            i['meta'] = 'backward'
            continue
        if i['name'][0] == '_' and i['name'][1] != '_':
            categorization['private'] += 1
            i['meta'] = 'private'
            continue
        if i['name'][-1] == '_':
            categorization['inplace'] += 1
            i['meta'] = 'inplace'
            continue
        if 'batch_norm' in i['name']:
            categorization['batch_norm'] += 1
            i['meta'] = 'batch_norm'
            continue
        if 'Tensor' not in i['func'] or'Tensor' not in i['ret_type']:
            categorization['non_tensor'] += 1
            i['meta'] = 'non_tensor'
            continue
        if 'cudnn' in i['name'] or 'mkldnn' in i['name'] or 'miopen' in i['name'] or 'native' in i['name'] or 'thnn' in i['name'] or 'slow' in i['name']:
            categorization['backend'] += 1
            i['meta'] = 'backend'
            continue
        if i['name'] in annotated_ops:
            categorization['core'] += 1
            i['meta'] = 'core ' + annotated_ops[i['name']]
        else:
            categorization['core'] += 1
            i['meta'] = 'core unknown'
    return categorization

categorization = annotate_ops(uniq_ops, True)
categorization = annotate_ops(ops, False)

for op in ops:
    info = [op['full_name'], op['meta'], not (op['full_name'] in noncomposite_ops), op['full_name'] in vmap_ops]
    print(','.join([str(i) for i in info]))