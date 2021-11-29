import yaml
import csv
import torch
import functorch
import re
import sys
import os

class CapturedOutput(object):
    """
    Class used to grab standard output.
    We need this instead of contextlib.redirect_stdout() if the printed text
    that we want to capture comes from C++.
    The result is stored in capturedtext.
    Pulled partially from https://www.py4u.net/discuss/66399.
    """
    escape_char = "\b"

    def __init__(self):
        self.origstream = sys.stdout
        self.origstreamfd = self.origstream.fileno()
        self.capturedtext = ""
        # Create a pipe so the stream can be captured:
        self.pipe_out, self.pipe_in = os.pipe()

    def __enter__(self):
        self.capturedtext = ""
        # Save a copy of the stream:
        self.streamfd = os.dup(self.origstreamfd)
        # Replace the original stream with our write pipe:
        os.dup2(self.pipe_in, self.origstreamfd)
        return self

    def __exit__(self, type, value, traceback):
        # Print the escape character to make the readOutput method stop:
        self.origstream.write(self.escape_char)
        # Flush the stream to make sure all our data goes in before
        # the escape character:
        self.origstream.flush()
        self.readOutput()
        # Close the pipe:
        os.close(self.pipe_in)
        os.close(self.pipe_out)
        # Restore the original stream:
        os.dup2(self.streamfd, self.origstreamfd)
        # Close the duplicate stream:
        os.close(self.streamfd)

    def readOutput(self):
        """
        Read the stream data (one byte at a time)
        and save the text in `capturedtext`.
        """
        while True:
            char = os.read(self.pipe_out, 1)
            if not char:
                break
            char = char.decode("utf-8")
            if self.escape_char in char:
                break
            self.capturedtext += char

def get_ops_for_key(key):
    all_out = CapturedOutput()
    with all_out:
        if key is None:
            torch._C._dispatch_print_registrations_for_dispatch_key()
        else:
            torch._C._dispatch_print_registrations_for_dispatch_key(key)

    ops = all_out.capturedtext.split('\n')
    cleaned_ops = []
    for i in ops:
        if 'aten::' not in i:
            continue
        cleaned_ops.append(i[6:].strip())
    return set(cleaned_ops)

def gen_data(special_op_lists, analysis_name):
    all_ops = get_ops_for_key(None)
    composite_ops = get_ops_for_key('CompositeImplicitAutograd')
    noncomposite_ops = all_ops - composite_ops

    ops = yaml.load(open('/home/chilli/fb/pytorch/aten/src/ATen/native/native_functions.yaml', 'r').read(), Loader=yaml.CLoader)

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
        for op in ops:
            old_tcnt = sum(categorization.values())
            if op['name'][-1] == '_':
                categorization['inplace'] += 1
                op['meta'] = 'inplace'
                continue
            if not is_unique and 'a!' in op['func'].lower():
                categorization['out'] += 1
                op['meta'] = 'out'
                continue
            if 'conv' in op['name']:
                categorization['conv'] += 1
                op['meta'] = 'conv'
                continue
            if 'pool' in op['name']:
                categorization['pool'] += 1
                op['meta'] = 'pool'
                continue
            if 'backward' in op['name']:
                categorization['backward'] += 1
                op['meta'] = 'backward'
                continue
            if op['name'][0] == '_' and op['name'][1] != '_':
                categorization['private'] += 1
                op['meta'] = 'private'
                continue
            if 'batch_norm' in op['name']:
                categorization['batch_norm'] += 1
                op['meta'] = 'batch_norm'
                continue
            if 'Tensor' not in op['func'] or'Tensor' not in op['ret_type']:
                categorization['non_tensor'] += 1
                op['meta'] = 'non_tensor'
                continue
            if 'cudnn' in op['name'] or 'mkldnn' in op['name'] or 'miopen' in op['name'] or 'native' in op['name'] or 'thnn' in op['name'] or 'slow' in op['name']:
                categorization['backend'] += 1
                op['meta'] = 'backend'
                continue
            if op['name'] in annotated_ops:
                categorization['core'] += 1
                op['meta'] = 'core ' + annotated_ops[op['name']]
            else:
                categorization['core'] += 1
                op['meta'] = 'core unknown'
        return categorization

    # categorization = annotate_ops(uniq_ops, True)
    categorization = annotate_ops(ops, False)

    with open(f"{analysis_name}", 'w') as f:
        for op in ops:
            info = [op['full_name'], op['meta'], not (op['full_name'] in noncomposite_ops)] + [op['name'] in op_list for op_list in special_op_lists]
            f.write(','.join([str(i) for i in info]) + '\n')

# Generates batching rule data
# gen_data([get_ops_for_key('FuncTorchBatched')], 'vmap')
if True:
    with open('run_ops.txt', 'r') as f:
        opinfo_ops = [i.strip() for i in f.readlines()]
    with open('run_decompositions.txt', 'r') as f:
        decomposed_ops = [i.strip() for i in f.readlines()]
    gen_data([opinfo_ops, decomposed_ops], 'decompositions')