import torch
import uuid
import sqlite3
conn = sqlite3.connect('operators_stats.sqlite')

# TODO: Add numel field

# Using sqlite3 table:
# CREATE TABLE usage (
#       operator_name TEXT NOT NULL,
#       operation_id TEXT NOT NULL,
#       type TEXT NOT NULL,
#       ord INT NOT NULL,
#       dim INT NOT NULL,
#       shape TEXT NOT NULL,
#       strides TEXT NOT NULL,
#       contiguous BOOL NOT NULL,
#       channels_last BOOL NOT NULL,
#       non_overlapping_and_dense BOOL NOT NULL,
#       sparse BOOL NOT NULL,
#       mkl_dnn BOOL NOT NULL,
#       dtype TEXT NOT NULL,
#       device TEXT NOT NULL);



def contains_tensors_4d_cl(args):
    for t in args:
        if isinstance(t, torch.Tensor):
            if t.dim() == 4 and t.is_contiguous(memory_format=torch.channels_last) and t.shape[1] > 1 and (t.shape[2] > 1 or t.shape[3] > 1):
                return True
        elif isinstance(t, list) or isinstance(t, tuple):
            if contains_tensors_4d_cl(list(t)):
                return True
    return False

def traverse_all_tensors(args, callback_fn, pass_type):
    if isinstance(args, list) or isinstance(args, tuple):
        for t in args:
            if isinstance(t, torch.Tensor):
                callback_fn(t, pass_type)
            elif isinstance(t, list) or isinstance(t, tuple):
                traverse_all_tensors(list(t), callback_fn, pass_type)
    else:
        traverse_all_tensors([args], callback_fn, pass_type)

def contains_tensors_4d(args):
    for t in args:
        if isinstance(t, torch.Tensor):
            if t.dim() == 4:
                return True
        elif isinstance(t, list) or isinstance(t, tuple):
            if contains_tensors_4d(list(t)):
                return True
    return False


def print_inputs(args, indent=''):
    res = ''
    for t in args:
        if isinstance(t, torch.Tensor):
            res += "%s %s %s %s %s\n" % (indent, t.stride(), t.shape, t.device, t.dtype)
        elif isinstance(t, list) or isinstance(t, tuple):
            res += "%s %s\n" % (indent, type(t))
            res += print_inputs(list(t), indent=indent + '    ') + "\n"
        else:
            res += "%s %s\n" % (indent, t)
    return res

def is_non_overlapping_and_dense(t):
    if t.dim() == 1:
        return t.shape[0] < 2 or t.stride()[0] == 1
    idx = list(range(t.dim()))
    strides = []
    for size, stride in zip(t.shape, t.stride()):
        if size < 2:
            stride = -1
        strides.append(stride)
    s_idx = sorted(idx, key = lambda x: strides[x])
    require_stride = 1
    for idx in s_idx:
        if strides[idx] == -1:
            continue
        if strides[idx] != require_stride:
            return False
        require_stride *= t.shape[idx]
    return True

records = []

def check_wrapper(fn):
    name = fn.__name__
    global records
    def check_cl(*args, **kwargs):
        order = [0]
        # records = []
        operation_id = str(uuid.uuid1())
        def callback(tensor, t):
            sparse = tensor.is_sparse
            mkldnn = tensor.is_mkldnn
            channels_last = not sparse and not mkldnn and tensor.is_contiguous(memory_format=torch.channels_last)
            non_overlapping_and_dense = not sparse and not mkldnn and is_non_overlapping_and_dense(tensor)
            contiguous = not sparse and not mkldnn and tensor.is_contiguous()
            if sparse or mkldnn:
                strides = '[]'
            else:
                strides = str(list(tensor.stride()))
            records.append(
                (name,
                operation_id,
                t,
                order[0],
                tensor.dim(),
                str(list(tensor.shape)),
                strides,
                contiguous,
                channels_last,
                non_overlapping_and_dense,
                sparse,
                mkldnn,
                str(tensor.dtype),
                str(tensor.device),
                )
            )
            order[0] += 1

        result = fn(*args, **kwargs)
        traverse_all_tensors(args, callback, 'args')
        if 'out' in kwargs:
            traverse_all_tensors(kwargs['out'], callback, 'out')
        traverse_all_tensors(result, callback, 'res')
        if len(records) > 1000:
            flush()
        return result
    return check_cl

def flush():
    cursor = conn.cursor()
    cursor.executemany('INSERT INTO usage VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)', records)
    #   operator_name TEXT NOT NULL,
    #   operation_id TEXT NOT NULL,
    #   type TEXT NOT NULL,
    #   ord INT NOT NULL,
    #   dim INT NOT NULL,
    #   shape TEXT NOT NULL,
    #   strides TEXT NOT NULL,
    #   contiguous BOOL NOT NULL,
    #   channels_last BOOL NOT NULL,
    #   non_overlapping_and_dense BOOL NOT NULL,
    #   sparse BOOL NOT NULL,
    #   mkl_dnn BOOL NOT NULL,
    #   dtype TEXT NOT NULL,
    #   device TEXT NOT NULL);
    conn.commit()
    records.clear()

def attribute(m):
    for i in dir(m):
        e = getattr(m, i)
        exclude_functions = ['backward', 'autograd', 'is_cuda', 'has_names', 'numel',
                             'stride', 'Tensor', 'is_contiguous', '__class__', 'dim',
                             'tensor', 'List', 'Optional', 'Tuple', 'Set']
        if i not in exclude_functions and not i.startswith('_') and '__call__' in dir(e):
            try:
                setattr(m, i, check_wrapper(e))
            except Exception as e:
                print(i)
                print(e)


attribute(torch.Tensor)
attribute(torch.nn.functional)
attribute(torch)
