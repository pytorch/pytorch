r""""Contains definitions of the methods used by the _DataLoaderIter workers to
collate samples fetched from dataset into Tensor(s).

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""

import torch
import re
from torch._six import container_abcs, string_classes, int_classes

np_str_obj_array_pattern = re.compile(r'[SaUO]')


def default_convert(data):
    r"""Puts each data field into a tensor"""

    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data
    elif elem_type.__module__ == 'numpy':
        return torch.as_tensor(data)
    elif isinstance(data, int_classes):
        return torch.tensor(data, dtype=torch.long)
    elif isinstance(data, float):
        return torch.tensor(data, dtype=torch.double)
    elif isinstance(data, string_classes):
        return data
    elif isinstance(data, container_abcs.Mapping):
        return {key: default_convert(data[key]) for key in data}
    elif isinstance(data, container_abcs.Sequence):
        return [default_convert(d) for d in data]

    raise TypeError(("default_convert: found unexpected input type {}".format(elem_type)))


_use_shared_memory = False
r"""Whether to use shared memory in default_collate"""


def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        # TODO: remove once CPU half tensors support stack
        if elem.dtype == torch.float16:
            out_shape = torch.Size([len(batch)] + list(elem.size()))
            if out is None:
                out = elem.new_empty(out_shape)
            else:
                strides = [1]
                for s in out_shape[1::-1]:
                    strides.insert(0, strides[0] * s)
                out = out.set_(out.storage, out_shape, strides)
            for src, dst in zip(batch, out):
                dst.copy_(src)
            return out
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(error_msg_fmt.format(elem.dtype))

            return default_collate([torch.from_numpy(b) for b in batch])
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError("default_collate: batch must contain tensors, numbers, dicts or lists; found {}".format(elem_type))
