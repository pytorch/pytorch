from __future__ import division
from __future__ import print_function

import os
import shutil

import torch
import lazy_tensor_core
import lazy_tensor_core.core.lazy_model as ltm


class TensorReference(object):
    __slots__ = ['tid']

    def __init__(self, tid):
        self.tid = tid


def _get_tensors_folder(path):
    return path + '.tensors'


def _get_tensor_file(path, tid):
    return os.path.join(path, 'tensor_{}.pt'.format(tid))


def _rewrite_data(path, data, save_tensors):

    def convert_fn(tensors):
        lazy_tensor_core._LAZYC._ltc_sync_multi(
            tensors, devices=[], wait=True, sync_ltc_data=True)
        rewritten_tensors = []
        for i, t in enumerate(tensors):
            if save_tensors:
                torch.save(t.cpu(), _get_tensor_file(path, i))
            rewritten_tensors.append(TensorReference(i))
        return rewritten_tensors

    def select_fn(v):
        return type(v) == torch.Tensor and ltm.is_lazy_tensor(v)

    if os.path.isdir(path):
        shutil.rmtree(path)
    os.mkdir(path)
    return ltm.ToLazyTensorArena(convert_fn, select_fn).transform(data)


def save(data, path, master_only=True, global_master=False):
    """Saves the input data into a file.

    The saved data is transferred to PyTorch CPU device before being saved, so a
    following `torch.load()` will load CPU data.
    Care must be taken when working with views. Instead of saving views it's
    recommended that you recreate them after the tensors have been loaded and
    moved to their destination device(s).

    Args:
      data: The input data to be saved. Any nested combination of Python objects
        (list, tuples, sets, dicts, ...).
      path: The destination file for the data saving operation. If `master_only`
        is ``False`` the path must point to different destinations as otherwise
        all the writes from the same host will override each other.
      master_only (bool, optional): Whether only the master device should save the
        data. If False, the `path` argument should be a different path for each of
        the ordinals taking part to the replication, otherwise all the replicas on
        the same host will be writing to the same location.
        Default: True
      global_master (bool, optional): When ``master_only`` is ``True`` this flag
        controls whether every host's master (if ``global_master`` is ``False``)
        saves the content, or only the global master (ordinal 0).
        Default: False
    """
    should_write_data = not master_only or ltm.is_master_ordinal(
        local=not global_master)

    ref_data = _rewrite_data(_get_tensors_folder(path), data, should_write_data)
    if should_write_data:
        torch.save(ref_data, path)
    ltm.rendezvous('lazy_tensor_core.utils.serialization.save')


def load(path):
    """Loads data previously saved with the `save()` API.

    Args:
      path (str): The path passed to the `save()` API.
    Returns:
      The loaded data.
    """
    ref_data = torch.load(path)
    tensor_folder = _get_tensors_folder(path)

    def convert_fn(tensors):
        rewritten_tensors = []
        for t in tensors:
            rewritten_tensors.append(
                torch.load(_get_tensor_file(tensor_folder, t.tid)))
        return rewritten_tensors

    def select_fn(v):
        return type(v) == TensorReference

    return ltm.ToLazyTensorArena(convert_fn, select_fn).transform(ref_data)
