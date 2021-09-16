from __future__ import division
from __future__ import print_function

import json
import io
import os
import torch
import lazy_tensor_core.core.lazy_model as ltm
import lazy_tensor_core.utils.gcsfs as gcs


def _index_split(index, split_size, split_count):
    parts = []
    while True:
        if parts:
            part = str(index % split_size)
        else:
            part = '{}.pt'.format(index)
        parts.append(part)
        index = index // split_size
        if index == 0:
            break
    while len(parts) < split_count:
        parts.append('0')
    parts.reverse()
    return parts


def _save_metadata(path, **kwargs):
    mpath = os.path.join(path, 'METADATA')
    jdata = json.dumps(kwargs)
    gcs.generic_write(jdata, mpath, makedirs=True)


def _load_metadata(path):
    mpath = os.path.join(path, 'METADATA')
    jdata = gcs.generic_read(mpath).decode()
    return json.loads(jdata)


class CachedDataset(torch.utils.data.Dataset):
    """Wraps an existing `torch.utils.data.Dataset` by providing file caching.

    The `CachedDataset` can be used to trade the CPU/RAM resources required to
    process a raw dataset, with storage/network resources.
    Example::

      train_dataset = datasets.MNIST(
          FLAGS.datadir,
          train=True,
          download=True,
          transform=transforms.Compose(
              [transforms.ToTensor(),
               transforms.Normalize((0.1307,), (0.3081,))]))
      train_dataset = CachedDataset(train_dataset, FLAGS.dscache_dir)

    The `CachedDataset` will transparently cache the original `Dataset` samples,
    so that every run after the first, will not trigger any more CPU/RAM usage
    related to the raw samples processing.
    Once a `CachedDataset` is fully cached, it can be exported (ie, tar.gz) and
    used in different machines.
    Just unpack the tar.gz and pass `None` as original `Dataset`:
    Example::

      train_dataset = CachedDataset(None, FLAGS.dscache_dir)

    To fully cache `CachedDataset` just run the `warmup()` API.
    A `CachedDataset` saved on GCS has the advantage to be able to be used from
    different machines without explicit exporting.

    Args:
      data_set (torch.utils.data.Dataset): The raw `torch.utils.data.Dataset` to be
        cached. It can be set to `None` in case all the input samples are stored
        within the `path` folder.
      path (string): The path where the dataset samples should be stored/loaded.
        The `path` needs to be writeable, unless all the samples are already stored.
        The `path` can be a GCS path (prefixed with `gs://`).
      max_files_per_folder (int): The maximum amount of files to be stored within a
        single folder. If `data_set` is `None` this value is ignored and taken from
        the cached metadata.
        Default: 1000
      compress (bool): Whether the saved samples should be compressed. Compression
        saves space at the expense of CPU required to compress/decompress.
        If `data_set` is `None` this value is ignored and taken from the cached
        metadata.
        Default: True
    """

    def __init__(self, data_set, path, max_files_per_folder=1000, compress=True):
        super(CachedDataset, self).__init__()
        self._data_set = data_set
        self._path = path
        if data_set is not None:
            self._max_files_per_folder = max_files_per_folder
            self._compress = compress
            self._count = len(data_set)
            if ltm.is_master_ordinal(local=not gcs.is_gcs_path(path)):
                _save_metadata(
                    path,
                    count=self._count,
                    compress=self._compress,
                    max_files_per_folder=self._max_files_per_folder)
        else:
            meta = _load_metadata(path)
            self._max_files_per_folder = meta['max_files_per_folder']
            self._count = meta['count']
            self._compress = meta['compress']
        self._split_count = len(
            _index_split(self._count, self._max_files_per_folder, 0))

    def _index_path(self, index):
        return os.path.join(
            self._path,
            *_index_split(index, self._max_files_per_folder, self._split_count))

    def _save_sample(self, data, path):
        bio = io.BytesIO()
        torch.save(data, bio, _use_new_zipfile_serialization=self._compress)
        gcs.generic_write(bio.getvalue(), path, makedirs=True)

    def _load_sample(self, path):
        try:
            data = gcs.generic_read(path)
            return torch.load(io.BytesIO(data))
        except BaseException:
            pass

    def warmup(self):
        for index in range(0, self._count):
            self.__getitem__(index)

    def __len__(self):
        return self._count

    def __getitem__(self, index):
        path = self._index_path(index)
        data = self._load_sample(path)
        if data is None:
            if self._data_set is None:
                raise RuntimeError(
                    'Source dataset not provided and sample {} is missing from cache folder {}'
                    .format(index, self._path))
            data = self._data_set[index]
            self._save_sample(data, path)
        return data
