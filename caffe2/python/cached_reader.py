## @package cached_reader
# Module caffe2.python.cached_reader
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from caffe2.python import core
from caffe2.python.dataio import Reader
from caffe2.python.dataset import Dataset
from caffe2.python.pipeline import pipe
from caffe2.python.task import Cluster, TaskGroup


class CachedReader(Reader):
    """
    Reader with persistent in-file cache.

    Example usage:
    cached_reader = CachedReader(reader)
    build_cache_step = cached_reader.build_cache('/tmp/cache.db')
    with LocalSession() as session:
        session.run(build_cache_step)

    Every time new reader is created, it's expected that build_cache will be
    called before setup_ex and usage of the reader. build_cache will check
    existence of provided file path and in case it's missing will initialize it
    by reading data from original reader. All consequent attempts to read will
    ignore original reader (i.e. no additional data will be read from it).
    """

    def __init__(self, reader, db_type='leveldb', name='cached_reader'):
        super(CachedReader, self).__init__(reader.schema())
        self.original_reader = reader
        self.cache_path = None
        self.ds_reader = None
        self.ds = Dataset(self._schema, name)
        self.db_type = db_type
        self.name = name
        self.field_names = self._schema.field_names()

    def setup_ex(self, init_net, finish_net):
        assert self.cache_path, 'build_cache must be called first'
        self._init_dataset(init_net)
        self._load_from_file(init_net)
        self.ds_reader = self.ds.reader(init_net, batch_size=100)

    def read(self, read_net):
        assert self.ds_reader, 'setup must be called first'
        return self.ds_reader.read(read_net)

    def has_cache(self):
        return self.cache_path and os.path.exists(self.cache_path)

    def build_cache(self, cache_path, overwrite=False):
        if not self.has_cache() or overwrite:
            self.cache_path = cache_path
        if self.has_cache() and not overwrite:
            # cache already exists, no need to rebuild it
            return core.execution_step('build_step', [])

        init_net = core.Net('init')
        self._init_dataset(init_net)
        with Cluster(), core.NameScope(self.name), TaskGroup() as copy_tg:
            pipe(self.original_reader, self.ds.writer(), num_threads=16)
            copy_step = copy_tg.to_task().get_step()
        save_net = core.Net('save')
        self._save_to_file(save_net)

        return core.execution_step('build_cache', [init_net, copy_step, save_net])

    def _init_dataset(self, init_net):
        with core.NameScope(self.name):
            self.ds.init_empty(init_net)

    def _save_to_file(self, net):
        net.Save(
            self.ds.content().field_blobs(),
            [],
            db=self.cache_path,
            db_type=self.db_type,
            blob_name_overrides=self.field_names,
            absolute_path=True,
        )

    def _load_from_file(self, net):
        net.Load(
            [],
            self.ds.content().field_blobs(),
            db=self.cache_path,
            db_type=self.db_type,
            absolute_path=True,
            source_blob_names=self.field_names,
        )
