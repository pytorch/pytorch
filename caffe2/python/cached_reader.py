## @package cached_reader
# Module caffe2.python.cached_reader
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from caffe2.python import core
from caffe2.python.db_file_reader import DBFileReader
from caffe2.python.pipeline import pipe
from caffe2.python.task import Cluster, TaskGroup


class CachedReader(DBFileReader):

    default_name_suffix = 'cached_reader'

    """Reader with persistent in-file cache.

    Example usage:
    cached_reader = CachedReader(
        reader,
        db_path='/tmp/cache.db',
        db_type='LevelDB',
    )
    build_cache_step = cached_reader.build_cache_step()
    with LocalSession() as session:
        session.run(build_cache_step)

    Every time new CachedReader is created, it's expected that
    db_path exists before calling .setup_ex(...) and .read(...).

    If db_path doesn't exist, it's expected build_cache_step to be called
    first to build a cache at db_path.

    build_cache_step will check existence of provided db_path and in case
    it's missing will initialize it by reading data from original reader.
    All consequent attempts to read will ignore original reader
    (i.e. no additional data will be read from it).

    Args:
        original_reader: Reader.
            If provided, it's the original reader used to build the cache file.
        db_path: str.
        db_type: str. DB type of file. A db_type is registed by
            `REGISTER_CAFFE2_DB(<db_type>, <DB Class>)`.
            Default to 'LevelDB'.
        name: str or None. Name of CachedReader.
            Optional name to prepend to blobs that will store the data.
            Default to '<db_name>_<default_name_suffix>'.
        batch_size: int.
            How many examples are read for each time the read_net is run.
    """
    def __init__(
        self,
        original_reader,
        db_path,
        db_type='LevelDB',
        name=None,
        batch_size=100,
    ):
        assert original_reader is not None, "original_reader can't be None"
        self.original_reader = original_reader

        super(CachedReader, self).__init__(
            db_path,
            db_type,
            name,
            batch_size,
        )

    def _init_reader_schema(self, *args, **kwargs):
        """Prepare the reader schema.

            Since an original reader is given,
            use it's schema as ground truth.

            Returns:
                schema: schema.Struct. Used in Reader.__init__(...).
        """
        return self.original_reader._schema

    def build_cache_step(self, overwrite=False):
        """Build a step for generating cache DB file.

            If self.db_path exists and not overwritting, build an empty step.
            Overwise, build a step as follows.
            Pipe original reader to the _DatasetWriter,
            so that dataset field blobs are populated.
            Then save these blobs into a file.

            Args:
                overwrite: bool. If true, ignore the existing file
                    and build a new one overwritting the existing one anyway.

            Returns:
                build_cache_step: ExcutionStep.
                    The step to be run for building a cache DB file.
        """
        if os.path.exists(self.db_path) and not overwrite:
            # cache already exists, no need to rebuild it
            return core.execution_step('build_step', [])

        init_net = core.Net('init')
        self._init_field_blobs_as_empty(init_net)
        with Cluster(), core.NameScope(self.name), TaskGroup() as copy_tg:
            pipe(self.original_reader, self.ds.writer(), num_threads=16)
            copy_step = copy_tg.to_task().get_step()
        save_net = core.Net('save')
        self._save_field_blobs_to_db_file(save_net)

        return core.execution_step('build_cache', [init_net, copy_step, save_net])

    def _save_field_blobs_to_db_file(self, net):
        """Save dataset field blobs to a DB file at db_path"""
        net.Save(
            self.ds.get_blobs(),
            [],
            db=self.db_path,
            db_type=self.db_type,
            blob_name_overrides=self.ds.field_names(),
            absolute_path=True,
        )
