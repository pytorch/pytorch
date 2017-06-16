## @package dataset
# Module caffe2.python.dataset
"""
Implementation of an in-memory dataset with structured schema.

Use this to store and iterate through datasets with complex schema that
fit in memory.

Iterating through entries of this dataset is very fast since the dataset
is stored as a set of native Caffe2 tensors, thus no type conversion or
deserialization is necessary.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, workspace
from caffe2.python.dataio import Reader, Writer
from caffe2.python.schema import (
    Struct, from_blob_list, from_column_list, InitEmptyRecord)
import numpy as np


class _DatasetReader(Reader):
    def __init__(self, dataset, name, batch_size=1):
        """Don't call this directly. Instead, use dataset.reader()"""
        Reader.__init__(self, dataset.content())
        self.dataset = dataset
        self.name = name or (dataset.name + '_cursor')
        self.batch_size = batch_size
        self.cursor = None

    def setup_ex(self, init_net, exit_net):
        if self.cursor is None:
            self.cursor = init_net.CreateTreeCursor(
                [],
                [self.name],
                fields=self.dataset.fields)

    def read(self, read_net):
        assert self.cursor, 'setup not called.'
        content = self.dataset.content()
        with core.NameScope(read_net.NextName(self.name)):
            fields = read_net.ReadNextBatch(
                [self.cursor] + content.field_blobs(),
                content.field_names(),
                batch_size=self.batch_size)
            if type(fields) is core.BlobReference:
                fields = [fields]
            return (read_net.IsEmpty([fields[0]]), fields)

    def reset(self, net):
        net.ResetCursor([self.cursor], [])


class _DatasetRandomReader(Reader):
    def __init__(self, dataset, name, indices, batch_size=1, loop_over=False):
        """Don't call this directly. Instead, use dataset.random_reader()"""
        Reader.__init__(self, dataset.content())
        self.dataset = dataset
        self.cursor = None
        self.name = name or (dataset.name + '_cursor')
        self.indices = indices
        self.batch_size = batch_size
        self.loop_over = loop_over

    def setup_ex(self, init_net, exit_net):
        if self.cursor is None:
            self.cursor = init_net.CreateTreeCursor(
                [],
                [self.name],
                fields=self.dataset.fields)

    def reset(self, net):
        net.ResetCursor([self.cursor], [])

    def computeoffset(self, net):
        self.reset(net)
        offsets = net.ComputeOffset(
            [self.cursor] + self.dataset.content().field_blobs(),
            'offsets')
        self.offsets = offsets

    def sort_and_shuffle(self, net, sort_by_field=None,
                         shuffle_size=1, batch_size=1):
        # no sorting by default
        content = self.dataset.content()
        sort_by_field_idx = -1
        if sort_by_field:
            assert sort_by_field in content.field_names(), (
                'Must be valid field.')
            sort_by_field_idx = content.field_names().index(sort_by_field)
        self.reset(net)

        indices = net.SortAndShuffle(
            [self.cursor] + content.field_blobs(),
            'indices',
            sort_by_field_idx=sort_by_field_idx,
            shuffle_size=shuffle_size,
            batch_size=batch_size)
        self.indices = indices

    def read(self, read_net):
        with core.NameScope(read_net.NextName(self.name)):
            fields = read_net.ReadRandomBatch(
                [self.cursor, self.indices, self.offsets] + (
                    self.dataset.content().field_blobs()),
                self.dataset.content().field_names(),
                batch_size=self.batch_size,
                loop_over=self.loop_over)
            return (read_net.IsEmpty([fields[0]]), fields)


class _DatasetWriter(Writer):
    def __init__(self, content):
        """Don't call this directly. Use dataset.writer() instead."""
        self._content = content
        self.mutex = None

    def setup_ex(self, init_net, exit_net):
        if self.mutex is None:
            self.mutex = init_net.CreateMutex([])

    def write(self, writer_net, fields):
        """
        Add operations to `net` that append the blobs in `fields` to the end
        of the dataset. An additional operator will also be added that checks
        the consistency of the data in `fields` against the dataset schema.

        Args:
            writer_net: The net that will contain the Append operators.
            fields: A list of BlobReference to be appeneded to this dataset.
        """
        assert self.mutex is not None, 'setup not called.'
        field_blobs = self._content.field_blobs()
        assert len(fields) == len(field_blobs), (
            'Expected %s fields, got %s.' % (len(field_blobs), len(fields)))
        writer_net.CheckDatasetConsistency(
            fields, [], fields=self._content.field_names())
        writer_net.AtomicAppend(
            [self.mutex] + field_blobs + list(fields),
            field_blobs)

    def commit(self, finish_net):
        """Commit is a no-op for an in-memory dataset."""
        pass


def Const(net, value, dtype=None, name=None):
    """
    Create a 'constant' by first creating an external input in the given
    net, and then feeding the corresponding blob with its provided value
    in the current workspace. The name is automatically generated in order
    to avoid clashes with existing blob names.
    """
    assert isinstance(net, core.Net), 'net must be a core.Net instance.'
    value = np.array(value, dtype=dtype)
    blob = net.AddExternalInput(net.NextName(prefix=name))
    workspace.FeedBlob(str(blob), value)
    return blob


def execution_step_with_progress(name, init_net, substeps, rows_read):
    # progress reporter
    report_net = core.Net('report_net')
    report_net.Print([rows_read], [])
    return core.execution_step(
        name,
        substeps,
        report_net=report_net,
        concurrent_substeps=True,
        report_interval=5)


class Dataset(object):
    """Represents an in-memory dataset with fixed schema.

    Use this to store and iterate through datasets with complex schema that
    fit in memory.

    Iterating through entries of this dataset is very fast since the dataset
    is stored as a set of native Caffe2 tensors, thus no type conversion or
    deserialization is necessary.
    """

    def __init__(self, fields, name=None):
        """Create an un-initialized dataset with schema provided by `fields`.

        Before this dataset can be used, it must be initialized, either by
        `init_empty` or `init_from_dataframe`.

        Args:
            fields: either a schema.Struct or a list of field names in a format
                    compatible with the one described in schema.py.
            name: optional name to prepend to blobs that will store the data.
        """
        assert isinstance(fields, list) or isinstance(fields, Struct), (
            'fields must be either a Struct or a list of raw field names.')
        if isinstance(fields, list):
            fields = from_column_list(fields)
        self.schema = fields
        self.fields = fields.field_names()
        self.field_types = fields.field_types()
        self.name = name or 'dataset'
        self.field_blobs = fields.field_blobs() if fields.has_blobs() else None

    def init_empty(self, init_net):
        """Initialize the blobs for this dataset with empty values.

        Empty arrays will be immediately fed into the current workspace,
        and `init_net` will take those blobs as external inputs.
        """
        self.field_blobs = InitEmptyRecord(
            init_net, self.schema.clone_schema()).field_blobs()

    def init_from_dataframe(self, net, dataframe):
        """Initialize the blobs for this dataset from a Pandas dataframe.

        Each column of the dataframe will be immediately fed into the current
        workspace, and the `net` will take this blobs as external inputs.
        """
        assert len(self.fields) == len(dataframe.columns)
        self.field_blobs = [
            Const(net, dataframe.as_matrix([col]).flatten(), name=field)
            for col, field in enumerate(self.fields)]

    def get_blobs(self):
        """
        Return the list of BlobReference pointing to the blobs that contain
        the data for this dataset.
        """
        assert self
        return self.field_blobs

    def content(self):
        """
        Return a Record of BlobReferences pointing to the full content of
        this dataset.
        """
        return from_blob_list(self.schema, self.field_blobs)

    def field_names(self):
        """Return the list of field names for this dataset."""
        return self.fields

    def field_types(self):
        """
        Return the list of field dtypes for this dataset.

        If a list of strings, not a schema.Struct, was passed to the
        constructor, this will return a list of dtype(np.void).
        """
        return self.field_types

    def reader(self, init_net=None, cursor_name=None, batch_size=1):
        """Create a Reader object that is used to iterate through the dataset.

        This will append operations to `init_net` that create a TreeCursor,
        used to iterate through the data.

        NOTE: Currently, it is not safe to append to a dataset while reading.

        Args:
            init_net: net that will be run once to create the cursor.
            cursor_name: optional name for the blob containing a pointer
                         to the cursor.
            batch_size: how many samples to read per iteration.

        Returns:
            A _DatasetReader that can be used to create operators that will
            iterate through the dataset.
        """
        assert self.field_blobs, 'Dataset not initialized.'
        reader = _DatasetReader(self, cursor_name, batch_size)
        if init_net is not None:
            reader.setup_ex(init_net, None)
        return reader

    def random_reader(self, init_net=None, indices=None, cursor_name=None,
                      batch_size=1, loop_over=False):
        """Create a Reader object that is used to iterate through the dataset.

        NOTE: The reader order depends on the order in indices.

        Args:
            init_net: net that will be run once to create the cursor.
            indices: blob of reading order
            cursor_name: optional name for the blob containing a pointer
                         to the cursor.
            batch_size: how many samples to read per iteration.
            loop_over: repeat the dataset indefinitely (in the same order)

        Returns:
            A DatasetReader that can be used to create operators that will
            iterate through the dataset according to indices.
        """
        assert self.field_blobs, 'Dataset not initialized.'
        reader = _DatasetRandomReader(
            self, cursor_name, indices, batch_size, loop_over)
        if init_net is not None:
            reader.setup_ex(init_net, None)
        return reader

    def writer(self, init_net=None):
        """Create a Writer that can be used to append entries into the dataset.

        NOTE: Currently, it is not safe to append to a dataset
              while reading from it.
        NOTE: Currently implementation of writer is not thread safe.
              TODO: fixme

        Args:
            init_net: net that will be run once in order to create the writer.
                      (currently not used)
        """
        assert self.field_blobs, 'Dataset not initialized.'
        writer = _DatasetWriter(self.content())
        if init_net is not None:
            writer.setup_ex(init_net, None)
        return writer
