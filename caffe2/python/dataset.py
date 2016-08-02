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
from caffe2.python.io import Reader, Writer
from caffe2.python.schema import Struct
import numpy as np


class _DatasetReader(Reader):
    def __init__(self, field_names, field_blobs, cursor, name):
        """Don't call this directly. Instead, use dataset.reader()"""
        self.field_names = field_names
        self.field_blobs = field_blobs
        self.cursor = cursor
        self.name = name

    def read(self, read_net, batch_size=1):
        with core.NameScope(read_net.NextName(self.name)):
            fields = read_net.ReadNextBatch(
                [self.cursor] + self.field_blobs,
                self.field_names,
                batch_size=batch_size)
            return (read_net.IsEmpty([fields[0]]), fields)

    def reset(self, net):
        net.ResetCursor([self.cursor], [])


class _DatasetRandomReader(Reader):
    def __init__(self, field_names, field_blobs, cursor, name, indices):
        """Don't call this directly. Instead, use dataset.random_reader()"""
        self.field_names = field_names
        self.field_blobs = field_blobs
        self.cursor = cursor
        self.name = name
        self.indices = indices

    def reset(self, net):
        net.ResetCursor([self.cursor], [])

    def computeoffset(self, net):
        self.reset(net)
        offsets = net.ComputeOffset(
            [self.cursor] + self.field_blobs,
            'offsets')
        self.offsets = offsets

    def sortAndShuffle(self, net, sort_by_field=None,
                       shuffle_size=1, batch_size=1):
        # no sorting by default
        sort_by_field_idx = -1
        if sort_by_field:
            assert sort_by_field in self.field_names, 'must be valid field'
            sort_by_field_idx = self.field_names.index(sort_by_field)
        self.reset(net)

        indices = net.SortAndShuffle(
            [self.cursor] + self.field_blobs,
            'indices',
            sort_by_field_idx=sort_by_field_idx,
            shuffle_size=shuffle_size,
            batch_size=batch_size)
        self.indices = indices

    def read(self, read_net, batch_size=1):
        fields = read_net.ReadRandomBatch(
            [self.cursor, self.indices, self.offsets] + self.field_blobs,
            self.field_names,
            batch_size=batch_size)
        return (read_net.IsEmpty([fields[0]]), fields)


class _DatasetWriter(Writer):
    def __init__(self, fields, field_blobs, init_net):
        """Don't call this directly. Use dataset.writer() instead."""
        self.fields = fields
        self.field_blobs = field_blobs
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
        assert len(fields) == len(self.fields), (
            'Expected %s fields, got %s.' % (len(self.fields), len(fields)))
        writer_net.CheckDatasetConsistency(fields, [], fields=self.fields)
        writer_net.AtomicAppend(
            [self.mutex] + list(self.field_blobs) + list(fields),
            self.field_blobs)

    def commit(self, finish_net):
        """Commit is a no-op for an in-memory dataset."""
        pass


def to_ndarray_list(values, schema):
    """
    Given a list of values and a dataset schema, produce list of ndarray in the
    right format.

    This function will perform some checks to make sure that the arrays
    produced have the right dtype and rank.
    """
    assert isinstance(schema, Struct), 'schema must be a Struct.'
    names = schema.field_names()
    types = schema.field_types()
    assert len(types) == len(values), (
        'Values must have %d elements, got %d' % (len(types), len(values)))

    arrays = []
    for value, dtype, name in zip(values, types, names):
        array = np.array(value, dtype=dtype.base)
        # if array is empty we may need to reshape a little
        if array.size == 0:
            array = array.reshape((0,) + dtype.shape)
        # check that the inner dimensions match the schema
        assert (array.shape[1:] == dtype.shape), (
            'Invalid array shape for field %s. Expected (%s), got (%s).' % (
                name,
                ', '.join(['_'] + map(str, dtype.shape)),
                ', '.join(map(str, array.shape))))
        arrays.append(array)
    return arrays


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
        self.schema = fields
        self.fields = (
            fields.field_names() if isinstance(fields, Struct) else fields)
        self.field_types = (
            fields.field_types() if isinstance(fields, Struct) else
            [np.dtype(np.void)] * len(self.fields))
        self.name = name or 'dataset'
        self.field_blobs = None

    def init_empty(self, init_net):
        """Initialize the blobs for this dataset with empty values.

        Empty arrays will be immediately fed into the current workspace,
        and `init_net` will take those blobs as external inputs.
        """
        self.field_blobs = [Const(init_net, [], name=f) for f in self.fields]

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

    def reader(self, init_net, cursor_name=None):
        """Create a Reader object that is used to iterate through the dataset.

        This will append operations to `init_net` that create a TreeCursor,
        used to iterate through the data.

        NOTE: Currently, it is not safe to append to a dataset while reading.

        Args:
            init_net: net that will be run once to create the cursor.
            cursor_name: optional name for the blob containing a pointer
                         to the cursor.

        Returns:
            A _DatasetReader that can be used to create operators that will
            iterate through the dataset.
        """
        assert self.field_blobs, 'Dataset not initialized.'
        cursor_name = cursor_name or (self.name + '_cursor')
        cursor = init_net.CreateTreeCursor(
            [],
            [cursor_name],
            fields=self.fields)
        return _DatasetReader(
            self.fields, self.field_blobs, cursor, cursor_name)

    def random_reader(self, init_net, indices=None, cursor_name=None):
        """Create a Reader object that is used to iterate through the dataset.

        NOTE: The reader order depends on the order in indices.

        Args:
            Similar to reader
            indices: blob of reading order

        Returns:
            A DatasetReader that can be used to create operators that will
            iterate through the dataset according to indices.
        """
        assert self.field_blobs, 'Dataset not initialized.'
        cursor_name = cursor_name or (self.name + '_cursor')
        cursor = init_net.CreateTreeCursor(
            [],
            [cursor_name],
            fields=self.fields)
        return _DatasetRandomReader(
            self.fields, self.field_blobs, cursor, cursor_name, indices)

    def writer(self, init_net):
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
        return _DatasetWriter(self.fields, self.field_blobs, init_net)
