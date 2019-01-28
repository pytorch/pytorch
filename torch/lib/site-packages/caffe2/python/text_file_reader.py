## @package text_file_reader
# Module caffe2.python.text_file_reader
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core
from caffe2.python.dataio import Reader
from caffe2.python.schema import Scalar, Struct, data_type_for_dtype


class TextFileReader(Reader):
    """
    Wrapper around operators for reading from text files.
    """
    def __init__(self, init_net, filename, schema, num_passes=1, batch_size=1):
        """
        Create op for building a TextFileReader instance in the workspace.

        Args:
            init_net   : Net that will be run only once at startup.
            filename   : Path to file to read from.
            schema     : schema.Struct representing the schema of the data.
                         Currently, only support Struct of strings.
            num_passes : Number of passes over the data.
            batch_size : Number of rows to read at a time.
        """
        assert isinstance(schema, Struct), 'Schema must be a schema.Struct'
        for name, child in schema.get_children():
            assert isinstance(child, Scalar), (
                'Only scalar fields are supported in TextFileReader.')
        field_types = [
            data_type_for_dtype(dtype) for dtype in schema.field_types()]
        Reader.__init__(self, schema)
        self._reader = init_net.CreateTextFileReader(
            [],
            filename=filename,
            num_passes=num_passes,
            field_types=field_types)
        self._batch_size = batch_size

    def read(self, net):
        """
        Create op for reading a batch of rows.
        """
        blobs = net.TextFileReaderRead(
            [self._reader],
            len(self.schema().field_names()),
            batch_size=self._batch_size)
        if type(blobs) is core.BlobReference:
            blobs = [blobs]

        is_empty = net.IsEmpty(
            [blobs[0]],
            core.ScopedBlobReference(net.NextName('should_stop'))
        )

        return (is_empty, blobs)
