"""
Defines the base interface for reading and writing operations.

Readers/Writers are objects that produce operations that read/write sequences
of data. Each operation reads or writes a list of BlobReferences.

Readers and Writers must be implemented such that read and write operations
are atomic and thread safe.

Examples of possible Readers and Writers:
    HiveReader, HiveWriter,
    QueueReader, QueueWriter,
    DatasetReader, DatasetWriter,
    DBReader, DBWriter,

See `dataset.py` for an example of implementation.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core


class Reader(object):
    """
    Reader is a abstract class to be implemented in order to provide
    operations capable of iterating through a dataset or stream of data.

    A Reader must implement at least one operation, `read`, which
    adds operations to a net that read the next batch of data. Readers can
    optionally support the `reset` operation, which is useful when multiple
    passes over the data are required.
    """
    def read(self, read_net, batch_size=1, *args):
        """
        Add operations to read_net that will read the read batch of data
        and return a list of BlobReference representing the blobs that will
        contain the batches produced.

        Operations added to `read_net` must be thread safe and atomic, that is,
        it should be possible to clone `read_net` and run multiple instances of
        it in parallel.

        Args:
            read_net: the net that will be appended with read operations
            batch_size: number of entires to read

        Returns:
            A tuple (should_stop, fields), with:

                should_stop: BlobReference pointing to a boolean scalar
                             blob that indicates whether the read operation
                             was succesfull or whether the end of data has
                             been reached.
                fields: A tuple of BlobReference containing the latest batch
                        of data that was read.
        """
        raise NotImplementedError('Readers must implement `read`.')

    def reset(self, net):
        """Append operations to `net` that will reset the reader.

        This can be used to read the data multiple times.
        Not all readers support this operation.
        """
        raise NotImplementedError('This reader cannot be resetted.')

    def execution_step(self, reader_net_name=None, batch_size=1):
        """Create an execution step with a net containing read operators.

        The execution step will contain a `stop_blob` that knows how to stop
        the execution loop when end of data was reached.

        E.g.:

            read_step, fields = reader.execution_step()
            consume_net = core.Net('consume')
            consume_net.Print(fields[0], [])
            p = core.Plan('reader')
            p.AddStep(read_step.AddNet(consume_net))
            core.RunPlan(p)

        Args:

            reader_net_name: (optional) the name of the reader_net to be
                             created. The execution step will
                             be named accordingly.
            batch_size: the batch size

        Returns:
            A tuple (read_step, fields), with:

                read_step: A newly created execution step containing a net with
                           read operations. The step will have `stop_blob` set,
                           in order to stop the loop on end of data.
                fields: A tuple of BlobReference containing the latest batch
                        of data that was read.
        """
        reader_net = core.Net(reader_net_name or 'reader')
        should_stop, fields = self.read(reader_net, batch_size=batch_size)
        read_step = core.execution_step(
            '{}_step'.format(reader_net_name),
            reader_net,
            should_stop_blob=should_stop)
        return (read_step, fields)


class Writer(object):
    """
    Writer is a abstract class to be implemented in order to provide
    operations capable of feeding a data stream or a dataset.

    A Writer must implement 2 operations:
    `write`, which adds operations to a net that write the write batch of
    data, and `commit`, which adds operations to a net in order to indicate
    that no more data will be written.
    """

    def write(self, writer_net, fields):
        """Add operations to `writer_net` that write the next batch of data.

        Operations added to the net must be thread-safe and unique, that is:
        multiple writers must be able to write to the dataset in parallel.

        Args:
            fields: a tuple of BlobReference containing the batch of data to
                    write.
        """
        raise NotImplementedError('Writers must implement write.')

    def commit(self, finish_net):
        """Add operations to `finish_net` that signal end of data.

        This must be implemented by all Writers, but may be no-op for some
        of them.
        """
        raise NotImplementedError('Writers must implement commit.')
