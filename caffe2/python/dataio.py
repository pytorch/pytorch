## @package dataio
# Module caffe2.python.dataio
"""
Defines the base interface for reading and writing operations.

Readers/Writers are objects that produce operations that read/write sequences
of data. Each operation reads or writes a list of BlobReferences.

Readers and Writers must be implemented such that read and write operations
are atomic and thread safe.

Examples of possible Readers and Writers:
    QueueReader, QueueWriter,
    DatasetReader, DatasetWriter,

See `dataset.py` for an example of implementation.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core
from caffe2.python.schema import Field, Struct, from_blob_list
import numpy as np


class Reader(object):
    def __init__(self, schema=None):
        if schema is not None:
            assert isinstance(schema, Field)
        self._schema = schema

    def schema(self):
        """
        Return the schema associated with the Reader
        """
        assert self._schema is not None, 'Schema not provided for this reader.'
        return self._schema

    def _set_schema(self, schema):
        self._schema = schema

    def setup_ex(self, init_net, finish_net):
        """Nets to be executed once at startup and finish.
           Experimental extension. Don't use yet"""
        pass

    def read_ex(self, local_init_net, local_finish_net):
        """Experimental extension to the interface. Don't use yet"""
        read_net = core.Net('reader_body')
        return ([read_net], ) + self.read(read_net)

    def read_record_ex(self, local_init_net, local_finish_net):
        """Experimental extension to the interface. Don't use yet"""
        nets, should_stop, fields = self.read_ex(
            local_init_net, local_finish_net)
        if self._schema:
            fields = from_blob_list(self._schema, fields)
        return nets, should_stop, fields

    """
    Reader is an abstract class to be implemented in order to provide
    operations capable of iterating through a dataset or stream of data.

    A Reader must implement at least one operation, `read`, which
    adds operations to a net that read the next batch of data. Readers can
    optionally support the `reset` operation, which is useful when multiple
    passes over the data are required.
    """
    def read(self, read_net):
        """
        Add operations to read_net that will read the read batch of data
        and return a list of BlobReference representing the blobs that will
        contain the batches produced.

        Operations added to `read_net` must be thread safe and atomic, that is,
        it should be possible to clone `read_net` and run multiple instances of
        it in parallel.

        Args:
            read_net: the net that will be appended with read operations

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

    def read_record(self, read_net):
        should_stop, fields = self.read(read_net)
        if self._schema:
            fields = from_blob_list(self._schema, fields)
        return should_stop, fields

    def execution_step(self, reader_net_name=None, external_should_stop=None):
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

        Returns:
            A tuple (read_step, fields), with:

                read_step: A newly created execution step containing a net with
                           read operations. The step will have `stop_blob` set,
                           in order to stop the loop on end of data.
                fields: A tuple of BlobReference containing the latest batch
                        of data that was read.
        """
        reader_net = core.Net(reader_net_name or 'reader')
        should_stop, fields = self.read_record(reader_net)
        if external_should_stop is not None:
            should_stop = reader_net.Or([external_should_stop, should_stop])
        read_step = core.execution_step(
            '{}_step'.format(reader_net_name),
            reader_net,
            should_stop_blob=should_stop)
        return (read_step, fields)


class Writer(object):
    """
    Writer is an abstract class to be implemented in order to provide
    operations capable of feeding a data stream or a dataset.

    A Writer must implement 2 operations:
    `write`, which adds operations to a net that write the write batch of
    data, and `commit`, which adds operations to a net in order to indicate
    that no more data will be written.
    """

    _schema = None

    def schema(self):
        return self._schema

    def write(self, writer_net, fields):
        """Add operations to `writer_net` that write the next batch of data.

        Operations added to the net must be thread-safe and unique, that is:
        multiple writers must be able to write to the dataset in parallel.

        Args:
            fields: a tuple of BlobReference containing the batch of data to
                    write.
        """
        raise NotImplementedError('Writers must implement write.')

    def write_record(self, writer_net, fields):
        if isinstance(fields, Field):
            self._schema = fields
            fields = fields.field_blobs()
        self.write(writer_net, fields)

    def setup_ex(self, init_net, finish_net):
        """Experimental, don't use yet"""
        self.commit(finish_net)

    def write_ex(self, fields, local_init_net, local_finish_net, stop_blob):
        """Experimental extension to the interface. Don't use yet"""
        write_net = core.Net('write_net')
        self.write(write_net, fields)
        return [write_net]

    def write_record_ex(
            self, fields, local_init_net, local_finish_net, stop_blob=None):
        """Experimental extension to the interface. Don't use yet."""
        if isinstance(fields, Field):
            self._schema = fields
            fields = fields.field_blobs()
        if stop_blob is None:
            stop_blob = local_init_net.NextName("dequeue_status")
        write_nets = self.write_ex(
            fields, local_init_net, local_finish_net, stop_blob)
        return (write_nets, stop_blob)

    def commit(self, finish_net):
        """Add operations to `finish_net` that signal end of data.

        This must be implemented by all Writers, but may be no-op for some
        of them.
        """
        pass


class ReaderBuilder(object):
    """ Allow usage of a reader in distributed fashion. """
    def schema(self):
        raise NotImplementedError()

    def enqueue_splits(self, net, split_queue):
        raise NotImplementedError()

    def splits(self, net):
        raise NotImplementedError()

    def new_reader(self, split_reader=None):
        raise NotImplementedError()


class PipedReaderBuilder(ReaderBuilder):
    """
    ReaderBuilder that modifies underlying builder by calling `piper`
    function on each new reader produced, and return the result of
    the function. This way, it is possible to append data processing
    pipelines that will be replicated for each reader that gets created.

    E.g.:

    PipedReaderBuilder(
        ReaderBuilder(...),
        lambda reader: pipe(reader, processor=my_proc))
    """

    def __init__(self, builder, piper):
        self._builder = builder
        self._piper = piper

    def schema(self):
        return self._builder.schema()

    def enqueue_splits(self, net, split_queue):
        return self._builder.enqueue_splits(net, split_queue)

    def splits(self, net):
        return self._builder.splits(net)

    def new_reader(self, split_reader=None):
        output = self._piper(self._builder.new_reader(split_reader))
        return output if isinstance(output, Reader) else output.reader()


class Pipe(object):
    def __init__(self, schema=None, obj_key=None):
        self._num_writers = 0
        self._num_readers = 0
        self._schema = schema
        self._obj_key = obj_key

    def schema(self):
        return self._schema

    def setup(self, global_init_net):
        pass

    def reader(self):
        raise NotImplementedError()

    def writer(self):
        raise NotImplementedError()

    def num_readers(self):
        return self._num_readers

    def num_writers(self):
        return self._num_writers

    def _new_writer(self, writer_schema, writer_init_net):
        if writer_schema is not None and self._schema is None:
            self._schema = writer_schema
        self._num_writers += 1
        if self._obj_key is not None:
            writer_init_net.add_attribute(self._obj_key, self)

    def _new_reader(self, reader_init_net):
        self._num_readers += 1
        if self._obj_key is not None:
            reader_init_net.add_attribute(self._obj_key, self)


class CounterReader(Reader):
    """ Reader that produces increasing integers. """
    def __init__(self):
        Reader.__init__(self, schema=Struct(('iter', np.int64)))
        self.counter = None
        self.should_stop = None

    def setup_ex(self, global_init_net, global_finish_net):
        if self.counter is None:
            self.counter = global_init_net.CreateCounter([], init_count=0)
            self.should_stop = global_init_net.ConstantFill(
                [], shape=[], dtype=core.DataType.BOOL, value=False)

    def read_ex(self, local_init_net, local_finish_net):
        count_net = core.Net('limited_reader_counter')
        value = count_net.CountUp([self.counter], 1)
        return [count_net], self.should_stop, [value]


class ReaderWithLimit(Reader):
    """
    Reader that stops after `num_iter` calls.

    If num_iter is None it becomes just a simple reader that exports a global
    flag for "out of data".
    """
    def __init__(self, reader, num_iter=1):
        Reader.__init__(self, schema=reader._schema)
        self.reader = reader
        self.counter = None
        self.num_iter = num_iter
        net = core.Net('reader_with_limit')
        self._data_finished = net.AddExternalInput(
            net.NextName('data_finished'))
        if self.num_iter is not None:
            self.counter = net.AddExternalInput(net.NextName('counter'))

    def setup_ex(self, global_init_net, global_finish_net):
        if self.counter:
            global_init_net.CreateCounter(
                [], [self.counter], init_count=int(self.num_iter))
        self.reader.setup_ex(global_init_net, global_finish_net)
        global_init_net.ConstantFill(
            [], [self._data_finished],
            shape=[], value=False, dtype=core.DataType.BOOL)

    def read_ex(self, local_init_net, local_finish_net):
        """ 1. check if we reached number of iterations and populate the same
        should_stop blob """
        count_net = core.Net('limited_reader_counter')
        if self.counter:
            should_stop = count_net.CountDown([self.counter], 1)
        else:
            should_stop = count_net.ConstantFill(
                [], 1,
                shape=[], value=False, dtype=core.DataType.BOOL)

        """ 2. call original reader """
        nets, local_data_finished, fields = self.reader.read_ex(
            local_init_net, local_finish_net)
        self._set_schema(self.reader._schema)

        """ 3. check if original reader is done. """
        check_done_net = core.Net('limited_reader_post')
        # copy to the same blob as the counter output to trigger reader
        # stopping
        check_done_net.Copy(local_data_finished, should_stop)
        # update global flag that underlying reader is done
        check_done_net.Or([self._data_finished, local_data_finished],
                          [self._data_finished])

        # this relies on `should_stop` being called after each net.
        return [count_net] + nets + [check_done_net], should_stop, fields

    def data_finished(self):
        """
        Return a blob that can be checked after the end of the reading task,
        which will contain a scalar float indicating whether the underlying
        reader has been exhausted (True) or whether we stopped because reached
        the limit of iterations (False).
        """
        return self._data_finished


def CountUntil(num_iter):
    return ReaderWithLimit(CounterReader(), num_iter)
