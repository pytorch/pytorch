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
    """
    Reader is an abstract class to be implemented in order to provide
    operations capable of iterating through a dataset or stream of data.

    A Reader must implement at least one operation, `read`, which
    adds operations to a net that read the next batch of data. Readers can
    optionally support the `reset` operation, which is useful when multiple
    passes over the data are required.
    """
    def __init__(self, schema=None):
        if schema is not None:
            assert isinstance(schema, Field)
        self._schema = schema

    def schema(self):
        assert self._schema is not None, 'Schema not provided for this reader.'
        return self._schema

    def _set_schema(self, schema):
        self._schema = schema

    def setup_ex(self, init_net, finish_net):
        """Setup nets to run at task initialization and cleanup time.

        Args:
            global_init_net: A net invoked at task init time.
            global_finish_net: A net invoked at task cleanup time.
        """
        pass

    def read_ex(self, local_init_net, local_finish_net):
        read_net = core.Net('reader_body')
        return ([read_net], ) + self.read(read_net)

    def read_record_ex(self, local_init_net, local_finish_net):
        nets, should_stop, fields = self.read_ex(
            local_init_net, local_finish_net)
        if self._schema:
            fields = from_blob_list(self._schema, fields)
        return nets, should_stop, fields

    def read(self, read_net):
        """Append operations to read_net that will read a batch from the
        underlying data soruce.

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

    def setup(self, **kwargs):
        """
        Optionally, perform one-time setup before calling new_reader().
        Subclass should make sure this function is only called once.
        """
        raise NotImplementedError()

    def new_reader(self, **kwargs):
        raise NotImplementedError()


class PipedReaderBuilder(ReaderBuilder):
    """ReaderBuilder that modifies underlying builder by calling `piper`
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

    def setup(self, **kwargs):
        self._builder.setup(**kwargs)

    def new_reader(self, **kwargs):
        # Passing everything down since you could wrap a PipedReaderBuilder in
        # another PipedReaderBuilder
        output = self._piper(
            reader=self._builder.new_reader(**kwargs),
            **kwargs
        )
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


class ReaderWithLimitBase(Reader):
    """Abstract Reader constrained by certain conditions.

    Base class for Reader classes which check for certain conditions to stop
    further processing (e.g. max number of iterations or time limit).
    Also produces a boolean blob (data_finished) that can be used to see if
    the reader exausted all input data (true) or stopped for another reason
    (false).
    """

    def __init__(self, reader):
        Reader.__init__(self, schema=reader._schema)
        self.reader = reader
        self.net = core.Net('reader_with_limit')
        self._data_finished = self.net.AddExternalInput(
            self.net.NextName('data_finished'))
        self.should_stop = None

    def setup_ex(self, global_init_net, global_finish_net):
        global_init_net.ConstantFill(
            [], [self._data_finished],
            shape=[], value=False, dtype=core.DataType.BOOL)
        self.reader.setup_ex(global_init_net, global_finish_net)
        self.setup_limiter(global_init_net, global_finish_net)

    def read_ex(self, local_init_net, local_finish_net):
        """Reads from an underlying Reader class, but may stop due to additional
        constraints.

        Build and return network(s) to read data from a Reader with
        additional constraints, depending on which derived class is used.
        Derived classes implement setup_limited and check_limiter_condition
        which determine the nature of the constraint imposed on the reader,
        e.g. iteration limits or time limit.

        Args:
            local_init_net: A net invoked at task instance init time (Once per
                parallel thread).
            local_finish_net: A net invoked at task instance cleanup time (Once
                per parallel thread).
        """

        # Check if limiting constraint is met.
        stop_condition_net = core.Net('limited_reader_condition')
        should_stop = self.check_limiter_condition(stop_condition_net)

        # Call original reader.
        nets, local_data_finished, fields = self.reader.read_ex(
            local_init_net, local_finish_net)
        self._set_schema(self.reader._schema)

        # Check if original reader is done.
        check_done_net = core.Net('limited_reader_post')
        # Copy to the same blob as the counter output to trigger reader
        # stopping - this is ok because execution will check should_stop_blob
        # after every single operation, so it has already been checked on this
        # iteration by this point.
        check_done_net.Copy(local_data_finished, should_stop)
        # Update externally-accessible flag indicating if reader is done
        check_done_net.Or([self._data_finished, local_data_finished],
                          [self._data_finished])

        return [stop_condition_net] + nets + [check_done_net], should_stop, fields

    def setup_limiter(self, global_init_net, global_finish_net):
        """Configure task level init/cleanup nets required to implement limit
        condition. Must be implemented by subclass.

        Args:
            global_init_net: A net invoked at task init time.
            global_finish_net: A net invoked at task cleanup time.
        """
        raise NotImplementedError("Subclass must implement `setup_limiter`")

    def check_limiter_condition(self, stop_condition_net):
        """Configure a net that is invoked between reading batches to see if
        limit condition is met. Must be implemented by subclass.

        Args:
            stop_condition_net: A net invoked to evaluate an early termination
                condition.
        """
        raise NotImplementedError("Subclass must implement `check_limiter_condition")

    def data_finished(self):
        """
        Return a blob that can be checked after the end of the reading task,
        which will contain a scalar float indicating whether the underlying
        reader has been exhausted (True) or whether we stopped because reached
        the limit of iterations (False).
        """
        return self._data_finished


class ReaderWithLimit(ReaderWithLimitBase):
    """Reader that stops after `num_iter` batches.

    If `num_iter` <= 0 or is None, reverts to an unconstrained reader that
    exports a boolean blob indicating that the reader has exhausted
    the data steam.
    """
    def __init__(self, reader, num_iter=1):
        """Class initializer.

        Args:
            reader: The underlying reader object doing the actual read.
            num_iter: Number of batches to read. If `None`,
                the class reverts to a normal reader except that it also
                produces a data_finished blob as a side effect to indicate
                whether the input stream is exhausted.
        """
        super(ReaderWithLimit, self).__init__(reader)
        self.counter = None
        self.num_iter = num_iter
        if self.num_iter is not None:
            self.counter = self.net.AddExternalInput(
                self.net.NextName('counter'))

    def setup_limiter(self, global_init_net, global_finish_net):
        if self.counter:
            global_init_net.CreateCounter(
                [], [self.counter], init_count=int(self.num_iter))

    def check_limiter_condition(self, stop_condition_net):
        if self.counter:
            return stop_condition_net.CountDown([self.counter], 1)
        else:
            return stop_condition_net.ConstantFill(
                [], 1,
                shape=[], value=False, dtype=core.DataType.BOOL)


def CountUntil(num_iter):
    return ReaderWithLimit(CounterReader(), num_iter)


class ReaderWithTimeLimit(ReaderWithLimitBase):
    """Reader that stops after `duration` seconds.

    If `duration` <= 0 or is None, reverts to an unconstrained reader that
    exports a boolean blob indicating that the reader has exhausted
    the data steam.
    """
    def __init__(self, reader, duration=0):
        """Class initializer.

        Args:
            reader: The underlying reader object doing the actual read.
            duration: Number of seconds to read. If un-specified, None, or <= 0,
                the class reverts to a normal reader except that it also
                produces a data_finished blob as a side effect to indicate
                whether the input stream is exhausted.
        """
        super(ReaderWithTimeLimit, self).__init__(reader)

        self.timer = None
        self.duration = duration
        self.duration_ns_blob = None

    def setup_limiter(self, global_init_net, global_finish_net):
        if self.duration is not None and self.duration > 0:
            duration_ns = int(self.duration * (10**9))

            self.timer = global_init_net.TimerBegin(
                [], counter_name='epoch_timer')
            start_time = global_init_net.TimerGet(self.timer)
            self.duration_ns_blob = global_init_net.ConstantFill(
                [start_time], value=duration_ns)

            global_finish_net.TimerEnd([self.timer], [])

    def check_limiter_condition(self, stop_condition_net):
        if self.duration:
            time_elapsed = stop_condition_net.TimerGet(self.timer)
            return stop_condition_net.GE(
                [time_elapsed, self.duration_ns_blob], str(self.should_stop))
        else:
            return stop_condition_net.ConstantFill(
                [], 1, shape=[], value=False, dtype=core.DataType.BOOL
            )
