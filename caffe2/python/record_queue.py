## @package record_queue
# Module caffe2.python.record_queue
"""
Implementation of a queue wrapper.
"""





from caffe2.python import core
from caffe2.python.dataio import Reader, Writer
from caffe2.python.schema import (
    Struct, Field, from_column_list)


class _QueueReader(Reader):
    def __init__(self, blobs_queue, schema, name=None):
        """Don't call this directly. Instead, use dataset.reader()"""
        super().__init__(schema)
        self.blobs_queue = blobs_queue
        self.name = name

    def read(self, read_net):
        with core.NameScope(read_net.NextName(self.name)):
            status = read_net.NextName()
            fields = read_net.SafeDequeueBlobs(
                self.blobs_queue, self._schema.field_names() + [status])
            return (fields[-1], fields[:-1])


class _QueueWriter(Writer):
    def __init__(self, blobs_queue, schema):
        self.blobs_queue = blobs_queue
        self.schema = schema

    def write(self, writer_net, fields):
        if isinstance(fields, Field):
            fields = fields.field_blobs()
        writer_net.CheckDatasetConsistency(
            fields, [], fields=self.schema.field_names())
        status = writer_net.NextName()
        writer_net.SafeEnqueueBlobs(
            [self.blobs_queue] + fields, fields + [status])
        return status


class RecordQueue:
    """ The class is used to feed data with some process from a reader into a
        queue and provider a reader interface for data fetching from the queue.
    """
    def __init__(self, fields, name=None, capacity=1,
                 enforce_unique_name=False, num_threads=1):
        assert isinstance(fields, list) or isinstance(fields, Struct), (
            'fields must be either a Struct or a list of raw field names.')
        if isinstance(fields, list):
            fields = from_column_list(fields)
        self.schema = fields
        self.name = name or 'queue'
        self.num_threads = num_threads
        num_blobs = len(self.schema.field_names())
        init_net = core.Net(self.name + '/init_net')
        self.blobs_queue = init_net.CreateBlobsQueue(
            [], 1,
            capacity=capacity,
            num_blobs=num_blobs,
            enforce_unique_name=enforce_unique_name)
        core.workspace.RunNetOnce(init_net)

        self.writer = _QueueWriter(self.blobs_queue, self.schema)
        reader_name = self.name + '_reader'
        self.reader = _QueueReader(self.blobs_queue, self.schema, reader_name)

        exit_net = core.Net(self.name + '/exit_net')
        exit_net.CloseBlobsQueue(self.blobs_queue, 0)
        self.exit_step = core.execution_step(
            '{}_close_step'.format(str(exit_net)),
            exit_net)

    def build(self, reader, process=None):
        """
        Build the producer_step to feed data from reader into the queue, and
        return the reader interface.
        Inputs:
            reader:           read data which will be stored in the queue.
            process:          preprocess data before enqueue.
        Outputs:
            reader:           reader to fetch the data from the queue.
            producer_step:    the step insert the data into the queue. Should be
                              run with comsume_step together.
            exit_step:        the step to close queue
            schema:           the schema for the reader.
        """
        producer_steps = []
        for i in range(self.num_threads):
            name = 'reader_' + str(i)
            net_reader = core.Net(name)
            should_stop, fields = reader.read_record(net_reader)
            step_read = core.execution_step(name, net_reader)

            name = 'queue_writer' + str(i)
            net_prod = core.Net(name)
            field_blobs = fields.field_blobs()
            if process:
                field_blobs = process(net_prod, fields).field_blobs()

            self.writer.write(net_prod, field_blobs)
            step_prod = core.execution_step(name, net_prod)
            step = core.execution_step(
                'producer_' + str(i),
                [step_read, step_prod],
                should_stop_blob=should_stop)
            producer_steps.append(step)
        producer_step = core.execution_step(
            'producers',
            producer_steps,
            concurrent_substeps=True)
        return self.reader, producer_step, self.exit_step, self.schema
