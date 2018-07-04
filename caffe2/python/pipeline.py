## @package pipeline
# Module caffe2.python.pipeline
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, queue_util
from caffe2.python.dataio import Reader, Writer
from caffe2.python.net_builder import NetBuilder, ops
from caffe2.python.schema import as_record, Field
from caffe2.python.task import Node, Task, TaskGroup


class Output(object):
    """
    Represents the result of a processor function. A processor can either
    return an Output, or it can return a record, in which case an Output will be
    created for it afterwards.
    """
    def __init__(self, nets=None, record=None, should_stop=None):
        builder_children = NetBuilder.current().get()
        assert nets is None or len(builder_children) == 0, (
            'Cannot both use `ops` syntax and return a list of nets.')
        if nets is None:
            nets = builder_children
        if isinstance(nets, core.Net):
            nets = [nets]
        self.nets = [] if nets is None else list(nets)
        self.record = None if record is None else as_record(record)
        self.should_stop = should_stop


DEFAULT_QUEUE_CAPACITY = 100


def _init_output(output, capacity, global_init_net, global_exit_net):
    if output is None:
        out_queue = queue_util.Queue(
            capacity=(
                capacity if capacity is not None
                else DEFAULT_QUEUE_CAPACITY))
        writer = out_queue.writer()
    elif isinstance(output, Writer):
        assert capacity is None, 'capacity would not be used.'
        out_queue = None
        writer = output
    elif hasattr(output, 'writer'):
        assert capacity is None, 'capacity would not be used.'
        out_queue = output
        writer = output.writer()
    else:
        raise ValueError('output must be a reader, queue or stream.')
    writer.setup_ex(global_init_net, global_exit_net)
    return out_queue, writer


def make_processor(processor):
    if processor is None:
        return lambda rec: rec
    elif isinstance(processor, core.Net):
        return NetProcessor(processor)
    else:
        return processor


def normalize_processor_output(output):
    """
    Allow for processors to return results in several formats.
    TODO(azzolini): simplify once all processors use NetBuilder API.
    """
    if isinstance(output, Output):
        """ Processor returned an Output. """
        return output
    elif isinstance(output, Field):
        """ Processor returned a record. """
        return Output(record=output)
    elif isinstance(output, tuple):
        is_record_and_blob = (
            len(output) == 2 and
            isinstance(output[0], Field) and
            isinstance(output[1], core.BlobReference))
        if is_record_and_blob:
            """ Processor returned (record, stop_blob) """
            return Output(None, *output)
        else:
            """ Processor returned (nets, record, stop_blob) """
            return Output(*output)
    else:
        """ Processor returned nets, no output """
        return Output(output)


def pipe(
        input, output=None, num_threads=1, processor=None, name=None,
        capacity=None, group=None, num_runtime_threads=1):
    """
    Given a Reader, Queue or DataStream in `input`, and optionally, a Writer,
    Queue or DataStream in `output`, creates a Task that, when run, will
    pipe the input into the output, using multiple parallel threads.
    Additionally, if a processor is given, it will be called between reading
    and writing steps, allowing it to transform the record.

    Args:
        input:       either a Reader, Queue or DataStream that will be read
                     until a stop is signaled either by the reader or the
                     writer.
        output:      either a Writer, a Queue or a DataStream that will be
                     writen to as long as neither reader nor writer signal
                     a stop condition. If output is not provided or is None,
                     a Queue is created with given `capacity` and writen to.
        num_threads: number of concurrent threads used for processing and
                     piping. If set to 0, no Task is created, and a
                     reader is returned instead -- the reader returned will
                     read from the reader passed in and process it.
                     ** DEPRECATED **. Use `num_runtime_threads` instead.
                     This option will be removed once all readers/processors
                     support `num_runtime_threads`.
        processor:   (optional) function that takes an input record and
                     optionally returns a record; this will be called
                     between read and write steps. If the processor does
                     not return a record, a writer will not be instantiated.
                     Processor can also be a core.Net with input and output
                     records properly set. In that case, a NetProcessor is
                     instantiated, cloning the net for each of the threads.
        name:        (optional) name of the task to be created.
        capacity:    when output is not passed, a queue of given `capacity`
                     is created and written to.
        group:       (optional) explicitly add the created Task to this
                     TaskGroup, instead of using the currently active one.
        num_runtime_threads: Similar to `num_threads`, but instead of expanding
                     the tasks with a `for` loop in python, does that at
                     runtime. This is preferable to `num_threads`, but some
                     processors/readers still require to be called multiple
                     times in python.

    Returns:
        Output Queue, DataStream, Reader, or None, depending on the parameters
        passed.
    """
    result, _ = _pipe_step(
        input, output, num_threads, processor, name, capacity, group,
        num_runtime_threads)
    return result


def pipe_and_output(
        input, output=None, num_threads=1, processor=None, name=None,
        capacity=None, group=None, num_runtime_threads=1, final_outputs=None):
    """
    Similar to `pipe`, with the additional ability for the pipe Task to
    return output values to the `Session` once done.

    Returns:
        Tuple (out_queue, *task_outputs)
            out_queue:    same as return value of `pipe`.
            task_outputs: TaskOutput object, fetchable from the client after
                          session.run() returns.
    """
    assert num_threads > 0
    result, task = _pipe_step(
        input, output, num_threads, processor, name, capacity, group,
        num_runtime_threads, final_outputs)
    output = None
    if final_outputs is not None:
        output = task.outputs()
        if type(final_outputs) not in (list, tuple):
            output = output[0]
    return result, output


def processor_name(processor):
    if hasattr(processor, 'name'):
        return processor.name
    if hasattr(processor, 'func_name'):
        if processor.func_name == '<lambda>':
            return processor.__module__
        if hasattr(processor, 'im_class'):
            return '%s.%s' % (processor.im_class.__name__, processor.func_name)
        return processor.func_name
    return processor.__class__.__name__


def _runtime_threads_task(name, group, final_outputs, reader, num_threads,
                          output, capacity):
    node_name = str(Node.current())
    profiler_name = "{0}/{1}/{2}/{3}/{4}".format(
        node_name,
        "pipe",
        name,
        processor_name(input) if input else "NoInput",
        processor_name(output) if output else "NoOutput")

    with Task(name=name, group=group, outputs=final_outputs,
              num_instances=num_threads) as task:
        global_exit_net = core.Net('pipe:exit')
        global_init_net = core.Net('pipe:init')
        reader.setup_ex(global_init_net, global_exit_net)

        init_net = core.Net('pipe:instance:init')
        exit_net = core.Net('pipe:instance:exit')
        read_nets, status, rec = reader.read_record_ex(init_net, exit_net)
        init_net.ConstantFill(
            [], [status],
            shape=[],
            value=False,
            dtype=core.DataType.BOOL
        )

        if rec is not None:
            out_queue, writer = _init_output(
                output, capacity, global_init_net, global_exit_net)
            write_nets, _ = writer.write_record_ex(
                rec, init_net, exit_net, status)
        else:
            out_queue = None
            write_nets = []

        with ops.task_init():
            ops.net(global_init_net)
        with ops.task_instance_init():
            ops.net(init_net)

        timer_start_net = core.Net('timer_start')
        timer = timer_start_net.TimerBegin([], counter_name=profiler_name)
        timer_end_net = core.Net('timer_end')
        timer_end_net.TimerEnd(timer, [])

        ops.net(core.execution_step(
            'body',
            [timer_start_net] + list(read_nets) + list(write_nets) +
            [timer_end_net],
            should_stop_blob=status))
        ops.net(timer_end_net)

        with ops.task_instance_exit():
            ops.net(exit_net)
        with ops.task_exit():
            ops.net(global_exit_net)

    return out_queue, task


def _static_threads_task(name, group, final_outputs, reader, num_threads,
                         output, capacity):
    node_name = str(Node.current())
    profiler_name = "{0}/{1}/{2}/{3}/{4}".format(
        node_name,
        "pipe",
        name,
        processor_name(input) if input else "NoInput",
        processor_name(output) if output else "NoOutput")

    with Task(name=name, group=group, outputs=final_outputs) as task:
        global_exit_net = core.Net('exit')
        global_init_net = core.Net('init')
        reader.setup_ex(global_init_net, global_exit_net)

        out_queue = None
        writer = None

        steps = []
        for thread_id in range(num_threads):
            with NetBuilder(name='t:%d' % thread_id) as nb:
                init_net = core.Net('init')
                exit_net = core.Net('exit')
                read_nets, status, rec = reader.read_record_ex(
                    init_net, exit_net)
                init_net.ConstantFill(
                    [], [status],
                    shape=[],
                    value=False,
                    dtype=core.DataType.BOOL
                )

                if rec is not None:
                    if writer is None:
                        # hack so that the out queue gets the right name prefix
                        # (otherwise they would be prefixed with the thread id)
                        with NetBuilder(_fullname=task.name):
                            out_queue, writer = _init_output(
                                output, capacity, global_init_net,
                                global_exit_net)
                    write_nets, _ = writer.write_record_ex(
                        rec, init_net, exit_net, status)
                else:
                    write_nets = []

                timer_start_net = core.Net('timer_start')
                timer = timer_start_net.TimerBegin([], counter_name=profiler_name)
                timer_end_net = core.Net('timer_end')
                timer_end_net.TimerEnd(timer, [])

                ops.net(init_net)
                ops.net(core.execution_step(
                    'body',
                    [timer_start_net] + list(read_nets) + list(write_nets) +
                    [timer_end_net],
                    should_stop_blob=status))
                ops.net(timer_end_net)
                ops.net(exit_net)
            steps.append(core.to_execution_step(nb))
        ops.net(global_init_net)
        ops.net(core.execution_step('body', steps, concurrent_substeps=True))
        ops.net(global_exit_net)
    return out_queue, task


def _pipe_step(
        input, output=None, num_threads=1, processor=None, name=None,
        capacity=None, group=None, num_runtime_threads=None, final_outputs=None):
    """
    """
    assert num_threads <= 1 or num_runtime_threads <= 1, (
        'Only one of num_threads or num_runtime_threads must be set.')

    if isinstance(input, Reader):
        reader = input
    elif hasattr(input, 'reader'):
        reader = input.reader()
    else:
        raise ValueError('in must be a reader, queue or stream.')

    if processor is not None:
        reader = ProcessingReader(reader, processor)

    if num_threads == 0 or num_runtime_threads == 0:
        assert output is None
        return reader, None

    if name is None and processor is not None:
        name = processor_name(processor)
    if name is None and output is not None:
        name = 'pipe_into:%s' % processor_name(output)
    if name is None:
        name = 'pipe_from:%s' % processor_name(input)

    if num_threads > 1:
        return _static_threads_task(
            name, group, final_outputs, reader, num_threads, output, capacity)
    else:
        return _runtime_threads_task(
            name, group, final_outputs, reader, num_runtime_threads, output,
            capacity)


class ProcessingReader(Reader):
    """
    Reader that reads from an upstream reader, calls the processor, and returns
    the processed record.
    """
    def __init__(self, reader, processor):
        Reader.__init__(self)
        self.reader = reader
        self.processor = make_processor(processor)

    def setup_ex(self, init_net, finish_net):
        self.reader.setup_ex(init_net, finish_net)

    def read_ex(self, init_net, exit_net):
        read_nets, status, rec = self.reader.read_record_ex(init_net, exit_net)
        # We don't use status as stop_blob of NetBuilder it's not guarantee that
        # it would end up being the true stob_blob. For example,
        # ReaderWithLimitBase doesn't pass the status through but rather copy
        # from it.
        with NetBuilder() as nb:
            # Current NetBuilder is optionally used inside the processor,
            # then its children are retrived inside of
            # normalize_processor_output.
            # Once readers and writers also use NetBuilder,
            # this logic will be more natural.
            result = normalize_processor_output(self.processor(rec))
        read_nets += result.nets
        if result.should_stop or nb._stop_blob:
            stop_net = core.Net('stop_net')
            if result.should_stop:
                stop_net.Or([status, result.should_stop], [status])
            if nb._stop_blob:
                stop_net.Or([status, nb._stop_blob], [status])
            read_nets.append(stop_net)
        if hasattr(self.processor, 'setup'):
            init_net.add_attribute(TaskGroup.LOCAL_SETUP, self.processor)
        self._set_schema(result.record)
        fields = result.record.field_blobs() if result.record else None
        return read_nets, status, fields


class NetProcessor(object):
    """
    Processor that clones a core.Net each time it's called, executing
    the cloned net as the processor. It requires the Net to have input
    and (optionally) output records set, with net.set_input_record() and
    net.set_output_record().
    """
    def __init__(self, net, stop_signal=None, thread_init_nets=None, name=None):
        assert isinstance(net, core.Net)
        assert stop_signal is None or isinstance(
            stop_signal, core.BlobReference)
        self.name = name or str(net)
        self.thread_init_nets = thread_init_nets or []
        self.net = net
        self._stop_signal = stop_signal
        self._blob_maps = []
        self._frozen = False
        self._cloned_init_nets = []

    def setup(self, init_net):
        self._frozen = True
        cloned_init_nets = self._cloned_init_nets
        self._cloned_init_nets = []
        return cloned_init_nets

    def __call__(self, rec):
        assert not self._frozen
        prefix = NetBuilder.current().name + '/'
        blob_remap = {}
        for net in self.thread_init_nets:
            new_net, _ = core.clone_and_bind_net(
                net, str(net) + prefix, prefix, blob_remap)
            self._cloned_init_nets.append(new_net)

        new_net, remappings = core.clone_and_bind_net(
            self.net, str(self.net) + prefix, prefix, blob_remap, rec)

        if self._stop_signal is None:
            stop_signal = None
        elif str(self._stop_signal) in remappings:
            stop_signal = core.BlobReference(
                remappings[str(self._stop_signal)],
                net=new_net)
        else:
            stop_signal = self._stop_signal

        self._blob_maps.append(remappings)
        return Output([new_net], new_net.output_record(), stop_signal)

    def blob_maps(self):
        self._frozen = True
        return self._blob_maps
