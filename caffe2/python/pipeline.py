from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, queue_util
from caffe2.python.dataio import Reader, Writer


def processor_step(
        reader, writer, num_threads=1, processor=None, name='processor'):
    """
    Given a reader and a writer, couple them through a processor, running
    across multiple threads.

    Args:
        reader:      an instance of dataio.Reader
        writer:      an instance of dataio.Wrier
        num_threads: number of processing threads
        processor:   if provided, a function taking form:
                     (nets, out_record) = processor(record)
                     where `record` is a schema.Struct containing the input,
                     `nets` is the list of nets doing the transformation, and
                     `out_record` is a schema.Struct with transformed data;
        name:        Name to be given to nets and execution steps created.

    Returns:
        Execution step that runs all threads of the processor in parallel.
    """
    assert isinstance(reader, Reader)
    assert isinstance(writer, Writer)
    global_init_net = core.Net(name + '_producer_global_init')
    global_exit_net = core.Net(name + '_producer_global_exit')

    reader.setup_ex(global_init_net, global_exit_net)
    writer.setup_ex(global_init_net, global_exit_net)

    def default_processor(fields):
        return [], fields

    if processor is None:
        processor = default_processor

    steps = []
    for thread_id in range(num_threads):
        init_net = core.Net(name + "_init_net_%d" % thread_id)
        exit_net = core.Net(name + "_exit_net_%d" % thread_id)

        read_nets, status, rec = reader.read_record_ex(init_net, exit_net)
        process_nets, rec = processor(rec)
        write_nets, _ = writer.write_record_ex(rec, init_net, exit_net, status)

        step = core.execution_step(
            name + "_thread_%d" % thread_id, [
                core.execution_step(name + "_init_step", init_net),
                core.execution_step(
                    name + "_worker_step",
                    list(read_nets) + list(process_nets) + list(write_nets),
                    should_stop_blob=status
                ), core.execution_step(name + "_exit_step", exit_net)
            ]
        )
        steps.append(step)

    return core.execution_step(
        "sender_step", [
            core.execution_step('init_step', global_init_net),
            core.execution_step(
                "sender_steps", steps, concurrent_substeps=True),
            core.execution_step('finish_step', global_exit_net),
        ]
    )


class LocalPipeline(object):
    """
    Create a data processing pipeline consisting of a sequence of
    multi-threaded processors communicating through queues.
    """
    def __init__(self):
        self.tasks = []
        self.init_net = core.Net('worker_init')

    def create_queue(self, capacity, schema):
        """
        Create a queue that will be used to communicate between processors.

        Args:
            capacity: max number of records in the queue
            schema:   a schema.Struct representing the schema of a record in
                      the queue.

        Returns:
            A QueueWrapper containing a queue.
        """
        return queue_util.QueueWrapper(self.init_net, capacity, schema)

    def add_task(self, task):
        """
        Add a task to the pipeline.
        This task will run in parallel to other tasks in the pipeline.
        """
        self.tasks.append(task)

    def link(self, reader, writer, num_threads=1, processor=None):
        """
        Add a task that will read from `reader`, and write to `writer`.
        See function `processor_step` above for description of the arguments.
        """
        self.add_task(processor_step(reader, writer, num_threads, processor))

    def get_step(self):
        """
        Create and return a Caffe2 execution step that will run all the tasks
        of this pipeline in parallel.
        """
        return core.execution_step('worker_step', [
            core.execution_step('worker_init', self.init_net),
            core.execution_step(
                'tasks_step', self.tasks, concurrent_substeps=True)
        ])

    def get_step_and_output(self):
        """
        Return a tuple (execution_step, output) to be used as one of the tasks
        in a distributed pipeline.
        """
        output = self.init_net.ConstantFill([], value=0.0)
        return self.get_step(), [output]
