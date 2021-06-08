## @package task
# Module caffe2.python.task

from caffe2.python import core, context
from caffe2.python.schema import Field, from_blob_list
from collections import defaultdict
from copy import copy
from future.utils import viewitems


def _merge_node_kwargs(a, b):
    # TODO(azzolini): consistency checks
    if a is None:
        return b
    if b is None:
        return a
    c = copy(a)
    c.update(b)
    return c


class Cluster(context.DefaultManaged):
    """
    Context that keeps track of all the node names used.
    Users shouldn't have to use them directly, since a Cluster is automatically
    generated at the first usage of 'Node'.
    """

    def __init__(self):
        # list instead of set to keep order
        self._nodes = []
        self._node_kwargs = {}

    def add_node(self, node):
        if str(node) not in self._nodes:
            self._nodes.append(str(node))
        self._node_kwargs[str(node)] = _merge_node_kwargs(
            node.kwargs(),
            self._node_kwargs.get(str(node)))

    def nodes(self):
        """
        Returns the list of unique node names used within this context.
        """
        return self._nodes

    def node_kwargs(self):
        return self._node_kwargs

    def __repr__(self):
        return "Cluster(nodes={}, node_kwargs={})".format(
            self.nodes(), self.node_kwargs())


class Node(context.DefaultManaged):
    """
    A Node context is used to indicate that all Tasks instantiated within will
    run on the given node name. (Only the name of the node actually counts.)
    Example:

        with TaskGroup() as tg:
            with Node('node1'):
                s1 = execution_step(...)
                Task(step=s1)
            with Node('node2'):
                s2 = execution_step(...)
            with Node('node1'):
                s3 = execution_step(...)

        In this example, all three execution steps will run in parallel.
        Moreover, s1 and s3 will run on the same node, and can see each
        others blobs.

        Additionally, a Node can be passed implementation-specific kwargs,
        in order to specify properties of the node.
    """

    def __init__(self, node='local', **kwargs):
        self._name = str(node)
        self._kwargs = kwargs
        Cluster.current().add_node(self)

    def __str__(self):
        return self._name

    def __repr__(self):
        return "Node(name={}, kwargs={})".format(self._name, self._kwargs)

    def kwargs(self):
        return self._kwargs


class WorkspaceType(object):
    """
    Determines whether tasks of a TaskGroup will run directly at the global
    workspace, which is kept alive across runs, or whether a new child
    workspace will be created for the run and destroyed afterwards.
    """
    PRIVATE = 'private'
    GLOBAL = 'global'


def get_setup_nets(key, steps_or_nets, target):
    init_net = core.Net(key + '/init')
    exit_net = core.Net(key + '/exit')
    init_nets = []
    exit_nets = []
    objs = []
    for step_or_net in steps_or_nets:
        if hasattr(step_or_net, 'get_all_attributes'):
            objs += step_or_net.get_all_attributes(key)
        elif hasattr(step_or_net, 'get_attributes'):
            objs += step_or_net.get_attributes(key)
    for obj in objs:
        # these are needed in order to allow nesting of TaskGroup, which
        # is a feature not yet implemented.
        if hasattr(obj, '_setup_used') and obj._setup_used:
            continue
        if hasattr(obj, '_setup_target') and obj._setup_target != target:
            continue
        if hasattr(obj, 'setup'):
            nets = obj.setup(init_net)
            if isinstance(nets, (list, tuple)):
                init_nets += nets
            elif isinstance(nets, (core.Net, core.ExecutionStep)):
                init_nets.append(nets)
            elif nets is not None:
                raise TypeError('Unsupported type for setup: %s' % type(nets))
            obj._setup_used = True
        if hasattr(obj, 'exit'):
            nets = obj.exit(exit_net)
            if isinstance(nets, (list, tuple)):
                exit_nets += nets
            elif isinstance(nets, (core.Net, core.ExecutionStep)):
                exit_nets.append(nets)
            elif nets is not None:
                raise TypeError('Unsupported type for setup: %s' % type(nets))
            obj._setup_used = True

    if len(init_net.Proto().op) > 0:
        init_nets.insert(0, init_net)
    if len(exit_net.Proto().op) > 0:
        exit_nets.insert(0, exit_net)
    return init_nets, exit_nets


def add_setup_steps(step, init_nets, exit_nets, name):
    if not init_nets and not exit_nets:
        return step
    steps = []
    if init_nets:
        steps.append(core.execution_step('%s:init' % name, init_nets))
    steps.append(step)
    if len(exit_nets) > 0:
        steps.append(core.execution_step('%s:exit' % name, exit_nets))
    return core.execution_step(name, steps)


class TaskGroup(context.Managed):
    """
    Context that gathers tasks which will run concurrently, potentially on
    multiple nodes. All tasks in the same node will share the same workspace
    and thus can share blobs, while tasks running in different nodes won't
    be able to directly share data.

    All tasks of the task group will start concurrently, and the task group
    will finish execution when the last task of the group finishes.

    Example:
        # suppose that s1 ... s5 are execution steps or nets.
        with TaskGroup() as tg:
            # these tasks go to default node 'local'
            Task(step=s1)
            Task(step=s2)

            with Node('n2'):
                Task(step=s3)
            with Node('n1'):
                Task(step=s4)
            with Node('n2'):
                Task(step=s5)

        # this will run all steps in parallel.
        # s1 and s2 will run at default node 'local'
        # s3 and s5 will run at node 'n2'
        # s4 will run at node 'n1'
        session.run(tg)
    """
    LOCAL_SETUP = 'local_setup'

    def __init__(self, workspace_type=None):
        self._plan_cache = None
        self._tasks = []
        self._already_used = False
        self._prev_active = None
        self._tasks_to_add = []
        self._report_nets = {}
        self._report_steps = []
        self._workspace_type = workspace_type
        self._tasks_by_node = None
        self._remote_nets = []

    def add_remote_net(self, net):
        self._remote_nets.append(net)

    def remote_nets(self):
        return self._remote_nets

    def add(self, task):
        assert not self._already_used, (
            'Cannot add Task to an already used TaskGroup.')
        assert (
            self._workspace_type is None or
            task._workspace_type is None or
            self._workspace_type == task._workspace_type)
        if task._workspace_type is None:
            task._workspace_type = (
                self._workspace_type or WorkspaceType.PRIVATE)
        if self._workspace_type is None:
            self._workspace_type = task._workspace_type
        task._notify_used()
        self._tasks.append(task)

    def tasks(self):
        for task in self._tasks_to_add:
            self.add(task)
        self._tasks_to_add = []
        self._already_used = True
        return self._tasks

    def num_registered_tasks(self):
        return len(self._tasks_to_add) + len(self._tasks)

    def used_nodes(self):
        # use list to keep order
        used = []
        for task in self._tasks + self._tasks_to_add:
            if task.node not in used:
                used.append(task.node)
        return used

    def report_step(self, step=None, node=None, interval_ms=1000):
        """
        Add a "report step" to this TaskGroup. This step will run repeatedly
        every `interval_ms` milliseconds for the duration of the TaskGroup
        execution on each of the nodes. It is guaranteed that this step
        will be run at least once after every Task in the node has finished.
        """
        step = core.to_execution_step(step)
        step.RunEveryMillis(interval_ms)
        self._report_steps.append((str(node or Node.current(node)), step))

    def report_net(self, net=None, node=None, report_interval=5):
        """
        DEPRECATED. Use report_step instead.
        """
        node = str(node or Node.current(node))
        assert net is None or node not in self._report_nets
        if node not in self._report_nets:
            self._report_nets[node] = (
                net if net else core.Net('%s/reporter' % node),
                report_interval)
        return self._report_nets[node][0]

    def tasks_by_node(self, node_remap=None):
        # tasks_by_node can't be called twice because the setup won't
        # work properly a second time.
        node_map = {}
        for task in self.tasks():
            node_map[task.node] =\
                node_remap(task.node) if node_remap else task.node
        if self._tasks_by_node is not None:
            tasks_by_node, prev_node_map = self._tasks_by_node
            assert prev_node_map == node_map, (
                'Cannot call tasks_by_node multiple times.')
            return tasks_by_node

        # now we have report_steps. report_net is deprecated
        for node, (net, interval) in viewitems(self._report_nets):
            self.report_step(net, node=node, interval_ms=interval * 1000)
        self._report_nets = {}

        tasks_by_node = defaultdict(list)
        for task in self.tasks():
            mapped_node = node_map[task.node]
            tasks_by_node[mapped_node].append(task)

        report_steps_by_node = defaultdict(list)
        for original_node, step in self._report_steps:
            report_steps_by_node[node_map[original_node]].append(step)

        grouped_by_node = TaskGroup()
        for node, tasks in viewitems(tasks_by_node):
            report_steps = report_steps_by_node[node]
            node_inits, node_exits = get_setup_nets(
                TaskGroup.LOCAL_SETUP,
                [t.get_step() for t in tasks] + report_steps,
                self)
            # shortcut for single task with no queue
            steps = report_steps
            outputs = []
            grouped_workspace_type = WorkspaceType.PRIVATE
            for task in tasks:
                step = task.get_step()
                step.SetCreateWorkspace(
                    task.workspace_type() == WorkspaceType.PRIVATE)
                if step is not None:
                    steps.append(step)
                outputs += task.outputs()
                # If any of the tasks in the node uses the global workspace,
                # then set the grouped task to use the global workspace as well
                if task.workspace_type() == WorkspaceType.GLOBAL:
                    grouped_workspace_type = WorkspaceType.GLOBAL
            if len(steps) == 0:
                steps.append(core.execution_step('empty', []))
            if len(steps) == 1:
                step = steps[0]
            else:
                step = core.execution_step(
                    '%s:body' % node, steps, concurrent_substeps=True)
            if len(node_inits) > 0 or len(node_exits) > 0:
                steps = []
                if len(node_inits) > 0:
                    steps.append(
                        core.execution_step('%s:init' % node, node_inits))
                steps.append(step)
                if len(node_exits) > 0:
                    steps.append(
                        core.execution_step('%s:exit' % node, node_exits))
                step = core.execution_step(node, steps)
            Task(
                node=node, step=step, outputs=outputs,
                name='grouped_by_node',
                group=grouped_by_node, workspace_type=grouped_workspace_type)
        self._tasks_by_node = (grouped_by_node, node_map)
        return grouped_by_node

    def to_task(self, node=None):
        node = str(Node.current(node))
        tasks = self.tasks_by_node(lambda x: node).tasks()
        if len(tasks) == 0:
            return Task()
        return tasks[0]

    def workspace_type(self):
        return self._workspace_type

    def __repr__(self):
        return "TaskGroup(tasks={}, workspace_type={}, remote_nets={})".format(
            self._tasks + self._tasks_to_add,
            self.workspace_type(),
            self.remote_nets())


class TaskOutput(object):
    """
    Represents the output of a task. An output can be a blob,
    a list of blob, or a record.
    """

    def __init__(self, names):
        self._schema = None
        self._is_scalar = False
        if isinstance(names, Field):
            self._schema = names
            names = self._schema.field_blobs()
        self._is_scalar = type(names) not in (tuple, list)
        if self._is_scalar:
            names = [names]
        self.names = names
        self._values = None

    def set(self, values, _fetch_func=None):
        assert len(values) == len(self.names)
        self._values = values
        self._fetch_func = _fetch_func

    def get(self):
        assert self._values is not None, 'Output value not set yet.'
        if self._is_scalar:
            return self._values[0]
        elif self._schema:
            return from_blob_list(self._schema, self._values)
        else:
            return self._values

    def fetch(self):
        assert self._fetch_func is not None, (
            'Cannot fetch value for this output.')
        fetched_vals = [self._fetch_func(v) for v in self._values]
        if self._is_scalar:
            return fetched_vals[0]
        elif self._schema:
            return from_blob_list(self._schema, fetched_vals)
        else:
            return fetched_vals

    def __repr__(self):
        return "TaskOutput(names={}, values={})".format(self.names, self._values)


def final_output(blob_or_record):
    """
    Adds an output to the current Task, or if no task is active,
    create a dummy task that returns the given blob or record
    to the client. This will return the value of the blob or record when
    the last task of the TaskGroup for a given node finishes.
    """
    cur_task = Task.current(required=False) or Task()
    return cur_task.add_output(blob_or_record)


class TaskOutputList(object):
    """ Keeps a list of outputs for a task """
    def __init__(self, outputs=None):
        self.outputs = outputs or []

    def names(self):
        """
        Retrive the output names.
        TODO(azzolini): make this schema-based.
        """
        names = []
        for o in self.outputs:
            names += o.names
        return names

    def set_values(self, values, _fetch_func=None):
        offset = 0
        for o in self.outputs:
            num = len(o.names)
            o.set(values[offset:offset + num], _fetch_func)
            offset += num
        assert offset == len(values), 'Wrong number of output values.'

    def __repr__(self):
        return "TaskOutputList(outputs={})".format(self.outputs)


class Task(context.Managed):
    """
    A Task is composed of an execution step and zero or more outputs.
    Tasks are executed in the context of a TaskGroup, which, in turn, can
    be run by a Session.

    Task outputs are fetched by the session at the end of the run.

    The recommended way of creating a task is by using `net_builder.ops`.
    Example:

        from net_builder import ops
        with Node('trainer'), Task(name='my_task', num_instances=2):
            with ops.task_init():
                globl = ops.Const(0)
            with ops.task_instance_init():
                local = ops.Const(0)
            with ops.loop(100):
                ops.Copy(globl, local)
            with ops.task_instance_exit():
                ops.Add([globl, local], [globl])
            with ops.task_exit():
                ops.Mul([globl, globl], [globl])

    The task above will create 2 instances that will run in parallel.
    Each instance will copy `local` to `globl` 100 times, Then Add `local`
    to `globl` once. The `Mul` will only execute once, after all the instances
    of the task have finished.
    """

    # TASK_SETUP runs once per task, before/after all
    # concurrent task instances start/finish.
    TASK_SETUP = 'task_setup'
    # Setup will run once for each instance of the task.
    TASK_INSTANCE_SETUP = 'task_instance_setup'
    REPORT_STEP = 'report_step'
    _global_names_used = set()

    @staticmethod
    def _get_next_name(node, group, name):
        basename = str(node) + '/' + str(name)
        names_used = (
            Task._global_names_used
            if group is None else
            set(t.name for t in group._tasks_to_add))
        cur_name = basename
        i = 0
        while cur_name in names_used:
            i += 1
            cur_name = '%s:%d' % (basename, i)
        return cur_name

    def __init__(
            self, step=None, outputs=None,
            workspace_type=None, group=None, node=None, name=None,
            num_instances=None):
        """
        Instantiate a Task and add it to the current TaskGroup and Node.

        Args:
           step:    If provided, this task will run this ExecutionStep.
           outputs: If provided, the task will return the provided outputs
                    to the client at completion time.
           node:    If provided, force task execution on the given node.
           name:    Name of the Task.
           num_instances: If provided, this task will be cloned num_instances
                          times at runtime, and all instances will run
                          concurrently.
        """
        if not name and isinstance(step, core.ExecutionStep):
            name = step.Proto().name
        if not name:
            name = 'task'
        # register this node name with active context
        self.node = str(Node.current(None if node is None else Node(node)))
        self.group = TaskGroup.current(group, required=False)

        self.name = Task._get_next_name(self.node, self.group, name)

        # may need to be temporarily removed later if Task used as a context
        if self.group is not None:
            self.group._tasks_to_add.append(self)

        self._already_used = False
        self._step = None
        self._step_with_setup = None
        self._outputs = []
        if step is not None:
            self.set_step(step)
        if outputs is not None:
            self.add_outputs(outputs)

        self._pipeline = None
        self._is_pipeline_context = False
        self._workspace_type = workspace_type
        self._report_net = None
        self._num_instances = num_instances

    def __enter__(self):
        super(Task, self).__enter__()

        # temporarily remove from _tasks_to_add to ensure correct order
        if self.group is not None:
            self.group._tasks_to_add.remove(self)
        self._assert_not_used()
        assert self._step is None, 'This Task already has an execution step.'
        from caffe2.python import net_builder
        self._net_builder = net_builder.NetBuilder(_fullname=self.name)
        self._net_builder.__enter__()
        return self

    def __exit__(self, type, value, traceback):
        super(Task, self).__exit__(type, value, traceback)

        self._net_builder.__exit__(type, value, traceback)
        if type is None:
            self.set_step(self._net_builder)
        if self.group is not None:
            self.group._tasks_to_add.append(self)
        self._net_builder = None

    def workspace_type(self):
        return self._workspace_type

    def _assert_not_used(self):
        assert not self._already_used, (
            'Cannot modify task since it is already been used.')

    def add_output(self, output):
        self._assert_not_used()
        output = (
            output if isinstance(output, TaskOutput) else TaskOutput(output))
        self._outputs.append(output)
        return output

    def add_outputs(self, outputs):
        self._assert_not_used()
        if type(outputs) not in (list, tuple):
            return self.add_output(outputs)
        else:
            return [self.add_output(output) for output in outputs]

    def set_step(self, step):
        self._assert_not_used()
        self._step = core.to_execution_step(step)

    def get_step(self):
        if self._step_with_setup is not None:
            return self._step_with_setup

        if self._step is None:
            self._step_with_setup = core.execution_step(self.name, [])
            return self._step_with_setup

        report_steps = [
            s
            for s in self._step.get_all_attributes(Task.REPORT_STEP)
            if not hasattr(s, '_report_step_used')
        ]
        for step in report_steps:
            step._report_step_used = True
            if not step.Proto().run_every_ms:
                step.RunEveryMillis(1000)
        task_init_nets, task_exit_nets = get_setup_nets(
            Task.TASK_SETUP, [self._step] + report_steps, self)
        instance_init_nets, instance_exit_nets = get_setup_nets(
            Task.TASK_INSTANCE_SETUP, [self._step] + report_steps, self)
        if len(self._outputs) == 0:
            output_net = core.Net('%s:output' % self.name)
            self.add_output(output_net.ConstantFill(
                [], 1, dtype=core.DataType.INT32, value=0))
            task_exit_nets.append(output_net)

        # Add instance-level report steps
        body = self._step if not report_steps else core.execution_step(
            '%s:body' % self.name, report_steps + [self._step])
        # Enclose with instance-level (thread-local) setup nets
        step_with_instance_setup = add_setup_steps(
            body, instance_init_nets, instance_exit_nets,
            self.name + ':instance')
        # Set up runtime concurrent instances
        if self._num_instances and self._num_instances > 1:
            step_with_instance_setup.SetCreateWorkspace(True)
            step_with_instance_setup = core.execution_step(
                '%s:parallel',
                [step_with_instance_setup],
                num_concurrent_instances=self._num_instances)
        # Enclose with task-level setup nets
        self._step_with_setup = add_setup_steps(
            step_with_instance_setup, task_init_nets, task_exit_nets, self.name)

        return self._step_with_setup

    def output_list(self):
        return TaskOutputList(self._outputs)

    def outputs(self):
        return self._outputs

    def _notify_used(self):
        self.get_step()
        self._already_used = True

    def __repr__(self):
        return "Task(name={}, node={}, outputs={})".format(
            self.name, self.node, self.outputs())


class SetupNets(object):
    """
    Allow to register a list of nets to be run at initialization
    and finalization of Tasks or TaskGroups.
    For example, let's say you have the following:

        init_net = core.Net('init')
        my_val = init_net.ConstantFill([], 'my_val', value=0)

        net = core.Net('counter')
        net.Add([my_val, net.Const(1),], [my_val])

        with TaskGroup() as task_group:
            with Node('trainer'):
                my_task = Task(step=[net])

    In order to have `init_net` run once before `net` runs for the
    first time, you can do one of the following:

        net.add_attribute(Task.TASK_SETUP, SetupNets([init_net]))

    or

        net.add_attribute(TaskGroup.LOCAL_SETUP, SetupNets([init_net]))

    - With Task.TASK_SETUP, init_net will run once at my_task startup.
    - With TaskGroup.LOCAL_SETUP, init_net will run once on node 'trainer',
      before any task of the task group is run on that node.

    The same SetupNets object can be added to multiple nets. It will only
    run once per Task/TaskGroup run.
    """

    def __init__(self, init_nets=None, exit_nets=None):
        self.init_nets = init_nets
        self.exit_nets = exit_nets

    def setup(self, init_net):
        return self.init_nets

    def exit(self, exit_net):
        return self.exit_nets

    def __repr__(self):
        return "SetupNets(init_nets={}, exit_nets={})".format(
            self.init_nets, self.exit_nets)
