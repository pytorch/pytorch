## @package net_builder
# Module caffe2.python.net_builder
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, context
from caffe2.python.task import Task, TaskGroup


@context.define_context()
class NetBuilder(object):
    """
    Scope-driven mechanism for building nets, loops and conditional blocks.
    Example:
        from caffe2.python.net_builder import NetBuilder, ops
        with NetBuilder() as nb:
            c = ops.Const(5)
            d = ops.Const(0)
            with ops.loop():
                ops.stop_if(ops.LE([c, ops.Const(0)]))
                ops.Add([c, ops.Const(-1)], [c])
                with ops.If(ops.GE([c, ops.Const(3)])):
                    ops.Add([d, ops.Const(10)])
            ops.Print(c, [])
            ops.Print(d, [])
        step = core.to_execution_step(nb)
    """
    def __init__(self, name=None, _stop_blob_required=False,
                 _stop_blob=None, _fullname=None):
        nb = NetBuilder.current(required=False)
        assert not _fullname or not name, 'Cannot set both _fullname and name'
        self.name = _fullname or '/'.join(
            n for n in (nb.name if nb else None, name) if n
        )
        self._frozen = False
        self._current_net = None
        self._children = []
        self._stop_blob = _stop_blob
        self._stop_blob_required = _stop_blob_required

    def stop_blob(self):
        """
        Returns the BlobReference to the stop_blob of this NetBuilder.
        If one is not yet available, creates one.
        This function assumes that the stop_blob() will be used immediatelly
        in the current net, so it doesn't initialize it if the current net is
        the first of the builder.
        """
        if self._stop_blob is None:
            net = self.current_net()
            self._stop_blob = core.BlobReference(
                net.NextName('stop_blob'), net=net)
            if self._current_net != self._children[0]:
                self._children.insert(0, core.Net('stop_blob_init'))
                self._children[0].Const(False, blob_out=self._stop_blob)
        return self._stop_blob

    def stop_if(self, blob):
        ops.Copy(blob, self.stop_blob())
        self._current_net = None

    def _assert_mutable(self):
        assert not self._frozen, (
            'This NetBuilder (%s) has been built already.' % self.name)

    def add(self, child):
        self._assert_mutable()
        self._current_net = None
        self._children.append(child)
        # to-do : check it's not a dag net
        if isinstance(child, core.Net):
            self._current_net = child
        return child

    def current_net(self, name=None):
        self._assert_mutable()
        if self._current_net is None or name is not None:
            self.add(core.Net(name))
        return self._current_net

    def freeze(self):
        for child in self._children:
            if hasattr(child, 'freeze'):
                child.freeze()
        self._current_net = None
        self._frozen = True

    def get(self):
        self.freeze()
        return self._children

    def __exit__(self, etype, *args):
        self.freeze()
        if etype is not None:
            return
        assert (not self._stop_blob_required) or self._stop_blob is not None, (
            'This NetBuilder (%s) requires a stop condition ' % self.name +
            'to be set with `stop` or `stop_if`')

    def __str__(self):
        return self.name or 'Un-named NetBuilder'


class Operations(object):
    """
    Operations to be used in the context of a NetBuilder.
    """
    def net(self, net=None, name=None):
        """
        Retrieves the current net, or add a new net to the builder.
        Args:
            net:   If provided, add the given net to the active builder.
                   Else, returns the current Net or creates a new one as needed.
            name:  if provided, creates a new Net with given name and makes
                   it the new current net of the active builder. Cannot
                   be provided if net is provided.
        """
        assert name is None or net is None, (
            'Cannot provide both `net` and `name`.')
        if net is not None:
            NetBuilder.current().add(net)
            return net
        return NetBuilder.current().current_net(name=name)

    def __getattr__(self, op_type):
        """
        Adds an operator call to the currently active Net.
        """
        if op_type.startswith('__'):
            raise AttributeError()
        # We want hasattr to work properly even if no context is active.
        if NetBuilder.current(required=False) is None:
            raise AttributeError('No active NetBuilder.')
        return getattr(self.net(), op_type)

    def task_group(self):
        """
        Creates a local task group which will execute as the next step of
        the current NetBuilder.
        """
        from caffe2.python import task
        group = NetBuilder.current()
        with task.Cluster():
            with task.Node('local'):
                tg = task.TaskGroup()
                group.add(tg)
                return tg

    def stop(self):
        """
        Stop execution of the current execution step.
            Example:
                ops.Print(a, 0)
                ops.stop()
                ops.Print(b, 0)
            In the example, 'b' will never be printed.
        """
        return self.stop_if(ops.Const(True))

    def stop_if(self, blob):
        """
        Stop execution of the current execution step if the
        condition `blob` is met.
            Example:
                ops.Print(a, 0)
                ops.stop_if(ops.LE([x, ops.Const(0)]))
                ops.Print(b, 0)
            In the example, 'b' will only be printed if the value of scalar
            tensor 'x' lower or equal to 0.
        """
        return NetBuilder.current().stop_if(blob)

    def loop(self, iters=None, name=None):
        """
        Creates a NetBuilder that will execute in a loop as the next step of
        the current NetBuilder. If `iters` is provided, the loop will execute
        for `iters` iterations and then stop. `iters` can be a constant or a
        BlobReference. If `iters` is not provided, the loop will execute
        until `ops.stop` or `ops.stop_if` is called.
            Examples:
                a = ops.Const(5)
                with ops.loop():
                    ops.stop_if(ops.LE([a, ops.Const(0)]))
                    ops.Print(a, 0)
                    ops.Add([a, ops.Const(-1)], [a])
            Above, 'a' will be printed 5 times, with values 5 to 1.

                with ops.loop(10) as loop:
                    ops.LogInfo(loop.iter())
            This will print the numbers from 0 to 9.

                x = ops.Add([ops.Const(10), ops.Const(10)])
                with ops.loop(x) as loop:
                    ops.LogInfo(loop.iter())
            This will print the numbers from 0 to 19.
        """
        return NetBuilder.current().add(_Loop(iters, name=name))

    def stop_guard(self, has_stopped_blob=None, name=None):
        """
        Creates a NetBuilder that will execute once as the next step of the
        current NetBuilder. After execution, a bool tensor will indicate
        whether the inner execution was halted with `stop` or `stop_if`.
            Example:
                a = ops.Const(True)
                with ops.stop_guard() as sg1:
                    ops.stop_if(a)
                    ops.Print(ops.Const('did not stop'))
                b = ops.Const(False)
                with ops.stop_guard() as sg2:
                    ops.stop_if(b)
                    ops.Print(ops.Const('did not stop'))
                ops.Print(sg1.has_stopped(), [])
                ops.Print(sg2.has_stopped(), [])
            In the example, 'did not stop' will be printed once,
            followed by True and False.
        """
        return NetBuilder.current().add(
            _StopGuard(has_stopped_blob=has_stopped_blob, name=name))

    def If(self, cond, name=None):
        """
        Creates a NetBuilder that will execute once as the next step of the
        current NetBuilder if the blob `cond` is True.
            Example:
                with ops.If(ops.Const(True)):
                    ops.Print(ops.Const('Will print'))
                with ops.If(ops.Const(False)):
                    ops.Print(ops.Const('Wont print'))
            The example will print 'Will print' once.
        """
        return NetBuilder.current().add(_RunIf(cond, name=name))

    def task_init(self):
        """
        Defines operations that will be executed once at task startup.
        Useful when implementing processors, that don't have access to the Task
        top-level structure.

        This setup will be run only once, even if multiple instances of the task
        will run in parallel. For instance-local initialization, use
        `task_instance_init` instead.

            Example:
                def my_processor(rec):
                    with ops.task_init():
                        one = ops.Const(1)
                        two = ops.Const(1)
                    return Tuple(
                        ops.Add(rec[0](), zero), ops.Add(rec[1](), two))
        """
        setup = _SetupBuilder(_SetupBuilder.INIT)
        self.net().add_attribute(Task.TASK_SETUP, setup)
        return setup

    def task_exit(self):
        """
        Define operations to be executed once at task shutdown.
        Useful when implementing processors, that don't have access to the Task
        top-level structure.

        This shutdown will be run only once, after all concurrent instances of
        the task have already finished. For instance-local shutdown,
        use `task_instance_exit` instead.

            Example:
                def read_queue(queue):
                    with ops.task_exit():
                        queue.close(ops.net())
                    return queue.read(ops.net())
        """
        setup = _SetupBuilder(_SetupBuilder.EXIT)
        self.net().add_attribute(Task.TASK_SETUP, setup)
        return setup

    def task_instance_init(self):
        """
        Defines operations that will be executed once at startup of each
        instance of a task. This can be seen as "thread_local" initialization.
        It is guaranteed to run only after all `task_init` logic finishes.

        This setup will be run concurrently for each instance of a task.
        For global task initialization, use `task_init` instead.
        """
        setup = _SetupBuilder(_SetupBuilder.INIT)
        self.net().add_attribute(Task.TASK_INSTANCE_SETUP, setup)
        return setup

    def task_instance_exit(self):
        """
        Defines operations that will be executed once at shutdown of each
        instance of a task. This can be seen as "thread_local" finalization.

        This shutdown will be run concurrently for each instance of a task.
        For global task shutdown, use `task_exit` instead.
        """
        setup = _SetupBuilder(_SetupBuilder.EXIT)
        self.net().add_attribute(Task.TASK_INSTANCE_SETUP, setup)
        return setup

    def local_init(self):
        """
        Similar to `task_init`, but executes at TaskGroup's startup instead,
        before any task of the group starts executing. This will run only
        once on each node, before initialization of any task, so it can be
        used e.g. to initialize blobs shared across tasks.
        """
        setup = _SetupBuilder(_SetupBuilder.INIT)
        self.net().add_attribute(TaskGroup.LOCAL_SETUP, setup)
        return setup

    def local_exit(self):
        """
        Similar to `task_exit`, but executes at TaskGroup's exit instead,
        after all tasks of the group finished execution.
        This will run only once on each node.
        """
        setup = _SetupBuilder(_SetupBuilder.EXIT)
        self.net().add_attribute(TaskGroup.LOCAL_SETUP, setup)
        return setup

    def task_reporter(self, interval_ms=1000, name=None):
        """
        Define operations to be executed at every time interval from
        task start-up to finish. These operations are guaranteed to
        execute at least once after all other operations of the task are
        finished.

            Example:
                with ops.task_reporter(interval_ms=10000):
                    ops.LogInfo('10s elapsed')
        """
        return _ReporterBuilder(interval_ms, net=self.net(), name=name)

    def local_reporter(self, interval_ms=1000, name=None):
        """
        Similar to task_report, but operations defined within this block
        will run repeatedly for as long as any of the tasks in the current
        TaskGroup have not finished.
        """
        return _ReporterBuilder(interval_ms, name=name)


ops = Operations()


class _ReporterBuilder(NetBuilder):
    def __init__(self, interval_ms, net=None, name=None):
        NetBuilder.__init__(self, name)
        self._net = net
        self.interval_ms = interval_ms

    def __exit__(self, etype, *args):
        if etype is None:
            step = core.to_execution_step(self)
            step.RunEveryMillis(self.interval_ms)
            if self._net:
                self._net.add_attribute(Task.REPORT_STEP, step)
            else:
                TaskGroup.current().report_step(
                    step, interval_ms=self.interval_ms)
        NetBuilder.__exit__(self, etype, *args)


class _SetupBuilder(NetBuilder):
    INIT = 'init'
    EXIT = 'exit'

    def __init__(self, type, name=None):
        NetBuilder.__init__(self, name)
        self.type = type

    def setup(self, net):
        if self.type == _SetupBuilder.INIT:
            return core.to_execution_step(self)

    def exit(self, net):
        if self.type == _SetupBuilder.EXIT:
            return core.to_execution_step(self)


class _RunOnce(NetBuilder):
    def __init__(self, name=None):
        NetBuilder.__init__(self, name)

    def __exit__(self, etype, *args):
        if etype is None and self._stop_blob is not None:
            ops.stop()
        NetBuilder.__exit__(self, etype, *args)


class _StopGuard(_RunOnce):
    def __init__(self, has_stopped_blob=None, name=None):
        _RunOnce.__init__(self, name)
        self._stopped = has_stopped_blob
        self._ran = False

    def __enter__(self):
        r = _RunOnce.__enter__(self)
        self._stopped = ops.Const(True, blob_out=self._stopped)
        return r

    def __exit__(self, etype, *args):
        if etype is None:
            self._ran = True
            ops.Const(False, blob_out=self._stopped)
        _RunOnce.__exit__(self, etype, *args)

    def has_stopped(self):
        """
        Return a blob that will be set to scalar bool `True` after
        this net builder ran, iff it was halted early.
        """
        assert self._ran, 'Context not used yet.'
        return self._stopped


class _Loop(NetBuilder):
    def __init__(self, iters=None, name=None):
        NetBuilder.__init__(self, name, _stop_blob_required=True)
        if iters is not None:
            self._inc = ops.Const(1)
            self._iter = ops.Const(0)
            self._num_iters = (
                iters if isinstance(iters, core.BlobReference)
                else ops.Const(iters))
        else:
            self._num_iters = None

    def iter(self):
        assert self._num_iters is not None, (
            'This loop does not have a number of iterations.')
        assert self._iter is not None, (
            'iter() must be called from inside the loop context')
        return self._iter

    def __enter__(self):
        builder = NetBuilder.__enter__(self)
        if self._num_iters is not None:
            ops.stop_if(ops.GE([self._iter, self._num_iters]))
        return builder

    def __exit__(self, type, *args):
        if type is None and self._num_iters is not None:
            self.current_net().Add([self._iter, self._inc], [self._iter])
        NetBuilder.__exit__(self, type, *args)


class _RunIf(_RunOnce):
    def __init__(self, cond_blob=None, name=None, _already_ran=None):
        _RunOnce.__init__(self, name)
        assert cond_blob or _already_ran
        self._is_else = cond_blob is None
        if _already_ran is None:
            self._else_blob = ops.Not(cond_blob)
            self._already_ran = ops.Const(False)
        else:
            self._already_ran = _already_ran
            self._else_blob = _already_ran if cond_blob is None else (
                ops.Or([_already_ran, ops.Not(cond_blob)]))

    def __enter__(self):
        r = _RunOnce.__enter__(self)
        ops.stop_if(self._else_blob)
        ops.Const(True, blob_out=self._already_ran)
        return r

    def Elif(self, cond, name=None):
        assert not self._is_else, 'Else not allowed for an Else.'
        return NetBuilder.current().add(_RunIf(
            cond, name=name or self.name, _already_ran=self._already_ran))

    def Else(self, name=None):
        assert not self._is_else, 'Elif not allowed for an Else.'
        return NetBuilder.current().add(
            _RunIf(name=name or self.name, _already_ran=self._already_ran))
