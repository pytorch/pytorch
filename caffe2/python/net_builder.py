from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, context


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
    def __init__(self, name=None, _stop_blob_required=False):
        self._name = name or ''
        self._prefix = name + '/' if name else ''
        self._frozen = False
        self._current_net = None
        self._children = []
        self._stop_blob = None
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
                self._children.insert(0, core.Net(
                    self._prefix + 'stop_blob_init'))
                self._children[0].Const(False, blob_out=self._stop_blob)
        return self._stop_blob

    def stop_if(self, blob):
        ops.Copy(blob, self.stop_blob())
        self._current_net = None

    def _assert_mutable(self):
        assert not self._frozen, (
            'This NetBuilder (%s) has been built already.' % self._name)

    def add(self, child):
        self._assert_mutable()
        self._current_net = None
        self._children.append(child)
        # to-do : check it's not a dag net
        if isinstance(child, core.Net):
            self._current_net = child
        return child

    def current_net(self):
        self._assert_mutable()
        if self._current_net is None:
            self.add(core.Net(self._prefix + 'net'))
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
            'This NetBuilder (%s) requires a stop condition ' % self._name +
            'to be set with `stop` or `stop_if`')


class Operations(object):
    """
    Operations to be used in the context of a NetBuilder.
    """
    def net(self, net=None):
        """
        Retrieves the current net, or add a new net to the builder.
        """
        if net is not None:
            NetBuilder.current().add(net)
            return net
        return NetBuilder.current().current_net()

    def __getattr__(self, op_type):
        """
        Adds an operator call to the currently active Net.
        """
        if op_type.startswith('__'):
            raise AttributeError()
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

    def loop(self):
        """
        Creates a NetBuilder that will execute in a loop as the next step of
        the current NetBuilder.
            Example:
                a = ops.Const(5)
                with ops.loop():
                    ops.stop_if(ops.LE([a, ops.Const(0)]))
                    ops.Print(a, 0)
                    ops.Add([a, ops.Const(-1)], [a])
            In the example, 'a' will be printed 5 times, with values 5 to 1.
        """
        return NetBuilder.current().add(NetBuilder(_stop_blob_required=True))

    def stop_guard(self, has_stopped_blob=None):
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
            _StopGuard(has_stopped_blob=has_stopped_blob))

    def If(self, cond):
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
        return NetBuilder.current().add(_RunIf(cond))


ops = Operations()


class _RunOnce(NetBuilder):
    def __init__(self, name=None):
        NetBuilder.__init__(self, name)

    def __exit__(self, *args):
        if self._stop_blob is not None:
            ops.stop()
        NetBuilder.__exit__(self, *args)


class _StopGuard(_RunOnce):
    def __init__(self, name=None, has_stopped_blob=None):
        _RunOnce.__init__(self, name)
        self._stopped = has_stopped_blob
        self._ran = False

    def __enter__(self):
        r = _RunOnce.__enter__(self)
        self._stopped = ops.Const(True, blob_out=self._stopped)
        return r

    def __exit__(self, *args):
        self._ran = True
        ops.Const(False, blob_out=self._stopped)
        _RunOnce.__exit__(self, args)

    def has_stopped(self):
        """
        Return a blob that will be set to scalar bool `True` after
        this net builder ran, iff it was halted early.
        """
        assert self._ran, 'Context not used yet.'
        return self._stopped


class _RunIf(_RunOnce):
    def __init__(self, cond_blob, name=None):
        _RunOnce.__init__(self, name)
        self._cond_blob = cond_blob

    def __enter__(self):
        r = _RunOnce.__enter__(self)
        ops.stop_if(self._cond_blob)
        return r
