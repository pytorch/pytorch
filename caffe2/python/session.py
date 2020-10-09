## @package session
# Module caffe2.python.session






from caffe2.python import core, workspace
from caffe2.python.task import Cluster, Task, TaskGroup, WorkspaceType


class CompiledRunnable(object):
    """ Wrapper for compiled runnable returned from session.compile() """
    def __init__(self, obj, session_class):
        self.obj = obj
        self.session_class = session_class


class Session(object):
    """
    Allows to run Nets, ExecutionSteps, Plans, Tasks and TaskGroups.
    A session can potentially run in multiple nodes concurrently.


    Example:
        from core import Net
        from caffe2.python.task import Task, TaskGroup, WorkspaceType

        net = Net('test1')
        net.Add([net.Const(1), net.Const(2)])

        net2 = net.Clone()
        step = core.execution_step('step1', [net2])

        with TaskGroup(WorkspaceType.GLOBAL) as init_tg:
            with Node('node1'):
                n1setup = net.Net('n1setup')
                n1msg = n1setup.Const('Hello from node 1.')
                Task(step=n1setup)

        with TaskGroup() as private_tg:
            with Node('node1'):
                n1 = net.Net('n1')
                n1.Print(n1msg, 0)
                Task(step=n1)
            with Node('node2'):
                n2 = net.Net('n2')
                n2.Print(n2.Const('Hello from node 2.'), 0)
                Task(step=n2)

        session = LocalSession()
        session.run(net)
        session.run(step)
        session.run(init_tg)
        session.run(private_tg)


    Global Workspace:
        At the beginning of the session, a global workspace is created and kept
        alive for the duration of the session.


    Private Workspace:
        Tasks can be run either directly on the global workspace, or they can
        instantiate a private child workspace that is released after each run.

    Blob visibility:
        Tasks running in different nodes in parallel will always run under
        different workspaces, so it must be assumed that they won't be able to
        access each other's blobs. Tasks running on the same node will follow
        Workspace hierarchy rules: tasks running on separate private workspaces
        will only be able to share blobs defined on a common parent Workspace.
    """

    _compiled_cache = {}

    def __init__(self):
        self._open = True

    def is_open(self):
        return self._open

    @classmethod
    def compile(cls, runnable, workspace_type=None, setup_net_list=None):
        if isinstance(runnable, CompiledRunnable):
            assert cls == runnable.session_class, (
                'Runnable was compiled for different session type. ' +
                'Need: %s, got: %s' % (
                    cls.__name__, runnable.session_class.__name__))
            return runnable

        if runnable in cls._compiled_cache:
            return cls._compiled_cache[runnable]

        if isinstance(runnable, TaskGroup):
            if workspace_type:
                if runnable.workspace_type():
                    assert runnable.workspace_type() == workspace_type, \
                        "Require {} but already have {}".format(
                            workspace_type, runnable.workspace_type())
                else:
                    runnable._workspace_type = workspace_type
            tg = runnable
        else:
            if workspace_type is None:
                workspace_type = WorkspaceType.GLOBAL
            tg = TaskGroup(workspace_type=workspace_type)
            if isinstance(runnable, Task):
                tg.add(runnable)
            elif isinstance(runnable, core.ExecutionStep):
                tg.add(Task(step=runnable))
            elif isinstance(runnable, core.Plan):
                # ExecutionSteps in Plan() object is supposed to run sequentially, while
                # tasks in TaskGroup run in parallel. So if we have multiple
                # ExecutionSteps in Plan() object, we choose to have a root
                # ExecutionStep to wrap all ExecutionSteps.
                assert len(runnable.Steps()) > 0
                if len(runnable.Steps()) == 1:
                    tg.add(Task(step=runnable.Steps()[0]))
                else:
                    # Task takes a list of ExecutionSteps and automatically wrap into
                    # a root ExecutionStep
                    tg.add(Task(step=runnable.Steps()))
            else:
                step = core.execution_step('runnable', runnable)
                tg.add(Task(step=step))
        compiled = CompiledRunnable(
            cls._compile_task_group(tg, setup_net_list), session_class=cls)
        cls._compiled_cache[runnable] = compiled
        return compiled

    def run(self, runnable, workspace_type=None, setup_net_list=None):
        """Run the given runnable.

        Args:
            runnable: Object recognized by the Session. Currently, we support
                TaskGroup, Task, Plan, ExecutionStep, and Net.
            workspace_type: A string defined in the WorkspaceType object.
            setup_net_list: A list of Net objects or a list of NetDef protos.
                So far this is only used by the DistributedSession, in which we
                need to pass a list of special nets to setup the master.
        """
        assert self.is_open(), 'Session is closed.'
        assert runnable is not None, 'Got a none runnable.'
        self._run_compiled(self.compile(runnable, workspace_type,
                                        setup_net_list).obj)

    def close(self):
        if self.is_open():
            self._do_close()
            self._open = False

    def fetch_output(self, output):
        raise NotImplementedError()

    def _run_compiled(self, task_group):
        raise NotImplementedError()

    @classmethod
    def _compile_task_group(cls, task_group, setup_net_list=None):
        return task_group

    def _do_close(self):
        pass

    def __enter__(self):
        assert self._open, 'Session already closed.'
        return self

    def __exit__(self, ex_type, value, traceback):
        if ex_type is None:
            self.close()


class LocalSession(Session):
    """
    Session that runs in a single node.
    Tasks are all remapped to run in parallel in the 'local' node.

    Currently, LocalSession runs all parallel tasks in the same workspace,
    but this behavior may change in the future. Only tasks pointing to the
    same logical node are guaranteed to always run in the same workspace.
    """
    def __init__(self, ws=None):
        Session.__init__(self)
        self._ws = ws or workspace.C.Workspace.current

    @classmethod
    def _compile_task_group(cls, task_group, setup_net_list=None):
        with Cluster():
            task = task_group.to_task()
        plan = core.Plan('task_group_plan')
        plan.AddStep(task.get_step())
        return (plan, task.output_list(), task.workspace_type)

    def _run_compiled(self, compiled):
        plan, output_list, workspace_type = compiled

        # make sure the output blobs belong to the parent workspace
        outputs = []
        for name in output_list.names():
            self._ws.create_blob(str(name))
            outputs.append(core.BlobReference(str(name)))
        output_list.set_values(outputs, _fetch_func=self._fetch_output)
        task_ws = (
            workspace.C.Workspace(self._ws)
            if workspace_type == WorkspaceType.PRIVATE else self._ws)
        with workspace.WorkspaceGuard(task_ws):
            task_ws.run(plan)

    def _fetch_output(self, output):
        return self._ws.blobs[str(output)].fetch()
