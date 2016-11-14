from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from caffe2.python import core, workspace
from caffe2.python.task import Task, TaskGroup, WorkspaceType


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
        At the beggining of the session, a global workspace is created and kept
        alive for the duration of the session.


    Private Workspace:
        Tasks can be run either directly on the global workspace, or they can
        instantiate a private child workspace that is released after each run.

    Blob visibility:
        Tasks running in different nodes in parallel will always run under
        different workspaces, so it must be assumed that they won't be able to
        access each other's blobs. On the other hand, tasks running on the same
        node are guaranteed to run on the same workspace within a run.
    """
    def __init__(self):
        self._open = True
        self._runnable_cache = {}

    def is_open(self):
        return self._open

    def run(self, runnable):
        assert self.is_open(), 'Session is closed.'
        if runnable not in self._runnable_cache:
            if isinstance(runnable, TaskGroup):
                tg = runnable
            else:
                tg = TaskGroup(workspace_type=WorkspaceType.GLOBAL)
                if isinstance(runnable, Task):
                    tg.add(runnable)
                elif isinstance(runnable, core.ExecutionStep):
                    tg.add(Task(step=runnable))
                else:
                    step = core.execution_step('runnable', runnable)
                    tg.add(Task(step=step))
            self._runnable_cache[runnable] = tg
        self._run_task_group(self._runnable_cache[runnable])

    def close(self):
        if self.is_open():
            self._do_close()
            self._open = False

    def fetch_output(self, output):
        raise NotImplementedError()

    def _run_task_group(self, task_group):
        raise NotImplementedError()

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
    def __init__(self, ws):
        Session.__init__(self)
        self._ws = ws
        self._plan_caches = {}

    def _run_task_group(self, task_group):
        if task_group not in self._plan_caches:
            task = task_group.to_task()
            plan = core.Plan('task_group_plan')
            plan.AddStep(task.get_step())
            self._plan_caches[task_group] = (plan, task)
        plan, task = self._plan_caches[task_group]

        # make sure the output blobs belong to the parent workspace
        outputs = []
        for name in task.output_names():
            self._ws.create_blob(str(name))
            outputs.append(core.BlobReference(str(name)))
        task.set_outputs(outputs, _fetch_func=self._fetch_output)
        task_ws = (
            workspace.C.Workspace(self._ws)
            if task.workspace_type == WorkspaceType.PRIVATE else self._ws)
        with workspace.WorkspaceGuard(task_ws):
            task_ws.run(plan)

    def _fetch_output(self, output):
        return self._ws.blobs[str(output)].fetch()
