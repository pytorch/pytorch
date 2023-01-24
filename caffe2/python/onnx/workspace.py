## @package onnx
# Module caffe2.python.onnx.workspace






import uuid

from caffe2.python import workspace

# Separating out the context manager part so that users won't
# (mis-)use Workspace instances as context managers
class _WorkspaceCtx(object):
    def __init__(self, workspace_id):
        self.workspace_id = workspace_id
        # A stack, so that the context manager is reentrant.
        self.workspace_stack = []

    def __enter__(self):
        self.workspace_stack.append(workspace.CurrentWorkspace())
        workspace.SwitchWorkspace(self.workspace_id, create_if_missing=True)

    def __exit__(self, exc_type, exc_value, traceback):
        w = self.workspace_stack.pop()
        # Strictly speaking, create_if_missing here is unnecessary, since a user
        # is not supposed to be allowed to destruct a workspace while we're in
        # it.  However, empirically, it has been observed that during abnormal
        # shutdown, Caffe2 deletes its default workspace fairly early in the
        # final calls to destructors.  In this case, we may attempt to exit
        # to a default workspace which no longer exists.  create_if_missing=True
        # will (harmlessly) recreate the workspace before we finally quit.)
        workspace.SwitchWorkspace(w, create_if_missing=True)


class Workspace(object):
    """
    An object representing a Caffe2 workspace.  It is a context manager,
    so you can say 'with workspace:' to use the represented workspace
    as your global workspace.  It also supports every method supported
    by caffe2.python.workspace, but instead of running these operations
    in the global workspace, it runs them in the workspace represented
    by this object.  When this object goes dead, the workspace (and all
    nets and blobs within it) are freed.

    Why do we need this class?  Caffe2's workspace model is very "global state"
    oriented, in that there is always some ambient global workspace you are
    working in which holds on to all of your networks and blobs.  This class
    makes it possible to work with workspaces more locally, and without
    forgetting to deallocate everything in the end.
    """
    def __init__(self):
        # Caffe2 (apparently) doesn't provide any native method of generating
        # a fresh, unused workspace, so we have to fake it by generating
        # a unique ID and hoping it's not used already / will not be used
        # directly in the future.
        self._ctx = _WorkspaceCtx(str(uuid.uuid4()))

    def __getattr__(self, attr):
        def f(*args, **kwargs):
            with self._ctx:
                return getattr(workspace, attr)(*args, **kwargs)
        return f

    def __del__(self):
        # NB: This is a 'self' call because we need to switch into the workspace
        # we want to reset before we actually reset it.  A direct call to
        # workspace.ResetWorkspace() will reset the ambient workspace, which
        # is not want we want.
        self.ResetWorkspace()
