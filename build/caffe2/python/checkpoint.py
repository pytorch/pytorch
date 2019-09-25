## @package checkpoint
# Module caffe2.python.checkpoint
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import logging
from caffe2.python import core, context
from caffe2.python.net_builder import ops
from caffe2.python.task import (
    final_output,
    Node,
    Task,
    TaskGroup,
    TaskOutput,
    WorkspaceType,
)

logger = logging.getLogger(__name__)



@context.define_context()
class Job(object):
    """
    A Job defines three TaskGroups: the `init_group`, the `epoch_group` and the
    `exit_group` which will be run by a JobRunner.

    The `init_group` will be run only once at startup. Its role is to
    initialize globally persistent blobs such as model weights, accumulators
    and data file lists.

    The `epoch_group` will be run in a loop after init_group. The loop will
    exit when any of the stop signals added with `add_stop_condition` is True
    at the end of an epoch.

    The download_group will be run only once, after all the executions of
    epoch_group finish. Its role is to collect the distribute scattered
    parameters back after training.

    The `exit_group` will be run only once at the very end of the job, the
    role of this group is to save the results of training in the end of the job.

    Jobs are context-driven, so that Tasks can be added to the active Job
    without having to explicitly pass the job object around.

    Example of usage:

    def build_reader(partitions):
        with Job.current().init_group:
            reader = HiveReader(init_reader, ..., partitions)
            Task(step=init_reader)
        with Job.current().epoch_group:
            limited_reader = ReaderWithLimit(reader, num_iter=10000)
            data_queue = pipe(limited_reader, num_threads=8)
            Job.current().add_stop_condition(limited_reader.data_finished())
        return data_queue

    def build_hogwild_trainer(reader, model):
        with Job.current().init_group:
            Task(step=model.param_init_net)
        with Job.current().epoch_group:
            pipe(reader, processor=model, num_threads=8)
        with Job.current().exit_group:
            Task(step=model.save_model_net)

    with Job() as job:
        reader = build_reader(partitions)
        model = build_model(params)
        build_hogwild_trainer(reader, model)
    """
    def __init__(self,
                 init_group=None, epoch_group=None,
                 download_group=None, exit_group=None,
                 stop_conditions=None, nodes_to_checkpoint=None):
        self.init_group = init_group or TaskGroup(
            workspace_type=WorkspaceType.GLOBAL)
        self.epoch_group = epoch_group or TaskGroup()
        self.download_group = download_group or TaskGroup()
        self.exit_group = exit_group or TaskGroup()
        self.stop_conditions = stop_conditions or []
        self._nodes_to_checkpoint = nodes_to_checkpoint

    def nodes_to_checkpoint(self):
        if self._nodes_to_checkpoint:
            return self._nodes_to_checkpoint
        else:
            return self.init_group.used_nodes()

    def compile(self, session_class):
        self._nodes_to_checkpoint = self.nodes_to_checkpoint()
        self.init_group = session_class.compile(self.init_group)
        self.epoch_group = session_class.compile(self.epoch_group)
        self.download_group = session_class.compile(self.download_group)
        self.exit_group = session_class.compile(self.exit_group)

    def __enter__(self):
        self.epoch_group.__enter__()
        return self

    def __exit__(self, *args):
        self.epoch_group.__exit__()

    def add_stop_condition(self, output):
        if isinstance(output, core.BlobReference):
            t = Task(outputs=[output], group=self.epoch_group)
            output = t.outputs()[0]
        assert isinstance(output, TaskOutput)
        self.stop_conditions.append(output)


def get_ckpt_filename(node_name, epoch):
    """Returns the checkpoint filename.

    Args:
        node_name: A string. The name of the node.
        epoch: An integer. The checkpoint epoch.

    Returns:
        ckpt_filename: A string. The filename of the checkpoint.
    """
    return node_name + '.' + str(epoch)


def db_name(epoch, node_name, db_prefix, path_prefix=None):
    """Returns the full db name where checkpoint files are saved.

    Args:
        epoch: An integer. The checkpoint epoch.
        node_name: A string. The name of the node.
        db_prefix: A string. The prefix used to construct full db name.
        path_prefix: A string. Optional param used to construct db name or path
            where checkpoint files are are stored.
    Returns:
        db_name: A string. The absolute path of full_db_name where checkpoint
            files are saved
    """
    if path_prefix:
        db_name = path_prefix + get_ckpt_filename(node_name, epoch)
    else:
        ckpt_filename = get_ckpt_filename(node_name, epoch)
        db_name = os.path.join(db_prefix, ckpt_filename)
    return db_name


class CheckpointManager(object):
    """
    Controls saving and loading of workspaces on every epoch boundary of a job.
    If a CheckpointManager instance is passed to JobRunner, then JobRunner will
    call `init`, `read` and `save` at different moments in between epoch runs.

    Args:
        db_prefix: The prefix used to construct full db name. Since `absolute_path`
            is set to True, this will be used as db_name in SaveOp.
        node_name: Name of the node where this checkpoint_manager is used.
        db_type: Type of database to use for storing checkpoint.
        metadata_handler: An optional object capable of reading/writing
            checkpoint info in storage of choice.
    """

    BLOB_NAMES = "blob_names"

    def __init__(self, db_prefix, node_name, db_type, metadata_handler=None):
        self._db_prefix = db_prefix
        self._node_name = node_name
        self._db_type = db_type
        self._metadata_handler = metadata_handler
        # make sure these blobs are the first in the checkpoint file.
        self._net = core.Net('!!checkpoint_mngr')
        self._blob_names = self._net.AddExternalInput(self.BLOB_NAMES)
        self._names_output = None
        self._path_prefix = None
        self._path_type = None
        self._current_db_name = None
        self._current_checkpoint_duration = None

    """
    Initialize the checkpoint manager. Determines all blobs that need to be saved
    or loads from a checkpoint.

    Args:
        nodes: An array of nodes where this checkpoint manager is running. Should
            only contain a single node.
        retrieve_from_epoch: Set to a number to load blobs from this epoch.
        path_prefix: Used to construct db name or path where checkpoint files are
            stored.
        path_type: Indicate the type of path where checkpoint files are stored.
    """
    def init(
        self,
        nodes=None,
        retrieve_from_epoch=None,
        path_prefix=None,
        path_type=None
    ):
        """
        Build a Task that will be run once after the job's `init_group` is run.
        This task will determine which blobs need to be checkpointed.
        If retrieve_from_epoch is not None, then the checkpoint metadata is
        retrieved from a previously saved checkpoint.
        """
        assert nodes is None or len(nodes) == 1, (
            'CheckpointManager only supports single node.')

        with Task(outputs=[self._blob_names]) as task:
            if retrieve_from_epoch is None:
                ops.GetAllBlobNames(
                    [],
                    self._blob_names,
                    include_shared=False)
            else:
                full_db_name = db_name(retrieve_from_epoch,
                                        self._node_name, self._db_prefix, path_prefix)
                db_type = path_type or self._db_type
                logger.info("Initializing checkpoints from = %s"
                            % full_db_name)
                ops.Load(
                    [], self._blob_names,
                    db=full_db_name,
                    db_type=db_type,
                    absolute_path=True,
                    keep_device=True,
                )
        self._names_output = task.outputs()[0]
        return task

    def blob_list(self):
        assert self._names_output
        return self._names_output.fetch().tolist()

    def _timed_task(self, cp_op_name, add_op):
        """
        Build a Task that will measure the time span of checkpoint operations,
        once operation is done, time can be read from _current_checkpoint_duration.

        Args:
            cp_op_name: A string name of the checkpoint operation.
            add_op: A functor to add the checkpoint operation.

        Returns:
            A task with timer.
        """
        with Task(name=cp_op_name) as task:
            with ops.task_init():
                timer = ops.TimerBegin([], counter_name=self._node_name)
            add_op()
            with ops.task_exit():
                time_span_blob = ops.TimerGetAndEnd(timer)
            self._current_checkpoint_duration = final_output(time_span_blob)
        return task

    def collect_checkpoint_stats(self, stats):
        """
        Add one checkpoint stats into the stats.

        Args:
            stats: A dict of checkpoint stats that will be reported.
        """
        if self._current_db_name and self._current_checkpoint_duration:
            stats[self._current_db_name] = self._current_checkpoint_duration.fetch()[0]
        else:
            logger.info(
                "Failed to collect checkpoint stats: {}".format(
                    self._current_db_name
                )
            )

    def load(self, epoch, path_prefix=None, path_type=None):
        """
        Build a Task that will be run by JobRunner when the job is to be
        resumed from a given epoch. This task will run a Load op that will
        load and deserialize all relevant blobs from a persistent storage.
        """
        self._current_db_name = db_name(
            epoch, self._node_name, self._db_prefix, path_prefix
        )
        db_type = path_type or self._db_type
        logger.info("Loading checkpoints from = %s" % self._current_db_name)

        def add_op():
            ops.Load(
                [],
                self.blob_list(),
                db=self._current_db_name,
                db_type=db_type,
                absolute_path=True,
                keep_device=True,
            )

        return self._timed_task('checkpoint_load', add_op)

    def load_blobs_from_checkpoint(self, blob_names, epoch):
        """
        Builds a Task that loads only the necessary blobs from a checkpoint of
        the given epoch. The necessary blobs are given in the blob_names
        argument.

        Args:
            blob_names: A list of strings. Each string is the name of a
                blob.
            epoch: The checkpoint epoch to load from.

        Returns:
            A Task which loads the specified blobs from the checkpoint of the
            given epoch.
        """
        self._current_db_name = db_name(epoch, self._node_name, self._db_prefix)
        logger.info('Load from %s' % self._current_db_name)

        def add_op():
            ops.Load(
                [],
                blob_names,
                db=self._current_db_name,
                db_type=self._db_type,
                absolute_path=True,
                allow_incomplete=True)

        return self._timed_task('checkpoint_partial_load', add_op)

    def check_db_exists(self, epoch):
        logger.info('Check existence of %s' %
                    db_name(epoch, self._node_name, self._db_prefix))
        with Task() as task:
            existence = ops.Const(False)
            ops.DBExists(
                [],
                [existence],
                db_name=db_name(epoch, self._node_name, self._db_prefix),
                db_type=self._db_type,
                absolute_path=True)
            task.add_output(existence)
        return task

    def report_checkpoint_stats(self, action_name):
        """
        Report checkpoint operation stats for current node.

        Args:
            action_name: A string of the name of checkpoint operation.
        """
        all_stats = {}
        self.collect_checkpoint_stats(all_stats)
        if self._metadata_handler:
            self._metadata_handler.report(action_name, all_stats)

    def save(self, epoch):
        """
        Build a Task that is run once after `init_group` and after each
        epoch is run. This will execute a Save ops to serialize and persist
        blobs present in the global workspace.
        """
        self._current_db_name = db_name(epoch, self._node_name, self._db_prefix)
        logger.info('Saving to %s' % self._current_db_name)

        def add_op():
            ops.Save(
                self.blob_list(), [],
                db=self._current_db_name,
                db_type=self._db_type,
                absolute_path=True)

        return self._timed_task('checkpoint_save', add_op)

    def write_checkpoint_metadata(self, epoch):
        """
        Write metadata for checkpoint

        Args:
            epoch: An integer. The epoch-id for which checkpoint metadata is
                written
        """
        if self._metadata_handler is not None:
            self._metadata_handler.write(epoch=epoch)

    def get_resume_from_epoch_id(self, user_epoch=None):
        """
        Identify the epoch-id from which Job must resume

        Args:
            user_epoch: An integer. Optional parameter for user to explicitly
                identify the epoch-id to load checkpoint from
        Retruns:
            epoch: the epoch-id to load checkpoints from
                or None if no checkpoints were written
        """
        last_epoch = user_epoch
        if self._metadata_handler is not None:
            last_epoch = self._metadata_handler.last_epoch(user_epoch=user_epoch)
        return last_epoch

    def set_params(self, nodes, path_prefix=None, path_type=None):
        """Set parameters associated with CP manager

        Args:
            nodes: An array of nodes where this checkpoint manager is running.
            path_prefix: Used to construct db name or path where checkpoint files are
                stored.
            path_type: Indicate the type of path where checkpoint files are stored.
        """
        if path_prefix:
            self._path_prefix = path_prefix
        if path_type:
            self._path_type = path_type
        if self._metadata_handler:
            self._metadata_handler.set_params(
                db_prefix=self._db_prefix,
                db_type=self._db_type,
                node_names=[str(self._node_name)],
                path_prefix=self._path_prefix,
                path_type=self._path_type)

    def cp_accessible(self, epoch=None):
        """Returns True if Checkpoint data is accessible

        Args:
            epoch: An integer. The epoch of the checkpoint. If None,
                it implies we need to check if checkpoint directory is accessible

        Returns:
            is_cp_accessible: A boolean. Returns True if Checkpoint data is accessible
        """
        if self._metadata_handler is not None:
            return self._metadata_handler.cp_accessible(epoch)
        else:
            return True


class MultiNodeCheckpointManager(object):
    """
    Coordinates checkpointing and checkpointing across multiple nodes.
    Each of `init`, `load` and `save` will build TaskGroups which will
    trigger checkpointing on each of the nodes involved in a distributed job.

    Args:
        db_prefix: The prefix used to construct full db name. Since `absolute_path`
            is set to True, this will be used as db_name in SaveOp.
        db_type: Type of database to use for storing checkpoint.
        metadata_handler: An optional object capable of reading/writing
            checkpoint info in storage of choice.
    """
    def __init__(self, db_prefix, db_type, metadata_handler=None):
        self._node_managers = None
        self._db_prefix = db_prefix
        self._db_type = db_type
        self._metadata_handler = metadata_handler
        self._path_prefix = None
        self._path_type = None

    def _task_group(self, func, *args, **kw):
        assert self._node_managers is not None, 'init must be called first.'
        with TaskGroup(WorkspaceType.GLOBAL) as task_group:
            for node, manager in self._node_managers:
                with Node(node):
                    func(manager, *args, **kw)
            return task_group

    """
    Args:
        nodes: An array of nodes where this checkpoint manager is running.
        retrieve_from_epoch: Set to a number to load blobs from this epoch.
        path_prefix: Used to construct db name or path where checkpoint files are
            stored.
        path_type: Indicate the type of path where checkpoint files are stored.
    """
    def init(
        self, nodes, retrieve_from_epoch=None, path_prefix=None, path_type=None
    ):
        if self._node_managers is not None:
            assert [node for node, _ in self._node_managers] == nodes
            return TaskGroup(WorkspaceType.GLOBAL)
        self._node_managers = []
        for node in nodes:
            with Node(node):
                manager = CheckpointManager(
                    db_prefix=self._db_prefix,
                    node_name=str(node),
                    db_type=self._db_type)
                self._node_managers.append((node, manager))
        return self._task_group(
            CheckpointManager.init,
            nodes=[node],
            retrieve_from_epoch=retrieve_from_epoch,
            path_prefix=path_prefix,
            path_type=path_type)

    def load(self, epoch, path_prefix=None, path_type=None):
        return self._task_group(
            CheckpointManager.load,
            epoch,
            path_prefix=path_prefix,
            path_type=path_type)

    def load_blobs_locally(self, nodes, blob_names, epoch, session):
        """Loads the necessary blobs from the checkpoints to the current node.

        Args:
            blob_names: A list of strings. Each string is the name of a
                blob.
            epoch: An integer. The checkpoint epoch to load from.
            session: A Session object to execute the Load ops.
        """
        if self._node_managers is not None:
            assert [node for node, _ in self._node_managers] == nodes
        else:
            self._node_managers = []
            for node in nodes:
                with Node(node):
                    manager = CheckpointManager(
                        db_prefix=self._db_prefix,
                        node_name=str(node),
                        db_type=self._db_type)
                    self._node_managers.append((node, manager))
        assert self._node_managers is not None, 'must initialize node managers'
        for _, manager in self._node_managers:
            existence_task = manager.check_db_exists(epoch)
            session.run(existence_task)
            existence = existence_task.outputs()[0].fetch()
            if not existence:
                logger.info('DB %s does not exist!' %
                            db_name(epoch, manager._node_name, manager._db_prefix))
                return False
            load_task = manager.load_blobs_from_checkpoint(blob_names, epoch)
            session.run(load_task)
        logger.info('Successfully loaded from checkpoints.')
        return True

    def get_ckpt_db_name(self, node_name, epoch):
        """Returns the DB name of the given node and the given epoch.

        The DB name is effectively the checkpoint path of the given node and
        the given epoch.

        Args:
            node_name: A string. The node name of interest.
            epoch: An integer. The epoch of the checkpoint.

        Returns:
            checkpoint_db_name: A string. The checkpoint path of the given
                node and the given epoch.
        """
        for node, manager in self._node_managers:
            if str(node) == node_name:
                return db_name(epoch, manager._node_name, manager._db_prefix)

    def report_checkpoint_stats(self, action_name):
        """
        Report the checkpoint stats for all the nodes, we need to aggregate all
        the node's stats together so that we know which node's checkpoint
        operation dominates.

        Args:
            action_name: A string of the name of checkpoint operation.
        """
        all_stats = {}
        for _, manager in self._node_managers:
            manager.collect_checkpoint_stats(all_stats)
        logger.debug("checkpoint stats: {}".format(all_stats))
        if self._metadata_handler:
            self._metadata_handler.report(action_name, all_stats)

    def save(self, epoch):
        """
        Build a Task that will execute a Save ops to serialize and persist
        blobs present in the global workspace.
        """
        return self._task_group(CheckpointManager.save, epoch)

    def write_checkpoint_metadata(self, epoch):
        """
        Write metadata for checkpoint

        Args:
            epoch: An integer. The epoch-id for which checkpoint metadata is
                written
        """
        if self._metadata_handler is not None:
            self._metadata_handler.write(epoch=epoch)

    def get_resume_from_epoch_id(self, user_epoch=None):
        """
        Identify the epoch-id from which Job must resume

        Args:
            user_epoch: An integer. Optional parameter for user to explicitly
                identify the epoch-id to load checkpoint from
        Retruns:
            epoch: the epoch-id to load checkpoints from
                or None if no checkpoints were written
        """
        last_epoch = user_epoch
        if self._metadata_handler is not None:
            last_epoch = self._metadata_handler.last_epoch(user_epoch=user_epoch)
        return last_epoch

    def set_params(self, nodes, path_prefix=None, path_type=None):
        """Set parameters associated with CP manager

        Args:
            nodes: An array of nodes where this checkpoint manager is running.
            path_prefix: Used to construct db name or path where checkpoint files are
                stored.
            path_type: Indicate the type of path where checkpoint files are stored.
        """
        self._node_names = [str(node) for node in nodes]
        if path_prefix:
            self._path_prefix = path_prefix
        if path_type:
            self._path_type = path_type
        if self._metadata_handler:
            self._metadata_handler.set_params(
                db_prefix=self._db_prefix,
                db_type=self._db_type,
                node_names=self._node_names,
                path_prefix=self._path_prefix,
                path_type=self._path_type)

    def cp_accessible(self, epoch=None):
        """Returns True if Checkpoint data is accessible

        Args:
            epoch: An integer. The epoch of the checkpoint. If None,
                it implies we need to check if checkpoint directory is accessible

        Returns:
            is_cp_accessible: A boolean. Returns True if Checkpoint data is accessible
        """
        if self._metadata_handler is not None:
            return self._metadata_handler.cp_accessible(epoch)
        else:
            return True


class UploadTaskGroupBuilder(object):
    """A simple class to upload checkpoints."""
    def build(self, epoch, checkpoint_manager):
        """Builds the task group to upload checkpoints.

        Args:
            epoch: An integer. The checkpoint epoch to be uploaded.
            checkpoint_manager: Can be a CheckpointManager for single machine
                or a MultiNodeCheckpointManager for multi-machine. The manager
                that initializes/saves/loads checkpoints.

        Raises:
            NotImplementedError: This base class only has the interface,
                the implementation will be in the subclasses.
        """
        raise NotImplementedError()


class JobRunner(object):
    """
    Implement the runtime logic for jobs with checkpointing at the level of
    epoch. Can be used to run either single-host or distributed jobs. Job
    runner is a callable to be called once from the master, passing a session
    as an argument. This call will block until the Job execution is complete.

    If a checkpoint_manager is passed, checkpoints will be taken after
    initialization and after each epoch execution. If, in addition,
    `resume_from_epoch` is an epoch number, the corresponding checkpoint will
    be loaded and job execution will continue from the given epoch. In
    this case, the job's init_group will not be run.

    Refer to checkpoint_test.py for an example.
    """
    def __init__(self, job, checkpoint_manager=None, resume_from_epoch=None,
                 upload_task_group_builder=None):
        """Initializes the JobRunner.

        Args:
            job: A Job object. The job to be executed.
            checkpoint_manager: Can be a CheckpointManager for single machine
                or a MultiNodeCheckpointManager for multi-machine. The manager
                that initializes/saves/loads checkpoints.
            resume_from_epoch: An integer. The epoch to resume from.
            upload_task_group_builder: A subclass of the
                UploadTaskGroupBuilder. Creates a task group to upload
                checkpoints.
        """
        self.resume_from_epoch = resume_from_epoch
        self.checkpoint_manager = checkpoint_manager
        self.job = job
        self.upload_task_group_builder = upload_task_group_builder

    def train(self, session):
        """Runs the training flow.

        Args:
            session: A Session object. Valid choises are: LocalSession,
                LocalHostScheduler, and DistributedSession. It is used to
                execute one TaskGroup a time.
        """
        # identify the epoch we must resume from
        if self.checkpoint_manager:
            self.checkpoint_manager.set_params(nodes=self.job.nodes_to_checkpoint())
            self.resume_from_epoch = self.checkpoint_manager.\
                get_resume_from_epoch_id(self.resume_from_epoch)
            if self.resume_from_epoch is not None:
                logger.info('Resuming from epoch {}'.format(self.resume_from_epoch))

        # Initialize all the nodes.
        from_scratch = self.resume_from_epoch is None
        if from_scratch:
            session.run(self.job.init_group)

        if self.checkpoint_manager:
            logger.info('Preparing checkpoints ...')
            session.run(self.checkpoint_manager.init(
                self.job.nodes_to_checkpoint(),
                retrieve_from_epoch=self.resume_from_epoch))
            # Save the first checkpoint before training starts, or resume from
            # a previously saved checkpoint.
            if from_scratch:
                self.save_checkpoints(0, session)
            else:
                logger.info('Loading checkpoints for epoch {} ...'.format(
                    self.resume_from_epoch))
                session.run(
                    self.checkpoint_manager.load(self.resume_from_epoch))
                self.checkpoint_manager.report_checkpoint_stats('checkpoint_load')
                logger.info('Checkpoint loaded')

        logger.info("Finished initializing")

        # Start training.
        epoch = 1 if from_scratch else self.resume_from_epoch + 1
        while True:
            logger.info('Starting epoch %d' % epoch)
            session.run(self.job.epoch_group)
            logger.info('Finished epoch %d' % epoch)
            stop_conditions = [o.fetch() for o in self.job.stop_conditions]

            if self.checkpoint_manager:
                self.save_checkpoints(epoch, session)

            if any(stop_conditions):
                logger.info('Stopping')
                break
            epoch += 1
        logger.info('Finished training')
        # Upload the checkpoints.
        if (self.upload_task_group_builder):
            upload_task_group = self.upload_task_group_builder.build(
                epoch, self.checkpoint_manager)
            session.run(upload_task_group)
            logger.info('Finished uploading the checkpoints')

        # Download the parameters to save
        session.run(self.job.download_group)
        logger.info('Finished downloading the parameters')

        # Finally run the exit step to save nets
        session.run(self.job.exit_group)
        logger.info('Finished running the exit group')
        return epoch

    def load_blobs_from_checkpoints(self, blob_names, epoch, session):
        """Loads the necessary blobs from the checkpoints.

        Checkpoints store the snapshots of the workspace in each node.
        Sometimes we only need to load a subset of the blobs from the
        checkpoints. One common scenario is to load only the model blobs from
        the checkpoints for evaluation purpose. Given the names of the
        necessary blobs, this function goes over all the checkpoints of all the
        nodes, but only loads the blobs specified in the blob_names to the
        current workspace.

        Args:
            blob_names: A list of strings. Each string is the name of a
                blob.
            epoch: An integer. The checkpoint epoch to load from.
            session: A Session object to execute the load ops.

        Raises:
            ValueError: When the checkpoint manager is invalid.
        """
        if not self.checkpoint_manager:
            raise ValueError('Checkpoint manager is None')
        logger.info('Loading checkpoint for epoch {} ...'.format(epoch))
        result = self.checkpoint_manager.load_blobs_locally(
            self.job.nodes_to_checkpoint(), blob_names, epoch, session)
        self.checkpoint_manager.report_checkpoint_stats('checkpoint_partial_load')
        return result

    def save_checkpoints(self, epoch, session):
        """Triggers operation to save checkpoints

        This method will trigger the Save ops to serialize and persist the
        blobs present in the global workspaace.

        Args:
            epoch: An integer. The checkpoint epoch-id that we are saving.
            session: A Session object to execute the save ops.

        Raises:
            ValueError: When the checkpoint manager is invalid.
        """
        if not self.checkpoint_manager:
            raise ValueError('Checkpoint manager is None')
        try:
            is_accessible = self.checkpoint_manager.cp_accessible(epoch=None)
            if is_accessible:
                logger.info('Saving checkpoints for epoch {}'.format(epoch))
                session.run(self.checkpoint_manager.save(epoch))
                self.checkpoint_manager.write_checkpoint_metadata(epoch)
                logger.info('Checkpoints saved')
                self.checkpoint_manager.report_checkpoint_stats('checkpoint_save')
            else:
                logger.warning("Checkpoint files cannot be accessed!")
        except Exception as ex:
            logger.warning("Unable to write checkpoint for epoch {}. Error={}".
                            format(epoch, ex))


def epoch_limiter(job, num_epochs):
    """
    Creates a task that will output True when a given
    number of epochs has finished.
    """
    with job.init_group:
        init_net = core.Net('epoch_counter_init')
        counter = init_net.CreateCounter([], init_count=num_epochs - 1)
        Task(step=init_net)

    with job.epoch_group:
        epoch_net = core.Net('epoch_countdown')
        finished = epoch_net.CountDown(counter)
        output = Task(step=epoch_net, outputs=finished).outputs()[0]
    job.add_stop_condition(output)
