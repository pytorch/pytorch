# pyre-strict
# mypy: allow-untyped-defs
import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union
from uuid import uuid4

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.checkpoint._async_executor import _AsyncCheckpointExecutor
from torch.distributed.checkpoint.logger import _dcp_method_logger, _init_logger
from torch.distributed.checkpoint.metadata import Metadata, STATE_DICT_TYPE
from torch.distributed.checkpoint.planner import SavePlanner
from torch.distributed.checkpoint.storage import StorageWriter
from torch.distributed.checkpoint.utils import _DistWrapper
from torch.distributed.elastic.agent.server.api import _get_fq_hostname
from torch.distributed.elastic.utils.distributed import get_free_port


logger = logging.getLogger()


class _CheckpointSaveProcessControlOpts(Enum):
    INIT_COMPLETE = "init_complete"
    TERMINATE = "terminate"


@dataclass(init=False, unsafe_hash=True)
class _CheckpointRequestIdentifier:
    checkpoint_id: Union[str, os.PathLike, None]
    uuid: str

    def __init__(self, checkpoint_id: Union[str, os.PathLike, None]):
        self.checkpoint_id = checkpoint_id
        self.uuid = str(uuid4())


@dataclass
class _AsyncCheckpointRequest:
    staged_state_dict: STATE_DICT_TYPE
    checkpoint_request_id: _CheckpointRequestIdentifier
    storage_writer: Optional[StorageWriter] = None
    planner: Optional[SavePlanner] = None


@dataclass(init=False)
class _ProcessGroupInitInfo:
    local_rank: int
    global_rank: int
    world_size: int
    tcp_store_master_addr: str
    tcp_store_master_port: int

    def __init__(self, process_group: Optional[dist.ProcessGroup] = None):
        self.local_rank = dist.get_node_local_rank(fallback_rank=0)
        self.global_rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)

        # Let coordinator rank find a free port on the localhost.
        # Broadcast the (master_addr, free_port) to all ranks; each rank in the
        # checkpoint daemon process will use TCPStore (master_addr, master_port)
        # for collective communication.
        dist_wrapper: _DistWrapper = _DistWrapper(
            group=process_group,
            use_dist=True,
            coordinator_rank=0,
        )

        def get_master_addr_and_port() -> tuple[str, int]:
            master_addr = os.environ.get("MASTER_ADDR")
            if master_addr is None:
                master_addr = _get_fq_hostname()
            return master_addr, get_free_port()

        self.tcp_store_master_addr, self.tcp_store_master_port = dist_wrapper.broadcast(
            step="get_master_addr_and_port",
            map_fun=get_master_addr_and_port,
        )


class _AsyncCheckpointProcess:
    def __init__(
        self,
        pg_init_info: _ProcessGroupInitInfo,
    ):
        self.ctx = mp.get_context("spawn")
        self._process_pipe, child_end = self.ctx.Pipe()

        self._save_process = self.ctx.Process(
            target=self._checkpointing_subprocess,
            args=(
                pg_init_info,
                child_end,
            ),
            daemon=True,
        )

        self._save_process.start()

        # Close the parent's copy of child end after we pass it into the child,
        # so the recv()s on it will fail-fast if the child process dies.
        child_end.close()

        # Wait for the checkpoint background process to initialize.
        # Using default GLOO init timeout.
        response = self._wait_for_response(timeout=1800)
        assert response == _CheckpointSaveProcessControlOpts.INIT_COMPLETE

    def __del__(self) -> None:
        if self._save_process.is_alive():
            logger.info("Terminating the checkpoint background process...")
            self._send(_CheckpointSaveProcessControlOpts.TERMINATE)
            self._save_process.join()

    def _send(self, data: Any) -> None:
        self._process_pipe.send(data)

    def _wait_for_response(self, timeout: Optional[float] = None) -> Any:
        if not self._save_process.is_alive():
            logger.info("Checkpoint background process is dead calling join()...")
            self._save_process.join()
            raise RuntimeError(
                f"Checkpoint background process is dead. Exit code: {self._save_process.exitcode}"
            )

        if timeout is not None and not self._process_pipe.poll(timeout=timeout):
            raise RuntimeError(
                f"Timed out after {timeout}s while waiting for response from checkpointer process pid: {self._save_process.pid}"
            )

        try:
            response = self._process_pipe.recv()
        except EOFError:
            raise RuntimeError(  # noqa: B904
                f"Checkpoint background process is dead. Exit code: {self._save_process.exitcode}"
            )

        if isinstance(response, BaseException):
            raise response

        return response

    def save(
        self,
        staged_state_dict: STATE_DICT_TYPE,
        *,
        checkpoint_id: Union[str, os.PathLike, None] = None,
        storage_writer: Optional[StorageWriter] = None,
        planner: Optional[SavePlanner] = None,
    ) -> Metadata:
        # Create a unique identifier to locate requests/responses
        # from the checkpoint daemon process.
        checkpoint_request_id = _CheckpointRequestIdentifier(checkpoint_id)
        async_cp_request = _AsyncCheckpointRequest(
            staged_state_dict=staged_state_dict,
            checkpoint_request_id=checkpoint_request_id,
            storage_writer=storage_writer,
            planner=planner,
        )
        self._send(async_cp_request)
        result = self._wait_for_response()
        assert isinstance(result, Metadata)
        return result

    @staticmethod
    def _execute_save(
        state_dict: STATE_DICT_TYPE,
        *,
        checkpoint_request_id: _CheckpointRequestIdentifier,
        storage_writer: Optional[StorageWriter] = None,
        planner: Optional[SavePlanner] = None,
    ) -> Metadata:
        from torch.distributed.checkpoint.state_dict_saver import save

        metadata = save(
            state_dict,
            checkpoint_id=checkpoint_request_id.checkpoint_id,
            storage_writer=storage_writer,
            planner=planner,
        )
        return metadata

    @staticmethod
    def _checkpointing_subprocess(
        pg_init_info: _ProcessGroupInitInfo,
        parent_conn,
    ) -> None:
        # Phase 1: Process Group Initialization
        # Only needs to execute once during the lifetime of the checkpoint background process.
        try:
            _init_logger(pg_init_info.global_rank)

            # Setup environment variables for process group initialization.
            os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"
            os.environ["MASTER_ADDR"] = pg_init_info.tcp_store_master_addr
            os.environ["MASTER_PORT"] = str(pg_init_info.tcp_store_master_port)
            os.environ["LOCAL_RANK"] = str(pg_init_info.local_rank)
            os.environ["RANK"] = str(pg_init_info.global_rank)
            os.environ["WORLD_SIZE"] = str(pg_init_info.world_size)

            logger.info(
                "Initializing dist.ProcessGroup in checkpoint background process"
            )
            # NOTE: GLOO backend is enforced here.
            dist.init_process_group(backend=dist.Backend.GLOO)
            dist.barrier()

            logger.info("Checkpoint background process is running...")
            parent_conn.send(_CheckpointSaveProcessControlOpts.INIT_COMPLETE)
        except BaseException as e:  # noqa: B036
            logger.error(
                f"Checkpoint background process failed during initialization: {e}"  # noqa: G004
            )
            parent_conn.send(e)
            return

        # Phase 2: Serving Loop
        try:
            while True:
                logger.info("Waiting for checkpoint save request...")
                obj = parent_conn.recv()
                if (
                    isinstance(obj, _CheckpointSaveProcessControlOpts)
                    and obj == _CheckpointSaveProcessControlOpts.TERMINATE
                ):
                    logger.info("Terminating the checkpoint background process.")
                    return
                assert isinstance(obj, _AsyncCheckpointRequest)
                logger.info(
                    f"Received async checkpoint request with id={obj.checkpoint_request_id.checkpoint_id}"  # noqa: G004
                )

                try:
                    response = _AsyncCheckpointProcess._execute_save(
                        obj.staged_state_dict,
                        checkpoint_request_id=obj.checkpoint_request_id,
                        storage_writer=obj.storage_writer,
                        planner=obj.planner,
                    )
                    parent_conn.send(response)
                    logger.info(
                        f"Completed checkpoint save request for checkpoint_id={obj.checkpoint_request_id}"  # noqa: G004
                    )
                except BaseException as e:  # noqa: B036
                    logger.error(
                        f"Checkpoint save failed for checkpoint_id={obj.checkpoint_request_id.checkpoint_id}: {e}"  # noqa: G004
                    )
                    parent_conn.send(e)
                    # Continue serving loop - don't exit process
        finally:
            logger.info("Checkpoint background process is shutting down...")
            dist.destroy_process_group()
            parent_conn.close()


_CHECKPOINT_PROCESS: Optional[_AsyncCheckpointProcess] = None


class _ProcessBasedAsyncCheckpointExecutor(_AsyncCheckpointExecutor):
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1)

    @staticmethod
    def _execute_save_impl(
        *,
        pg_init_info: Optional[_ProcessGroupInitInfo],
        staging_future_or_state_dict: Union[Future[STATE_DICT_TYPE], STATE_DICT_TYPE],
        checkpoint_id: Union[str, os.PathLike, None] = None,
        storage_writer: Optional[StorageWriter] = None,
        planner: Optional[SavePlanner] = None,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> Metadata:
        if _CHECKPOINT_PROCESS is None:
            assert pg_init_info is not None
            ckpt_kwargs = {}
            if (ckpt_id := getattr(storage_writer, "checkpoint_id", None)) is not None:
                ckpt_kwargs["checkpoint_id"] = ckpt_id
                ckpt_kwargs["process_group"] = process_group

            @_dcp_method_logger(**ckpt_kwargs)
            def create_checkpoint_daemon_process() -> None:
                global _CHECKPOINT_PROCESS
                _CHECKPOINT_PROCESS = _AsyncCheckpointProcess(pg_init_info=pg_init_info)

            create_checkpoint_daemon_process()

        assert _CHECKPOINT_PROCESS is not None
        staged_state_dict = (
            staging_future_or_state_dict.result()
            if isinstance(staging_future_or_state_dict, Future)
            else staging_future_or_state_dict
        )
        return _CHECKPOINT_PROCESS.save(
            staged_state_dict=staged_state_dict,
            checkpoint_id=checkpoint_id,
            storage_writer=storage_writer,
            planner=planner,
        )

    def execute_save(
        self,
        staging_future_or_state_dict: Union[Future[STATE_DICT_TYPE], STATE_DICT_TYPE],
        *,
        checkpoint_id: Union[str, os.PathLike, None] = None,
        storage_writer: Optional[StorageWriter] = None,
        planner: Optional[SavePlanner] = None,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> Future:
        """
        NOTE:

        - Checkpoint process is implemented as a daemon process.
        The AsyncCheckpointProcess' lifetime is tied to the lifetime of the
        main process (e.g. trainer process).

        - The first call to execute_save_in_process() will initialize the checkpoint
        daemon process. Subsequent async checkpoint requests will not need process
        initialization. Therefore, the first async checkpoint request will take longer to complete.

        - Process initialization can have significant overhead, dominated by latency for all ranks to spawn
        a background process + process group initialization in the background process.
        """
        pg_init_info: Optional[_ProcessGroupInitInfo] = None
        if _CHECKPOINT_PROCESS is None:
            # Find a free port on coordinator rank and broadcast
            # to all ranks.
            pg_init_info = _ProcessGroupInitInfo(process_group)

        f: Future = self._executor.submit(
            self._execute_save_impl,
            pg_init_info=pg_init_info,
            staging_future_or_state_dict=staging_future_or_state_dict,
            checkpoint_id=checkpoint_id,
            storage_writer=storage_writer,
            planner=planner,
        )
        f.add_done_callback(lambda f: self._executor.shutdown(wait=False))

        return f
