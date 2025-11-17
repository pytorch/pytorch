import logging
import os
import traceback
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from multiprocessing.connection import Connection
from typing import Any, Optional, Union

import torch.multiprocessing as mp
from torch.multiprocessing.spawn import ProcessExitedException

from .checkpoint_writer import CheckpointWriter
from .types import RankInfo, STATE_DICT


logger = logging.getLogger(__name__)


@dataclass
class CheckpointProcessConfig:
    """
    Configuration options for the CheckpointProcess.

    This class provides configuration options for the checkpoint process,
    including initialization functions, timeouts, and writer configuration.

    Attributes:
        subprocess_init_timeout_secs: Maximum time in seconds to wait for subprocess initialization.
        subprocess_shutdown_timeout_secs: Maximum time in seconds to wait for subprocess shutdown.
    """

    subprocess_init_timeout_secs: int = 30
    subprocess_shutdown_timeout_secs: int = 60


class RequestType(Enum):
    PING = "ping"
    WRITE_CHECKPOINT = "write_checkpoint"
    TERMINATE_PROCESS = "exit"


@dataclass
class WorkerRequest:
    """
    A dataclass for storing the command to be sent to the worker process.
    Note: This relies on pickling to send the command to the worker process. Handle
    backward compatibility accordingly.
    """

    request_type: RequestType
    payload: dict[str, Any]


@dataclass
class WorkerResponse:
    request_type: RequestType
    success: bool
    error_msg: Optional[str] = None
    payload: Optional[dict[str, Any]] = None


class CheckpointProcess:
    """
    A checkpoint writer that writes checkpoints to a remote process.
    """

    def __init__(
        self,
        rank_info: RankInfo,
        config: CheckpointProcessConfig,
        subprocess_init_fn: Callable[[Any], None],
        subprocess_init_args: tuple[Any, ...],
        checkpoint_writer_init_fn: Callable[..., CheckpointWriter],
        checkpoint_writer_init_args: dict[str, Any],
    ):
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._rank_info = rank_info
        self._config = config
        self._subprocess_init_fn = subprocess_init_fn
        self._subprocess_init_args = subprocess_init_args
        self._checkpoint_writer_init_fn = checkpoint_writer_init_fn
        self._checkpoint_writer_init_args = checkpoint_writer_init_args
        self.process = None
        self._parent_end: Optional[Connection] = None
        self._child_end: Optional[Connection] = None

        self.process_creation_future = self._executor.submit(
            self._create_subprocess,
            config,
        )

    def _create_subprocess(
        self,
        config: CheckpointProcessConfig,
    ) -> None:
        logger.info(
            "Creating checkpoint subprocess for rank %d", self._rank_info.global_rank
        )

        spawn_context = mp.get_context("spawn")
        self._parent_end, child_end = spawn_context.Pipe()

        # Known workaround for https://github.com/pytorch/pytorch/issues/37377
        os.environ["MKL_SERVICE_FORCE_INTEL"] = "GNU"

        logger.debug("Spawning subprocess for rank_info=%s", self._rank_info)
        self.process = mp.spawn(
            fn=CheckpointProcess._subprocess,
            args=(
                self._rank_info,
                child_end,
                self._subprocess_init_fn,
                self._subprocess_init_args,
                self._checkpoint_writer_init_fn,
                self._checkpoint_writer_init_args,
            ),
            nprocs=1,
            join=False,
            daemon=True,
        )

        # close the child end of the pipe so recv on it will fail
        # fast when the child process is terminated unexpectedly.
        child_end.close()
        self._send(
            request_type=RequestType.PING,
            payload={},
        )

        logger.debug(
            "Waiting for checkpoint subprocess to initialize (timeout: %ds)",
            config.subprocess_init_timeout_secs,
        )

        # wait for the timeout or a response from subprocess
        assert self._parent_end is not None, "Parent end of pipe should be initialized"
        if not self._parent_end.poll(timeout=config.subprocess_init_timeout_secs):
            msg = f"Timed out after {config.subprocess_init_timeout_secs}s waiting for checkpoint subprocess to initialize"
            logger.error(msg)
            raise TimeoutError(msg)

        self._recv()
        logger.info("Checkpoint subprocess initialized successfully")

    @staticmethod
    def _subprocess(
        sub_rank: int,
        rank_info: RankInfo,
        parent_pipe: Connection,
        subprocess_init_fn: Callable[[Any], None],
        subprocess_init_args: tuple[Any, ...],
        checkpoint_writer_init_fn: Callable[..., CheckpointWriter],
        checkpoint_writer_init_args: dict[str, Any],
    ) -> None:
        logger.debug(
            "Checkpoint subprocess started for rank %d/%d (PID: %d)",
            rank_info.global_rank,
            rank_info.global_world_size,
            os.getpid(),
        )

        assert sub_rank == 0, "We need only one checkpointer per parent training"
        request = WorkerRequest(request_type=RequestType.PING, payload={})

        try:
            # Calling initialize callback, so we can perform app-specific initialization of the subprocess.
            subprocess_init_fn(*subprocess_init_args)

            # Initialize checkpoint writer - automatically include rank_info in init_args
            writer_init_args = dict(checkpoint_writer_init_args)
            if "rank_info" not in writer_init_args:
                writer_init_args["rank_info"] = rank_info
            checkpoint_writer = checkpoint_writer_init_fn(**writer_init_args)

            while True:
                request = parent_pipe.recv()

                if request.request_type == RequestType.PING:
                    parent_pipe.send(
                        WorkerResponse(request_type=RequestType.PING, success=True)
                    )
                elif request.request_type == RequestType.WRITE_CHECKPOINT:
                    path = request.payload["path"]
                    logger.info("Writing checkpoint to %s", path)

                    checkpoint_writer.write(
                        path=path,
                        state_dict=request.payload["state_dict"],
                        **request.payload["kwargs"],
                    )

                    logger.info("Checkpoint written successfully to %s", path)
                    parent_pipe.send(
                        WorkerResponse(RequestType.WRITE_CHECKPOINT, success=True)
                    )
                elif request.request_type == RequestType.TERMINATE_PROCESS:
                    logger.debug("Received termination request.")
                    parent_pipe.send(
                        WorkerResponse(RequestType.TERMINATE_PROCESS, success=True)
                    )
                    logger.info("Subprocess terminated gracefully")
                    break
                else:
                    error_msg = f"Unknown request type: {request.request_type}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

        except Exception as e:
            error_text = traceback.format_exc()
            logger.error(
                "Exception in subprocess  (%s): %s", type(e).__name__, error_text
            )

            # Communicating exception via the queue to the main process
            parent_pipe.send(
                WorkerResponse(
                    request_type=request.request_type,
                    success=False,
                    error_msg=error_text,
                )
            )
            parent_pipe.close()
            logger.error("Subprocess terminated due to exception: %s", e)

    def _send(self, request_type: RequestType, payload: dict[str, Any]) -> None:
        try:
            assert self._parent_end is not None, (
                "Parent end of pipe should be initialized"
            )
            self._parent_end.send(
                WorkerRequest(
                    request_type=request_type,
                    payload=payload,
                )
            )
        except OSError as e:
            error_msg = "Child process terminated unexpectedly"
            logger.error(
                "Communication failed during %s request: %s", request_type.value, e
            )
            raise RuntimeError(error_msg) from e

    def _recv(self) -> Optional[dict[str, Any]]:
        try:
            assert self._parent_end is not None, (
                "Parent end of pipe should be initialized"
            )
            response = self._parent_end.recv()
            if response.success is False:
                error_msg = (
                    f"Unexpected response from worker process: {response.error_msg}"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            return response.payload
        except (EOFError, BrokenPipeError, ConnectionResetError) as e:
            error_msg = f"Child process terminated unexpectedly: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def write(
        self,
        state_dict: Union[STATE_DICT, Future[STATE_DICT]],
        path: str,
        **kwargs: Any,
    ) -> Optional[Future[None]]:
        logger.debug("Waiting for subprocess initialization to complete")

        # wait until the process is started
        self.process_creation_future.result()

        return self._executor.submit(
            self._write,
            state_dict,
            path,
            **kwargs,
        )

    def _write(
        self,
        state_dict: Union[STATE_DICT, Future[STATE_DICT]],
        path: str,
        **kwargs: Any,
    ) -> None:
        logger.debug("Starting checkpoint write to %s", path)

        # wait for staging state_dict to be available
        if isinstance(state_dict, Future):
            logger.debug("Waiting for state_dict Future to resolve")
            sd = state_dict.result()
        else:
            sd = state_dict

        # Log state_dict info only if debug logging is enabled (performance-conscious)
        if logger.isEnabledFor(logging.DEBUG):
            if hasattr(sd, "keys"):
                logger.debug("State_dict contains %d keys", len(sd.keys()))

        self._send(
            request_type=RequestType.WRITE_CHECKPOINT,
            payload={
                "state_dict": sd,
                "path": path,
                "kwargs": kwargs,
            },
        )

        logger.debug("Waiting for write completion response")
        # wait for response
        self._recv()
        logger.debug("Checkpoint write to %s completed successfully", path)

    def close(self) -> None:
        logger.debug(
            "Closing CheckpointProcess for rank %d", self._rank_info.global_rank
        )
        self._executor.shutdown(wait=True, cancel_futures=True)

        if self.process and self.process.processes[0].is_alive():
            subprocess_pid = self.process.processes[0].pid
            # send graceful termination to sub process
            try:
                self._parent_end.send(
                    WorkerRequest(
                        request_type=RequestType.TERMINATE_PROCESS,
                        payload={},
                    )
                )
            except BrokenPipeError:
                logger.warning(
                    "BrokenPipeError when sending termination request - subprocess (PID: %d) may have already terminated",
                    subprocess_pid,
                )
                # subprocess terminated unexpectedly and below code will raise a
                # ProcessExitedException.

            logger.debug(
                "Waiting for subprocess to terminate gracefully (timeout: %ds)",
                self._config.subprocess_shutdown_timeout_secs,
            )

            try:
                if not self.process.join(
                    timeout=self._config.subprocess_shutdown_timeout_secs
                ):
                    # graceful shutdown failed, kill the process.
                    logger.warning(
                        "Subprocess (PID: %d) did not terminate gracefully within %ds, killing it",
                        subprocess_pid,
                        self._config.subprocess_shutdown_timeout_secs,
                    )
                    self.process.processes[0].kill()
                    logger.info("Subprocess killed forcefully")
            except ProcessExitedException as e:
                logger.error(
                    "ProcessExitedException during subprocess termination: %s", e
                )
                raise

        logger.debug("CheckpointProcess closed successfully")
