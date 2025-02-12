import abc
import json
import os
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from enum import Enum
from multiprocessing.connection import PipeConnection
from typing import Any, Dict, Optional, Union

import torch
import torch.multiprocessing as mp

from torch.distributed.checkpoint._v2._base import (
    Barrier,
    CheckpointContext,
    CheckpointerBase,
    CheckpointingConfig,
    CheckpointWriterBase,
    ManifestBuilder,
    ModelStore,
    RankInfo,
    SerializationFormat,
)
from torch.distributed.checkpoint._v2._checkpoint_layout import CheckpointLayoutBase
from torch.distributed.checkpoint._v2._metadata import Metadata
from torch.distributed.checkpoint._v2._utils import wrap_future


class StagingMethod(abc.ABC):
    """
    StagingMethod is an abstract base class for staging methods.
    """

    @abc.abstractmethod
    def stage(
        self,
        state_dict: Dict[str, Any],
        metadata: Metadata,
        context: CheckpointContext,
    ) -> Future[Dict[str, Any]]:
        pass

    @abc.abstractmethod
    def close(self) -> None:
        pass


class A0CStagingMethod(StagingMethod):
    def __init__(
        self,
        rank_info: RankInfo,
        config: CheckpointingConfig,
    ):
        self._rank_info = rank_info
        self._staging_executor = None
        self._staging_stream = None

        self._init_buffer_in_thread = config.init_buffer_in_thread

        if torch.cuda.is_available():
            self.staging_stream = torch.cuda.Stream()

        if config.stage_in_thread:
            self._staging_executor = ThreadPoolExecutor(max_workers=1)

    def init_buffer(self, state_dict: Dict[str, Any]):
        pass

    def stage(
        self,
        state_dict: Dict[str, Any],
        metadata: Metadata,
        context: CheckpointContext,
    ) -> Future[Dict[str, Any]]:
        if self._staging_executor is None:
            result = Future()
            result.set_result(self._stage(state_dict, metadata, context))
        else:
            result = self._staging_executor.submit(
                self._stage,
                state_dict,
                metadata,
                context,
            )
        return result

    def _stage(
        self, state_dict: Dict[str, Any], metadata: Metadata, context: CheckpointContext
    ) -> Dict[str, Any]:
        with (
            torch.cuda.stream(self._staging_stream)
            if torch.cuda.is_available()
            else nullcontext()
        ):
            copy = {}
            # copy = self._copy_state_dict(
            #     checkpoint.model_state,
            #     self.model_cache,
            #     non_blocking=True,
            #     block_every_n_tensors=self.block_every_n_tensors,
            # )
            # waits for the enqued copy operations to finish.
            if torch.cuda.is_available():
                self.staging_stream.synchronize()

        return copy

    def close(self) -> None:
        # clean up the buffer
        pass


class RequestType(Enum):
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
    payload: Dict[str, Any]


@dataclass
class WorkerResponse:
    request_type: RequestType
    success: bool = False
    error_msg: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None


class CheckpointWorkerProcess:
    def __init__(self, parent: PipeConnection, writer: CheckpointWriterBase):
        """
        For use in async checkpointing. Initialize the SubProcessWorker with a connection object.
        :param conn: A connection object for communication.
        """
        self._parent = parent
        self._writer = writer

    def run(self):
        """Run the worker to process messages from the parent."""
        while True:
            request = self._parent.recv()

            if request.request_type == RequestType.TERMINATE_PROCESS:
                print("Exiting subprocess.")
                self._parent.send(WorkerResponse(RequestType.TERMINATE_PROCESS))
                break
            elif request.request_type == RequestType.WRITE_CHECKPOINT:
                print(f"Subprocess received: {request.op}")
                self._writer.write(
                    state_dict=request.attrs["state_dict"],
                    metadata=request.attrs["metadata"],
                    context=request.context,
                    root_dir=request.attrs["root_dir"],
                )
                self._parent.send(WorkerResponse(RequestType.WRITE_CHECKPOINT))


class CheckpointWriter(CheckpointWriterBase):
    def __init__(
        self,
        config: Any,
        rank_info: RankInfo,
        storage: ModelStore,
        checkpoint_layout: CheckpointLayoutBase,
        serialization_format: SerializationFormat,
        barrier: Barrier,
    ):
        """
        Writes the state_dict to storage.

        Args:
            config (Any): The config to use for the checkpoint.
            rank_info (RankInfo): The rank info to use for the checkpoint.
            storage (Storage): The storage to use for the checkpoint.
            checkpoint_layout (CheckpointLayout): The layout to use for the checkpoint.
            serialization_format (SerializationFormat): The serialization format to use for the checkpoint.
        """

        self._config = config
        self._rank_info = rank_info
        self._storage = storage
        self._layout = checkpoint_layout
        self._serialization_format = serialization_format
        self._barrier = barrier

    def write(
        self,
        state_dict: Union[Future[dict[str, Any]], dict[str, Any]],
        metadata: Metadata,
        context: Any,
        root_dir: str,
    ) -> Optional[Future[None]]:
        """
        Writes the state_dict to storage.

        Args:
            state_dict (dict[str, Any]): The state_dict to write.
            manifest (Optional[Manifest]): The manifest to write.
            context (Any): The context to write.
            path (str): The path to write the checkpoint to.

        Returns:
            str: The path to the checkpoint.
        """
        # naive example for now
        if isinstance(state_dict, Future):
            state_dict = state_dict.result()

        metadata_path = "metadata.json"
        if self._config.save_manifest_with_checkpoint:
            with self._storage.open(os.path.join(root_dir, metadata_path)) as f:
                f.write(json.dumps(asdict(metadata)).encode("utf-8"))

        file_paths = self._layout.get_file_mappings_for_write(
            self._rank_info.global_rank, state_dict
        )

        for file_path, obj in file_paths.items():
            with self._storage.open(os.path.join(root_dir, file_path)) as f:
                self._serialization_format.serialize(obj, f)

        if self._config.use_barrier_for_save_completion:
            self._barrier.wait(self._config.barrier_timeout)


class RemoteWriter(CheckpointWriterBase):
    """
    A checkpoint writer that writes checkpoints to a remote process.
    """

    def __init__(self, writer: CheckpointWriter):
        self._writer = writer
        self._write_executor = ThreadPoolExecutor(max_workers=1)

        spawn_context = mp.get_context("spawn")
        self._parent_end, self._child_end = spawn_context.Pipe()
        # close the child end of the pipe so recv on it will fail
        # fast when the child process is terminated unexpectedly.
        self._child_end.close()
        # init remote worker
        self._remote_worker = CheckpointWorkerProcess(self._child_end, self._writer)
        self._remote_worker_process = spawn_context.Process(
            target=self._remote_worker.run, args=()
        )
        self._remote_worker_process.start()

    def write(
        self,
        state_dict: Union[Future[Dict[str, Any]], Dict[str, Any]],
        metadata: Metadata,
        context: CheckpointContext,
        root_dir: str,
    ) -> Optional[Future[None]]:

        # wait for staging state_dict to be available
        if isinstance(state_dict, Future):
            state_dict = state_dict.result()

        return self._write_executor.submit(
            self._write,
            state_dict,
            metadata,
            context,
            root_dir,
        )

    def _write(
        self,
        state_dict: Dict[str, Any],
        metadata: Metadata,
        context: CheckpointContext,
        root_dir: str,
    ) -> None:

        self._send(
            request_type=RequestType.WRITE_CHECKPOINT,
            payload={
                "state_dict": state_dict,
                "metadata": metadata,
                "root_dir": root_dir,
                "context": context,
            },
        )
        # wait for response
        self._recv()

    def _send(self, request_type: RequestType, payload: Dict[str, Any]):
        self._parent_end.send(
            WorkerRequest(
                request_type=request_type,
                payload=payload,
            )
        )

    def _recv(self) -> Optional[Dict[str, Any]]:
        try:
            response = self._parent_end.recv()
            if response.success is False:
                raise RuntimeError(
                    f"Unexpected response from worker process: {response.error_msg}"
                )
            return response.payload
        except EOFError:
            raise RuntimeError("Child process terminated unexpectedly.")

    def close(self):
        self._send(request_type=RequestType.TERMINATE_PROCESS, payload={})
        self._recv()  # wait for response
        self._write_executor.shutdown(wait=True)


class Checkpointer(CheckpointerBase):

    def __init__(
        self,
        config: CheckpointingConfig,
        rank_info: RankInfo,
        staging_method: StagingMethod,
        writer: CheckpointWriter,
        manifest_builder: Optional[ManifestBuilder] = None,
    ):
        self._config = config
        self._rank_info = rank_info
        self._staging_method = staging_method
        self._pending_future = None
        self._staging_method = A0CStagingMethod(rank_info, config)
        self._cached_metadata: Optional[Metadata] = None
        self._manifest_builder = manifest_builder
        self._writer = writer

    def save(
        self,
        state_dict: dict[str, Any],
        context: CheckpointContext,
        root_dir: str,
        use_cached_manifest: bool = False,
    ) -> Optional[tuple[Future[None], Future[None]]]:

        # wait for previous checkpoint to finish
        if self._pending_future is not None:
            self._pending_future.result()

        if (
            self._config.save_manifest_with_checkpoint
            and self._manifest_builder is not None
        ):
            if not use_cached_manifest or self._cached_metadata is None:
                manifest = self._manifest_builder.buid_manifest(
                    state_dict=state_dict,
                    context=context,
                )
                self._cached_metadata = Metadata(manifest)

        if self._cached_metadata is None:
            self._cached_metadata = Metadata(None)

        staging_result = state_dict
        if self._config.async_checkpointing:
            staging_result = self._staging_method.stage(
                state_dict=state_dict,
                metadata=self._cached_metadata,
                context=context,
            )

            self._pending_future = self._writer.write(
                staging_result, self._cached_metadata, context, root_dir
            )
            assert self._pending_future is not None
            return wrap_future(staging_result), self._pending_future

        else:
            self._writer.write(staging_result, self._cached_metadata, context, root_dir)
            return None
