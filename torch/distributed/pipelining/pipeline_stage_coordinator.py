# Copyright (c) Meta Platforms, Inc. and affiliates
"""
Pipeline Stage Coordinator Interface for Pipeline Parallelism

This module provides a pluggable interface for device-specific pipeline stage coordination
in pipeline parallelism. It allows different device types (CUDA, CPU, XLA) to extend
the existing pipeline infrastructure with device-specific coordination logic while maintaining
backward compatibility.

Custom pipeline stage coordinators can be registered by:

    from torch.distributed.pipelining.pipeline_stage_coordinator import register_pipeline_stage_coordinator

    def create_my_coordinator():
        return MyCustomPipelineStageCoordinator()

    device = torch.device("my_device")
    register_pipeline_stage_coordinator(device, create_my_coordinator)
"""

import copy
import logging
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist


logger = logging.getLogger(__name__)

__all__ = [
    "PipelineStageCoordinatorError",
    "PipelineStageCoordinator",
    "PipelineStageCoordinatorRegistry",
    "register_pipeline_stage_coordinator",
    "get_pipeline_stage_coordinator",
    "pipeline_stage_coordinator_registry",
]


class PipelineStageCoordinatorError(Exception):
    """Base exception for pipeline stage coordination errors."""


class PipelineStageCoordinator:
    """Base class for device-specific pipeline stage coordination with default implementations.

    This class provides default implementations.
    Device-specific coordinators can inherit from this class and override only the methods
    that need custom behavior.

    Note:
        Pipeline parallelism users normally **do not** need to implement their own
        ``PipelineStageCoordinator``. The default implementation is suitable for most users.
    """

    def __init__(self, device: torch.device, group: dist.ProcessGroup):
        self._device = device
        self._group = group

    def create_stage_tensor_metadata(self, tensor: torch.Tensor) -> torch.Tensor:
        """Create metadata object from tensor for pipeline stage communication.

        Args:
            tensor: The tensor to create metadata from

        Returns:
            Meta tensor that can be sent via dist.send_object_list
        """
        logger.debug(
            "Creating stage tensor metadata from tensor with shape %s, dtype %s, device %s",
            tensor.shape,
            tensor.dtype,
            tensor.device,
        )
        return tensor.to("meta")

    def create_stage_communication_buffer(
        self, metadata: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Create tensor buffer from metadata for pipeline stage communication.

        Args:
            metadata: The metadata object received from another stage
            device: The device to create the buffer on

        Returns:
            Empty tensor buffer for pipeline stage communication
        """
        logger.debug(
            "Creating stage communication buffer from metadata with shape %s, dtype %s on device %s",
            metadata.shape,
            metadata.dtype,
            device,
        )
        return torch.empty(metadata.shape, dtype=metadata.dtype, device=device)

    def infer_pipeline_stage_outputs(
        self,
        submod: torch.nn.Module,
        input_metadata: tuple[Any, ...],
        kwargs: Optional[dict] = None,
    ) -> tuple[Any, ...]:
        """Infer pipeline stage output metadata from input metadata. Override for device-specific logic.

        This method is used during pipeline stage initialization to determine the shapes and dtypes
        of outputs that will be produced during the forward pass, which are needed for setting up
        pipeline communication buffers between pipeline stages.

        Args:
            submod: The submodule to run forward inference on
            input_metadata: tuple of input metadata objects (e.g., meta tensors, XLA tensor specs)
            kwargs: Optional keyword arguments for the submodule forward pass

        Returns:
            tuple of output metadata objects representing the forward pass outputs
        """
        logger.debug(
            "Inferring pipeline stage outputs for submodule %s with %d input metadata objects",
            submod.__class__.__name__,
            len(input_metadata),
        )

        # Default implementation: convert to meta tensors and run inference
        processed_args = []
        for meta in input_metadata:
            if hasattr(meta, "to"):
                processed_args.append(meta.to("meta"))
            else:
                processed_args.append(meta)

        with torch.no_grad():
            cloned_submod = copy.deepcopy(submod)
            meta_submod = cloned_submod.to_empty(device="meta")
            outputs = meta_submod(*processed_args, **(kwargs or {}))

        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]

        # Convert outputs to metadata
        outputs_meta = []
        for output in outputs:
            metadata = self.create_stage_tensor_metadata(output)
            outputs_meta.append(metadata)

        logger.debug(
            "Inferred %d pipeline stage output metadata objects", len(outputs_meta)
        )
        return tuple(outputs_meta)


PipelineStageCoordinatorCreator = Callable[[], PipelineStageCoordinator]


class PipelineStageCoordinatorRegistry:
    """Registry for pipeline stage coordinators following PyTorch's dispatch pattern."""

    _registry: dict[str, PipelineStageCoordinatorCreator]
    _default_coordinator_creator: Optional[PipelineStageCoordinatorCreator]

    def __init__(self) -> None:
        self._registry = {}
        self._default_coordinator_creator = None

    def register(
        self, device: torch.device, creator: PipelineStageCoordinatorCreator
    ) -> None:
        """Register a pipeline stage coordinator for a device.

        Args:
            device: The device associated with the coordinator (e.g. 'cuda', 'cpu', 'xla')
            creator: Factory function that creates the coordinator instance
        """
        if not device:
            raise ValueError("The device type must be a non-empty string.")

        current_creator: Optional[PipelineStageCoordinatorCreator]
        try:
            current_creator = self._registry[device.type]
        except KeyError:
            current_creator = None

        if current_creator is not None and current_creator != creator:
            raise ValueError(
                f"The device type '{device.type}' cannot be registered with '{creator}' as it "
                f"is already registered with '{current_creator}'."
            )

        logger.debug(
            "Registering pipeline stage coordinator for device type: %s", device.type
        )
        self._registry[device.type] = creator

    def set_default_coordinator(self, creator: PipelineStageCoordinatorCreator) -> None:
        """Set the default coordinator creator for unregistered device types.

        Args:
            creator: Factory function that creates the default coordinator instance
        """
        logger.debug("Setting default pipeline stage coordinator")
        self._default_coordinator_creator = creator

    def create_coordinator(
        self, device: torch.device, group: dist.ProcessGroup
    ) -> PipelineStageCoordinator:
        """Create a pipeline stage coordinator for the given device.

        Args:
            device: The device to create a coordinator for

        Returns:
            The appropriate pipeline stage coordinator for the device

        Raises:
            NotImplementedError: If no coordinator is registered for the device type and no default is set
        """
        try:
            creator = self._registry[device.type]
            logger.debug(
                "Using registered coordinator for device type: %s", device.type
            )
        except KeyError:
            if self._default_coordinator_creator is not None:
                creator = self._default_coordinator_creator
                logger.debug(
                    "Using default coordinator for device type: %s", device.type
                )
            else:
                raise NotImplementedError(  # noqa: B904
                    f"No pipeline stage coordinator registered for device type '{device.type}' and no default coordinator set. "
                    f"Available device types: {list(self._registry.keys())}. "
                    f"Did you forget to call `register_pipeline_stage_coordinator`?"
                )

        coordinator = creator(device, group)
        logger.debug(
            "Created pipeline stage coordinator %s for device %s",
            coordinator.__class__.__name__,
            device.type,
        )
        return coordinator


def _create_default_coordinator(device, group) -> PipelineStageCoordinator:
    """Factory function for default pipeline stage coordinator."""
    return PipelineStageCoordinator(device, group)


# Global registry instance
pipeline_stage_coordinator_registry = PipelineStageCoordinatorRegistry()


def register_pipeline_stage_coordinator(
    device: torch.device, creator: PipelineStageCoordinatorCreator
) -> None:
    """Register a pipeline stage coordinator for a device type.

    Args:
        device: The device (e.g., torch.device('cuda'), torch.device('xla'))
        creator: Factory function that creates the coordinator instance

    Example:
        >>> def create_my_coordinator():
        ...     return MyCustomPipelineStageCoordinator()
        >>> register_pipeline_stage_coordinator(
        ...     torch.device("my_device"), create_my_coordinator
        ... )
    """
    pipeline_stage_coordinator_registry.register(device, creator)


def get_pipeline_stage_coordinator(
    device: torch.device, group: dist.ProcessGroup
) -> PipelineStageCoordinator:
    """Get appropriate pipeline stage coordinator for device.

    Args:
        device: The device to get a coordinator for

    Returns:
        The appropriate pipeline stage coordinator for the device

    Raises:
        NotImplementedError: If no coordinator is registered for the device type and no default is set
    """
    return pipeline_stage_coordinator_registry.create_coordinator(device, group)


def _register_default_stage_coordinator() -> None:
    """Register default pipeline stage coordinator."""
    # Set the default coordinator that will be used for any unregistered device types
    pipeline_stage_coordinator_registry.set_default_coordinator(
        _create_default_coordinator
    )
    logger.debug("Registered default pipeline stage coordinator")
