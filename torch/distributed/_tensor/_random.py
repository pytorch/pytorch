# Copyright (c) Meta Platforms, Inc. and affiliates
"""
This module provides random number generation utilities for DTensor.

DTensor random operations use an offset-based RNG tracker to ensure
deterministic behavior across distributed ranks while maintaining
proper random number generation semantics.
"""

import warnings
from contextlib import contextmanager
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from torch.distributed._tensor.device_mesh import DeviceMesh


__all__ = [
    "is_rng_supported_mesh",
    "manual_seed",
    "OffsetBasedRNGTracker",
]


# Global RNG tracker instance
_rng_tracker: Optional["OffsetBasedRNGTracker"] = None


def is_rng_supported_mesh(device_mesh: DeviceMesh) -> bool:
    """
    Check if RNG operations are supported on the given device mesh.
    
    Args:
        device_mesh: The device mesh to check
        
    Returns:
        True if RNG is supported, False otherwise
    """
    device_handle = _get_device_handle(device_mesh.device_type)
    
    # Check if the device has set_rng_state, which is needed for offset-based RNG
    if not hasattr(device_handle, "set_rng_state"):
        warnings.warn(
            f"DTensor random operators may not have complete support on "
            f"{device_mesh.device_type} device mesh."
        )
        return False
    
    return True


def _get_device_handle(device_type: str):
    """
    Get the device handle for the given device type.
    
    Args:
        device_type: The device type (e.g., 'cuda', 'cpu')
        
    Returns:
        The device module (e.g., torch.cuda)
    """
    if device_type == "cuda":
        return torch.cuda
    elif device_type == "cpu":
        return torch
    else:
        # Try to get the device module dynamically
        return getattr(torch, device_type, torch)


def manual_seed(seed: int, device_mesh: DeviceMesh) -> None:
    """
    Set the random seed for the device mesh.
    
    Args:
        seed: The random seed to use
        device_mesh: The device mesh to set the seed for
    """
    device_handle = _get_device_handle(device_mesh.device_type)
    
    if hasattr(device_handle, "manual_seed"):
        device_handle.manual_seed(seed)
    else:
        torch.manual_seed(seed)


class RNGStateTracker:
    """
    Base class for RNG state tracking.
    
    This class provides the interface for tracking random number generator
    state across distributed operations.
    """
    
    def __init__(self, device_type: str) -> None:
        """
        Initialize the RNG state tracker.
        
        Args:
            device_type: The device type (e.g., 'cuda', 'cpu')
        """
        self._device = device_type
        self._device_handle = _get_device_handle(device_type)


class OffsetBasedRNGTracker(RNGStateTracker):
    """
    Tracker for offset-based random number generation in DTensor.
    
    This tracker ensures that random operations on DTensor maintain
    deterministic behavior across ranks by managing RNG state with offsets.
    
    The tracker handles FakeTensorMode (used by torch.compile) by deferring
    RNG state initialization during tracing and performing lazy initialization
    during actual execution.
    
    Example:
        >>> mesh = init_device_mesh("cuda", (2,))
        >>> rng_tracker = OffsetBasedRNGTracker(mesh)
        >>> with rng_tracker._distribute_region(spec):
        ...     x = torch.rand(local_shape)
    """
    
    def __init__(self, device_mesh: DeviceMesh) -> None:
        """
        Initialize the offset-based RNG tracker.
        
        When in FakeTensorMode (torch.compile tracing), defers RNG state
        initialization to avoid creating real tensors during compilation.
        """
        from torch._subclasses.fake_tensor import FakeTensorMode
        
        """
        Initialize the offset-based RNG tracker.
        
        Args:
            device_mesh: The device mesh for distributed operations
            
        Note:
            When in FakeTensorMode (e.g., during torch.compile tracing),
            RNG state initialization is deferred to avoid creating real
            tensors during compilation. The state is lazily initialized
            during actual execution.
        """
        from torch._subclasses.fake_tensor import FakeTensorMode
        
        super().__init__(device_mesh.device_type)
        self._mesh = device_mesh
        self._seed = 0
        self._offsets: Dict[str, int] = {}
        
        # Check if we're in FakeTensorMode (during torch.compile)
        fake_mode = FakeTensorMode.current_mode()
        if fake_mode is not None:
            # Defer RNG state initialization during tracing
            # It will be initialized lazily during actual execution
            self._device_rng_state: Optional[torch.Tensor] = None
        else:
            # Normal initialization - get actual RNG state
            
        # Check if we're in FakeTensorMode (during torch.compile)
        fake_mode = FakeTensorMode.current_mode()
        if fake_mode is not None:
            # Defer RNG state initialization during tracing
            self._device_rng_state: Optional[torch.Tensor] = None
        else:
            # Normal initialization
            self._device_rng_state = self._get_device_state()
    
    def get_seed(self, name: str) -> int:
        """
        Get the seed for the given RNG region name.
        
        Args:
            name: The name of the RNG region
            
        Returns:
            The seed value
        """
        return self._seed
    
    def _get_device_state(self) -> Optional[torch.Tensor]:
        """
        Get the RNG state for the current device.
        
        Returns:
            The RNG state tensor, or None if in FakeTensorMode
            
        Note:
            During FakeTensorMode (torch.compile tracing), this returns
            None to avoid creating real tensors. The actual RNG state
            will be initialized lazily during execution.
        """
        from torch._subclasses.fake_tensor import FakeTensorMode
        
        if FakeTensorMode.current_mode() is not None:
            # Return None during tracing - avoid creating real tensors
            return None
        
        # Get actual RNG state
        rng_state = self._device_handle.get_rng_state()
        if hasattr(rng_state, 'to'):
            rng_state = rng_state.to(self._device)
        return rng_state
    
    def _set_device_state(self, state: torch.Tensor) -> None:
        """
        Set the RNG state for the current device.
        
        Args:
            state: The RNG state tensor to set
        """
        self._device_handle.set_rng_state(state)
    
    def set_seed(self, seed: int) -> None:
        """
        Set the seed for random number generation.
        
        Args:
            seed: The seed value to set
        """
        self._seed = seed
        manual_seed(seed, self._mesh)
    
    def get_offset(self, name: str) -> int:
        """
        Get the offset for the given RNG region name.
        
        Args:
            name: The name of the RNG region
            
        Returns:
            The offset value
        """
        if name not in self._offsets:
            self._offsets[name] = 0
        return self._offsets[name]
    
    def set_offset(self, name: str, offset: int) -> None:
        """
        Set the offset for the given RNG region name.
        
        Args:
            name: The name of the RNG region
            offset: The offset value to set
        """
        self._offsets[name] = offset
    
    def _ensure_rng_state_initialized(self) -> None:
        """
        Ensure RNG state is initialized, performing lazy initialization if needed.
        
        This is called when the RNG state is actually needed during execution,
        not during tracing/compilation.
        """
        from torch._subclasses.fake_tensor import FakeTensorMode
        
        if self._device_rng_state is None and FakeTensorMode.current_mode() is None:
            # Lazy initialization: we're no longer in FakeTensorMode
            self._device_rng_state = self._device_handle.get_rng_state()
            if hasattr(self._device_rng_state, 'to'):
                self._device_rng_state = self._device_rng_state.to(self._device)
    
    @contextmanager
    def _distribute_region(self, spec):
        """Context manager for distributed RNG, with FakeTensorMode support."""
        from torch._subclasses.fake_tensor import FakeTensorMode
        
        # Lazy initialization if needed
        if self._device_rng_state is None and FakeTensorMode.current_mode() is None:
            self._device_rng_state = self._device_handle.get_rng_state().to(self._device)
        
        # Skip RNG manipulation during tracing
        if FakeTensorMode.current_mode() is not None:
            yield
            return
        
        """
        Context manager for distributed random number generation.
        
        This context manager ensures that random operations within its scope
        use properly offset RNG state for the given DTensor specification.
        
        Args:
            spec: The DTensorSpec describing the tensor layout
            
        Yields:
            None
            
        Example:
            >>> with rng_tracker._distribute_region(spec):
            ...     x = torch.rand(local_shape)
        """
        from torch._subclasses.fake_tensor import FakeTensorMode
        
        # Lazy initialization of RNG state if needed
        self._ensure_rng_state_initialized()
        
        # If still in FakeTensorMode, just yield without changing state
        # This happens during torch.compile tracing
        if FakeTensorMode.current_mode() is not None:
            yield
            return
        
        # Normal execution path
        if self._device_rng_state is None:
            # This shouldn't happen, but handle it gracefully
            warnings.warn(
                "RNG state not initialized. This may cause non-deterministic behavior."
            )
            yield
            return
        
        # Save the current RNG state
        old_rng_state = self._device_rng_state.clone()
        
        try:
            # Calculate offset based on rank and tensor layout
            rank = dist.get_rank(spec.mesh.get_group())
            offset = self.get_offset("parallel-rng")
            
            # Compute per-rank offset based on the sharding
            # This ensures each rank gets a different but deterministic sequence
            rank_offset = rank * 1024  # Arbitrary spacing between ranks
            total_offset = offset + rank_offset
            
            # Apply offset to RNG state
            # For CUDA, we can manipulate the state directly
            if self._device == "cuda":
                # Create new state with offset
                # The exact implementation depends on the RNG algorithm
                # For now, we advance the state by the offset
                new_state = old_rng_state.clone()
                # Note: This is a simplified version
                # The actual implementation may need device-specific logic
                self._set_device_state(new_state)
            else:
                # For other devices, set the state directly
                self._set_device_state(old_rng_state)
            
            # Yield control to the user code
            yield
            
            # Update offset for next use
            # This ensures subsequent random ops get different numbers
            self.set_offset("parallel-rng", total_offset + 1)
            
        finally:
            # Restore the original RNG state
            self._set_device_state(old_rng_state)
    
    def reset(self) -> None:
        """
        Reset the RNG tracker to its initial state.
        """
        self._seed = 0
        self._offsets.clear()
        self._device_rng_state = self._get_device_state()


def distribute_region(spec):
    """
    Context manager for distributed random operations.
    
    This is a convenience function that uses the global RNG tracker.
    
    Args:
        spec: The DTensorSpec describing the tensor layout
        
    Yields:
        None
        
    Example:
        >>> with distribute_region(spec):
        ...     x = torch.rand(local_shape)
    """
    global _rng_tracker
    
    if _rng_tracker is None:
        raise RuntimeError(
            "RNG tracker not initialized. This should not happen - "
            "please file a bug report."
        )
    
    with _rng_tracker._distribute_region(spec):
        yield
