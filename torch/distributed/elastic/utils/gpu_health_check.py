#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
GPU Health Check utilities for distributed elastic training.

This module provides GPU health monitoring capabilities based on the NVIDIA
resiliency extension implementation, including GPU recovery action detection
and health status monitoring.
"""

import asyncio
import logging
import os
import threading
import traceback
from collections import defaultdict
from functools import wraps
from typing import Callable, Optional, Union

from torch.distributed.elastic.utils.logging import get_logger

# Get the logger
logger = get_logger(__name__)

# Adds basic thread safety, allowing to run health checks from multiple threads.
# This is needed for rendezvous unit tests. NOTE: It will work as long as each
# function/method that uses NVML performs NVML initialization and shutdown.
# Please follow this pattern when adding new code.
_nvml_lock = threading.RLock()


def with_pynvml_lock(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with _nvml_lock:
            return func(*args, **kwargs)

    return wrapper


class PynvmlMixin:
    def __init__(self):
        # Initialize pynvml to None
        self.pynvml = None

    def check_pynvml_availability(self) -> bool:
        try:
            import pynvml

            self.pynvml = pynvml
            return True
        except ImportError:
            logger.warning("Pynvml is not installed.")
            return False

    def is_gb200_platform(self) -> bool:
        """
        Detect if the current platform is GB200.

        Since all nodes are homogeneous on GPUs, we only need to check GPU 0.
        This allows for platforms with different numbers of GPUs (4, 8, etc.).

        Returns:
            bool: True if GB200 platform is detected, False otherwise.
        """
        if not self.pynvml:
            return False

        try:
            self.pynvml.nvmlInit()
            num_gpus = self.pynvml.nvmlDeviceGetCount()

            # Need at least one GPU to check
            if num_gpus == 0:
                return False

            # Check only GPU 0 since all nodes are homogeneous
            handle = self.pynvml.nvmlDeviceGetHandleByIndex(0)
            name = self.pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')

            # GB200 GPUs have names like "GB200" or "B200"
            return any(gpu_type in name.upper() for gpu_type in ["GB200", "B200"])

        except self.pynvml.NVMLError as e:
            logger.debug(f"NVML Error while detecting GB200: {e}")
            return False
        finally:
            try:
                self.pynvml.nvmlShutdown()
            except self.pynvml.NVMLError:
                pass

        return False

    def get_gb200_static_mapping(self) -> dict:
        """
        Get the static GPU to NIC mapping for GB200 platform.

        Returns:
            dict: Mapping from GPU rank to NIC name for GB200.
        """
        return {0: "mlx5_0", 1: "mlx5_1", 2: "mlx5_3", 3: "mlx5_4"}

    @with_pynvml_lock
    def get_gpu_pci_mapping(self):
        """
        Retrieve GPU local rank to PCI Bus mapping using pynvml.
        """
        assert (
            self.pynvml is not None
        ), "pynvml is not initialized. Ensure check_pynvml_availability() is called first."

        gpu_pci_map = {}
        try:
            self.pynvml.nvmlInit()
            num_gpus = self.pynvml.nvmlDeviceGetCount()

            for i in range(num_gpus):
                handle = self.pynvml.nvmlDeviceGetHandleByIndex(i)
                pci_info = self.pynvml.nvmlDeviceGetPciInfo(handle)
                bus_id = pci_info.busId
                if isinstance(bus_id, bytes):
                    bus_id = bus_id.decode('utf-8')
                bus_id = bus_id.lower()
                # Extract the last 12 characters (standard PCI format)
                gpu_pci_map[i] = bus_id[-12:]

        except self.pynvml.NVMLError as e:
            logger.error(f"NVML Error: {e}\n{traceback.format_exc()}")

        finally:
            try:
                self.pynvml.nvmlShutdown()
            except self.pynvml.NVMLError as e:
                logger.error(f"Failed to shut down NVML: {e}")

        return gpu_pci_map


class GPUHealthCheck(PynvmlMixin):
    def __init__(
        self,
        device_index: Optional[int] = None,
        interval: int = 60,
        on_failure: Optional[Callable] = None,
    ):
        """
        Initializes the GPUHealthCheck class.

        Args:
            device_index (Optional[int]): GPU device index to check. If None, checks all GPUs.
            interval (int): Interval in seconds between asynchronous health checks.
            on_failure (Optional[Callable]): Callback function to handle health check failures.
        """
        super().__init__()
        self.device_index = device_index
        self.interval = interval
        self.on_failure = on_failure
        self.pynvml_available = self.check_pynvml_availability()
        self.enabled = self._check_driver_version()

    @with_pynvml_lock
    def _check_driver_version(self) -> bool:
        """
        Checks if the GPU driver version supports health checks (version r570 or newer).

        Returns:
            bool: True if the driver supports health checks, False otherwise.
        """
        GPU_RECOVERY_API_MIN_DRIVER_VERSION = 570

        if not self.pynvml_available:
            logger.warning("GPU Health checks are disabled because pynvml is not available.")
            return False

        try:
            self.pynvml.nvmlInit()
            driver_version = self.pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver_version, bytes):
                driver_version = driver_version.decode('utf-8')
            self.pynvml.nvmlShutdown()

            major_version = int(driver_version.split('.')[0])

            if major_version < GPU_RECOVERY_API_MIN_DRIVER_VERSION:
                logger.warning(
                    f"Health checks disabled: GPU driver version r{major_version} is older than "
                    f"required r{GPU_RECOVERY_API_MIN_DRIVER_VERSION} for the GPU Recovery API."
                )
                return False
            return True

        except Exception as e:
            logger.warning(
                f"GPU Health checks disabled: Unable to determine driver version due to: {e}"
            )
            return False

    async def async_check(self) -> None:
        """
        Asynchronous GPU health check that runs periodically.

        Periodically checks GPU health and handles any failures if they occur.
        """
        if not self.enabled:
            return

        while True:
            await asyncio.sleep(self.interval)
            result = await self._check_health()
            if not result and self.on_failure:
                await self.on_failure()

    async def _check_health(self) -> bool:
        """
        Performs the asynchronous GPU health check.

        Returns:
            bool: True if all GPUs are healthy, False if any GPU has an issue.
        """
        return self._perform_health_check()

    def __call__(self) -> Union[Optional[Exception], bool]:
        """
        Synchronous GPU health check callable.

        Returns:
            bool: Returns True if GPUs are healthy.
        """
        if not self.enabled:
            logger.warning("Health checks are disabled; skipping synchronous check.")
            return True

        result = self._perform_health_check()
        return result

    @with_pynvml_lock
    def _perform_health_check(self) -> bool:
        """
        Core method to perform GPU health check. Used by both sync and async checks.

        Checks the recovery action needed for the specified GPU device(s).

        Returns:
            bool: True if all specified GPUs are healthy (no recovery action needed), False otherwise.
        """
        try:
            self.pynvml.nvmlInit()

            # Determine which devices to check
            devices_to_check = (
                [self.device_index]
                if self.device_index is not None
                else range(self.pynvml.nvmlDeviceGetCount())
            )

            # Check all specified devices
            for device_id in devices_to_check:
                if not self._check_gpu_health(device_id):
                    return False

            return True

        except self.pynvml.NVMLError as e:
            logger.warning(f"NVML Error: {str(e)}")
            return False
        except Exception as e:
            logger.warning(f"Unexpected Error: {str(e)}")
            return False
        finally:
            try:
                self.pynvml.nvmlShutdown()
            except Exception as e:
                logger.warning(f"Error during NVML shutdown: {str(e)}")

    def _check_gpu_health(self, device_id: int) -> bool:
        """
        Check health for a specific GPU device.

        Args:
            device_id (int): GPU device index to check.

        Returns:
            bool: True if GPU is healthy, False otherwise.
        """
        try:
            handle = self.pynvml.nvmlDeviceGetHandleByIndex(device_id)

            if not hasattr(self.pynvml, "NVML_FI_DEV_GET_GPU_RECOVERY_ACTION"):
                # PyNVML is not available so we assume the GPU is healthy
                return True

            # Get the GPU recovery action status
            recovery_action = self.pynvml.nvmlDeviceGetFieldValues(
                handle, [self.pynvml.NVML_FI_DEV_GET_GPU_RECOVERY_ACTION]
            )[0].value.uiVal

            # Interpret the recovery action
            if recovery_action == self.pynvml.NVML_GPU_RECOVERY_ACTION_NONE:
                return True
            elif recovery_action == self.pynvml.NVML_GPU_RECOVERY_ACTION_GPU_RESET:
                logger.warning(
                    f"GPU {device_id}: Requires a reset to recover. Terminate GPU processes and reset the GPU."
                )
                return False
            elif recovery_action == self.pynvml.NVML_GPU_RECOVERY_ACTION_NODE_REBOOT:
                logger.warning(
                    f"GPU {device_id}: Requires a node reboot to recover. Reboot the system."
                )
                return False
            elif recovery_action == self.pynvml.NVML_GPU_RECOVERY_ACTION_DRAIN_P2P:
                logger.warning(
                    f"GPU {device_id}: Requires peer-to-peer traffic to be drained. Terminate related processes."
                )
                return False
            elif recovery_action == self.pynvml.NVML_GPU_RECOVERY_ACTION_DRAIN_AND_RESET:
                logger.warning(
                    f"GPU {device_id}: Operating at reduced capacity. Drain existing work and reset the GPU."
                )
                return False
            else:
                logger.warning(
                    f"GPU {device_id}: Unknown recovery action status: {recovery_action}"
                )
                return False

        except self.pynvml.NVMLError as e:
            logger.warning(f"NVML Error: {str(e)}")
            return False
        except Exception as e:
            logger.warning(f"Unexpected Error: {str(e)}")
            return False


# Convenience functions for easy access
def create_gpu_health_check(
    device_index: Optional[int] = None,
    interval: int = 60,
    on_failure: Optional[Callable] = None,
) -> GPUHealthCheck:
    """
    Create a GPU health check instance.
    
    Args:
        device_index: GPU device index to check. If None, checks all GPUs.
        interval: Interval in seconds between asynchronous health checks.
        on_failure: Callback function to handle health check failures.
        
    Returns:
        GPUHealthCheck instance
    """
    return GPUHealthCheck(
        device_index=device_index,
        interval=interval,
        on_failure=on_failure,
    )


def quick_gpu_health_check(device_index: Optional[int] = None) -> bool:
    """
    Perform a quick GPU health check without creating a persistent instance.
    
    Args:
        device_index: GPU device index to check. If None, checks all GPUs.
        
    Returns:
        bool: True if GPUs are healthy, False otherwise.
    """
    health_check = create_gpu_health_check(device_index=device_index)
    try:
        return health_check()
    except Exception as e:
        logger.error(f"Quick GPU health check failed: {e}")
        return False
