#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional, Dict, Any
import time
import threading

from torch.distributed.elastic.utils.logging import get_logger
from torch.distributed.elastic.utils.gpu_health_check import GPUHealthCheck


log = get_logger(__name__)

__all__ = ["HealthCheckServer", "create_healthcheck_server", "GPUHealthCheckServer", "create_gpu_healthcheck_server"]


class HealthCheckServer:
    """
    Interface for health check monitoring server, which can be extended
    by starting tcp/http server on the specified port.

    Args:

        alive_callback: Callable[[], int], callback to last progress time of agent

        port: int, port number to start tcp/http server

        timeout: int, timeout seconds to decide agent is alive/dead
    """

    _alive_callback: Callable[[], int]
    _port: int
    _timeout: int

    def __init__(
        self, alive_callback: Callable[[], int], port: int, timeout: int
    ) -> None:
        self._alive_callback = alive_callback
        self._port = port
        self._timeout = timeout

    def start(self) -> None:
        """
        Unsupported functionality for Pytorch, doesn't start any health check server
        """
        log.warning("No health check server started")

    def stop(self) -> None:
        """
        Function to stop health check server
        """
        log.info("Stopping noop health check server.")


class GPUHealthCheckServer(HealthCheckServer):
    """
    GPU-aware health check monitoring server for distributed elastic training.
    
    This server extends the basic HealthCheckServer with GPU health monitoring
    capabilities based on the NVIDIA resiliency extension implementation.
    
    Args:
        alive_callback: Callable[[], int], callback to last progress time of agent
        port: int, port number to start tcp/http server
        timeout: int, timeout seconds to decide agent is alive/dead
        gpu_device_index: Optional[int], specific GPU device to monitor. If None, monitors all GPUs.
        gpu_check_interval: int, interval in seconds for GPU health checks
        enable_gpu_monitoring: bool, whether to enable GPU health monitoring
    """
    
    def __init__(
        self,
        alive_callback: Callable[[], int],
        port: int,
        timeout: int,
        gpu_device_index: Optional[int] = None,
        gpu_check_interval: int = 60,
        enable_gpu_monitoring: bool = True,
    ) -> None:
        super().__init__(alive_callback, port, timeout)
        
        self._gpu_health_check: Optional[GPUHealthCheck] = None
        self._gpu_monitoring_enabled = enable_gpu_monitoring
        self._gpu_device_index = gpu_device_index
        self._gpu_check_interval = gpu_check_interval
        
        self._gpu_health_thread: Optional[threading.Thread] = None
        self._gpu_health_stop_event = threading.Event()
        self._last_gpu_health_status: bool = True
        self._gpu_health_lock = threading.Lock()
        
        if self._gpu_monitoring_enabled:
            self._init_gpu_monitoring()
    
    def _init_gpu_monitoring(self) -> None:
        """Initialize GPU health monitoring."""
        try:
            self._gpu_health_check = GPUHealthCheck(
                device_index=self._gpu_device_index,
                interval=self._gpu_check_interval,
                on_failure=self._on_gpu_failure,
            )
            
            if self._gpu_health_check.enabled:
                log.info(f"GPU health monitoring initialized for device {self._gpu_device_index or 'all'}")
            else:
                log.warning("GPU health monitoring not available or disabled")
                self._gpu_monitoring_enabled = False
        except Exception as e:
            log.error(f"Failed to initialize GPU health monitoring: {e}")
            self._gpu_monitoring_enabled = False
    
    def _on_gpu_failure(self) -> None:
        """Callback function for GPU health check failures."""
        log.warning("GPU health check detected a failure")
        # This could be extended to trigger specific recovery actions
    
    def start(self) -> None:
        """
        Start the health check server with GPU monitoring.
        """
        log.info(f"Starting GPU health check server on port {self._port}")
        
        if self._gpu_monitoring_enabled and self._gpu_health_check is not None:
            self._start_gpu_monitoring()
        
        # For now, this is a no-op server as per the original implementation
        # In a real implementation, this would start a TCP/HTTP server
        log.warning("Health check server started (no-op implementation)")
    
    def stop(self) -> None:
        """
        Stop the health check server and GPU monitoring.
        """
        log.info("Stopping GPU health check server")
        
        if self._gpu_monitoring_enabled:
            self._stop_gpu_monitoring()
    
    def _start_gpu_monitoring(self) -> None:
        """Start GPU health monitoring in a separate thread."""
        if self._gpu_health_thread is not None and self._gpu_health_thread.is_alive():
            return
        
        self._gpu_health_stop_event.clear()
        self._gpu_health_thread = threading.Thread(
            target=self._gpu_monitoring_loop,
            name="GPUHealthMonitor",
            daemon=True
        )
        self._gpu_health_thread.start()
        log.info("GPU health monitoring thread started")
    
    def _stop_gpu_monitoring(self) -> None:
        """Stop GPU health monitoring thread."""
        if self._gpu_health_thread is not None:
            self._gpu_health_stop_event.set()
            self._gpu_health_thread.join(timeout=5.0)
            if self._gpu_health_thread.is_alive():
                log.warning("GPU health monitoring thread did not stop gracefully")
            self._gpu_health_thread = None
            log.info("GPU health monitoring thread stopped")
    
    def _gpu_monitoring_loop(self) -> None:
        """Main loop for GPU health monitoring."""
        while not self._gpu_health_stop_event.is_set():
            try:
                if self._gpu_health_check is not None:
                    # Perform synchronous health check
                    is_healthy = self._gpu_health_check()
                    
                    with self._gpu_health_lock:
                        self._last_gpu_health_status = is_healthy
                    
                    if not is_healthy:
                        log.warning("GPU health check detected unhealthy state")
                    else:
                        log.debug("GPU health check passed")
                
                # Wait for next check or stop event
                self._gpu_health_stop_event.wait(self._gpu_check_interval)
                
            except Exception as e:
                log.error(f"Error in GPU monitoring loop: {e}")
                # Wait a bit before retrying
                self._gpu_health_stop_event.wait(5.0)
    
    def get_gpu_health_status(self) -> Dict[str, Any]:
        """
        Get current GPU health status.
        
        Returns:
            Dictionary containing GPU health status information.
        """
        if not self._gpu_monitoring_enabled or self._gpu_health_check is None:
            return {"error": "GPU monitoring not enabled"}
        
        with self._gpu_health_lock:
            return {
                "enabled": self._gpu_health_check.enabled,
                "device_index": self._gpu_device_index,
                "is_healthy": self._last_gpu_health_status,
                "pynvml_available": self._gpu_health_check.pynvml_available,
                "timestamp": time.time()
            }
    
    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive health summary including both process and GPU health.
        
        Returns:
            Dictionary containing overall health status.
        """
        summary = {
            "process_health": {
                "alive": True,  # This would be determined by alive_callback in real implementation
                "last_progress_time": self._alive_callback() if self._alive_callback else 0,
            },
            "gpu_health": self.get_gpu_health_status(),
            "timestamp": time.time()
        }
        
        # Determine overall health
        gpu_health = summary["gpu_health"]
        if "error" in gpu_health:
            summary["overall_healthy"] = True  # Process is healthy even if GPU monitoring fails
        else:
            summary["overall_healthy"] = gpu_health.get("is_healthy", True)
        
        return summary


def create_healthcheck_server(
    alive_callback: Callable[[], int],
    port: int,
    timeout: int,
) -> HealthCheckServer:
    """
    creates health check server object
    """
    return HealthCheckServer(alive_callback, port, timeout)


def create_gpu_healthcheck_server(
    alive_callback: Callable[[], int],
    port: int,
    timeout: int,
    gpu_device_index: Optional[int] = None,
    gpu_check_interval: int = 60,
    enable_gpu_monitoring: bool = True,
) -> GPUHealthCheckServer:
    """
    Creates a GPU-aware health check server object.
    
    Args:
        alive_callback: Callable[[], int], callback to last progress time of agent
        port: int, port number to start tcp/http server
        timeout: int, timeout seconds to decide agent is alive/dead
        gpu_device_index: Optional[int], specific GPU device to monitor. If None, monitors all GPUs.
        gpu_check_interval: int, interval in seconds for GPU health checks
        enable_gpu_monitoring: bool, whether to enable GPU health monitoring
        
    Returns:
        GPUHealthCheckServer instance
    """
    return GPUHealthCheckServer(
        alive_callback=alive_callback,
        port=port,
        timeout=timeout,
        gpu_device_index=gpu_device_index,
        gpu_check_interval=gpu_check_interval,
        enable_gpu_monitoring=enable_gpu_monitoring,
    )
