GPU Health Check for PyTorch Distributed Elastic Training
=========================================================

This module provides GPU health monitoring capabilities for distributed elastic training, based on the NVIDIA resiliency extension implementation. It enables detection of GPU recovery actions and health status monitoring to improve the reliability of distributed training workloads.

Features
--------

- **GPU Recovery Action Detection**: Monitors GPU health using NVIDIA's GPU Recovery API
- **Driver Version Validation**: Ensures compatibility with NVIDIA driver version r570 or newer
- **Thread-Safe Operations**: Safe for use in multi-threaded environments
- **Asynchronous Monitoring**: Supports both synchronous and asynchronous health checks
- **GB200 Platform Support**: Special handling for NVIDIA GB200 platforms
- **Integration with Health Check Servers**: Seamless integration with PyTorch's distributed elastic health check system

Requirements
------------

- NVIDIA GPU with CUDA support
- NVIDIA driver version r570 or newer
- PyNVML library (``pip install pynvml``)
- Python 3.7+

Quick Start
-----------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from torch.distributed.elastic.utils.gpu_health_check import create_gpu_health_check

    # Create a GPU health check for all GPUs
    gpu_check = create_gpu_health_check(
        device_index=None,  # Check all GPUs
        interval=60,        # Check every 60 seconds
        on_failure=lambda: print("GPU health check failed!")
    )

    # Perform a synchronous health check
    is_healthy = gpu_check()
    print(f"GPU health status: {'Healthy' if is_healthy else 'Unhealthy'}")

Quick Health Check
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from torch.distributed.elastic.utils.gpu_health_check import quick_gpu_health_check

    # Perform a quick health check without creating a persistent instance
    is_healthy = quick_gpu_health_check()
    print(f"Quick health check: {'Healthy' if is_healthy else 'Unhealthy'}")

Integration with Health Check Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from torch.distributed.elastic.agent.server.health_check_server import create_gpu_healthcheck_server

    # Mock callback for process health
    def process_alive_callback():
        return int(time.time())

    # Create a GPU health check server
    server = create_gpu_healthcheck_server(
        alive_callback=process_alive_callback,
        port=8080,
        timeout=30,
        gpu_device_index=None,  # Monitor all GPUs
        gpu_check_interval=30,  # Check every 30 seconds
        enable_gpu_monitoring=True,
    )

    # Start the server
    server.start()

    # Get health status
    gpu_status = server.get_gpu_health_status()
    print(f"GPU health: {gpu_status}")

    # Stop the server
    server.stop()

API Reference
-------------

GPUHealthCheck Class
~~~~~~~~~~~~~~~~~~~~

The main class for GPU health monitoring.

Constructor
^^^^^^^^^^^

.. code-block:: python

    GPUHealthCheck(
        device_index: Optional[int] = None,
        interval: int = 60,
        on_failure: Optional[Callable] = None,
    )

**Parameters:**
- ``device_index``: GPU device index to check. If ``None``, checks all GPUs.
- ``interval``: Interval in seconds between asynchronous health checks.
- ``on_failure``: Callback function to handle health check failures.

Methods
^^^^^^^

- ``__call__() -> bool``: Perform a synchronous health check
- ``async_check() -> None``: Start asynchronous health monitoring
- ``is_gb200_platform() -> bool``: Check if running on GB200 platform

PynvmlMixin Class
~~~~~~~~~~~~~~~~~

Mixin class providing PyNVML functionality.

Methods
^^^^^^^

- ``check_pynvml_availability() -> bool``: Check if PyNVML is available
- ``is_gb200_platform() -> bool``: Detect GB200 platform
- ``get_gb200_static_mapping() -> dict``: Get GB200 GPU-to-NIC mapping

GPUHealthCheckServer Class
~~~~~~~~~~~~~~~~~~~~~~~~~~

Health check server with GPU monitoring capabilities.

Constructor
^^^^^^^^^^^

.. code-block:: python

    GPUHealthCheckServer(
        alive_callback: Callable[[], int],
        port: int,
        timeout: int,
        gpu_device_index: Optional[int] = None,
        gpu_check_interval: int = 60,
        enable_gpu_monitoring: bool = True,
    )

Methods
^^^^^^^

- ``start()``: Start the health check server
- ``stop()``: Stop the health check server
- ``get_gpu_health_status() -> dict``: Get current GPU health status
- ``get_health_summary() -> dict``: Get comprehensive health summary

GPU Recovery Actions
--------------------

The health check system can detect the following GPU recovery actions:

- **NVML_GPU_RECOVERY_ACTION_NONE**: GPU is healthy, no action needed
- **NVML_GPU_RECOVERY_ACTION_GPU_RESET**: GPU requires a reset
- **NVML_GPU_RECOVERY_ACTION_NODE_REBOOT**: Node requires a reboot
- **NVML_GPU_RECOVERY_ACTION_DRAIN_P2P**: Peer-to-peer traffic needs to be drained
- **NVML_GPU_RECOVERY_ACTION_DRAIN_AND_RESET**: GPU operating at reduced capacity

Error Handling
--------------

The system gracefully handles various error conditions:

- **PyNVML not available**: Health checks are disabled with a warning
- **Driver version too old**: Health checks are disabled for drivers older than r570
- **NVML errors**: Errors are logged and health checks return ``False``

Thread Safety
-------------

All GPU health check operations are thread-safe using a reentrant lock (``threading.RLock``). This allows multiple threads to safely perform health checks simultaneously.

Examples
--------

See the following example files:
- ``torch/distributed/elastic/examples/example_gpu_health_check.py``: Comprehensive usage examples
- ``test/distributed/elastic/agent/server/test/gpu_health_check_nvidia_test.py``: Test suite demonstrating functionality

Integration with Distributed Training
-------------------------------------

The GPU health check system integrates seamlessly with PyTorch's distributed elastic training:

1. **Health Check Servers**: Use ``GPUHealthCheckServer`` in your elastic agent
2. **Monitoring Loops**: Integrate health checks into training loops
3. **Failure Handling**: Implement custom failure callbacks for recovery actions
4. **Status Reporting**: Use health status APIs for monitoring and alerting

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

1. **"PyNVML not installed"**: Install PyNVML with ``pip install pynvml``
2. **"Driver version too old"**: Update to NVIDIA driver version r570 or newer
3. **"GPU health monitoring not available"**: Check GPU availability and driver compatibility

Debug Information
~~~~~~~~~~~~~~~~~

Enable debug logging to see detailed health check information:

.. code-block:: python

    import logging
    logging.getLogger('torch.distributed.elastic.utils.gpu_health_check').setLevel(logging.DEBUG)

License
-------

This code is licensed under the BSD-style license found in the PyTorch repository.
