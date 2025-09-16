#!/usr/bin/env python3

"""
Example script demonstrating GPU health check usage in PyTorch distributed elastic training.

This script shows how to integrate GPU health monitoring into distributed training workflows.
"""

import time

from torch.distributed.elastic.utils.gpu_health_check import (
    GPUHealthCheck,
    create_gpu_health_check,
    quick_gpu_health_check,
)
from torch.distributed.elastic.agent.server.health_check_server import (
    create_gpu_healthcheck_server,
)


def example_basic_usage():
    """Example of basic GPU health check usage."""
    print("=== Basic GPU Health Check Usage ===")
    
    # Create a GPU health check for all GPUs
    gpu_check = create_gpu_health_check(
        device_index=None,  # Check all GPUs
        interval=60,        # Check every 60 seconds
        on_failure=lambda: print("⚠️  GPU health check detected a failure!")
    )
    
    print(f"GPU health monitoring enabled: {gpu_check.enabled}")
    print(f"PyNVML available: {gpu_check.pynvml_available}")
    
    if gpu_check.enabled:
        # Perform a health check
        is_healthy = gpu_check()
        print(f"Current GPU health status: {'✅ Healthy' if is_healthy else '❌ Unhealthy'}")
        
        # Quick health check
        quick_result = quick_gpu_health_check()
        print(f"Quick health check result: {'✅ Healthy' if quick_result else '❌ Unhealthy'}")
    else:
        print("GPU health monitoring not available (pynvml not installed or driver version too old)")
    
    print()


def example_server_usage():
    """Example of GPU health check server usage."""
    print("=== GPU Health Check Server Usage ===")
    
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
    
    print("Starting GPU health check server...")
    server.start()
    
    # Wait a moment for monitoring to initialize
    time.sleep(1)
    
    # Get health status
    gpu_status = server.get_gpu_health_status()
    print("GPU Health Status:")
    if "error" in gpu_status:
        print(f"  Error: {gpu_status['error']}")
    else:
        print(f"  Monitoring enabled: {gpu_status['enabled']}")
        print(f"  Device index: {gpu_status['device_index']}")
        print(f"  Is healthy: {'✅ Yes' if gpu_status['is_healthy'] else '❌ No'}")
        print(f"  PyNVML available: {gpu_status['pynvml_available']}")
    
    # Get overall health summary
    summary = server.get_health_summary()
    print("\nOverall Health Summary:")
    print(f"  Process healthy: {'✅ Yes' if summary['process_health']['alive'] else '❌ No'}")
    print(f"  Overall healthy: {'✅ Yes' if summary['overall_healthy'] else '❌ No'}")
    
    # Stop the server
    print("\nStopping GPU health check server...")
    server.stop()
    print("Server stopped.")
    print()


def example_custom_failure_handler():
    """Example of custom failure handling."""
    print("=== Custom Failure Handler Example ===")
    
    def custom_failure_handler():
        print("🚨 Custom failure handler triggered!")
        print("   This could trigger recovery actions like:")
        print("   - Restarting the training process")
        print("   - Notifying the orchestrator")
        print("   - Saving checkpoint before restart")
    
    # Create GPU health check with custom failure handler
    gpu_check = create_gpu_health_check(
        device_index=0,  # Monitor specific GPU
        interval=30,
        on_failure=custom_failure_handler,
    )
    
    print(f"GPU health check created for device 0")
    print(f"Enabled: {gpu_check.enabled}")
    
    if gpu_check.enabled:
        # Perform health check
        result = gpu_check()
        print(f"Health check result: {'✅ Healthy' if result else '❌ Unhealthy'}")
    
    print()


def example_integration_with_training():
    """Example of how to integrate with training loops."""
    print("=== Integration with Training Loop Example ===")
    
    # This is a simplified example of how you might integrate GPU health checks
    # into a training loop
    
    gpu_check = create_gpu_health_check(interval=60)
    
    if not gpu_check.enabled:
        print("GPU health monitoring not available, proceeding without monitoring")
        return
    
    print("Starting training loop with GPU health monitoring...")
    
    # Simulate training epochs
    for epoch in range(3):
        print(f"Epoch {epoch + 1}/3")
        
        # Simulate some training work
        time.sleep(0.5)
        
        # Check GPU health periodically
        if epoch % 2 == 0:  # Check every other epoch
            is_healthy = gpu_check()
            if not is_healthy:
                print("❌ GPU health check failed! Consider stopping training.")
                break
            else:
                print("✅ GPU health check passed")
    
    print("Training completed.")
    print()


def main():
    """Main example function."""
    print("=" * 60)
    print("PyTorch Distributed Elastic GPU Health Check Examples")
    print("=" * 60)
    print()
    
    try:
        example_basic_usage()
        example_server_usage()
        example_custom_failure_handler()
        example_integration_with_training()
        
        print("=" * 60)
        print("All examples completed!")
        print("=" * 60)
        print()
        print("Note: GPU health monitoring requires:")
        print("1. PyNVML library installed (pip install pynvml)")
        print("2. NVIDIA GPU driver version r570 or newer")
        print("3. CUDA-compatible GPU")
        
    except Exception as e:
        print(f"Example failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
