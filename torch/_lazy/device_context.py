import threading
from typing import Any, Dict, Optional

import torch._C._lazy


class DeviceContext:
    """
    A context manager for device-specific operations.
    
    This class maintains a cache of device contexts to avoid recreating
    contexts for the same device multiple times.
    """
    _CONTEXTS: Dict[str, 'DeviceContext'] = {}
    _CONTEXTS_LOCK = threading.Lock()

    def __init__(self, device: str) -> None:
        """
        Initialize a device context.
        
        Args:
            device: The device identifier string
        """
        self.device = device
    
    @classmethod
    def get_context(cls, device: Optional[str] = None) -> 'DeviceContext':
        """
        Get or create a device context for the specified device.
        
        Args:
            device: The device identifier string, or None to use the default device
            
        Returns:
            The device context for the specified device
        """
        if device is None:
            device = torch._C._lazy._get_default_device_type()
        else:
            device = str(device)
            
        with cls._CONTEXTS_LOCK:
            devctx = cls._CONTEXTS.get(device)
            if devctx is None:
                devctx = cls(device)
                cls._CONTEXTS[device] = devctx
            return devctx


def get_device_context(device: Optional[str] = None) -> DeviceContext:
    """
    Get a device context for the specified device.
    
    Args:
        device: The device identifier string, or None to use the default device
        
    Returns:
        The device context for the specified device
    """
    return DeviceContext.get_context(device)