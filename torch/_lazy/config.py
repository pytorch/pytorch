import torch._C._lazy

def get_force_fallback():
    """Get the config used to force LTC fallback"""
    return torch._C._lazy._get_force_fallback()

def set_force_fallback(configval):
    """Set the config used to force LTC fallback"""
    torch._C._lazy._set_force_fallback(configval)

def enable_reuse_ir():
    """Enable reusing IR nodes for faster tracing"""
    torch._C._lazy._enable_reuse_ir()
