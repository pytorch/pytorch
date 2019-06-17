"""
This global flag controls whether to change the existing parameters
in-place instead of assigning new tensors to the parameters when
converting a module using such as `module._apply(fn)`.

Default: False
"""
_overwrite_module_params_on_conversion = False

def set_overwrite_module_params_on_conversion(value):
	global _overwrite_module_params_on_conversion
	_overwrite_module_params_on_conversion = value

def get_overwrite_module_params_on_conversion():
	return _overwrite_module_params_on_conversion
