"""
This global flag controls whether to change the existing parameters
in-place instead of assigning new tensors to the parameters when
converting a module using `module._apply(fn)`.

Default: False
"""
TORCH_FLAG_overwrite_module_params_on_conversion_ = False

def set_overwrite_module_params_on_conversion(value):
	global TORCH_FLAG_overwrite_module_params_on_conversion_
	TORCH_FLAG_overwrite_module_params_on_conversion_ = value

def get_overwrite_module_params_on_conversion():
	return TORCH_FLAG_overwrite_module_params_on_conversion_
