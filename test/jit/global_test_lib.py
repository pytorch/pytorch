import torch


global_var = None 


def set_global_var(v: str):
    global global_var
    global_var = v


def set_global_var_and_capture(v: str):
    global global_var
    global_var = v
    torch.jit.capture_global_constant_value("global_var")


def reset():
    global global_var
    global_var = None
    torch.jit.reset_captured_global_constant_values_registry()


def use_global_var():
    global global_var
    localized_var: str = str(global_var)
    return localized_var
