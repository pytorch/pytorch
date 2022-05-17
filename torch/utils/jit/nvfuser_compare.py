import torch

class CompareLog():
    def __init__(self):
        pass

    def callback(self, arg1, arg, arg3):
        pass

def register_compare_log(options):
    logger = CompareLog()
    torch._C._jit_set_nvfuser_comparison_callback(logger, True)
