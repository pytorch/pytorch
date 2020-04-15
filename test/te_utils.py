import torch

class ExecutionCounter(object):
    def try_get_trigger_value(self):
        try:
            return torch._C._jit_get_trigger_value(self.name)
        except Exception:
            return 0

    def __init__(self, name):
        self.name = name
        self.start_value = self.try_get_trigger_value()

    def elapsed_value(self):
        value = self.try_get_trigger_value()
        return value - self.start_value

class CudaCodeGenCreated(ExecutionCounter):
    def __init__(self):
        super(CudaCodeGenCreated, self).__init__("cuda_codegen_created")

class CudaCodeGenExecuted(ExecutionCounter):
    def __init__(self):
        super(CudaCodeGenExecuted, self).__init__("cuda_codegen_executed")

class LLVMCodeGenCreated(ExecutionCounter):
    def __init__(self):
        super(LLVMCodeGenCreated, self).__init__("llvm_codegen_created")

class LLVMCodeGenExecuted(ExecutionCounter):
    def __init__(self):
        super(LLVMCodeGenExecuted, self).__init__("llvm_codegen_executed")

class SimpleIREvalExecuted(ExecutionCounter):
    def __init__(self):
        super(SimpleIREvalExecuted, self).__init__("simple_ir_eval_executed")
