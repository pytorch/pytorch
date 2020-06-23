import torch

print(torch.version.__version__)

def scriptAndSave(module, fileName):
    print('-' * 80)
    script_module = torch.jit.script(module)
    print(script_module.graph)
    outputFileName = fileName
    script_module.save(outputFileName)
    print("Saved to " + outputFileName)
    print('=' * 80)

class Test(torch.jit.ScriptModule):
    def __init__(self):
        super(Test, self).__init__()

    @torch.jit.script_method
    def forward(self, input):
        return None

    @torch.jit.script_method
    def sumTensorsList(self, input):
        # type: (List[Tensor]) -> Tensor
        sum = torch.zeros_like(input[0])
        for x in input:
            sum += x
        return sum
scriptAndSave(Test(), "sum_tensors_list.pt")
