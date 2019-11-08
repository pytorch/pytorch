import torch

OUTPUT_DIR = "src/androidTest/assets/"

def scriptAndSave(module, fileName):
    print('-' * 80)
    script_module = torch.jit.script(module)
    print(script_module.graph)
    outputFileName = OUTPUT_DIR + fileName
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
    def eqBool(self, input):
        # type: (bool) -> bool
        return input

    @torch.jit.script_method
    def eqInt(self, input):
        # type: (int) -> int
        return input

    @torch.jit.script_method
    def eqFloat(self, input):
        # type: (float) -> float
        return input

    @torch.jit.script_method
    def eqStr(self, input):
        # type: (str) -> str
        return input

    @torch.jit.script_method
    def eqTensor(self, input):
        # type: (Tensor) -> Tensor
        return input

    @torch.jit.script_method
    def eqDictStrKeyIntValue(self, input):
        # type: (Dict[str, int]) -> Dict[str, int]
        return input

    @torch.jit.script_method
    def eqDictIntKeyIntValue(self, input):
        # type: (Dict[int, int]) -> Dict[int, int]
        return input

    @torch.jit.script_method
    def eqDictFloatKeyIntValue(self, input):
        # type: (Dict[float, int]) -> Dict[float, int]
        return input

    @torch.jit.script_method
    def listIntSumReturnTuple(self, input):
        # type: (List[int]) -> Tuple[List[int], int]
        sum = 0
        for x in input:
            sum += x
        return (input, sum)

    @torch.jit.script_method
    def listBoolConjunction(self, input):
        # type: (List[bool]) -> bool
        res = True
        for x in input:
            res = res and x
        return res

    @torch.jit.script_method
    def listBoolDisjunction(self, input):
        # type: (List[bool]) -> bool
        res = False
        for x in input:
            res = res or x
        return res

    @torch.jit.script_method
    def tupleIntSumReturnTuple(self, input):
        # type: (Tuple[int, int, int]) -> Tuple[Tuple[int, int, int], int]
        sum = 0
        for x in input:
            sum += x
        return (input, sum)

    @torch.jit.script_method
    def optionalIntIsNone(self, input):
        # type: (Optional[int]) -> bool
        return input is None

    @torch.jit.script_method
    def intEq0None(self, input):
        # type: (int) -> Optional[int]
        if input == 0:
            return None
        return input

    @torch.jit.script_method
    def str3Concat(self, input):
        # type: (str) -> str
        return input + input + input

scriptAndSave(Test(), "test.pt")
