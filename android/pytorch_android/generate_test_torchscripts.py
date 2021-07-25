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
    def eqBool(self, input: bool) -> bool:
        return input

    @torch.jit.script_method
    def eqInt(self, input: int) -> int:
        return input

    @torch.jit.script_method
    def eqFloat(self, input: float) -> float:
        return input

    @torch.jit.script_method
    def eqStr(self, input: str) -> str:
        return input

    @torch.jit.script_method
    def eqTensor(self, input: Tensor) -> Tensor:
        return input

    @torch.jit.script_method
    def eqDictStrKeyIntValue(self, input: Dict[str, int]) -> Dict[str, int]:
        return input

    @torch.jit.script_method
    def eqDictIntKeyIntValue(self, input: Dict[int, int]) -> Dict[int, int]:
        return input

    @torch.jit.script_method
    def eqDictFloatKeyIntValue(self, input: Dict[float, int]) -> Dict[float, int]:
        return input

    @torch.jit.script_method
    def listIntSumReturnTuple(self, input: List[int]) -> Tuple[List[int], int]:
        sum = 0
        for x in input:
            sum += x
        return (input, sum)

    @torch.jit.script_method
    def listBoolConjunction(self, input: List[bool]) -> bool:
        res = True
        for x in input:
            res = res and x
        return res

    @torch.jit.script_method
    def listBoolDisjunction(self, input: List[bool]) -> bool:
        res = False
        for x in input:
            res = res or x
        return res

    @torch.jit.script_method
    def tupleIntSumReturnTuple(self, input: Tuple[int, int, int]) -> Tuple[Tuple[int, int, int], int]:
        sum = 0
        for x in input:
            sum += x
        return (input, sum)

    @torch.jit.script_method
    def optionalIntIsNone(self, input: Optional[int]) -> bool:
        return input is None

    @torch.jit.script_method
    def intEq0None(self, input: int) -> Optional[int]:
        if input == 0:
            return None
        return input

    @torch.jit.script_method
    def str3Concat(self, input: str) -> str:
        return input + input + input

    @torch.jit.script_method
    def newEmptyShapeWithItem(self, input):
        return torch.tensor([int(input.item())])[0]

    @torch.jit.script_method
    def testAliasWithOffset(self) -> List[Tensor]:
        x = torch.tensor([100, 200])
        a = [x[0], x[1]]
        return a

    @torch.jit.script_method
    def testNonContiguous(self):
        x = torch.tensor([100, 200, 300])[::2]
        assert not x.is_contiguous()
        assert x[0] == 100
        assert x[1] == 300
        return x

    @torch.jit.script_method
    def conv2d(self, x: Tensor, w: Tensor, toChannelsLast: bool) -> Tensor:
        r = torch.nn.functional.conv2d(x, w)
        if (toChannelsLast):
            r = r.contiguous(memory_format=torch.channels_last)
        else:
            r = r.contiguous()
        return r

    @torch.jit.script_method
    def contiguous(self, x: Tensor) -> Tensor:
        return x.contiguous()

    @torch.jit.script_method
    def contiguousChannelsLast(self, x: Tensor) -> Tensor:
        return x.contiguous(memory_format=torch.channels_last)

    @torch.jit.script_method
    def contiguousChannelsLast3d(self, x: Tensor) -> Tensor:
        return x.contiguous(memory_format=torch.channels_last_3d)

scriptAndSave(Test(), "test.pt")
