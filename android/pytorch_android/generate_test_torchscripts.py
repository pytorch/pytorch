import torch
from typing import List, Tuple, Dict

OUTPUT_DIR = "src/androidTest/assets/"

def do(module, fileName):
    print('-'*80)
    script_module = torch.jit.script(module)
    print(script_module.graph)
    outputFileName = OUTPUT_DIR + fileName
    script_module.save(outputFileName)
    print("Saved to " + outputFileName)
    print('='*80)

# region Eqs
class EqBool(torch.jit.ScriptModule):
    def __init__(self):
        super(EqBool, self).__init__()

    @torch.jit.script_method
    def forward(self, input):
        # type: (bool) -> bool
        return input

class EqInt(torch.jit.ScriptModule):
    def __init__(self):
        super(EqInt, self).__init__()

    @torch.jit.script_method
    def forward(self, input):
        # type: (int) -> int
        return input

class EqFloat(torch.jit.ScriptModule):
    def __init__(self):
        super(EqFloat, self).__init__()

    @torch.jit.script_method
    def forward(self, input):
        # type: (float) -> float
        return input

class EqTensor(torch.jit.ScriptModule):
    def __init__(self):
        super(EqTensor, self).__init__()

    @torch.jit.script_method
    def forward(self, input):
        # type: (Tensor) -> Tensor
        return input

class EqDictStrKeyIntValue(torch.jit.ScriptModule):
    def __init__(self):
        super(EqDictStrKeyIntValue, self).__init__()

    @torch.jit.script_method
    def forward(self, input):
        # type: (Dict[str, int]) -> Dict[str, int]
        return input

class EqDictIntKeyIntValue(torch.jit.ScriptModule):
    def __init__(self):
        super(EqDictIntKeyIntValue, self).__init__()

    @torch.jit.script_method
    def forward(self, input):
        # type: (Dict[int, int]) -> Dict[int, int]
        return input

class EqDictFloatKeyIntValue(torch.jit.ScriptModule):
    def __init__(self):
        super(EqDictFloatKeyIntValue, self).__init__()

    @torch.jit.script_method
    def forward(self, input):
        # type: (Dict[float, int]) -> Dict[float, int]
        return input
# endregion Eqs        

class ListIntSumReturnTuple(torch.jit.ScriptModule):
    def __init__(self):
        super(ListIntSumReturnTuple, self).__init__()

    @torch.jit.script_method
    def forward(self, input):
        # type: (List[int]) -> Tuple[List[int], int]
        sum = 0
        for x in input:
            sum += x
        return (input, sum)


class TupleIntSumReturnTuple(torch.jit.ScriptModule):
    def __init__(self):
        super(TupleIntSumReturnTuple, self).__init__()

    @torch.jit.script_method
    def forward(self, input):
        # type: (Tuple[int, int, int]) -> Tuple[Tuple[int, int, int], int]
        sum = 0
        for x in input:
            sum += x
        return (input, sum)


class IntEq0None(torch.jit.ScriptModule):
    def __init__(self):
        super(IntEq0None, self).__init__()

    @torch.jit.script_method
    def forward(self, input):
        # type: (int) -> Optional[int]
        if input == 0:
            return None
        return input

do(EqBool(), "EqBool.pt")
do(EqInt(), "EqInt.pt")
do(EqFloat(), "EqFloat.pt")
do(EqTensor(), "EqTensor.pt")
do(EqDictStrKeyIntValue(), "EqDictStrKeyIntValue.pt")
do(EqDictIntKeyIntValue(), "EqDictIntKeyIntValue.pt")
do(EqDictFloatKeyIntValue(), "EqDictFloatKeyIntValue.pt")




do(ListIntSumReturnTuple(), "ListIntSumReturnTuple.pt")
do(TupleIntSumReturnTuple(), "TupleIntSumReturnTuple.pt")
do(IntEq0None(), "IntEq0None.pt")
