import torch
from typing import TypeVar, Union, List, Opional, Tuple, Dict, Any, Callable, OrderedDict
from dataclasses import dataclass, fields


T = TypeVar("T", bound="JSONSerializable")
LeafType = Union[None, bool, int, float, str, OrderedDict[Union["LeafType", T], Union["LeafType", T]], torch.dtype, list[Union["LeafType", T]]]
JSONType = Union[T, LeafType]
# perf_model_registry = {
#     "NVIDIA H100": {"mm": h100_model},
#     # show how the same model can be used for
#     # two funcs, but does not have to be
#     {"addmm": h100_model},
# }

@dataclass
class JSONSerializable:
    arst: int
    tsra: str
    _is_leaf: bool = False
    def to_json_dict(self) -> OrderedDict[JSONType, JSONType]:
        ret = OrderedDict()
        f = [f.name for f in fields(self) if not f.name.startswith("_")]
        f.sort()
        
        
        ret = OrderedDict((f, getattr(self, f)) for f in f)

        # recursively resolve, calling 
        # str(self) if is_leaf == True
        # TODO
        result = {}
        result['version'] = self.version()
        return ret

    @classmethod
    def version(cls) -> int:
        # Incrementing version will invalidate all 
        # LUT entries, in the case of major perf update or
        # changes to the Ontology.
        return 1

breakpoint()


def validate_jsonable_fields(cls):
    allowed_leaf_types = {int, ...}
    # validate class members recursively
    # TODO
    return cls

@validate_jsonable_fields
@dataclass
class TritonTemplate(JSONSerializable):
      name: str
      grid: int

@validate_jsonable_fields
@dataclass
class Problem(JSONSerializable):
    _is_leaf = True
    def __repr__(self):
        raise NotImplemented("implement string conversion")

@validate_jsonable_fields
@dataclass
class TritonMMSolution(JSONSerializable):
    template: TritonTemplate
    BLOCK_M: int
    BLOCK_N: int
    _is_leaf: bool = True
    def __repr__(self) -> str:
        return f"{self.template};block_m={self.BLOCK_M}"
    def choice_caller(self, problem: Problem) -> ir.ChoiceCaller:
        pass


@validate_jsonable_fields
@dataclass
class MMProblem(Problem):
    _is_leaf = True
    M: int
    M_dtype: torch.dtype
    N: int
    N_dtype: torch.dtype
    K: int
    out_dtype: torch.dtype
    #layout: TODO
    def __repr__(self) -> str:
        # TODO
        return ""
    @classmethod
    def from_input(cls, t1: ir.IRNode, t2: ir.IRNode) -> MMProblem:
        pass

class AddMMProblem(MMProblem):
    add_dtype: torch.dtype
    def __repr__(self) -> str:
        # TODO
        return ""
    @classmethod
    def from_input(cls, t1: ir.IRNode, t2: ir.IRNode) -> MMProblem:
        ...

class Solutions(JSONSerializable):
	solutions: OrderedDict[Problem, list[TritonMMSolution]]

class Hardware(JSONSerializable):
	problems: OrderedDict[str, Solutions]

class Table(JSONSerializable):
	hardware: OrderedDict[str, Hardware]

# table from somewhere
table = Table()

def lookup_valid_configs(hardware_name: str, problem_name: str, problem: Problem) -> list[TritonMMSolution]:
	return table.hardware[hardware_name].problems[problem_name].solutions[problem]

def create_choice(problem, solution) -> TritonTemplateCaller:
	pass

def keyfn(config: JSONSerializable) -> str:
	return str(config.to_json_dict())

class Model(torch.nn.Module):
    def encode(self, problem: Problem) -> torch.Tensor:
        # TODO
        pass
    def decode(self, ret_tensor: torch.Tensor) -> TritonMMSolution:
        # TODO
        pass

def topk_configs(hardware_name: str, problem_name: str, problem: Problem, topk: int) -> list[TritonMMSolution]:
	model = perf_model_registry[problem_name]
	inp_tensor = model.encode(problem)
	ret_tensor = model(inp_tensor)
	return model.decode(ret_tensor)[topk:]


output_result = """
recorded_table = {
    "hardware": {
        "gfx942:sramecc+:xnack-": {
            "problems": {
                "addmm": {
                "solutions": {
                    "((torch.float16, [4096, 2048], [0, 1]), (torch.float16, [4096, 8192], [8192, 1]), (torch.float16, [8192, 2048], [2048, 1]))": [
                        "template=triton_mm;block_m=128;block_k=64;block_n=128;num_warps=4;num_stages=2;group_m=16;matrix_instr_nonkdim=16;waves_per_eu=0;kpack=2",
                        "template=triton_mm;block_m=128;block_k=64;block_n=128;num_warps=8;num_stages=2;group_m=8;matrix_instr_nonkdim=16;waves_per_eu=0;kpack=2",
                        "template=triton_mm;block_m=128;block_k=32;block_n=128;num_warps=4;num_stages=2;group_m=16;matrix_instr_nonkdim=16;waves_per_eu=2;kpack=2",
                    ]
                }
            }
        }
    }
}
"""
