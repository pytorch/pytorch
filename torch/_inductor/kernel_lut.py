import json
import logging
import typing
from collections import OrderedDict
from dataclasses import asdict, dataclass, field, fields
from functools import lru_cache
from typing import Any, get_origin, Optional, TYPE_CHECKING, TypeVar, Union


if TYPE_CHECKING:
    from triton import Config as TritonConfig


try:
    from typing_extensions import Self
except ImportError:
    from typing import Self

import torch
from torch.utils._ordered_set import OrderedSet


# Set up logging for kernel LUT
logger = logging.getLogger(__name__)


T = TypeVar("T", bound="JSONSerializable")
LeafType = Union[
    None, bool, int, float, str, OrderedDict[str, Any], torch.dtype, list[Any]
]
JSONType = Union[T, LeafType]


@dataclass(kw_only=True)
class JSONSerializable:
    """
    This class implements a system similar to Pydantic Models for validating and serializing dataclasses.
    """

    # Incrementing version will invalidate all LUT entries, in the case of major perf update or
    # changes to the Ontology.
    version: int = 1
    _is_leaf: bool = False

    @classmethod
    def from_dict(cls, inp: OrderedDict[str, Any] | str) -> Self:
        """
        Convert a dictionary representation of the object.
        """
        try:
            ret = OrderedDict()
            if isinstance(inp, str):
                if cls._is_leaf:
                    return cls.parse(inp)
                else:
                    raise NotImplementedError(
                        f"String representation not implemented for base {cls.__name__}"
                    )
            for k, v in inp.items():
                v_type = cls.__dataclass_fields__[k].type
                if get_origin(v_type) is OrderedDict:
                    k1_type, v1_type = typing.get_args(v_type)
                    if isinstance(k1_type, type) and issubclass(
                        k1_type, JSONSerializable
                    ):

                        def kp(tmpk: Any) -> Any:
                            return k1_type.from_dict(tmpk)

                        k_process = kp
                    else:

                        def k_process(tmpk: Any) -> Any:
                            return tmpk

                    if isinstance(v1_type, type) and issubclass(
                        v1_type, JSONSerializable
                    ):

                        def vp(tmpv: Any) -> Any:
                            return v1_type.from_dict(tmpv)

                        v_process = vp
                    else:

                        def v_process(tmpv: Any) -> Any:
                            return tmpv

                    v_new: Any = OrderedDict(
                        (k_process(key), v_process(val)) for key, val in v.items()
                    )

                elif get_origin(v_type) is list:
                    elem_type = typing.get_args(v_type)[0]
                    if isinstance(elem_type, type) and issubclass(
                        elem_type, JSONSerializable
                    ):
                        v_new = [elem_type.from_dict(x) for x in v]
                    else:
                        v_new = v
                elif isinstance(v_type, type) and issubclass(v_type, JSONSerializable):
                    v_new = v_type.from_dict(v)
                else:
                    v_new = v
                ret[k] = v_new
            return cls(**ret)  # type: ignore[arg-type]
        except Exception as e:
            logger.error("Failed to deserialize %s from dict: %s", cls.__name__, e)
            raise ValueError(f"Malformed data for {cls.__name__}: {e}") from e

    def to_dict(self) -> OrderedDict[str, Any]:
        """
        Convert the object to a dictionary representation.
        Will be written to and from using json.dumps and json.loads.
        """
        # get the fields of the dataclass
        field_list = fields(self)
        # filter out the _ fields
        field_list = [field for field in field_list if not field.name.startswith("_")]
        # ensure the fields are sorted for consistent serialization
        field_list.sort(key=lambda x: x.name)
        ret: OrderedDict[str, Any] = OrderedDict()
        for field_obj in field_list:
            field_val = getattr(self, field_obj.name)
            if isinstance(field_val, JSONSerializable):
                if field_val._is_leaf:
                    ret[field_obj.name] = str(field_val)
                else:
                    ret[field_obj.name] = field_val.to_dict()
            elif isinstance(field_val, list):
                if len(field_val) == 0:
                    ret[field_obj.name] = []
                elif isinstance(field_val[0], JSONSerializable):
                    if field_val[0]._is_leaf:
                        ret[field_obj.name] = [str(x) for x in field_val]
                    else:
                        ret[field_obj.name] = [x.to_dict() for x in field_val]
                else:
                    ret[field_obj.name] = field_val
            elif isinstance(field_val, OrderedDict):
                tmp: OrderedDict[Any, Any] = OrderedDict()
                for k, v in field_val.items():
                    if isinstance(v, JSONSerializable):
                        if v._is_leaf:
                            new_v: Any = str(v)
                        else:
                            new_v = v.to_dict()
                    else:
                        new_v = v
                    if isinstance(k, JSONSerializable):
                        if k._is_leaf:
                            new_k: Any = str(k)
                        else:
                            new_k = k.to_dict()
                    else:
                        new_k = k
                    tmp[new_k] = new_v
                ret[field_obj.name] = tmp
            else:
                ret[field_obj.name] = field_val
        return ret

    def __str__(self) -> str:
        """
        Return a string representation of the object.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def parse(cls, string: str) -> Self:
        """
        Parse the string representaiton of the object. Only reqiured for leaf nodes.
        """
        raise NotImplementedError(
            f"String representation not implemented for base {cls.__name__}"
        )


@dataclass(kw_only=True)
class TritonGEMMConfig(JSONSerializable):
    _is_leaf: bool = True
    name: str
    grid: int
    block_m: int
    block_n: int
    block_k: int
    group_m: int
    num_stages: int
    num_warps: int
    EVEN_K: bool = False
    ALLOW_TF32: bool = False
    USE_FAST_ACCUM: bool = False
    ACC_TYPE: str = "tl.float32"

    def __hash__(self) -> int:
        return hash(
            (
                self.name,
                self.grid,
                self.block_m,
                self.block_n,
                self.block_k,
                self.group_m,
                self.num_stages,
                self.num_warps,
                self.EVEN_K,
                self.ALLOW_TF32,
                self.USE_FAST_ACCUM,
                self.ACC_TYPE,
            )
        )

    @classmethod
    def parse(cls, string: str) -> Self:
        d = json.loads(string, object_pairs_hook=OrderedDict)
        # validate types, yay python :P
        if "name" not in d:
            raise KeyError("Missing required field: name")
        if not isinstance(d["name"], str):
            raise TypeError(f"name must be a string, got {type(d['name'])}")
        if "grid" not in d:
            raise KeyError("Missing required field: grid")
        if not isinstance(d["grid"], int):
            raise TypeError(f"grid must be an int, got {type(d['grid'])}")
        if "block_m" not in d:
            raise KeyError("Missing required field: block_m")
        if not isinstance(d["block_m"], int):
            raise TypeError(f"block_m must be an int, got {type(d['block_m'])}")
        if "block_n" not in d:
            raise KeyError("Missing required field: block_n")
        if not isinstance(d["block_n"], int):
            raise TypeError(f"block_n must be an int, got {type(d['block_n'])}")
        if "block_k" not in d:
            raise KeyError("Missing required field: block_k")
        if not isinstance(d["block_k"], int):
            raise TypeError(f"block_k must be an int, got {type(d['block_k'])}")
        if "group_m" not in d:
            raise KeyError("Missing required field: group_m")
        if not isinstance(d["group_m"], int):
            raise TypeError(f"group_m must be an int, got {type(d['group_m'])}")
        if "num_stages" not in d:
            raise KeyError("Missing required field: num_stages")
        if not isinstance(d["num_stages"], int):
            raise TypeError(f"num_stages must be an int, got {type(d['num_stages'])}")
        if "num_warps" not in d:
            raise KeyError("Missing required field: num_warps")
        if not isinstance(d["num_warps"], int):
            raise TypeError(f"num_warps must be an int, got {type(d['num_warps'])}")
        if "EVEN_K" in d and not isinstance(d["EVEN_K"], bool):
            raise TypeError(f"EVEN_K must be a bool, got {type(d['EVEN_K'])}")
        if "ALLOW_TF32" in d and not isinstance(d["ALLOW_TF32"], bool):
            raise TypeError(f"ALLOW_TF32 must be a bool, got {type(d['ALLOW_TF32'])}")
        if "USE_FAST_ACCUM" in d and not isinstance(d["USE_FAST_ACCUM"], bool):
            raise TypeError(
                f"USE_FAST_ACCUM must be a bool, got {type(d['USE_FAST_ACCUM'])}"
            )
        if "ACC_TYPE" in d and not isinstance(d["ACC_TYPE"], str):
            raise TypeError(f"ACC_TYPE must be a string, got {type(d['ACC_TYPE'])}")
        return cls(**d)


@dataclass(kw_only=True)
class MMProblem(JSONSerializable):
    _is_leaf: bool = True
    B: int
    M: int
    M_dtype: torch.dtype
    N: int
    K: int
    K_dtype: torch.dtype
    out_dtype: torch.dtype
    out_size: tuple[int, int, int]
    out_stride: tuple[int, int, int]

    def __hash__(self) -> int:
        return hash(
            (
                self.B,
                self.M,
                self.M_dtype,
                self.N,
                self.K_dtype,
                self.K,
                self.out_dtype,
                self.out_size,
                self.out_stride,
            )
        )

    def __str__(self) -> str:
        """
        Return a string representation of the object.
        """
        d = asdict(self)
        d["M_dtype"] = str(d["M_dtype"]).split(".")[-1]
        d["K_dtype"] = str(d["K_dtype"]).split(".")[-1]
        d["out_dtype"] = str(d["out_dtype"]).split(".")[-1]
        d["out_size"] = list(d["out_size"])
        d["out_stride"] = list(d["out_stride"])
        d = OrderedDict((k, v) for k, v in d.items() if not k.startswith("_"))
        return json.dumps(d)

    @classmethod
    def parse(cls, string: str) -> Self:
        d = json.loads(string, object_pairs_hook=OrderedDict)
        # validate types, yay python :P
        if "B" not in d:
            raise KeyError("Missing required field: B")
        if not isinstance(d["B"], int):
            raise TypeError(f"B must be an int, got {type(d['B'])}")
        if "M" not in d:
            raise KeyError("Missing required field: M")
        if not isinstance(d["M"], int):
            raise TypeError(f"M must be an int, got {type(d['M'])}")
        if "N" not in d:
            raise KeyError("Missing required field: N")
        if not isinstance(d["N"], int):
            raise TypeError(f"N must be an int, got {type(d['N'])}")
        if "K" not in d:
            raise KeyError("Missing required field: K")
        if not isinstance(d["K"], int):
            raise TypeError(f"K must be an int, got {type(d['K'])}")
        if "M_dtype" not in d:
            raise KeyError("Missing required field: M_dtype")
        if not isinstance(d["M_dtype"], str):
            raise TypeError(f"M_dtype must be a string, got {type(d['M_dtype'])}")
        if "K_dtype" not in d:
            raise KeyError("Missing required field: K_dtype")
        if not isinstance(d["K_dtype"], str):
            raise TypeError(f"K_dtype must be a string, got {type(d['K_dtype'])}")
        if "out_dtype" not in d:
            raise KeyError("Missing required field: out_dtype")
        if not isinstance(d["out_dtype"], str):
            raise TypeError(f"out_dtype must be a string, got {type(d['out_dtype'])}")
        if "out_size" not in d:
            raise KeyError("Missing required field: out_size")
        if not isinstance(d["out_size"], list):
            raise TypeError(f"out_size must be a list, got {type(d['out_size'])}")
        if "out_stride" not in d:
            raise KeyError("Missing required field: out_stride")
        if not isinstance(d["out_stride"], list):
            raise TypeError(f"out_stride must be a list, got {type(d['out_stride'])}")

        # Validate torch dtype strings
        try:
            d["M_dtype"] = getattr(torch, d["M_dtype"])
        except AttributeError:
            raise ValueError(f"Invalid torch dtype: {d['M_dtype']}") from None
        try:
            d["K_dtype"] = getattr(torch, d["K_dtype"])
        except AttributeError:
            raise ValueError(f"Invalid torch dtype: {d['K_dtype']}") from None
        try:
            d["out_dtype"] = getattr(torch, d["out_dtype"])
        except AttributeError:
            raise ValueError(f"Invalid torch dtype: {d['out_dtype']}") from None

        d["out_size"] = tuple(d["out_size"])
        d["out_stride"] = tuple(d["out_stride"])
        return cls(**d)


@dataclass(kw_only=True)
class Solution(JSONSerializable):
    # like mm or addmm
    name: str
    # mapping
    config: list[TritonGEMMConfig]


@dataclass(kw_only=True)
class Operation(JSONSerializable):
    name: str
    solution: OrderedDict[MMProblem, Solution]


@dataclass(kw_only=True)
class Hardware(JSONSerializable):
    # like gfx942:sramecc+:xnack-
    operation: OrderedDict[str, Operation]


@dataclass(kw_only=True)
class Table(JSONSerializable):
    hardware: OrderedDict[str, Hardware]
    _set_cache: OrderedDict[
        tuple[str, str, MMProblem], OrderedSet[TritonGEMMConfig]
    ] = field(default_factory=OrderedDict)

    def serialize(self) -> str:
        foo = self.to_dict()
        return json.dumps(foo, indent=2)

    @classmethod
    def deserialize(cls, s: str) -> Optional[Self]:
        try:
            return cls.from_dict(json.loads(s, object_pairs_hook=OrderedDict))
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.error("Failed to deserialize table: %s", e)
            return None

    def lookup(
        self, hardware: str, op_name: str, problem: MMProblem
    ) -> Optional[list[TritonGEMMConfig]]:
        """
        Lookup the best TritonGEMMConfig for a given problem.
        """
        if hardware not in self.hardware:
            return None
        tmp = self.hardware[hardware].operation
        if op_name not in tmp:
            return None
        tmp = tmp[op_name].solution
        if problem not in tmp:
            return None
        return tmp[problem].config

    def lookup_set(
        self, hardware: str, op_name: str, problem: MMProblem
    ) -> Optional[OrderedSet[TritonGEMMConfig]]:
        """
        Easier and faster to check membership in a set, but cache the sets for runtime.
        """
        if (hardware, op_name, problem) in self._set_cache:
            return self._set_cache[(hardware, op_name, problem)]
        problem_list = self.lookup(hardware, op_name, problem)
        problem_set = OrderedSet(problem_list) if problem_list is not None else None
        if problem_set is None:
            return None
        self._set_cache[(hardware, op_name, problem)] = problem_set
        return problem_set

    def filter(
        self,
        hardware: str,
        op_name: str,
        problem: MMProblem,
        to_filter: list[TritonGEMMConfig],
    ) -> Optional[list[TritonGEMMConfig]]:
        """
        Filter a list of TritonGEMMConfig for a given problem.
        """

        problem_set = self.lookup_set(hardware, op_name, problem)
        if problem_set is None:
            return None
        ret = [x for x in to_filter if x in problem_set]
        if len(ret) == 0:
            return None
        return ret


def convert_triton_configs_to_gemm_configs(
    triton_configs: list["TritonConfig"], name_prefix: str = "triton_config"
) -> list[TritonGEMMConfig]:
    """
    Convert a list of triton.runtime.autotuner.Config objects to TritonGEMMConfig objects.

    Args:
        triton_configs: List of triton.runtime.autotuner.Config objects
        name_prefix: Prefix for generated config names (default: "triton_config")

    Returns:
        List of TritonGEMMConfig objects
    """
    gemm_configs = []

    for i, config in enumerate(triton_configs):
        # Extract kwargs which contain the block sizes
        kwargs = getattr(config, "kwargs", {})

        # Handle case where kwargs is None
        if kwargs is None:
            kwargs = {}

        # Extract required parameters from kwargs
        block_m = kwargs.get("BLOCK_M", 64)  # Default fallback values
        block_n = kwargs.get("BLOCK_N", 64)
        block_k = kwargs.get("BLOCK_K", 32)
        group_m = kwargs.get("GROUP_M", 8)

        # Extract other parameters directly from config object
        num_stages = getattr(config, "num_stages", 2)
        num_warps = getattr(config, "num_warps", 4)

        # Extract optional parameters with defaults
        even_k = kwargs.get("EVEN_K", False)
        allow_tf32 = kwargs.get("ALLOW_TF32", False)
        use_fast_accum = kwargs.get("USE_FAST_ACCUM", False)
        acc_type = kwargs.get("ACC_TYPE", "tl.float32")

        # Generate a unique name for this config
        config_name = f"{name_prefix}_{i}"

        # Create TritonGEMMConfig object
        gemm_config = TritonGEMMConfig(
            name=config_name,
            grid=1,  # Default grid value, can be adjusted based on requirements
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            group_m=group_m,
            num_stages=num_stages,
            num_warps=num_warps,
            EVEN_K=even_k,
            ALLOW_TF32=allow_tf32,
            USE_FAST_ACCUM=use_fast_accum,
            ACC_TYPE=acc_type,
        )

        gemm_configs.append(gemm_config)

    return gemm_configs


@lru_cache
def get_table(path: str) -> Optional[Table]:
    """Load a table from a file path."""
    try:
        with open(path) as f:
            return Table.deserialize(f.read())
    except OSError as e:
        logger.error("Failed to read table from %s: %s", path, e)
        return None


def get_table_safe(path: str) -> Optional[Table]:
    """Safely load a table from a file path without caching."""
    try:
        with open(path) as f:
            return Table.deserialize(f.read())
    except OSError as e:
        logger.error("Failed to read table from %s: %s", path, e)
        return None
