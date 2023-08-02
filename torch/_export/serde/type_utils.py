from typing import List, Dict, Tuple, Set, Union
import inspect

class Succeed:
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Succeed)

    def __repr__(self) -> str:
        return "Succeed()"

class Fail:
    def __init__(self, msg: List[str]) -> None:
        self.msg = msg

    def __repr__(self) -> str:
        return f"Fail(msg={self.msg})"

    def __eq__(self, other: object) -> bool:
        return type(other) == Fail and self.msg == other.msg


class NotSure:
    def __init__(self, v: object, t: object) -> None:
        self.v = v
        self.t = t

    def __repr__(self) -> str:
        return f"NotSure(v={self.v}, t={self.t})"

    def __eq__(self, other: object) -> bool:
        return type(other) == NotSure and self.v == other.v and self.t == other.t


def pp(t: object) -> str:
    """
    pretty print a type object
    """

    head, args = deconstruct(t)
    if head == list:
        return f"List[{pp(args[0])}]"
    elif head == set:
        return f"Set[{pp(args[0])}]"
    elif head == dict:
        return f"Dict[{pp(args[0])}, {pp(args[1])}]"
    elif head == Union:
        args_str = ", ".join(pp(arg) for arg in args)
        return f"Union[{args_str}]"
    else:
        return t.__name__


def check(v: object, t: object) -> Union[Succeed, Fail, NotSure]:
    """
    Tries to decide if v is of type t

    Succeeds
        - when v is a simple Python value (e.g., ints, floats, nested built-in structs like Dict, Set, List, Union)
          and v's type matches t
        - or when `isinstance(v, t)`

    Fails
        - when v is a simple Python value (as described above)
          and v's type does not match t

    Gives NotSure
        - when the type of v is not included in this logic
    """

    head, args = deconstruct(t)
    if head == list:
        if not isinstance(v, list):
            return Fail(msg=[f"Expected {v} to be a list, but got {type(v).__name__}."])
        element_type = args[0]
        for element in v:
            check_result = check(element, element_type)
            if isinstance(check_result, Fail):
                return Fail([f"{v} is not a {pp(t)}."] + check_result.msg)
        return Succeed()
    elif head == dict:
        if not isinstance(v, dict):
            return Fail(msg=[f"Expected {v} to be a dict, but got {type(v).__name__}."])
        key_type, val_type = args[0], args[1]
        for key, val in v.items():
            check_result = check(key, key_type)
            if isinstance(check_result, Fail):
                return Fail([f"{v} is not a {pp(t)}."] + check_result.msg)
            check_result = check(val, val_type)
            if isinstance(check_result, Fail):
                return Fail([f"{v} is not a {pp(t)}."] + check_result.msg)
        return Succeed()
    elif head == set:
        if not isinstance(v, set):
            return Fail(msg=[f"Expected {v} to be a set, but got {type(v).__name__}."])
        element_type = args[0]
        for element in v:
            check_result = check(element, element_type)
            if isinstance(check_result, Fail):
                return Fail([f"{v} is not a {pp(t)}."] + check_result.msg)
        return Succeed()
    elif head == Union:
        for arg in args:
            check_result = check(v, arg)
            if isinstance(check_result, Succeed):
                return Succeed()
        return Fail([f"{v} is not of type {pp(t)}."])
    elif head in [int, float, str, bool] or inspect.isclass(head):
        if isinstance(v, head):
            return Succeed()
        else:
            return Fail([f"Expected {head.__name__} from {v}, but got {type(v).__name__}."])
    else:
        return NotSure(t=t, v=v)


def deconstruct(t: object) -> Tuple[object, Tuple[object, ...]]:
    """
    examples:
    - deconstruct(List[int]) -> List, (int)
    - deconstruct(Dict[int, List[str]]) -> Dict, (int, List[int])
    - deconstruct(int) -> int, ()
    """
    if hasattr(t, "__origin__") and hasattr(t, "__args__"):
        return t.__origin__, t.__args__
    else:
        return t, ()
