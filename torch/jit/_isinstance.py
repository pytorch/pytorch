from typing import List, Dict, Tuple, Union, Optional


def get_origin(target_type):
    return getattr(target_type, "__origin__", None)


def get_args(target_type):
    return getattr(target_type, "__args__", None)


def check_args_exist(target_type):
    if target_type is List or target_type is list:
        raise RuntimeError(
            "Attempted to use List without a "
            "contained type. Please add a contained type, e.g. "
            "List[int]"
        )
    elif target_type is Tuple or target_type is tuple:
        raise RuntimeError(
            "Attempted to use Tuple without a "
            "contained type. Please add a contained type, e.g. "
            "Tuple[int]"
        )
    elif target_type is Dict or target_type is dict:
        raise RuntimeError(
            "Attempted to use Dict without "
            "contained types. Please add contained type, e.g. "
            "Dict[int, int]"
        )
    elif target_type is None or target_type is Optional:
        raise RuntimeError(
            "Attempted to use Optional without a "
            "contained type. Please add a contained type, e.g. "
            "Optional[int]"
        )

# supports List/Dict/Tuple and Optional types
# TODO support future
def generics_checker(obj, target_type):
    origin_type = get_origin(target_type)
    check_args_exist(target_type)
    if origin_type is None:
        pass
    elif origin_type is list or origin_type is List:
        if not isinstance(obj, list):
            return False
        for el in obj:
            # check if nested generics, ex: List[List[str]]
            arg_type = get_args(target_type)[0]
            arg_origin = get_origin(arg_type)
            if arg_origin:  # processes nested generics, ex: List[List[str]]
                if not generics_checker(el, arg_type):
                    return False
            elif not isinstance(el, arg_type):
                return False
    elif origin_type is Dict or origin_type is dict:
        if not isinstance(obj, dict):
            return False
        key_type = get_args(target_type)[0]
        val_type = get_args(target_type)[1]
        for key, val in obj.items():
            # check if keys are of right type
            if not isinstance(key, key_type):
                return False
            val_origin = get_origin(val_type)
            if val_origin:
                if not generics_checker(val, val_type):
                    return False
            elif not isinstance(val, val_type):
                return False
    elif origin_type is Tuple or origin_type is tuple:
        if not isinstance(obj, tuple):
            return False
        arg_types = get_args(target_type)
        if len(obj) != len(arg_types):
            return False
        for el, el_type in zip(obj, arg_types):
            el_origin = get_origin(el_type)
            if el_origin:
                if not generics_checker(el, el_type):
                    return False
            elif not isinstance(el, el_type):
                return False
    elif origin_type is Union:  # actually handles Optional Case
        if obj is None:  # check before recursion because None is always fine
            return True
        optional_type = get_args(target_type)[0]
        optional_origin = get_origin(optional_type)
        if optional_origin:
            return generics_checker(obj, optional_type)
        elif isinstance(obj, optional_type):
            return True
        else:
            return False
    return True


def _isinstance(obj, target_type) -> bool:
    origin_type = get_origin(target_type)
    if origin_type:
        return generics_checker(obj, target_type)
    # handle odd case of non typed optional origin returning as none
    if origin_type is None and target_type is Optional:
        check_args_exist(target_type)
    # handle non-generics
    return isinstance(obj, target_type)
