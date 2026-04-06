from . import compiled_autograd, eval_frame, guards  # noqa: F401

def strip_function_call(name: str) -> str: ...
def is_valid_var_name(name: str) -> bool | int: ...
def get_type_slots(obj: type | object) -> tuple[int, int, int, int]: ...
def has_slot(slots: int, slot_bit: int) -> bool: ...

class PySequenceSlots:
    SQ_LENGTH: int
    SQ_CONCAT: int
    SQ_REPEAT: int
    SQ_ITEM: int
    SQ_CONTAINS: int
    SQ_ASS_ITEM: int
    SQ_INPLACE_CONCAT: int
    SQ_INPLACE_REPEAT: int

class PyMappingSlots:
    MP_LENGTH: int
    MP_SUBSCRIPT: int
    MP_ASS_SUBSCRIPT: int

class PyNumberSlots:
    NB_ADD: int
    NB_SUBTRACT: int
    NB_MULTIPLY: int
    NB_REMAINDER: int
    NB_POWER: int
    NB_NEGATIVE: int
    NB_POSITIVE: int
    NB_ABSOLUTE: int
    NB_BOOL: int
    NB_INVERT: int
    NB_LSHIFT: int
    NB_RSHIFT: int
    NB_AND: int
    NB_XOR: int
    NB_OR: int
    NB_INT: int
    NB_FLOAT: int
    NB_INPLACE_ADD: int
    NB_INPLACE_SUBTRACT: int
    NB_INPLACE_MULTIPLY: int
    NB_INPLACE_REMAINDER: int
    NB_INPLACE_POWER: int
    NB_INPLACE_LSHIFT: int
    NB_INPLACE_RSHIFT: int
    NB_INPLACE_AND: int
    NB_INPLACE_XOR: int
    NB_INPLACE_OR: int
    NB_FLOOR_DIVIDE: int
    NB_TRUE_DIVIDE: int
    NB_INPLACE_FLOOR_DIVIDE: int
    NB_INPLACE_TRUE_DIVIDE: int
    NB_INDEX: int
    NB_MATRIX_MULTIPLY: int
    NB_INPLACE_MATRIX_MULTIPLY: int

class PyTypeSlots:
    TP_HASH: int
    TP_ITER: int
    TP_ITERNEXT: int
    TP_CALL: int
    TP_REPR: int
    TP_RICHCOMPARE: int
    TP_GETATTRO: int
    TP_SETATTRO: int
    TP_DESCR_GET: int
    TP_DESCR_SET: int
