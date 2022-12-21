import ctypes
import dataclasses
import sys
import types
from typing import Callable, Dict, List, NamedTuple, Optional, OrderedDict, Union

from typing_extensions import Protocol

if sys.version_info >= (3, 11):
    # In latest CPython version, the eval frame API was changed to
    # provide this struct instead of the types.FrameType.
    # This structure mimicks the current version in internal/pycore_frame.h:
    # typedef struct _PyInterpreterFrame {
    #     /* "Specials" section */
    #     PyFunctionObject *f_func; /* Strong reference */
    #     PyObject *f_globals; /* Borrowed reference */
    #     PyObject *f_builtins; /* Borrowed reference */
    #     PyObject *f_locals; /* Strong reference, may be NULL */
    #     PyCodeObject *f_code; /* Strong reference */
    #     PyFrameObject *frame_obj; /* Strong reference, may be NULL */
    #     /* Linkage section */
    #     struct _PyInterpreterFrame *previous;
    #     // NOTE: This is not necessarily the last instruction started in the given
    #     // frame. Rather, it is the code unit *prior to* the *next* instruction. For
    #     // example, it may be an inline CACHE entry, an instruction we just jumped
    #     // over, or (in the case of a newly-created frame) a totally invalid value:
    #     _Py_CODEUNIT *prev_instr;
    #     int stacktop;     /* Offset of TOS from localsplus  */
    #     bool is_entry;  // Whether this is the "root" frame for the current _PyCFrame.
    #     char owner;
    #     /* Locals and stack */
    #     PyObject *localsplus[1];
    # } _PyInterpreterFrame;

    class _PyInterpreterFrame(ctypes.Structure):
        pass

    _PyInterpreterFrame._fields_ = [
        ("f_func", ctypes.py_object),
        ("f_globals", ctypes.py_object),
        ("f_builtins", ctypes.py_object),
        ("f_locals", ctypes.py_object),
        ("f_code", ctypes.py_object),
        ("frame_obj", ctypes.py_object),
        ("previous", ctypes.POINTER(_PyInterpreterFrame)),
        ("prev_instr", ctypes.c_ushort),
        ("stacktop", ctypes.c_int),
        ("is_entry", ctypes.c_bool),
        ("owner", ctypes.c_char),
        ("localsplus", ctypes.py_object * 1),
    ]

    DynamoFrameType = _PyInterpreterFrame
else:
    DynamoFrameType = types.FrameType


class GuardFail(NamedTuple):
    # A string repr of the piece of failed guard code we eval-ed
    reason: str
    # A code object where we failed a guard
    orig_code: types.CodeType


class GuardFn(Protocol):
    closure_vars: OrderedDict[str, object]
    args: List[str]
    code_parts: List[str]
    verbose_code_parts: List[str]
    global_scope: Dict[str, object]
    guard_fail_fn: Optional[Callable[[GuardFail], None]]

    # maps locals of user function to bool
    def __call__(self, *maybe_dotzero: object, **f_locals: object) -> bool:
        ...


@dataclasses.dataclass
class GuardedCode:
    code: types.CodeType
    check_fn: GuardFn


class DynamoCallbackFn(Protocol):
    def __call__(
        self, frame: DynamoFrameType, cache_size: int, lasti: int
    ) -> Optional[GuardedCode]:
        ...


DynamoCallback = Union[DynamoCallbackFn, None, bool]


class DynamoGuardHook(Protocol):
    def __call__(
        self,
        guard_fn: GuardFn,
        code: types.CodeType,
        f_locals: Dict[str, object],
        last: bool,
    ) -> None:
        ...
