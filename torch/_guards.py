import dataclasses
import enum
import logging
import weakref
from typing import Callable, List, NamedTuple, Optional

log = logging.getLogger(__name__)

# TODO(voz): Stolen pattern, not sure why this is the case,
# but mypy complains.
try:
    import sympy  # type: ignore[import]
except ImportError:
    log.warning("No sympy found")

"""
torch._guards is the definitional source of truth for general purpose guard structures.

An important thing to keep in mind here is the preservation of layering. There should be no dynamo notions,
and no guard installation notions here.
"""


class GuardSource(enum.Enum):
    LOCAL = 0
    GLOBAL = 1
    LOCAL_NN_MODULE = 2
    GLOBAL_NN_MODULE = 3
    CONSTANT = 4
    RANDOM_VALUE = 5
    SHAPE_ENV = 6

    def select(self, locals_, globals_):
        if self in (GuardSource.LOCAL, GuardSource.LOCAL_NN_MODULE):
            return locals_
        if self in (GuardSource.GLOBAL, GuardSource.GLOBAL_NN_MODULE):
            return globals_
        raise NotImplementedError()

    def is_nn_module(self) -> bool:
        return self in (GuardSource.GLOBAL_NN_MODULE, GuardSource.LOCAL_NN_MODULE)

    def is_local(self):
        return self in (GuardSource.LOCAL, GuardSource.LOCAL_NN_MODULE)


"""
Base class for a "GuardBuilder" role.

The GuardBuilderBase role is to represent a scope within which to build a guard. The name is a little
confusing, as its not a builder, but for the sake of avoiding a lot of renames and keeping the original reference
to torchdynamo's GuardBuilder.

Note: create_fn is invoked with a GuardBuilderBase and a Guard. A GuardBuilder is chosen based
on GuardSource's select function.

There is value in keeping this GuardBuilderBase empty to keep layering clean.
"""


class GuardBuilderBase:
    pass


class ShapeGuard(NamedTuple):
    expr: sympy.Expr
    stack: str


@dataclasses.dataclass
class Guard:
    # The name of a Guard specifies what exactly it is the guard is guarding
    # on.  The meaning of the name is dependent on the create_fn; you must
    # look at the use-site inside create_fn to know what name means.
    #
    # That being said, although you might think this is just a "name", name is
    # usually an arbitrary Python expression that will be evaluated with all
    # globals (and locals, if you create a LOCAL guard) to extract the Python
    # object that we want to perform guard tests on.  This evaluation
    # typically happens in GuardBuilder.eval.  In these cases, name is
    # typically produced by Source.name() (not to be confused with
    # GuardSource)--morally, we could have stored a Source here.
    #
    # Occasionally, name is not a valid Python expression; sometimes
    # it is meaningless.  Example create_fns that are like this include
    # GRAD_MODE and SYMBOL_MATCH.
    name: str
    source: GuardSource
    create_fn: Callable[[GuardBuilderBase, "Guard"], None]
    is_volatile: bool = False

    # Export only. These values are written to at time of guard check_fn creation.
    guard_types: Optional[List[str]] = None
    code_list: Optional[List[str]] = None
    obj_weakref: Optional[object] = None
    guarded_class_weakref: Optional[type] = None

    def __hash__(self):
        return hash((self.name, self.source, id(self.create_fn)))

    def sort_key(self):
        return (
            self.source.value if self.source else -1,
            len(self.name),
            self.name,
            self.create_fn.__code__.co_firstlineno,
        )

    def __lt__(self, other):
        return self.sort_key() < other.sort_key()

    @staticmethod
    def weakref_to_str(obj_weakref):
        """
        This is a workaround of a Python weakref bug.

        `obj_weakref` is instance returned by `weakref.ref`,
        `str(obj_weakref)` is buggy if the original obj overrides __getattr__, e.g:

            class MyConfig(dict):
                def __getattr__(self, x):
                    return self[x]

            obj = MyConfig(offset=5)
            obj_weakref = weakref.ref(obj)
            str(obj_weakref)  # raise error: KeyError: '__name__'
        """
        if isinstance(obj_weakref, weakref.ReferenceType):
            obj = obj_weakref()
            if obj is not None:
                return f"<weakref at {hex(id(obj_weakref))}; to '{obj.__class__.__name__}' at {hex(id(obj))}>"
            else:
                return f"<weakref at {hex(id(obj_weakref))}; dead>"
        else:
            return str(obj_weakref)

    def __str__(self):
        s = f"""
            {self.source.name.lower() if self.source else ""} {repr(self.name)} {self.create_fn.__name__}
            {{
                'guard_types': {self.guard_types},
                'code': {self.code_list},
                'obj_weakref': {self.weakref_to_str(self.obj_weakref)}
                'guarded_class': {self.guarded_class_weakref}
            }}
            """
        return s

    def create(self, local_builder: GuardBuilderBase, global_builder: GuardBuilderBase):
        return self.create_fn(self.source.select(local_builder, global_builder), self)

    def is_nn_module(self):
        return self.source.is_nn_module()

    def is_local(self):
        return self.source.is_local()

    def set_export_info(self, guard_type, guarded_class, code_list, obj_weakref):
        if not self.guard_types:
            self.guard_types = list()

        self.guard_types.append(guard_type)

        assert self.guarded_class_weakref in (
            guarded_class,
            None,
        ), "Guarded class id must be identical, or None"
        self.guarded_class_weakref = guarded_class

        if not self.code_list:
            self.code_list = code_list
        else:
            self.code_list.extend(code_list)

        assert self.obj_weakref in (
            obj_weakref,
            None,
        ), "Guarded object must be identical, or None"
        self.obj_weakref = obj_weakref
