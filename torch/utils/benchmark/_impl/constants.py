import dataclasses
import enum
import textwrap
import typing


class Language(enum.Enum):
    PYTHON = 0
    CPP = 1


@dataclasses.dataclass(init=False, repr=False, eq=True, frozen=True)
class WorkSpec:
    """Container for information used to define a Timer. (except globals)"""
    stmt: str
    setup: str
    global_setup: str = ""
    num_threads: int = 1
    language: Language

    def __init__(
        self,
        stmt: str,
        setup: str,
        global_setup: str,
        num_threads: int,
        language: typing.Union[Language, str]
    ) -> None:
        # timeit.Timer allows a callable, however due to the use of
        # subprocesses in some code paths we must be less permissive.
        if not isinstance(stmt, str):
            raise ValueError("Only a `str` stmt is supported.")

        if language in (Language.PYTHON, "py", "python"):
            language = Language.PYTHON
            if global_setup:
                raise ValueError(
                    f"global_setup is C++ only, got `{global_setup}`. Most "
                    "likely this code can simply be moved to `setup`."
                )

        elif language in (Language.CPP, "cpp", "c++"):
            language = Language.CPP

        else:
            raise ValueError(f"Invalid language `{language}`.")

        if language == Language.CPP and setup == "pass":
            setup = ""

        # Convenience adjustment so that multi-line code snippets defined in
        # functions do not IndentationError (Python) or look odd (C++). The
        # leading newline removal is for the initial newline that appears when
        # defining block strings. For instance:
        #   textwrap.dedent("""
        #     print("This is a stmt")
        #   """)
        # produces '\nprint("This is a stmt")\n'.
        #
        # Stripping this down to 'print("This is a stmt")' doesn't change
        # what gets executed, but it makes __repr__'s nicer.
        stmt = textwrap.dedent(stmt)
        stmt = (stmt[1:] if stmt and stmt[0] == "\n" else stmt).rstrip()

        setup = textwrap.dedent(setup)
        setup = (setup[1:] if setup and setup[0] == "\n" else setup).rstrip()

        object.__setattr__(self, "stmt", stmt)
        object.__setattr__(self, "setup", setup)
        object.__setattr__(self, "global_setup", global_setup)
        object.__setattr__(self, "num_threads", num_threads)
        object.__setattr__(self, "language", language)


@dataclasses.dataclass(init=True, repr=False, eq=True, frozen=True)
class WorkMetadata:
    """Container for user provided metadata."""
    label: typing.Optional[str] = None
    sub_label: typing.Optional[str] = None
    description: typing.Optional[str] = None
    env: typing.Optional[str] = None


COMPILED_MODULE_NAME = "CompiledTimerModule"
