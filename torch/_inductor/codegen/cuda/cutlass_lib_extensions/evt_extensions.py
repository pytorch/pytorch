from collections.abc import Callable
from typing import Any, Union

from sympy import Expr

from torch._inductor.ir import (
    ComputedBuffer,
    InputBuffer,
    is_contiguous_strides_for_shape,
)
from torch.utils._ordered_set import OrderedSet

from ..cutlass_utils import torch_dtype_to_cutlass_type, try_import_cutlass


EpilogueFunctor = Any  # EpilogueFunctor local class defined in _trace
Buffer = Union[ComputedBuffer, InputBuffer]
CutlassTupleType = Any  # cutlass.backend.c_types.tuple_factory_.<locals>.TupleType
CutlassVisitorType = Any  # cutlass.backend.c_types.visitor_factory.<locals>.VisitorType
CutlassArgType = (
    Any  # Can be a CutlassTupleType, CutlassVisitorType, EmptyByte, or ctype.c_void_p
)


if try_import_cutlass():
    import ast
    import ctypes
    import textwrap
    from typing import Union

    from cutlass_cppgen.backend.c_types import (  # type: ignore[import-not-found]
        EmptyByte,
    )
    from cutlass_cppgen.backend.epilogue import (  # type: ignore[import-not-found]
        dtype2ctype,
    )
    from cutlass_cppgen.backend.evt import (  # type: ignore[import-not-found]
        EpilogueFunctorVisitor,
    )
    from cutlass_cppgen.backend.evt.backend.emitter_base import (  # type: ignore[import-not-found]
        FusionCallbacks,
    )
    from cutlass_cppgen.backend.evt.backend.sm90_emitter import (  # type: ignore[import-not-found]
        CollectiveEpilogue,
    )
    from cutlass_cppgen.backend.evt.frontend import (  # type: ignore[import-not-found]
        PythonASTFrontend,
    )
    from cutlass_cppgen.backend.evt.ir.tensor import (  # type: ignore[import-not-found]
        Tensor as CutlassTensor,
    )
    from cutlass_library import (
        DataType,
        EpilogueScheduleType,
        LayoutType,
        TileDescription,
    )

    from torch._inductor.codegen.cuda import cuda_env
    from torch._inductor.utils import IndentedBuffer

    _CUTLASS_C_DTYPES = OrderedSet(dtype2ctype.values())  # type: ignore[var-annotated]

    class EVTArgRenames:
        """Handles mapping buffer names to variable names in the cpp kernel signature and body"""

        def __init__(self) -> None:
            self.buf_renames: dict[str, str] = {}

        def new_name(self, name: str) -> str:
            if name in self.buf_renames:
                return self.buf_renames[name]
            else:
                new_name = f"ptr_{len(self.buf_renames)}"
                self.buf_renames[name] = new_name
                return new_name

        def get(self, name: str) -> str:
            return self.buf_renames.get(name, name)

    def create_example_tensors(
        var_name_to_buffer_name: dict[str, str],
        name_to_buffer: dict[str, Buffer],
        size_hint_fn: Callable[[Union[Expr, int]], int],
    ) -> dict[str, CutlassTensor]:
        def cutlass_tensor_from_buffer(
            buffer: Buffer,
        ) -> CutlassTensor:
            shape = buffer.get_layout().size
            stride = buffer.get_layout().stride
            shape = tuple(size_hint_fn(x) for x in shape)
            stride = tuple(size_hint_fn(x) for x in stride)

            is_row_major = is_contiguous_strides_for_shape(stride, shape)
            is_column_major = is_contiguous_strides_for_shape(stride[::-1], shape[::-1])

            if not is_row_major and not is_column_major:
                raise RuntimeError(
                    f"Cannot create example tensor for {buffer.get_name()} with \
non-contiguous layout, received stride: {stride} and shape: {shape}"
                )

            return CutlassTensor(
                shape=shape,
                layout_tag=(
                    LayoutType.RowMajor if is_row_major else LayoutType.ColumnMajor
                ),
                element=torch_dtype_to_cutlass_type(buffer.get_layout().dtype),
            )

        return {
            key: cutlass_tensor_from_buffer(name_to_buffer[name])
            for key, name in var_name_to_buffer_name.items()
        }

    def trace(
        fn_src: str,
        example_tensors: dict[str, CutlassTensor],
        accum_type: DataType,
        output_type: DataType,
        tile_description: TileDescription,
        epilogue_schedule: EpilogueScheduleType,
        name_to_buffer: dict[str, Buffer],
        size_hint_fn: Callable[[Union[Expr, int]], int],
        **kwargs: dict[str, Any],
    ) -> tuple[str, str, str, EVTArgRenames]:
        cuda_arch = int(cuda_env.get_cuda_arch())  # type: ignore[arg-type]
        assert cuda_arch >= 90, "Only SM90+ is supported for EVT"
        epilogue_functor = _trace(fn_src, example_tensors, cuda_arch, **kwargs)
        visitor = EpilogueFunctorVisitor(cuda_arch, epilogue_functor)
        fusion_callbacks = FusionCallbacks(visitor.graph, cuda_arch, emit_CD=False)
        collective_epilogue = CollectiveEpilogue(
            tile_description,
            epilogue_schedule,
            accum_type,
            output_type,
            fusion_callbacks,
        )
        evt_name, evt_code = collective_epilogue.emit()
        evt_args, arg_renames = _render_argument_type(
            epilogue_functor, name_to_buffer, size_hint_fn
        )
        return evt_name, evt_args, evt_code, arg_renames

    # Based off of
    # https://github.com/NVIDIA/cutlass/blob/df18f5e4f5de76bed8be1de8e4c245f2f5ec3020/python/cutlass/epilogue/epilogue.py#L117
    # This is modified to enable directly passing the source code of the epilogue vs getting it from a bona-fide python function
    # The reason for this is that inspect.getsource does not work with functions defined at runtime via exec/eval
    def _trace(
        fn_src: str,
        example_tensors: dict[str, CutlassTensor],
        cc: int,
        **kwargs: Any,
    ) -> EpilogueFunctor:
        class EpilogueFunctor(PythonASTFrontend):
            def __init__(self, cc: int, **kwargs: Any):
                self.source = textwrap.dedent(fn_src)
                super().__init__(cc, **kwargs)

            def parse(
                self,
                example_inputs: dict[str, CutlassTensor],
            ) -> None:
                self.example_inputs = example_inputs
                self.ast = ast.parse(self.source)
                self.visit(self.ast)

        cc = int(cuda_env.get_cuda_arch())
        epilogue_functor = EpilogueFunctor(cc=cc, **kwargs)
        epilogue_functor.trace(example_tensors)
        return epilogue_functor

    def _render_argument_type(
        epilogue_functor: EpilogueFunctor,
        name_to_buffer: dict[str, Buffer],
        size_hint_fn: Callable[[Union[Expr, int]], int],
    ) -> tuple[str, EVTArgRenames]:
        epilogue_thread_type = epilogue_functor.epilogue_thread_type
        arg_renames = EVTArgRenames()

        # Fragile, but this is the only way to guarantee t is expected type because t is a local class
        def is_nested_visitor_type(t: type) -> bool:
            return (
                ".".join([t.__module__, t.__qualname__])
                == "cutlass_cppgen.backend.c_types.visitor_factory.<locals>.VisitorType"
            )

        buffer = IndentedBuffer()
        with buffer.set_tabwidth(2):

            def render_argument_type(name: str, t: CutlassArgType) -> None:
                if issubclass(t, ctypes.c_byte):
                    buffer.writeline(f"{{}}, /* {name} */")
                else:
                    fields = [
                        (
                            fname,
                            _get_arg_from_node(
                                ty, name_to_buffer[name], size_hint_fn, arg_renames
                            ),
                        )
                        for fname, ty in t._fields_
                    ]
                    field_strs = [
                        f"/* {fname} */ {str(field)}" for fname, field in fields
                    ]
                    buffer.writeline(f"{{{', '.join(field_strs)}}}, /* {name} */")

            def render_thread_type(name: str, t: CutlassArgType) -> None:
                if is_nested_visitor_type(t):
                    buffer.writeline(f"{{ /* {name} */")
                    with buffer.indent():
                        for name, inner_t in t._fields_:
                            render_thread_type(name, inner_t)
                    buffer.writeline("},")
                else:
                    render_argument_type(name, t)

            # unroll the recursion once to address special case formatting
            # namely, no ending comma and no indentation for the outermost thread type
            buffer.writeline("{ /* thread */")
            with buffer.indent(3):
                if is_nested_visitor_type(epilogue_thread_type):
                    with buffer.indent():
                        for name, inner_t in epilogue_thread_type._fields_:
                            render_thread_type(name, inner_t)
                else:
                    render_argument_type("thread", epilogue_thread_type)
                buffer.writeline("}")

        return buffer.getvalue(), arg_renames

    def _get_arg_from_node(
        arg_ty: type,
        node: Buffer,
        size_hint_fn: Callable[[Union[Expr, int]], int],
        arg_renames: EVTArgRenames,
    ) -> str:
        from ..cuda_template import CUTLASSTemplate

        # Today, arguments are either a pointer to the
        # node's memory, a stride tuple, the datatype
        # Once again, need to check for local class type for stride tuple
        if (
            str(arg_ty)
            == "<class 'cutlass_cppgen.backend.c_types.tuple_factory_.<locals>.TupleType'>"
        ):
            DEFAULT_STRIDE_LEN = 3
            assert len(node.get_layout().stride) <= DEFAULT_STRIDE_LEN
            stride = [size_hint_fn(x) for x in node.get_layout().stride]
            for _ in range(DEFAULT_STRIDE_LEN - len(stride)):
                stride.append(0)

            def render_stride(x: int) -> str:
                # Handle EBO for 0 and 1
                if x == 0:
                    return "_0{}"
                elif x == 1:
                    return "_1{}"
                else:
                    return str(x)

            return f"{{{', '.join([render_stride(x) for x in stride])}}}"

        elif issubclass(arg_ty, ctypes.c_void_p):
            name = arg_renames.new_name(node.get_name())
            return f"({CUTLASSTemplate._DTYPE_TO_CUTLASS[node.get_layout().dtype]}*) ({name} + {name}_offset)"
        elif (
            arg_ty in _CUTLASS_C_DTYPES
        ):  # Assumption: this is the element dtype, this holds for all cutlass ir nodes currently
            return f"{CUTLASSTemplate._DTYPE_TO_CUTLASS[node.get_layout().dtype]}(0)"
        elif issubclass(arg_ty, EmptyByte):
            return "{}"

        raise NotImplementedError(f"Unsupported arg type: {arg_ty}")
