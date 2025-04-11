# mypy: allow-untyped-defs

from torch._inductor.ir import ComputedBuffer, InputBuffer

from ..cutlass_utils import try_import_cutlass


if try_import_cutlass():
    import ast
    import ctypes
    import textwrap

    from cutlass.backend.evt import (  # type: ignore[import-untyped, import-not-found]
        EpilogueFunctorVisitor,
    )
    from cutlass.backend.evt.backend.emitter_base import (  # type: ignore[import-untyped, import-not-found]
        FusionCallbacks,
    )
    from cutlass.backend.evt.backend.sm90_emitter import (  # type: ignore[import-untyped, import-not-found]
        CollectiveEpilogue,
    )
    from cutlass.backend.evt.frontend import (  # type: ignore[import-untyped, import-not-found]
        PythonASTFrontend,
    )
    from cutlass.backend.evt.ir.tensor import (  # type: ignore[import-untyped, import-not-found]
        Tensor as CutlassTensor,
    )
    from cutlass_library import (
        DataType,
        EpilogueScheduleType,
        LayoutType,
        TileDescription,
    )

    from torch._inductor.codegen.cuda import cuda_env
    import torch
    from torch._inductor.utils import IndentedBuffer

    TORCH_TO_CUTLASS_DTYPE = {
        torch.float32: DataType.f32,
        torch.float16: DataType.f16,
        torch.bfloat16: DataType.bf16,
    }

    def create_example_tensors(
        read_names: list[str],
        write_names: list[str],
        buffer_renames: dict[str, str],
        name_to_buffer: dict[str, ComputedBuffer],
        name_to_input: dict[str, InputBuffer],
    ):
        example_tensors = {}

        def cutlass_tensor_from_buffer(buffer: ComputedBuffer):
            shape = tuple(int(x) for x in buffer.get_layout().size)
            stride = tuple(int(x) for x in buffer.get_layout().stride)

            is_column_major = True
            for i in range(1, len(shape)):
                if shape[i] == 1:
                    continue
                if stride[i] != stride[i - 1] * shape[i - 1]:
                    is_column_major = False

            is_row_major = True
            for i in range(len(shape) - 1):
                if shape[i] == 1:
                    continue

                if stride[i] != stride[i + 1] * shape[i + 1]:
                    is_row_major = False

            if not is_row_major and not is_column_major:
                raise RuntimeError(
                    f"Cannot create example tensor for {buffer.get_name()} with non-contiguous layout, recieved stride: {stride} and shape: {shape}"
                )

            return CutlassTensor(
                shape=shape,
                layout_tag=LayoutType.RowMajor
                if is_row_major
                else LayoutType.ColumnMajor,
                element=TORCH_TO_CUTLASS_DTYPE[buffer.get_layout().dtype],
            )

        def get_buffer(name):
            if name in name_to_buffer:
                return name_to_buffer[name]

            if name in name_to_input:
                return name_to_input[name]

            raise RuntimeError(
                f"Buffer {name} not found in name_to_buffer or name_to_input"
            )

        for name in read_names + write_names:
            key = name

            if name in buffer_renames:
                key = buffer_renames[
                    name
                ]  # Need to rewrite some special args (e.g. acc is a required arg name)

            example_tensors[key] = cutlass_tensor_from_buffer(get_buffer(name))

        return example_tensors

    def trace(
        fn_src: str,
        example_tensors: dict[str, CutlassTensor],
        accum_type: DataType,
        output_type: DataType,
        tile_description: TileDescription,
        epilogue_schedule: EpilogueScheduleType,
        **kwargs,
    ):
        cuda_arch = int(cuda_env.get_cuda_arch())  # type: ignore[arg-type]
        assert cuda_arch >= 90, "Only SM90+ is supported for EVT"
        epilogue_functor = _trace(fn_src, example_tensors, **kwargs)
        visitor = EpilogueFunctorVisitor(cuda_arch, epilogue_functor)
        fusion_callbacks = FusionCallbacks(visitor.graph, cuda_arch, emit_CD=False)
        collective_epilogue = CollectiveEpilogue(
            tile_description,
            epilogue_schedule,
            accum_type,
            output_type,
            fusion_callbacks,
        )

        return collective_epilogue.emit()

    # Based off of
    # https://github.com/NVIDIA/cutlass/blob/df18f5e4f5de76bed8be1de8e4c245f2f5ec3020/python/cutlass/epilogue/epilogue.py#L117
    # This is modified to enable directly passing the source code of the epilogue vs getting it from a bona-fide python function
    # The reason for this is that inspect.getsource does not work with functions defined at runtime via exec/eval
    def _trace(fn_src, example_tensors, **kwargs):
        class EpilogueFunctor(PythonASTFrontend):
            def __init__(self, **kwargs):
                self.source = textwrap.dedent(fn_src)
                super().__init__(**kwargs)

            def parse(self, example_inputs):
                self.example_inputs = example_inputs
                self.ast = ast.parse(self.source)
                self.visit(self.ast)

        epilogue_functor = EpilogueFunctor(**kwargs)
        epilogue_functor.trace(example_tensors)
        return epilogue_functor

    def _render_argument_type(epilogue_functor):
        epilogue_thread_type = epilogue_functor.epilogue_thread_type

        # Fragile, but this is the only way to guarantee t is expected type because t is a local class
        def is_nested_visitor_type(t):
            return (
                ".".join([t.__module__, t.__qualname__])
                == "cutlass.backend.c_types.visitor_factory.<locals>.VisitorType"
            )

        buffer = IndentedBuffer()

        def render_argument_type(name, t):
            fnames = []
            if issubclass(t, ctypes.c_byte):
                buffer.writeline(f"{{}}, /* {name} */")
            else:
                for fname, _ in t._fields_:
                    fnames.append(fname)
                buffer.writeline(f"{{{', '.join(fnames)}}}, /* {name} */")

        def render_thread_type(name, t):
            if is_nested_visitor_type(t):
                buffer.writeline(f"{{ /* {name} */")
                with buffer.indent():
                    for name, inner_t in t._fields_:
                        render_thread_type(name, inner_t)
                buffer.writeline("},")
            else:
                render_argument_type(name, t)

        buffer.writeline("{{")
        with buffer.indent():
            render_thread_type("thread", epilogue_thread_type)

        buffer.writeline("}};")

        return buffer.getvalue()
