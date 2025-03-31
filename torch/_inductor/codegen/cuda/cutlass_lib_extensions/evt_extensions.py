# mypy: allow-untyped-defs

from ..cutlass_utils import try_import_cutlass


if try_import_cutlass():
    import ast
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
    from cutlass_library import DataType, EpilogueScheduleType, TileDescription

    def generate(
        fn_src: str,
        example_tensors: dict[str, CutlassTensor],
        accum_type: DataType,
        output_type: DataType,
        tile_description: TileDescription,
        epilogue_schedule: EpilogueScheduleType,
        **kwargs,
    ):
        epilogue_functor = _trace(fn_src, example_tensors, **kwargs)
        visitor = EpilogueFunctorVisitor(90, epilogue_functor)
        fusion_callbacks = FusionCallbacks(visitor.graph, 90, emit_CD=False)
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
