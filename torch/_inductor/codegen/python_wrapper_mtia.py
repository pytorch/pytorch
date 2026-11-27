from typing_extensions import override

from torch._inductor import ir

from .wrapper import PythonWrapperCodegen


class PythonWrapperMtia(PythonWrapperCodegen):
    """
    A thin wrapper of PythonWrapperCodegen with MTIA specific logic
    """

    @override
    def write_header(self) -> None:
        super().write_header()

        # MITA specific imports
        self.imports.splice("import mtia.host_runtime.torch_mtia.dynamic_library")

    @override
    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: str | None,
        parent_wrapper: PythonWrapperCodegen | None,
        partition_signatures: ir.GraphPartitionSignature | None = None,
    ) -> PythonWrapperCodegen:
        if is_subgraph:
            # Delegate to the parent class to handle the case of subgraph
            return PythonWrapperCodegen.create(
                is_subgraph, subgraph_name, parent_wrapper, partition_signatures
            )
        return PythonWrapperMtia()
