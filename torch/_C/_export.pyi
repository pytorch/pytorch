# Defined in torch/csrc/export/pybind.cpp

class CppExportedProgram: ...

def deserialize_exported_program(
    serialized_program: str,
) -> CppExportedProgram: ...
def serialize_exported_program(
    cpp_exported_program: CppExportedProgram,
) -> str: ...
