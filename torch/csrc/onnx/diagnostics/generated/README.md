This folder contains generated sources for the ONNX exporter diagnostic rules.

The source of truth for the diagnostic rules is [../../../../onnx/_internal/diagnostics/rules.yaml](../../../../onnx/_internal/diagnostics/rules.yaml).

The code generator lives at `tools/onnx/gen_diagnostics.py`.

It is called automatically by the torch codegen (`tools/setup_helpers/generate_code.py`)
as a part of the build process.
