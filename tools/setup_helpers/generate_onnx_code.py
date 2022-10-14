from tools.onnx import gen_diagnostics as diagnostics_generator

_RULES_PATH = "torch/onnx/_internal/diagnostics/rules.yaml"
_OUT_PY_DIRS = "torch/onnx/_internal/diagnostics/generated"
_OUT_CPP_DIRS = "torch/csrc/onnx/diagnostics/generated"
_TEMPLATE_DIR = "tools/onnx/templates"


def main() -> None:
    rules = diagnostics_generator.load_rules(_RULES_PATH)
    diagnostics_generator.gen_diagnostics_cpp(rules, _OUT_CPP_DIRS, _TEMPLATE_DIR)
    diagnostics_generator.gen_diagnostics_python(rules, _OUT_PY_DIRS, _TEMPLATE_DIR)


if __name__ == "__main__":
    main()
