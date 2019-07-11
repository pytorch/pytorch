from torch.utils.cpp_extension import load
from string import Template
import tempfile
from . import tensorextension

cpp_header = "torch/prototypes/nestedtensor/codegen/tensor_list.h"

cpp_source = open(cpp_header).read() + "\n"

def build_unary_functions():
    cpp_template = """
void tensor_list_${op}(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::${op}_out(out[i], input1[i]);
  }
}
    """
    cpp_template = Template(cpp_template)
    cpp_source = ""
    for op in tensorextension.get_unary_functions():
        cpp_source += cpp_template.substitute(op=op)
    return cpp_source + "\n"

def build_binary_functions():
    cpp_template = """
void tensor_list_${op}(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& input2,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::${op}_out(out[i], input1[i], input2[i]);
  }
}
    """
    cpp_template = Template(cpp_template)
    cpp_source = ""
    for op in tensorextension.get_binary_functions():
        cpp_source += cpp_template.substitute(op=op)
    return cpp_source + "\n"

def build_comparison_functions():
    cpp_template = """
void tensor_list_${op}(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& input2,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::${op}_out(out[i], input1[i], input2[i]);
  }
}
    """
    cpp_template = Template(cpp_template)
    cpp_source = ""
    for op in tensorextension.get_comparison_functions():
        cpp_source += cpp_template.substitute(op=op)
    return cpp_source + "\n"

cpp_source += build_unary_functions()
cpp_source += build_binary_functions()
cpp_source += build_comparison_functions()

def build_bindings():
    pybind_template = '  m.def("${op}", &tensor_list_${op}, "${op}");'
    pybind_template = Template(pybind_template)

    cpp_source = ""
    for op in tensorextension.get_unary_functions():
        cpp_source += pybind_template.substitute(op=op) + "\n"
    for op in tensorextension.get_binary_functions():
        cpp_source += pybind_template.substitute(op=op) + "\n"
    for op in tensorextension.get_comparison_functions():
        cpp_source += pybind_template.substitute(op=op) + "\n"
    return cpp_source

cpp_source += "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n"
cpp_source += build_bindings()
cpp_source += '}'

# TODO: Needs to happen in the build folder
with open("/tmp/tensor_list.cpp", "w") as source_out:
    source_out.write(cpp_source)
    tensor_list = load(name="tensor_list",
                       sources=[source_out.name],
                       verbose=True)
