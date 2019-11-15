h_method_dispatch_header = """
#include <torch/csrc/python_headers.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/nested_tensor/python_nested_tensor.h>
namespace torch {
namespace nested_tensor {
"""

h_method_dispatch_template = """
PyObject *_ListNestedTensorVariable_{method_name}(PyObject *self_);
"""

h_method_dispatch_footer = """
}
}
"""

cpp_method_dispatch_header = """
#include <torch/csrc/nested_tensor/generated/dispatch.h>
namespace torch {
namespace nested_tensor {
"""

cpp_method_dispatch_template = """
PyObject *_ListNestedTensorVariable_{method_name}(PyObject *self_) {{
  HANDLE_TH_ERRORS
  auto &self = reinterpret_cast<_ListNestedTensorVariable *>(self_)->cdata;
  return torch::autograd::utils::wrap(self.{method_name}());
  END_HANDLE_TH_ERRORS
}}
"""

cpp_method_dispatch_footer = """
}
}
"""

method_names = [
    "__repr__",
    "__str__",
    "detach",
    "dim",
    "element_size",
    "grad",
    "is_contiguous",
    "is_pinned",
    "numel",
    "pin_memory",
    "requires_grad",
    "to_tensor",
]


if __name__ == "__main__":
    with open("generated/dispatch.cpp", 'w') as f:
        f.write(cpp_method_dispatch_header)
        for method_name in method_names:
            f.write(cpp_method_dispatch_template.format(method_name=method_name))
        f.write(cpp_method_dispatch_footer)

    with open("generated/dispatch.h", 'w') as f:
        f.write(h_method_dispatch_header)
        for method_name in method_names:
            f.write(h_method_dispatch_template.format(method_name=method_name))
        f.write(h_method_dispatch_footer)
