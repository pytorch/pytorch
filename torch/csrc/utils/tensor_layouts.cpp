#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/tensor_layouts.h>

namespace torch::utils {

static void registerLayout(
    PyObject* torch_module,
    at::Layout layout,
    const char* name,
    const char* qualified_name) {
  THPObjectPtr obj(THPLayout_New(layout, qualified_name));
  if (PyModule_AddObjectRef(torch_module, name, obj.get()) != 0) {
    throw python_error();
  }
  registerLayoutObject((THPLayout*)obj.get(), layout);
}

void initializeLayouts() {
  auto torch_module = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!torch_module)
    throw python_error();

  registerLayout(torch_module, at::Layout::Strided, "strided", "torch.strided");
  registerLayout(
      torch_module, at::Layout::Sparse, "sparse_coo", "torch.sparse_coo");
  registerLayout(
      torch_module, at::Layout::SparseCsr, "sparse_csr", "torch.sparse_csr");
  registerLayout(
      torch_module, at::Layout::SparseCsc, "sparse_csc", "torch.sparse_csc");
  registerLayout(
      torch_module, at::Layout::SparseBsr, "sparse_bsr", "torch.sparse_bsr");
  registerLayout(
      torch_module, at::Layout::SparseBsc, "sparse_bsc", "torch.sparse_bsc");
  registerLayout(torch_module, at::Layout::Mkldnn, "_mkldnn", "torch._mkldnn");
  registerLayout(torch_module, at::Layout::Jagged, "jagged", "torch.jagged");
}

} // namespace torch::utils
